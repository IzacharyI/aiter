#!/usr/bin/env python3
"""
Benchmark harness for Hunyun rope_norm_store_kv fusion.

This is intentionally separate from the correctness pytest.  GEAK can use it as
the performance gate after `op_tests/test_hunyun_rope_norm_store_kv.py` verifies
semantics.

Output format is line-oriented and easy to parse:

    BASELINE_CASE api=... mode=... ... median_us=123.45
    TARGET_CASE api=... mode=... ... median_us=12.34
    BASELINE_SUM_US=1234.56
    PERF_SUM_US=123.45

If the target `hpc` API is missing, baseline still runs and the script prints
TARGET_MISSING=1.  Lower PERF_SUM_US is better when TARGET_MISSING=0.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


def _add_build_lib_to_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    build_libs = sorted((repo / "build").glob("lib.*"))
    for path in reversed(build_libs):
        sys.path.insert(0, str(path))


_add_build_lib_to_path()

try:
    import hpc  # type: ignore
except ModuleNotFoundError:
    hpc = None  # type: ignore


@dataclass(frozen=True)
class BenchCase:
    api: str
    mode: str
    num_req: int
    num_q_heads: int
    num_kv_heads: int
    qk_head_dim: int
    qk_norm_policy: int
    quant_policy: int | None = None


def generate_cos_sin_cache(max_position: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_position).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def generate_kv_block_indices(kcache: torch.Tensor, req_length: list[int]) -> torch.Tensor:
    kv_block_size = kcache.shape[1]
    num_blocks_per_req = [(length + kv_block_size - 1) // kv_block_size for length in req_length]
    shuffled = torch.randperm(kcache.shape[0])
    kv_idx = torch.ones(len(req_length), max(num_blocks_per_req) + 4, dtype=torch.int32) * -1
    offset = 0
    for i, num_blocks in enumerate(num_blocks_per_req):
        kv_idx[i, :num_blocks] = shuffled[offset : offset + num_blocks]
        offset += num_blocks
    return kv_idx


def apply_rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def apply_rotary_pos_emb_neox_reference(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos_sin[:, :half].unsqueeze(1)
    sin = cos_sin[:, half:].unsqueeze(1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def hadamard_matrix(n: int, device, dtype=torch.float32) -> torch.Tensor:
    matrix = torch.tensor([[1.0]], device=device, dtype=dtype)
    size = 1
    while size < n:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
        size *= 2
    return matrix


def apply_hadamard_per_head(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    matrix = hadamard_matrix(head_dim, x.device, dtype=torch.float32)
    return torch.matmul(x.to(torch.float32), matrix.t()) * (1.0 / math.sqrt(head_dim))


def pad_decode_inputs_to_align8(
    qkv: torch.Tensor,
    num_seqlen: torch.Tensor,
    q_index: torch.Tensor,
    kv_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nr = qkv.shape[0]
    nb = num_seqlen.shape[0]
    pb = (nb + 7) // 8 * 8
    pr = (nr + 7) // 8 * 8

    if pr > nr:
        qkv = torch.cat([qkv, torch.zeros(pr - nr, qkv.shape[1], dtype=qkv.dtype, device=qkv.device)])
    if pb > nb:
        num_seqlen = torch.cat([num_seqlen, torch.zeros(pb - nb, dtype=num_seqlen.dtype, device=num_seqlen.device)])
        q_index = torch.cat([q_index, torch.full((pb - nb,), pr, dtype=q_index.dtype, device=q_index.device)])
        kv_indices = torch.cat(
            [
                kv_indices,
                torch.zeros(pb - nb, kv_indices.shape[1], dtype=kv_indices.dtype, device=kv_indices.device),
            ]
        )
    return qkv, num_seqlen, q_index, kv_indices


def prepare_inputs(
    case: BenchCase,
    *,
    kv_block_size: int = 64,
    max_num_kv_blocks: int = 1024,
    max_rope_position: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[torch.Tensor, ...]:
    v_head_dim = case.qk_head_dim
    hidden = (
        case.num_q_heads * case.qk_head_dim
        + case.num_kv_heads * case.qk_head_dim
        + case.num_kv_heads * v_head_dim
    )

    cos_sin = generate_cos_sin_cache(max_rope_position, case.qk_head_dim).to(dtype=torch.float32, device=device)
    kcache = torch.randn(
        max_num_kv_blocks, kv_block_size, case.num_kv_heads, case.qk_head_dim, dtype=dtype, device=device
    )
    vcache = torch.randn(max_num_kv_blocks, kv_block_size, case.num_kv_heads, v_head_dim, dtype=dtype, device=device)
    q_norm_w = torch.randn(case.qk_head_dim, dtype=torch.float32, device=device)
    k_norm_w = torch.randn(case.qk_head_dim, dtype=torch.float32, device=device)

    real_rows: int | None = None
    if case.mode == "prefill":
        req_len = torch.randint(20, 200, (case.num_req,)).tolist()
        qkv_full = torch.randn(sum(req_len), hidden, dtype=dtype, device=device)
        req_len_t = torch.tensor(req_len, device=device)
        q_len_t = torch.min((torch.rand(case.num_req, device=device) * req_len_t).long() + 1, req_len_t)
        cumsum = torch.cumsum(req_len_t, dim=0)
        qkv = torch.cat([qkv_full[cumsum[i] - q_len_t[i] : cumsum[i]] for i in range(case.num_req)])
        q_index = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), torch.cumsum(q_len_t, 0)]).to(
            torch.int32
        )
        num_seqlen = torch.tensor(req_len, dtype=torch.int32, device=device)
        kv_indices = generate_kv_block_indices(kcache, req_len).to(device)
    else:
        mtp = 0 if case.mode == "decode" else 1
        tokens_per_req = mtp + 1
        existing_len = torch.randint(20, 200, (case.num_req,)).tolist()
        updated_len = [x + tokens_per_req for x in existing_len]
        qkv = torch.randn(case.num_req * tokens_per_req, hidden, dtype=dtype, device=device)
        q_index = torch.arange(0, (case.num_req + 1) * tokens_per_req, tokens_per_req, device=device, dtype=torch.int32)
        num_seqlen = torch.tensor(updated_len, dtype=torch.int32, device=device)
        kv_indices = generate_kv_block_indices(kcache, updated_len).to(device)
        real_rows = qkv.shape[0]
        qkv, num_seqlen, q_index, kv_indices = pad_decode_inputs_to_align8(qkv, num_seqlen, q_index, kv_indices)

    return qkv, num_seqlen, q_index, kcache, vcache, kv_indices, q_norm_w, k_norm_w, cos_sin, real_rows


def rope_norm_reference(
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    num_seqlen: torch.Tensor,
    q_index: torch.Tensor,
    kv_indices: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    qk_norm_policy: int,
    apply_hadamard: bool,
) -> torch.Tensor:
    dtype = qkv.dtype
    num_kv = kcache.shape[2]
    v_dim = vcache.shape[3]
    qk_dim = kcache.shape[3]
    num_q = (qkv.shape[1] - num_kv * qk_dim - num_kv * v_dim) // qk_dim
    num_req = num_seqlen.shape[0]
    q_lens = (q_index[1:] - q_index[:-1]).tolist()
    num_rows = int(q_index[-1].item())
    block_size = kcache.shape[1]

    q = qkv[:, : num_q * qk_dim].to(torch.float32).view(num_rows, num_q, qk_dim)
    k = qkv[:, num_q * qk_dim : (num_q + num_kv) * qk_dim].to(torch.float32).view(num_rows, num_kv, qk_dim)
    v = qkv[:, (num_q + num_kv) * qk_dim :].view(num_rows, num_kv, v_dim)

    token_cos_sin = torch.zeros(num_rows, qk_dim, dtype=torch.float32, device=qkv.device)
    offset = 0
    for req_idx in range(num_req):
        seq_len = int(num_seqlen[req_idx].item())
        q_len = int(q_lens[req_idx])
        if q_len > 0:
            token_cos_sin[offset : offset + q_len] = cos_sin[seq_len - q_len : seq_len]
        offset += q_len

    if qk_norm_policy == 2:
        q = apply_rms_norm_reference(q, q_norm_w)
        k = apply_rms_norm_reference(k, k_norm_w)
    q = apply_rotary_pos_emb_neox_reference(q, token_cos_sin)
    k = apply_rotary_pos_emb_neox_reference(k, token_cos_sin)
    if qk_norm_policy == 1:
        q = apply_rms_norm_reference(q, q_norm_w)
        k = apply_rms_norm_reference(k, k_norm_w)

    if apply_hadamard:
        q = apply_hadamard_per_head(q, qk_dim)
        k = apply_hadamard_per_head(k, qk_dim)

    token = 0
    for req_idx in range(num_req):
        seq_len = int(num_seqlen[req_idx].item())
        q_len = int(q_lens[req_idx])
        for pos in range(seq_len - q_len, seq_len):
            block_idx = pos // block_size
            pos_in_block = pos % block_size
            cache_block = int(kv_indices[req_idx, block_idx].item())
            kcache[cache_block, pos_in_block] = k[token].to(dtype)
            vcache[cache_block, pos_in_block] = v[token].to(dtype)
            if pos == seq_len - 1 and pos_in_block + 1 < block_size:
                kcache[cache_block, pos_in_block + 1 :] = 0
                vcache[cache_block, pos_in_block + 1 :] = 0
            token += 1

    return q.to(dtype)


def make_baseline_call(case: BenchCase):
    qkv, num_seqlen, q_index, kcache, vcache, kv_indices, q_norm_w, k_norm_w, cos_sin, real_rows = prepare_inputs(case)

    if real_rows is not None:
        qkv_ref = qkv[:real_rows]
        num_seqlen_ref = num_seqlen[: case.num_req]
        q_index_ref = q_index[: case.num_req + 1]
        kv_indices_ref = kv_indices[: case.num_req]
    else:
        qkv_ref, num_seqlen_ref, q_index_ref, kv_indices_ref = qkv, num_seqlen, q_index, kv_indices

    def call():
        kcache_work = kcache.clone()
        vcache_work = vcache.clone()
        q_out = rope_norm_reference(
            kcache_work,
            vcache_work,
            qkv_ref,
            cos_sin,
            num_seqlen_ref,
            q_index_ref,
            kv_indices_ref,
            q_norm_w,
            k_norm_w,
            case.qk_norm_policy,
            apply_hadamard=case.quant_policy == 3,
        )
        if case.api == "bf16":
            return q_out, kcache_work, vcache_work

        quant_policy = int(case.quant_policy or 0)
        q_fp8 = q_out.to(torch.float8_e4m3fn)
        if quant_policy in (0, 3):
            v_scale = torch.rand(case.num_kv_heads, dtype=torch.float32, device=qkv.device) * 0.2 + 0.05
            kcache_fp8 = kcache_work.to(torch.float8_e4m3fn)
            vcache_fp8 = vcache_work.to(torch.float8_e4m3fn)
            return q_fp8, kcache_fp8, vcache_fp8, v_scale
        k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
        v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
        kcache_fp8 = (kcache_work / k_scale[0]).to(torch.float8_e4m3fn)
        vcache_fp8 = (vcache_work / v_scale[0]).to(torch.float8_e4m3fn)
        return q_fp8, kcache_fp8, vcache_fp8

    return call


def make_target_call(case: BenchCase):
    if hpc is None:
        return None

    qkv, num_seqlen, q_index, kcache, vcache, kv_indices, q_norm_w, k_norm_w, cos_sin, _real_rows = prepare_inputs(case)

    if case.api == "bf16":
        def call():
            return hpc.rope_norm_store_kv(
                kcache,
                vcache,
                qkv,
                cos_sin,
                num_seqlen,
                q_index,
                kv_indices,
                case.mode == "prefill",
                q_norm_weight=q_norm_w if case.qk_norm_policy > 0 else None,
                k_norm_weight=k_norm_w if case.qk_norm_policy > 0 else None,
                qk_norm_policy=case.qk_norm_policy,
            )

        return call

    quant_policy = int(case.quant_policy or 0)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)
    kv_block_size = kcache.shape[1]
    if quant_policy in (0, 3):
        scale_l = case.qk_head_dim // 4
        scale_r = kv_block_size // scale_l
        k_scale = torch.zeros(kcache.shape[0], scale_r, case.num_kv_heads, scale_l, dtype=torch.float32, device=qkv.device)
        v_scale = torch.rand(case.num_kv_heads, dtype=torch.float32, device=qkv.device) * 0.2 + 0.05
    else:
        k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
        v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)

    if case.mode == "prefill":
        max_seqlens = int((q_index[1:] - q_index[:-1]).max().item())
    else:
        max_seqlens = 1 if case.mode == "decode" else 2

    q_scale_inv = torch.tensor([0.5], dtype=torch.float32, device=qkv.device) if quant_policy == 2 else None

    def call():
        return hpc.rope_norm_store_kv_fp8(
            key_cache=kcache_fp8,
            value_cache=vcache_fp8,
            qkv=qkv,
            cos_sin=cos_sin,
            num_seqlen_per_req=num_seqlen,
            q_index=q_index,
            kvcache_indices=kv_indices,
            is_prefill=case.mode == "prefill",
            k_scale=k_scale,
            v_scale=v_scale,
            quant_policy=hpc.QuantType(quant_policy),
            max_seqlens=max_seqlens,
            q_scale_inv=q_scale_inv,
            q_norm_weight=q_norm_w if case.qk_norm_policy > 0 else None,
            k_norm_weight=k_norm_w if case.qk_norm_policy > 0 else None,
            qk_norm_policy=case.qk_norm_policy,
        )

    return call


def synchronize() -> None:
    torch.cuda.synchronize()


def benchmark_call(call, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        call()
    synchronize()

    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        call()
        end.record()
        synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def quick_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []
    for num_q_heads, num_kv_heads in [(8, 1), (64, 8)]:
        cases.extend(
            [
                BenchCase("bf16", "prefill", 16, num_q_heads, num_kv_heads, 128, 0),
                BenchCase("bf16", "decode", 16, num_q_heads, num_kv_heads, 128, 1),
                BenchCase("bf16", "mtp", 16, num_q_heads, num_kv_heads, 128, 2),
                BenchCase("fp8", "prefill", 16, num_q_heads, num_kv_heads, 128, 1, 0),
                BenchCase("fp8", "decode", 16, num_q_heads, num_kv_heads, 128, 2, 2),
                BenchCase("fp8", "mtp", 16, num_q_heads, num_kv_heads, 128, 1, 3),
            ]
        )
    return cases


def full_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []
    for num_req in [7, 16]:
        for num_q_heads, num_kv_heads in [(8, 1), (64, 8)]:
            for mode in ["prefill", "decode", "mtp"]:
                for qk_norm_policy in [0, 1, 2]:
                    cases.append(BenchCase("bf16", mode, num_req, num_q_heads, num_kv_heads, 128, qk_norm_policy))
                    for quant_policy in [0, 1, 2, 3]:
                        cases.append(
                            BenchCase(
                                "fp8",
                                mode,
                                num_req,
                                num_q_heads,
                                num_kv_heads,
                                128,
                                qk_norm_policy,
                                quant_policy,
                            )
                        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["quick", "full"], default="quick")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cases = quick_cases() if args.suite == "quick" else full_cases()
    baseline_total = 0.0
    target_total = 0.0
    target_missing = hpc is None
    print(f"=== rope_norm_store_kv benchmark | suite={args.suite} | cases={len(cases)} ===", flush=True)
    if target_missing:
        print("TARGET_MISSING=1 reason=no_hpc_module", flush=True)
    for case in cases:
        baseline_call = make_baseline_call(case)
        baseline_us = benchmark_call(baseline_call, args.warmup, args.iters)
        baseline_total += baseline_us
        print(
            "BASELINE_CASE "
            f"api={case.api} mode={case.mode} num_req={case.num_req} "
            f"q_heads={case.num_q_heads} kv_heads={case.num_kv_heads} dim={case.qk_head_dim} "
            f"norm={case.qk_norm_policy} quant={case.quant_policy if case.quant_policy is not None else 'NA'} "
            f"median_us={baseline_us:.2f}",
            flush=True,
        )

        target_call = make_target_call(case)
        if target_call is None:
            continue
        target_us = benchmark_call(target_call, args.warmup, args.iters)
        target_total += target_us
        speedup = baseline_us / target_us if target_us > 0 else 0.0
        print(
            "TARGET_CASE "
            f"api={case.api} mode={case.mode} num_req={case.num_req} "
            f"q_heads={case.num_q_heads} kv_heads={case.num_kv_heads} dim={case.qk_head_dim} "
            f"norm={case.qk_norm_policy} quant={case.quant_policy if case.quant_policy is not None else 'NA'} "
            f"median_us={target_us:.2f} speedup={speedup:.4f}",
            flush=True,
        )

    print(f"BASELINE_SUM_US={baseline_total:.2f}", flush=True)
    if target_missing:
        print("PERF_SUM_US=inf", flush=True)
    else:
        print(f"PERF_SUM_US={target_total:.2f}", flush=True)
        if target_total > 0:
            print(f"SPEEDUP={baseline_total / target_total:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
