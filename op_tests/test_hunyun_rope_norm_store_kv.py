#!/usr/bin/env python3
"""
Correctness tests for Hunyun rope_norm_store_kv fusion.

This file is adapted from the ticket attachment and focuses on the new APIs:

- hpc.rope_norm_store_kv
- hpc.rope_norm_store_kv_fp8

It intentionally avoids the attachment's brittle `../build/lib.*/[0]` import
path.  Instead it tries repo-local build/lib paths first, then normal Python
import resolution.  If `hpc` is not implemented yet, pytest collection still
succeeds and the tests fail with a clear message.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest
import torch


def _add_build_lib_to_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    build_libs = sorted((repo / "build").glob("lib.*"))
    for path in reversed(build_libs):
        sys.path.insert(0, str(path))


def import_hpc():
    _add_build_lib_to_path()
    try:
        return importlib.import_module("hpc")
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Unable to import hpc. Implement/build the hpc extension that exposes "
            "rope_norm_store_kv, rope_norm_store_kv_fp8, and QuantType.",
            pytrace=False,
        )
        raise exc


def allclose(ref_tensor, real_tensor, atol=1e-8, rtol=1e-5):
    assert ref_tensor.dtype == real_tensor.dtype
    assert ref_tensor.device == real_tensor.device
    assert ref_tensor.shape == real_tensor.shape
    return torch.allclose(ref_tensor.to(torch.float32), real_tensor.to(torch.float32), atol=atol, rtol=rtol)


def generate_cos_sin_cache(max_position, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_position).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def generate_kv_block_indices(kcache, req_length):
    kv_block_size = kcache.shape[1]
    num_blocks_per_req = [(length + kv_block_size - 1) // kv_block_size for length in req_length]
    shuffled = torch.randperm(kcache.shape[0])
    kv_idx = torch.ones(len(req_length), max(num_blocks_per_req) + 4, dtype=torch.int32) * -1
    offset = 0
    for i, num_blocks in enumerate(num_blocks_per_req):
        kv_idx[i, :num_blocks] = shuffled[offset : offset + num_blocks]
        offset += num_blocks
    return kv_idx


def apply_rms_norm_reference(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def apply_rotary_pos_emb_neox_reference(x, cos_sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos_sin[:, :half].unsqueeze(1)
    sin = cos_sin[:, half:].unsqueeze(1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def hadamard_matrix(n, device, dtype=torch.float32):
    """Build an unscaled Hadamard matrix of size n, where n is a power of two."""
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


def apply_hadamard_per_head(x, head_dim):
    matrix = hadamard_matrix(head_dim, x.device, dtype=torch.float32)
    return torch.matmul(x.to(torch.float32), matrix.t()) * (1.0 / math.sqrt(head_dim))


def rope_norm_ref(
    kcache,
    vcache,
    qkv,
    cos_sin,
    num_seqlen_per_req,
    q_index,
    kv_indices,
    q_norm_weight,
    k_norm_weight,
    qk_norm_policy,
    apply_hadamard=False,
):
    """PyTorch reference: reshape + optional RMSNorm + RoPE + optional Hadamard + paged KV write."""
    dtype = qkv.dtype
    num_kv = kcache.shape[2]
    v_dim = vcache.shape[3]
    qk_dim = kcache.shape[3]
    num_q = (qkv.shape[1] - num_kv * qk_dim - num_kv * v_dim) // qk_dim
    num_req = num_seqlen_per_req.shape[0]
    q_lens = (q_index[1:] - q_index[:-1]).tolist()
    num_rows = q_index[-1].item()
    block_size = kcache.shape[1]

    q = qkv[:, : num_q * qk_dim].to(torch.float32).view(num_rows, num_q, qk_dim)
    k = qkv[:, num_q * qk_dim : (num_q + num_kv) * qk_dim].to(torch.float32).view(num_rows, num_kv, qk_dim)
    v = qkv[:, (num_q + num_kv) * qk_dim :].view(num_rows, num_kv, v_dim)

    token_cos_sin = torch.zeros(num_rows, qk_dim, dtype=torch.float32, device=qkv.device)
    offset = 0
    for req_idx in range(num_req):
        seq_len = num_seqlen_per_req[req_idx].item()
        q_len = q_lens[req_idx]
        if q_len > 0:
            token_cos_sin[offset : offset + q_len] = cos_sin[seq_len - q_len : seq_len]
        offset += q_len

    if qk_norm_policy == 2:
        q = apply_rms_norm_reference(q, q_norm_weight)
        k = apply_rms_norm_reference(k, k_norm_weight)
    q = apply_rotary_pos_emb_neox_reference(q, token_cos_sin)
    k = apply_rotary_pos_emb_neox_reference(k, token_cos_sin)
    if qk_norm_policy == 1:
        q = apply_rms_norm_reference(q, q_norm_weight)
        k = apply_rms_norm_reference(k, k_norm_weight)

    if apply_hadamard:
        q = apply_hadamard_per_head(q, qk_dim)
        k = apply_hadamard_per_head(k, qk_dim)

    token = 0
    for req_idx in range(num_req):
        seq_len = int(num_seqlen_per_req[req_idx].item())
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


def pad_decode_inputs_to_align8(qkv, num_seqlen, q_index, kv_indices):
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
    return qkv, num_seqlen, q_index, kv_indices, nr


def prepare_inputs(
    num_req,
    is_prefill,
    mtp,
    num_q_heads,
    num_kv_heads,
    qk_head_dim,
    v_head_dim=None,
    kv_block_size=64,
    max_num_kv_blocks=1024,
    max_rope_position=2048,
    dtype=torch.bfloat16,
    device="cuda",
):
    if v_head_dim is None:
        v_head_dim = qk_head_dim
    hidden = num_q_heads * qk_head_dim + num_kv_heads * qk_head_dim + num_kv_heads * v_head_dim

    cos_sin = generate_cos_sin_cache(max_rope_position, qk_head_dim).to(dtype=torch.float32, device=device)
    kcache = torch.randn(max_num_kv_blocks, kv_block_size, num_kv_heads, qk_head_dim, dtype=dtype, device=device)
    vcache = torch.randn(max_num_kv_blocks, kv_block_size, num_kv_heads, v_head_dim, dtype=dtype, device=device)
    q_norm_w = torch.randn(qk_head_dim, dtype=torch.float32, device=device)
    k_norm_w = torch.randn(qk_head_dim, dtype=torch.float32, device=device)

    if is_prefill:
        req_len = torch.randint(20, 200, (num_req,)).tolist()
        qkv_full = torch.randn(sum(req_len), hidden, dtype=dtype, device=device)
        req_len_t = torch.tensor(req_len, device=device)
        q_len_t = torch.min((torch.rand(num_req, device=device) * req_len_t).long() + 1, req_len_t)
        cumsum = torch.cumsum(req_len_t, dim=0)
        qkv = torch.cat([qkv_full[cumsum[i] - q_len_t[i] : cumsum[i]] for i in range(num_req)])
        q_index = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), torch.cumsum(q_len_t, 0)]).to(
            torch.int32
        )
        num_seqlen = torch.tensor(req_len, dtype=torch.int32, device=device)
        kv_indices = generate_kv_block_indices(kcache, req_len).to(device)
        real_rows = None
    else:
        tokens_per_req = mtp + 1
        existing_len = torch.randint(20, 200, (num_req,)).tolist()
        updated_len = [x + tokens_per_req for x in existing_len]
        qkv_raw = torch.randn(num_req * tokens_per_req, hidden, dtype=dtype, device=device)
        q_index_raw = torch.arange(0, (num_req + 1) * tokens_per_req, tokens_per_req, device=device, dtype=torch.int32)
        num_seqlen_raw = torch.tensor(updated_len, dtype=torch.int32, device=device)
        kv_indices_raw = generate_kv_block_indices(kcache, updated_len).to(device)
        qkv, num_seqlen, q_index, kv_indices, real_rows = pad_decode_inputs_to_align8(
            qkv_raw, num_seqlen_raw, q_index_raw, kv_indices_raw
        )

    return qkv, num_seqlen, q_index, kcache, vcache, kv_indices, q_norm_w, k_norm_w, cos_sin, real_rows


@pytest.mark.parametrize("num_q_heads,num_kv_heads,qk_head_dim", [(8, 1, 128), (64, 8, 128)])
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("num_req", [7, 16])
@pytest.mark.parametrize("is_prefill,mtp", [(True, 0), (False, 0), (False, 1)])
def test_rope_norm_store_kv(num_q_heads, num_kv_heads, qk_head_dim, qk_norm_policy, num_req, is_prefill, mtp):
    hpc = import_hpc()
    qkv, num_seqlen, q_index, kcache, vcache, kv_indices, q_norm_w, k_norm_w, cos_sin, real_rows = prepare_inputs(
        num_req, is_prefill, mtp, num_q_heads, num_kv_heads, qk_head_dim
    )
    kcache_ref, vcache_ref = kcache.clone(), vcache.clone()

    out_q = hpc.rope_norm_store_kv(
        kcache,
        vcache,
        qkv,
        cos_sin,
        num_seqlen,
        q_index,
        kv_indices,
        is_prefill,
        q_norm_weight=q_norm_w if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_w if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    if real_rows is not None:
        qkv_ref, num_seqlen_ref = qkv[:real_rows], num_seqlen[:num_req]
        q_index_ref, kv_indices_ref = q_index[: num_req + 1], kv_indices[:num_req]
    else:
        qkv_ref, num_seqlen_ref, q_index_ref, kv_indices_ref = qkv, num_seqlen, q_index, kv_indices

    ref_q = rope_norm_ref(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_ref,
        q_index_ref,
        kv_indices_ref,
        q_norm_w,
        k_norm_w,
        qk_norm_policy,
    )

    rows = real_rows if real_rows is not None else out_q.shape[0]
    assert allclose(ref_q, out_q[:rows], atol=8e-2)
    assert allclose(kcache_ref, kcache, atol=8e-2)
    assert allclose(vcache_ref, vcache, atol=8e-2)


@pytest.mark.parametrize("num_q_heads,num_kv_heads,qk_head_dim", [(8, 1, 128), (64, 8, 128)])
@pytest.mark.parametrize("qk_norm_policy", [0, 1, 2])
@pytest.mark.parametrize("quant_policy", [0, 1, 2, 3])
@pytest.mark.parametrize("num_req", [7, 16])
@pytest.mark.parametrize("is_prefill,mtp", [(True, 0), (False, 0), (False, 1)])
def test_rope_norm_store_kv_fp8(
    num_q_heads,
    num_kv_heads,
    qk_head_dim,
    qk_norm_policy,
    quant_policy,
    num_req,
    is_prefill,
    mtp,
):
    hpc = import_hpc()
    qkv, num_seqlen, q_index, kcache, vcache, kv_indices, q_norm_w, k_norm_w, cos_sin, real_rows = prepare_inputs(
        num_req, is_prefill, mtp, num_q_heads, num_kv_heads, qk_head_dim
    )
    kcache_ref, vcache_ref = kcache.clone(), vcache.clone()

    kv_block_size = kcache.shape[1]
    if quant_policy in (0, 3):
        scale_l = qk_head_dim // 4
        scale_r = kv_block_size // scale_l
        k_scale = torch.zeros(kcache.shape[0], scale_r, num_kv_heads, scale_l, dtype=torch.float32, device=qkv.device)
        v_scale = torch.rand(num_kv_heads, dtype=torch.float32, device=qkv.device) * 0.2 + 0.05
    else:
        k_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)
        v_scale = torch.tensor([0.1], dtype=torch.float32, device=qkv.device)

    q_scale_val = 2.0
    q_scale_inv = torch.tensor([1.0 / q_scale_val], dtype=torch.float32, device=qkv.device)
    kcache_fp8 = kcache.to(torch.float8_e4m3fn)
    vcache_fp8 = vcache.to(torch.float8_e4m3fn)
    max_seqlens = int((q_index[1:] - q_index[:-1]).max().item()) if is_prefill else mtp + 1

    q_fp8, q_scale_out, split_k_flag = hpc.rope_norm_store_kv_fp8(
        key_cache=kcache_fp8,
        value_cache=vcache_fp8,
        qkv=qkv,
        cos_sin=cos_sin,
        num_seqlen_per_req=num_seqlen,
        q_index=q_index,
        kvcache_indices=kv_indices,
        is_prefill=is_prefill,
        k_scale=k_scale,
        v_scale=v_scale,
        quant_policy=hpc.QuantType(quant_policy),
        max_seqlens=max_seqlens,
        q_scale_inv=q_scale_inv if quant_policy == 2 else None,
        q_norm_weight=q_norm_w if qk_norm_policy > 0 else None,
        k_norm_weight=k_norm_w if qk_norm_policy > 0 else None,
        qk_norm_policy=qk_norm_policy,
    )

    assert split_k_flag.shape == (num_seqlen.shape[0], num_kv_heads)
    assert split_k_flag.dtype == torch.int32

    if quant_policy in (0, 1, 3):
        if is_prefill:
            pad128 = ((max_seqlens + 127) // 128) * 128
            assert q_scale_out.shape == (num_seqlen.shape[0], num_q_heads, pad128)
            seqlens = (q_index[1:] - q_index[:-1]).to(qkv.device)
            mask = torch.arange(pad128, device=qkv.device).expand(num_seqlen.shape[0], pad128) < seqlens.unsqueeze(1)
            scale_flat = q_scale_out.permute(0, 2, 1)[mask]
            rows = int(q_index[-1].item())
            q_bf16 = (q_fp8[:rows].to(torch.bfloat16) * scale_flat[:, :, None]).to(torch.bfloat16)
        else:
            assert q_scale_out.shape == (qkv.shape[0], num_q_heads)
            rows = real_rows
            q_bf16 = (q_fp8[:rows].to(torch.bfloat16) * q_scale_out[:rows, :, None]).to(torch.bfloat16)
    else:
        assert q_scale_out is None
        rows = real_rows if real_rows is not None else q_fp8.shape[0]
        q_bf16 = (q_fp8[:rows].to(torch.float32) * q_scale_val).to(torch.bfloat16)

    if real_rows is not None:
        qkv_ref, num_seqlen_ref = qkv[:real_rows], num_seqlen[:num_req]
        q_index_ref, kv_indices_ref = q_index[: num_req + 1], kv_indices[:num_req]
    else:
        qkv_ref, num_seqlen_ref, q_index_ref, kv_indices_ref = qkv, num_seqlen, q_index, kv_indices

    ref_q = rope_norm_ref(
        kcache_ref,
        vcache_ref,
        qkv_ref,
        cos_sin,
        num_seqlen_ref,
        q_index_ref,
        kv_indices_ref,
        q_norm_w,
        k_norm_w,
        qk_norm_policy,
        apply_hadamard=(quant_policy == 3),
    )
    assert allclose(ref_q, q_bf16, atol=0.8)

    q_lens_ref = (q_index_ref[1:] - q_index_ref[:-1]).tolist()
    scale_l = qk_head_dim // 4 if quant_policy in (0, 3) else None
    token = 0
    for req_idx in range(num_req):
        seq_len = int(num_seqlen_ref[req_idx].item())
        q_len = int(q_lens_ref[req_idx])
        for pos in range(seq_len - q_len, seq_len):
            block_idx = pos // kv_block_size
            pos_in_block = pos % kv_block_size
            cache_block = int(kv_indices_ref[req_idx, block_idx].item())
            for head in range(num_kv_heads):
                v_fp8 = vcache_fp8[cache_block, pos_in_block, head].to(torch.float32)
                v_ref = vcache_ref[cache_block, pos_in_block, head].to(torch.float32)
                v_dequant = v_fp8 * (v_scale[head] if quant_policy in (0, 3) else v_scale[0])
                assert allclose(v_ref, v_dequant, atol=0.8)

                k_fp8 = kcache_fp8[cache_block, pos_in_block, head].to(torch.float32)
                k_ref = kcache_ref[cache_block, pos_in_block, head].to(torch.float32)
                if quant_policy in (0, 3):
                    assert scale_l is not None
                    k_dequant = k_fp8 * k_scale[cache_block, pos_in_block // scale_l, head, pos_in_block % scale_l]
                else:
                    k_dequant = k_fp8 * k_scale[0]
                assert allclose(k_ref, k_dequant, atol=0.8)
            token += 1
