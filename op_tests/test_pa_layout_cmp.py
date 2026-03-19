# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# KV Cache Layout Comparison Benchmark
#
# Fair comparison at the same block_size:
#   R) paged_attention_ragged + V 4D + BF16 KV (SGLang actual decode path)
#   A) paged_attention_rocm   + V 4D + FP8 KV  (vLLM v0.x layout, baseline)
#   B) paged_attention_rocm   + V 5D + FP8 KV  (preshuffle layout)
#   C) pa_fwd_asm             + V 5D + FP8 KV  (preshuffle + ASM, non-persistent)
#   D) pa_persistent_fwd     + V 5D + FP8 KV  (preshuffle + ASM, persistent)
#
# NOTE: Test R uses BF16 KV because paged_attention_ragged does not yet
#       support FP8 KV.  SGLang converts FP8 → BF16 before calling it
#       (see TODO in aiter_backend.py).  So R reflects the real SGLang path
#       (including the 2x KV bandwidth penalty vs FP8 tests).
#
# speedup(A/R) = FP8 rocm vs BF16 ragged (real SGLang gap)
# speedup(B/A) = pure V layout gain within the same kernel
# speedup(C/A) = V layout + ASM kernel gain (non-persistent)
# speedup(D/A) = V layout + ASM kernel + persistent scheduling gain
# speedup(D/C) = pure persistent scheduling gain (same ASM kernel type)

import argparse
import itertools
import random
from typing import List, Tuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter import pertoken_quant
from aiter.test_common import benchmark, checkAllclose, perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

uniform_range = (-1, 1)

_PARTITION_SIZE_ROCM = 256


# ---------------------------------------------------------------------------
# Data preparation utilities (mirrored from test_pa_ps.py)
# ---------------------------------------------------------------------------

def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    model_dtype: torch.dtype,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create BF16/FP16 KV caches in vLLM v0.x storage format.

    Returns:
        k_cache: [num_blocks, num_heads, head_size // x, block_size, x]  (5D)
        v_cache: [num_blocks, num_heads, head_size, block_size]          (4D)
    """
    x = 16 // model_dtype.itemsize
    k_cache = torch.empty(
        (num_blocks, num_heads, head_size // x, block_size, x),
        dtype=model_dtype, device=device,
    )
    k_cache.uniform_(*uniform_range)

    v_cache = torch.empty(
        (num_blocks, num_heads, head_size, block_size),
        dtype=model_dtype, device=device,
    )
    v_cache.uniform_(*uniform_range)
    return k_cache, v_cache


def pertoken_quant_kvcache_symm(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    quant_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token symmetric FP8 quantization for KV cache.

    Returns:
        k_quant:     [num_blocks, num_heads, head_dim // qx, block_size, qx]  (5D)
        v_quant:     [num_blocks, num_heads, head_dim, block_size]             (4D)
        k_scale_asm: [num_blocks, num_kv_heads, block_size, 1]
        v_scale_asm: [num_blocks, num_kv_heads, block_size, 1]
    """
    num_blocks = k_cache.shape[0]
    num_heads = k_cache.shape[1]
    head_dim = v_cache.shape[2]
    block_size = v_cache.shape[3]

    k_cache_permute = (
        k_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    v_cache_permute = (
        v_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    k_quant, k_scale_asm = pertoken_quant(k_cache_permute, quant_dtype=quant_dtype)
    v_quant, v_scale_asm = pertoken_quant(v_cache_permute, quant_dtype=quant_dtype)

    qx = 16 // quant_dtype.itemsize

    k_quant = (
        k_quant.view(num_blocks, num_heads, block_size, head_dim // qx, qx)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    v_quant = (
        v_quant.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )

    return k_quant, v_quant, k_scale_asm, v_scale_asm


def asm_V_shuffle(vc: torch.Tensor) -> torch.Tensor:
    """Preshuffle V from 4D to 5D layout for CDNA3 MFMA alignment.

    [num_blocks, num_kv_heads, head_size, block_size]
      -> [num_blocks, num_kv_heads, block_size // x, head_size, x]
    """
    x = 16 // vc.element_size()
    nb, nh, hd, bs = vc.shape
    return (
        vc.view(nb, nh, hd, bs // x, x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# Kernel wrappers with @perftest for timing
# ---------------------------------------------------------------------------

@perftest()
def run_rocm_pa(
    output, exp_sums, max_logits, tmp_out,
    query, key_cache, value_cache,
    num_kv_heads, scale, block_tables, context_lens,
    block_size, max_context_len, kv_cache_dtype,
    k_scale, v_scale,
):
    aiter.paged_attention_rocm(
        out=output,
        exp_sums=exp_sums,
        max_logits=max_logits,
        tmp_out=tmp_out,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        scale=scale,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=block_size,
        max_context_len=max_context_len,
        alibi_slopes=None,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        partition_size=_PARTITION_SIZE_ROCM,
    )
    return output


@perftest()
def run_asm_pa(
    query, k_cache, v_cache,
    block_tables, seq_lens, block_tables_stride0,
    max_qlen, k_scale, v_scale, qo_indptr,
):
    return aiter.pa_fwd_asm(
        query, k_cache, v_cache,
        block_tables, seq_lens, block_tables_stride0,
        max_qlen, k_scale, v_scale,
        None, qo_indptr,
    )


@perftest()
def run_ragged_pa(
    output, workspace_buffer,
    query, key_cache, value_cache,
    scale, kv_indptr, kv_page_indices, kv_last_page_lens,
    block_size, max_num_partitions,
    kv_cache_dtype, k_scale, v_scale,
):
    aiter.paged_attention_ragged(
        out=output,
        workspace_buffer=workspace_buffer,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        scale=scale,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        block_size=block_size,
        max_num_partitions=max_num_partitions,
        alibi_slopes=None,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_layout="HND",
        logits_soft_cap=0.0,
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_out_scale=None,
        partition_size=_PARTITION_SIZE_ROCM,
    )
    return output


@perftest(num_rotate_args=20)
def run_pa_persistent(
    Q, K, V, output, max_qlen,
    qo_indptr, kv_indptr, kv_indices, context_lens,
    K_QScale, V_QScale,
    work_indptr, work_info,
    reduce_indptr, reduce_final_map, reduce_partial_map,
    softmax_scale, mask,
):
    return aiter.pa_persistent_fwd(
        Q=Q, K=K, V=V, output=output,
        max_qlen=max_qlen,
        qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
        context_lens=context_lens,
        K_QScale=K_QScale, V_QScale=V_QScale,
        work_indptr=work_indptr, work_info=work_info,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        softmax_scale=softmax_scale, mask=mask,
    )


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

@benchmark()
def test_pa_layout_cmp(
    ctx_lens: int,
    batch_size: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> dict:
    ret = {}
    device = "cuda:0"
    torch.set_default_device(device)
    num_query_heads, num_kv_heads = num_heads

    assert num_query_heads % num_kv_heads == 0
    max_seq_len = max(ctx_lens, block_size)
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * batch_size
    num_blocks_per_seq = (ctx_lens + block_size - 1) // block_size

    scale = float(1.0 / (head_size ** 0.5))

    # ---- Query (decode: qlen=1 per sequence) ----
    query = torch.empty(batch_size, num_query_heads, head_size, dtype=dtype)
    query.uniform_(*uniform_range)

    # ---- Block tables ----
    block_tables_lst: List[List[int]] = []
    for _ in range(batch_size):
        block_tables_lst.append(
            [random.randint(0, num_blocks - 1) for _ in range(num_blocks_per_seq)]
        )
    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)
    seq_lens = torch.full((batch_size,), ctx_lens, dtype=torch.int)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int)

    # ---- Create BF16 KV caches (vLLM v0.x storage format) ----
    k_cache, v_cache = kv_cache_factory(
        num_blocks, block_size, num_kv_heads, head_size, dtype, device,
    )

    # ---- FP8 per-token quantization (shared across all tests) ----
    k_quant, v_quant_4d, k_scale_asm, v_scale_asm = pertoken_quant_kvcache_symm(
        k_cache, v_cache, quant_dtype=dtypes.fp8,
    )
    v_quant_5d = asm_V_shuffle(v_quant_4d)

    # ---- Workspace for paged_attention_rocm ----
    max_num_partitions = (ctx_lens + _PARTITION_SIZE_ROCM - 1) // _PARTITION_SIZE_ROCM
    exp_sums = torch.empty(
        (batch_size, num_query_heads, max_num_partitions),
        dtype=dtypes.fp32,
    )
    max_logits = torch.empty_like(exp_sums)
    tmp_out = torch.empty(
        (batch_size, num_query_heads, max_num_partitions, head_size),
        dtype=dtype,
    )

    # ---- Ragged-style indexing (kv_indptr / kv_page_indices / kv_last_page_lens) ----
    actual_blocks = (seq_lens + block_size - 1) // block_size
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr[1:batch_size + 1] = torch.cumsum(actual_blocks, dim=0)

    kv_page_indices_lst = []
    for i in range(batch_size):
        kv_page_indices_lst += block_tables_lst[i][:actual_blocks[i]]
    kv_page_indices = torch.tensor(kv_page_indices_lst, dtype=torch.int)

    last_page_len = ctx_lens % block_size if ctx_lens % block_size != 0 else block_size
    kv_last_page_lens = torch.full((batch_size,), last_page_len, dtype=torch.int)

    # ---- Workspace for paged_attention_ragged ----
    ws_size = (
        batch_size * num_query_heads * max_num_partitions * (2 * 4 + head_size * dtype.itemsize)
    )
    workspace_buffer = torch.empty(ws_size, dtype=torch.uint8, device=device)

    # ---- Dummy scales for ragged kernel (BF16 mode, scales are ignored) ----
    k_scale_ragged = torch.tensor([1.0], dtype=torch.float32, device=device)
    v_scale_ragged = torch.tensor([1.0], dtype=torch.float32, device=device)

    # ==================================================================
    # Test R: paged_attention_ragged + V 4D  (SGLang aiter decode path)
    #   SGLang converts FP8 KV → BF16 before calling ragged (see TODO in
    #   aiter_backend.py), so we use the original BF16 KV caches here to
    #   match the real-world code path.  kv_cache_dtype="auto" (BF16).
    # ==================================================================
    output_r = torch.empty_like(query)
    us_r = float("nan")
    try:
        _, us_r = run_ragged_pa(
            output_r, workspace_buffer,
            query, k_cache, v_cache,
            scale, kv_indptr, kv_page_indices, kv_last_page_lens,
            block_size, max_num_partitions,
            "auto", k_scale_ragged, v_scale_ragged,
        )
    except Exception as e:
        print(f"  [R] paged_attention_ragged skipped: {e}")

    # ==================================================================
    # Test A: paged_attention_rocm + V 4D  (vLLM v0.x layout, baseline)
    # ==================================================================
    output_a = torch.empty_like(query)
    _, us_a = run_rocm_pa(
        output_a, exp_sums, max_logits, tmp_out,
        query, k_quant, v_quant_4d,
        num_kv_heads, scale, block_tables, seq_lens,
        block_size, ctx_lens, "fp8",
        k_scale_asm, v_scale_asm,
    )

    # ==================================================================
    # Test B: paged_attention_rocm + V 5D  (preshuffle, same kernel)
    # ==================================================================
    output_b = torch.empty_like(query)
    _, us_b = run_rocm_pa(
        output_b, exp_sums, max_logits, tmp_out,
        query, k_quant, v_quant_5d,
        num_kv_heads, scale, block_tables, seq_lens,
        block_size, ctx_lens, "fp8",
        k_scale_asm, v_scale_asm,
    )

    err_ab = checkAllclose(
        output_a, output_b,
        msg=f"[A vs B] rocm V4D vs V5D (bs={block_size})",
    )

    # ==================================================================
    # Test C: pa_fwd_asm + V 5D  (non-persistent)
    #   Available: block_size=16 (gqa=8,10,16), block_size=1024 (gqa=10 only)
    # ==================================================================
    us_c = float("nan")
    try:
        _, us_c = run_asm_pa(
            query, k_quant, v_quant_5d,
            block_tables, seq_lens, block_tables.size(1),
            1,
            k_scale_asm, v_scale_asm,
            qo_indptr,
        )
    except RuntimeError as e:
        print(f"  [C] pa_fwd_asm skipped (no kernel for gqa={num_query_heads // num_kv_heads}, "
              f"block_size={block_size}): {e}")

    # ==================================================================
    # Test D: pa_persistent_fwd + V 5D  (persistent scheduling)
    #   Available: block_size=1024 (ps=1 kernels)
    # ==================================================================
    us_d = float("nan")
    kv_indices = kv_page_indices

    try:
        (
            (wm_size, wm_type), (wi_size, wi_type), (wis_size, wis_type),
            (ri_size, ri_type), (rfm_size, rfm_type), (rpm_size, rpm_type),
        ) = aiter.get_pa_metadata_info_v1(batch_size, num_kv_heads)

        work_metadata_ptrs = torch.empty(wm_size, dtype=wm_type)
        work_indptr = torch.empty(wi_size, dtype=wi_type)
        work_info = torch.empty(wis_size, dtype=wis_type)
        reduce_indptr = torch.empty(ri_size, dtype=ri_type)
        reduce_final_map = torch.empty(rfm_size, dtype=rfm_type)
        reduce_partial_map = torch.empty(rpm_size, dtype=rpm_type)

        # warmup
        aiter.get_pa_metadata_v1(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            1, 1, True,
            work_metadata_ptrs, work_indptr, work_info,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            kv_granularity=max(block_size, 16), block_size=block_size,
            max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=True,
            max_split_per_batch=-1,
        )
        torch.cuda.synchronize()

        aiter.get_pa_metadata_v1(
            qo_indptr, kv_indptr, seq_lens,
            num_query_heads // num_kv_heads, num_kv_heads, True,
            work_metadata_ptrs, work_indptr, work_info,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            kv_granularity=max(block_size, 16), block_size=block_size,
            max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=True,
            max_split_per_batch=-1,
        )

        output_d = torch.empty(
            batch_size, num_query_heads, head_size, dtype=dtype,
        )
        _, us_d = run_pa_persistent(
            Q=query, K=k_quant, V=v_quant_5d,
            output=output_d, max_qlen=1,
            qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
            context_lens=seq_lens,
            K_QScale=k_scale_asm, V_QScale=v_scale_asm,
            work_indptr=work_indptr, work_info=work_info,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            softmax_scale=scale, mask=1,
        )
    except RuntimeError as e:
        print(f"  [D] pa_persistent_fwd skipped (no kernel for "
              f"block_size={block_size}): {e}")

    # ==================================================================
    # Results
    # ==================================================================
    def safe_div(a, b):
        if b > 0 and a == a and b == b:  # NaN check
            return a / b
        return float("nan")

    ret["ctx_lens"] = ctx_lens
    ret["batch_size"] = batch_size
    ret["num_heads"] = f"{num_query_heads},{num_kv_heads}"
    ret["head_size"] = head_size
    ret["block_size"] = block_size
    ret["us_ragged(R)"] = us_r
    ret["us_rocm_v4d(A)"] = us_a
    ret["us_rocm_v5d(B)"] = us_b
    ret["us_asm(C)"] = us_c
    ret["us_ps(D)"] = us_d
    ret["A/R"] = safe_div(us_r, us_a)
    ret["B/A"] = safe_div(us_a, us_b)
    ret["C/A"] = safe_div(us_a, us_c)
    ret["D/A"] = safe_div(us_a, us_d)
    ret["D/C"] = safe_div(us_c, us_d)
    ret["err(A~B)"] = err_ab

    return ret


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description=(
        "KV Cache Layout Comparison Benchmark (FP8)\n\n"
        "Fair comparison at each block_size:\n"
        "  R: paged_attention_ragged + V 4D + BF16 KV (SGLang actual decode path)\n"
        "  A: paged_attention_rocm   + V 4D + FP8 KV  (vLLM v0.x baseline)\n"
        "  B: paged_attention_rocm   + V 5D + FP8 KV  (preshuffle)\n"
        "  C: pa_fwd_asm            + V 5D + FP8 KV  (ASM, non-persistent)\n"
        "  D: pa_persistent_fwd     + V 5D + FP8 KV  (ASM, persistent)\n\n"
        "  A/R = FP8 rocm vs BF16 ragged (real SGLang gap)\n"
        "  B/A = pure V layout gain\n"
        "  C/A = layout + ASM kernel gain\n"
        "  D/A = layout + ASM + persistent gain\n"
        "  D/C = pure persistent scheduling gain\n\n"
        "Kernel availability (gfx942):\n"
        "  block_size=16:   C available (gqa=8,10,16), D not available\n"
        "  block_size=1024: C available (gqa=10 only), D available\n"
    ),
)
parser.add_argument(
    "-d", "--dtype", type=dtypes.str2Dtype, nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    help="Model dtype. e.g.: -d bf16",
)
parser.add_argument(
    "-n", "--num_heads", type=dtypes.str2tuple, nargs="*",
    default=[(10, 1)],
    help="(num_query_heads, num_kv_heads). e.g. -n 10,1  (gqa=10 works for all block_sizes)",
)
parser.add_argument(
    "-hd", "--head_dim", type=int, default=128,
    help="Head dimension. e.g. -hd 128",
)
parser.add_argument(
    "-c", "--ctx_len", type=int, nargs="*",
    default=[1024, 4096, 8192],
    help="Context length. e.g. -c 1024",
)
parser.add_argument(
    "-b", "--batch_size", type=int, nargs="*",
    default=[32, 64, 128, 256],
    help="Batch size. e.g. -b 128",
)
parser.add_argument(
    "--block_size", type=int, nargs="*",
    default=[16, 1024],
    help="Block size(s). e.g. --block_size 16 1024",
)
args = parser.parse_args()

for model_dtype in args.dtype:
    df = []
    for num_heads, block_size, ctx_len, batch_size in itertools.product(
        args.num_heads, args.block_size, args.ctx_len, args.batch_size,
    ):
        if ctx_len < block_size:
            continue
        ret = test_pa_layout_cmp(
            ctx_len, batch_size, num_heads, args.head_dim,
            block_size, model_dtype,
        )
        df.append(ret)

    df = pd.DataFrame(df)
    print("\n" + "=" * 80)
    print("KV Cache Layout Comparison (FP8)")
    print("  R: paged_attention_ragged + V 4D + BF16 KV (SGLang actual decode path)")
    print("  A: paged_attention_rocm   + V 4D + FP8 KV  (vLLM v0.x baseline)")
    print("  B: paged_attention_rocm   + V 5D + FP8 KV  (preshuffle)")
    print("  C: pa_fwd_asm            + V 5D + FP8 KV  (ASM, non-persistent)")
    print("  D: pa_persistent_fwd     + V 5D + FP8 KV  (ASM, persistent)")
    print("  A/R = FP8 rocm vs BF16 ragged | B/A = layout | C/A = ASM | D/A = ASM+PS | D/C = PS")
    print("=" * 80)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("layout_cmp summary (markdown):\n%s", df_md)
    df.to_csv("pa_layout_cmp.csv")
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(df)
