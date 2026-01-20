# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import random
from typing import List, Optional, Tuple, Union
import torch
import aiter
from aiter import dtypes
from aiter import paged_attn as ops
from aiter.test_common import (
    checkAllclose,
    perftest,
    tensor_dump,
    tensor_load,
    benchmark,
)
from aiter import pertoken_quant
import argparse
import pandas as pd

uniform_range = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": dtypes.bf16,
    "float": dtypes.fp32,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}
ck_naive_quant_algo = [
    "NO",
    "KV_8BIT_PER_HEAD",
    # // FP8/INT8 quant for KVCache, per-token quant
    # // [num_tokens, nhead, hdim] -> [nhead, num_tokens]
    "KV_8BIT_PER_TOKEN",
    # // same as 8bit per token quant but 4 bit
    "KV_4BIT_PER_TOKEN",
    "KV_8BIT_PER_TENSOR",
]


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    x = 16 // torch_dtype.itemsize
    k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    k_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        k_cache = torch.empty(size=k_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            k_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        k_caches.append(k_cache)

    v_cache_shape = (num_blocks, num_heads, head_size, block_size)
    v_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        v_cache = torch.empty(size=v_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            v_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        v_caches.append(v_cache)
    return k_caches, v_caches


FLOAT32_BYTES = torch.finfo(dtypes.fp32).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = 65536
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 32768  # Arbitrary values for testing
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {dtypes.fp16, dtypes.bf16}
DTYPES = [torch.half, dtypes.bf16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 80, 96, 112, 120, 128, 192, 256]

BLOCK_SIZES = [16, 32]
USE_ALIBI = [False, True]
KV_CACHE_DTYPE = ["auto", "fp8"]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]

# 0: no quant. 1: (ignore this), FP8, 2: K/V per-token(prefer this)
PA_QUANT = 2


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    nkvhead,
    k_scale=torch.Tensor,  # [1] or [nkvhead, 1, seq_lenth]
    v_scale=torch.Tensor,  # [1] or [nkvhead, 1, seq_lenth]
    attn_mask: Optional[torch.Tensor] = None,
    dtype=None,
) -> torch.Tensor:
    p_scale = 1.0
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query.float(), key.float())

    # [nqhead, q_len, ctx_len]
    nqhead, q_len, ctx_len = attn_weights.shape
    attn_weights = attn_weights.view(nqhead // nkvhead, nkvhead, q_len, ctx_len)
    attn_weights *= k_scale
    attn_weights = attn_weights.view(nqhead, q_len, ctx_len)

    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()

    attn_weights = torch.softmax(attn_weights, dim=-1)

    attn_weights = attn_weights.view(nqhead // nkvhead, nkvhead, q_len, ctx_len)
    attn_weights *= v_scale
    attn_weights = attn_weights.view(nqhead, q_len, ctx_len)
    # if v_scale != 1.0:
    #     attn_weights, p_scale = aiter.per_tensor_quant(
    #         attn_weights,  quant_dtype=dtypes.i8)
    #     # attn_weights,  quant_dtype=key.dtype)
    #     # attn_weights = attn_weights.float()*p_scale

    out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
    out *= p_scale
    return out.to(dtype)


def pertoken_quant_kvcache_symm(
    # [num_blocks, num_heads, head_size // x, block_size, x]
    k_cache: torch.Tensor,
    # [num_blocks, num_heads, head_size, block_size]
    v_cache: torch.Tensor,
    quant_dtype: torch.dtype,  # e.g. dtypes.fp8
    scale_dtype: torch.dtype = dtypes.fp32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = k_cache.shape[0]
    num_heads = k_cache.shape[1]
    head_dim = v_cache.shape[2]
    block_size = v_cache.shape[3]
    # x          = k_cache.shape[4]
    total_tokens = num_blocks * block_size

    # print(f"{k_cache.shape=}{k_cache.stride()=}")
    # print(f"{v_cache.shape=}{v_cache.stride()=}")

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

    # NOTE: quant_x and original x could be different
    quant_x = 16 // quant_dtype.itemsize

    k_quant = (
        k_quant.view(num_blocks, num_heads, block_size, head_dim // quant_x, quant_x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    k_scale = k_scale_asm.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    v_quant = (
        v_quant.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    v_scale = v_scale_asm.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)

    # print(f"{k_quant.shape=}{k_quant.stride()=}")
    # print(f"{k_scale.shape=}{k_scale.stride()=}")
    # print(f"{v_quant.shape=}{v_quant.stride()=}")
    # print(f"{v_scale.shape=}{v_scale.stride()=}")
    # print(f"k_cache_permute:{k_cache_permute[0, :, :, :]}, k_quant:{k_quant[0, :, :, :, :]}, k_scale:{k_scale[:, 0]}")

    return k_quant, k_scale, v_quant, v_scale, k_scale_asm, v_scale_asm


@perftest(num_iters=2)
def run_native(
    query,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    k_scale_cache,
    v_scale_cache,
    num_queries_per_kv,
    dtype,
):
    output = torch.zeros_like(query).to(dtype)
    num_query_heads = query.shape[1]
    num_kv_heads = v_cache.shape[1]
    head_size = v_cache.shape[2]
    block_size = v_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()

    # (num_blocks, num_heads, head_size // x, block_size, x)
    k_cache = (
        k_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_kv_heads, head_size)
    )
    # (num_blocks, num_heads, head_size, block_size)
    v_cache = v_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_kv_heads, head_size)
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        ctx_len = int(seq_lens_lst[i])

        idx = [
            int(block_table[j // block_size]) * block_size + (j % block_size)
            for j in range(ctx_len)
        ]
        if k_cache.dtype == dtypes.fp8:
            keys = k_cache.view(dtypes.i8)[idx].view(dtypes.fp8)
            values = v_cache.view(dtypes.i8)[idx].view(dtypes.fp8)
        else:
            keys = k_cache[idx]
            values = v_cache[idx]
        if k_scale_cache.numel() > 1:
            k_scale = k_scale_cache[:, idx].contiguous().view(num_kv_heads, 1, ctx_len)
            v_scale = v_scale_cache[:, idx].contiguous().view(num_kv_heads, 1, ctx_len)
        else:
            k_scale = k_scale_cache  # [1]
            v_scale = v_scale_cache  # [1]

        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(ctx_len).int()
            alibi_bias = (position_ids - ctx_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(
            q, keys, values, scale, num_kv_heads, k_scale, v_scale, alibi_bias, dtype
        )
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output  # , 1


@perftest()
def run_aiter(
    query,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    k_scale,
    v_scale,
):
    return ops.PagedAttention.forward_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )


@perftest(num_iters=2)
def run_aiter_naive(
    query,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    k_dequant_scales,
    v_dequant_scales,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    k_scale,
    v_scale,
    block_size,
    quant_algo=0,
):
    return aiter.pa_fwd_naive(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        k_dequant_scales,
        v_dequant_scales,
        max_seq_len,
        num_kv_heads,
        scale,
        k_scale,
        v_scale,
        block_size,
        quant_algo,
    )


@perftest()
def run_aiter_asm(
    query,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    block_tables_stride0,
    k_scale=None,
    v_scale=None,
    high_precision=0,
):
    return aiter.pa_fwd_asm(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        block_tables_stride0,
        K_QScale=k_scale,
        V_QScale=v_scale,
        out_=None,
        high_precision=high_precision,
    )


def dump_input(
    path,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float,
    out_golden,
    out_test,
):
    # path = '/mnt/raid0/ljin1/dk/ater/debug_ctx7'
    tensor_dump(query, "Q", path)
    # qbk = tensor_load('Q.bin')
    # checkAllclose(query, qbk)
    tensor_dump(k_cache, "K_cache", path)
    tensor_dump(v_cache, "V_cache", path)
    tensor_dump(block_tables, "block_tables", path)
    tensor_dump(seq_lens, "seq_lens", path)
    tensor_dump(k_scale, "k_scale", path)
    tensor_dump(v_scale, "v_scale", path)
    tensor_dump(out_golden, "out_golden", path)
    tensor_dump(out_test, "out_test", path)


def load_input():
    # return (tensor_load('Q.bin'),
    #         tensor_load('K_cache.bin'),
    #         tensor_load('V_cache.bin'),
    #         tensor_load('block_tables.bin'),
    #         tensor_load('seq_lens.bin'),
    #         tensor_load('out_aiter.bin'))
    # return (tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/Q_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/K_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/V_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/block_tables.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/seq_lens.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/OUT_16.bin'),
    #         )
    return (
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/Q_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/K_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/V_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/block_tables.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/seq_lens.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/OUT_BF16.bin"),
    )


DUMP = 1
VERIFY = 2
# debug_mode = DUMP
# debug_mode = VERIFY
debug_mode = 0
torch.set_printoptions(sci_mode=False)


def asm_V_shuffle(VC):
    # [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    # [num_blocks, num_kv_heads, block_size/X, head_size, X]
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC


@benchmark()
def test_paged_attention(
    ctx_lens: int,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:
    torch.set_default_device(device)
    # Using default kv_scale
    k_scale = v_scale = torch.tensor(1.0, device=device, dtype=dtypes.fp32)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=dtypes.fp32)
    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    max_seq_len = ctx_lens
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs
    print(f"{debug_mode=}")

    if debug_mode == VERIFY:
        (query, k_cache, v_cache, block_tables, seq_lens, out_golden) = load_input()
    else:
        query = torch.empty_strided(
            (num_seqs, num_query_heads, head_size),
            ((num_query_heads + 2 * num_kv_heads) * head_size, head_size, 1),
            dtype=dtype,
        )
        query.uniform_(*uniform_range)

        # seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
        seq_lens = [ctx_lens for _ in range(num_seqs)]
        seq_lens = torch.tensor(seq_lens, dtype=torch.int)

        # Create the block tables.
        block_tables_lst: List[List[int]] = []
        for _ in range(num_seqs):
            block_table = [
                random.randint(0, num_blocks - 1) for _ in range(max_num_blocks_per_seq)
            ]
            block_tables_lst.append(block_table)

        block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

        # Create the KV caches.
        k_caches, v_caches = kv_cache_factory(
            num_blocks,
            block_size,
            1,
            num_kv_heads,
            head_size,
            kv_cache_dtype,
            dtype,
            seed,
            device,
        )
        k_cache, v_cache = k_caches[0], v_caches[0]

    out_aiter, time_aiter = run_aiter(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )
    if debug_mode != VERIFY:
        out_golden = out_aiter
    checkAllclose(
        out_golden, out_aiter, msg=f"golden vs aiter_shomy:{time_aiter:>8.2f} us......"
    )
    # tensor_dump(out_aiter, 'out_aiter')

    time_aiter_asm = None
    if dtype == dtypes.bf16:
        out_aiter_asm, time_aiter_asm = run_aiter_asm(
            query.contiguous(),  # this kernel need contiguous buffer
            k_cache,
            asm_V_shuffle(v_cache),
            block_tables,
            seq_lens,
            max_seq_len,
            kv_cache_dtype,
            num_kv_heads,
            scale,
            alibi_slopes,
            block_tables.stride(0),
        )

        checkAllclose(
            out_golden,
            out_aiter_asm,
            msg=f"golden vs aiter_asm:{time_aiter_asm:>8.2f} us......",
        )
        # tensor_dump(out_aiter, 'out_aiter')

    for quant_algo_, cache_type_ in [
        (0, k_cache.dtype),
        (2, dtypes.fp8),
        (2, dtypes.i8),
        (4, dtypes.fp8),
    ]:
        quant_algo = ck_naive_quant_algo[quant_algo_]
        if quant_algo == "NO":
            k_quant_, k_scale_, v_quant_, v_scale_ = (
                k_cache,
                torch.empty((0)),
                v_cache,
                torch.empty((0)),
            )
        elif quant_algo == "KV_8BIT_PER_TOKEN":
            k_quant_, k_scale_, v_quant_, v_scale_, k_scale_asm, v_scale_asm = (
                pertoken_quant_kvcache_symm(k_cache, v_cache, quant_dtype=cache_type_)
            )
        elif quant_algo == "KV_8BIT_PER_TENSOR":
            k_quant_, k_scale_ = aiter.per_tensor_quant(
                k_cache, quant_dtype=cache_type_
            )

            x = 16 // cache_type_.itemsize
            k_quant_ = (
                k_quant_.permute(0, 1, 3, 2, 4)
                .reshape(num_blocks, num_kv_heads, block_size, -1)
                .contiguous()
            )
            k_quant_ = (
                k_quant_.view(num_blocks, num_kv_heads, block_size, head_size // x, x)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

            v_quant_, v_scale_ = aiter.per_tensor_quant(
                v_cache, quant_dtype=cache_type_
            )

            k_scale_asm = torch.empty(
                num_blocks, num_kv_heads, block_size, dtype=dtypes.fp32, device=device
            )
            v_scale_asm = torch.empty(
                num_blocks, num_kv_heads, block_size, dtype=dtypes.fp32, device=device
            )
            k_scale_asm.fill_(k_scale_.item())
            v_scale_asm.fill_(v_scale_.item())
            k_scale_hip_ = torch.empty(
                num_blocks * block_size, num_kv_heads, dtype=dtypes.fp32, device=device
            )
            v_scale_hip_ = torch.empty(
                num_blocks * block_size, num_kv_heads, dtype=dtypes.fp32, device=device
            )
            k_scale_hip_.fill_(k_scale_.item())
            v_scale_hip_.fill_(v_scale_.item())

            out_aiter, time_aiter = run_aiter(
                query,
                k_quant_,
                v_quant_,
                block_tables,
                seq_lens,
                max_seq_len,
                "fp8",
                num_kv_heads,
                scale,
                alibi_slopes,
                k_scale_hip_,
                v_scale_hip_,
            )
            checkAllclose(
                out_golden,
                out_aiter,
                msg=f"golden vs shomy:{time_aiter:>8.2f} us......(quant:{ck_naive_quant_algo[quant_algo_]}, kvcache:{cache_type_})",
            )
        # if quant_algo != "KV_8BIT_PER_TENSOR":
        # out_aiter_naive, time_aiter_naive = run_aiter_naive(
        #     query,
        #     k_quant_,
        #     v_quant_,
        #     block_tables,
        #     seq_lens,
        #     k_scale_,
        #     v_scale_,
        #     max_seq_len,
        #     kv_cache_dtype,
        #     num_kv_heads,
        #     scale,
        #     alibi_slopes,
        #     k_scale,
        #     v_scale,
        #     block_size,
        #     quant_algo_
        # )
        # checkAllclose(out_aiter_asm, out_aiter_naive,
        #             msg=f'golden vs ck_naive(quant:{ck_naive_quant_algo[quant_algo_]}, kvcache:{cache_type_}):{time_aiter_naive:>8.2f} us......')
        if quant_algo_ != 0:
            out_aiter_asm, time_aiter_asm = run_aiter_asm(
                query,
                k_quant_,
                asm_V_shuffle(v_quant_),
                block_tables,
                seq_lens,
                max_seq_len,
                kv_cache_dtype,
                num_kv_heads,
                scale,
                alibi_slopes,
                block_tables.stride(0),
                k_scale_asm,
                v_scale_asm,
            )
            checkAllclose(
                out_golden,
                out_aiter_asm,
                msg=f"golden vs aiter_asm:{time_aiter_asm:>8.2f} us......(quant:{ck_naive_quant_algo[quant_algo_]}, kvcache:{cache_type_})",
            )

            if (
                dtype in [dtypes.bf16, dtypes.fp16]
                and quant_algo_ == 2
                and cache_type_ == dtypes.fp8
            ):
                if dtype == dtypes.bf16:
                    high_precision_list = [1, 2]
                else:
                    high_precision_list = [1]
                for high_precision in high_precision_list:
                    try:
                        out_aiter_asm, time_aiter_asm = run_aiter_asm(
                            query,
                            k_quant_,
                            asm_V_shuffle(v_quant_),
                            block_tables,
                            seq_lens,
                            max_seq_len,
                            kv_cache_dtype,
                            num_kv_heads,
                            scale,
                            alibi_slopes,
                            block_tables.stride(0),
                            k_scale_asm,
                            v_scale_asm,
                            high_precision,
                        )
                        checkAllclose(
                            out_golden,
                            out_aiter_asm,
                            msg=f"golden vs aiter_asm high_precision {high_precision}:{time_aiter_asm:>8.2f} us......(quant:{ck_naive_quant_algo[quant_algo_]}, kvcache:{cache_type_})",
                        )
                    except RuntimeError as e:
                        if "get_heuristic_kernel" in str(e):
                            print(f"[SKIP] high_precision={high_precision} not supported for GQA={num_query_heads//num_kv_heads}, skipping...")
                        else:
                            raise

            # if quant_algo == "KV_8BIT_PER_TENSOR":
            #     q_quant_, q_scale_ = aiter.per_tensor_quant(
            #         query,  quant_dtype=cache_type_)
            # Skip native test for large batch sizes (extremely slow due to sequential processing)
            NATIVE_BS_LIMIT = 256
            if num_seqs <= NATIVE_BS_LIMIT:
                out_native, time_native = run_native(
                    query,
                    # q_quant_,
                    k_quant_,
                    v_quant_,
                    block_tables,
                    seq_lens,
                    max_seq_len,
                    kv_cache_dtype,
                    num_kv_heads,
                    scale,
                    # scale*q_scale_.item(),
                    alibi_slopes,
                    k_scale_,
                    v_scale_,
                    num_queries_per_kv,
                    dtype,
                )
                checkAllclose(
                    out_golden,
                    out_native,
                    msg=f"golden vs torch_native: {time_native:>8.2f} us...... (quant:{ck_naive_quant_algo[quant_algo_]}, kvcache:{cache_type_})",
                )
            else:
                print(f"[SKIP] torch_native test for bs={num_seqs} > {NATIVE_BS_LIMIT} (too slow)")

    if debug_mode == DUMP:
        dump_input(
            query,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            kv_cache_dtype,
            num_kv_heads,
            scale,
            alibi_slopes,
            k_scale,
            v_scale,
            out_golden,
        )

    # out_native, time_native = run_native(
    #     query,
    #     k_cache,
    #     v_cache,
    #     block_tables,
    #     seq_lens,
    #     max_seq_len,
    #     kv_cache_dtype,
    #     num_kv_heads,
    #     scale,
    #     alibi_slopes,
    #     k_scale,
    #     v_scale,
    #     num_queries_per_kv,
    #     dtype
    # )
    # checkAllclose(out_golden, out_native,
    #               msg=f'golden vs torch_native: {time_native:>8.2f} us......')
    # tensor_dump(out_native, 'out_native')

    # atol, rtol = 1e-2, 1e-2
    # msg = f"[perf] dim: {str((num_seqs, num_heads, head_size)):<20}, dtype: {dtype}, {time_native=:<8.2f} us, {time_aiter=:<8.2f} us, uplift: {time_native/time_aiter-1:<5.1%}"
    # checkAllclose(out_native, out_aiter, atol=atol, rtol=rtol, msg=msg)
    # print(
    #     f"[test] dim: {str((ctx_lens, num_seqs, num_heads, head_size)):<20}, dtype: {dtype}, finished)\n")
    print(
        f"finish~ {ctx_lens=}, {num_seqs=}, {num_heads=}, {head_size=}, {use_alibi=}, {block_size=}, {dtype=}, {kv_cache_dtype=}\n"
    )
    return {"aiter_shomy": time_aiter, "aiter_asm": time_aiter_asm}


df = []

# ============================================================================
# LLaMA 3.1 405B Baseline Test Configurations
# Full model: num_heads=128, num_kv_heads=8, head_dim=128
# With TP=2: num_q_heads=64, num_kv_heads=4
# ============================================================================

# LLaMA 3.1 405B with TP=2
LLAMA_NUM_HEADS = (64, 4)  # (num_query_heads, num_kv_heads) for TP=2
LLAMA_HEAD_SIZE = 128
LLAMA_BLOCK_SIZE = 16

# Prefill baseline cases (adjusted for single GPU memory ~256GB)
# Original targets: bs: 1000–22000; ctx: 476–19988
# Memory formula: num_blocks = (ctx_len/16) * bs, KV cache ~= num_blocks * 4 * 128 * 16 * 2 * 2 bytes
# Safe limit: num_blocks < 2M (~65GB KV cache)
# P1: ctx_len=512,   bs=1024   -> blocks=32K,   ~1GB   ✓
# P2: ctx_len=4096,  bs=2048   -> blocks=524K,  ~17GB  ✓
# P3: ctx_len=8192,  bs=1024   -> blocks=524K,  ~17GB  ✓
# P4: ctx_len=16384, bs=512    -> blocks=524K,  ~17GB  ✓
PREFILL_CASES = [
    {"name": "P1", "ctx_len": 512,   "bs": 1024,  "desc": "min baseline"},
    {"name": "P2", "ctx_len": 4096,  "bs": 2048,  "desc": "mid ctx + reduced bs"},
    {"name": "P3", "ctx_len": 8192,  "bs": 1024,  "desc": "near input mean, reduced bs"},
    {"name": "P4", "ctx_len": 16384, "bs": 512,   "desc": "max ctx, reduced bs"},
]

# Decode baseline cases (bs: 5–896; ctx: 476–19988)
# D1: ctx_len=512,   bs=8   (接近最小，快速 sanity)
# D2: ctx_len=4096,  bs=128 (中等 ctx + 中等并发)
# D3: ctx_len=8192,  bs=384 (接近/略低于均值附近的重负载)
# D4: ctx_len=16384, bs=896 (高 ctx + 最大 bs，压到上限)
DECODE_CASES = [
    {"name": "D1", "ctx_len": 512,   "bs": 8,   "desc": "min sanity"},
    {"name": "D2", "ctx_len": 4096,  "bs": 128, "desc": "mid ctx + mid concurrency"},
    {"name": "D3", "ctx_len": 8192,  "bs": 384, "desc": "near mean heavy load"},
    {"name": "D4", "ctx_len": 16384, "bs": 896, "desc": "max ctx + max bs"},
]

l_dtype = ["fp16", "bf16"]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="LLaMA 3.1 405B PagedAttention Baseline Test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default="bf16",
    help="""Data type (default: bf16).
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["prefill", "decode", "all"],
    default="all",
    help="""Test mode: prefill, decode, or all (default: all).
    e.g.: -m decode""",
)
parser.add_argument(
    "-c",
    "--case",
    type=str,
    nargs="?",
    const=None,
    default=None,
    help="""Specific case to run: P1-P4 for prefill, D1-D4 for decode.
    e.g.: -c P2 or -c D3""",
)
parser.add_argument(
    "--kv_cache_dtype",
    type=str,
    choices=["auto", "fp8"],
    default="auto",
    help="""KV cache dtype (default: auto).
    e.g.: --kv_cache_dtype fp8""",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="""CUDA device (default: cuda:0).
    e.g.: --device cuda:1""",
)

args = parser.parse_args()

# Setup dtype
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]

# Setup test cases based on mode
test_cases = []
if args.mode in ["prefill", "all"]:
    for case in PREFILL_CASES:
        if args.case is None or args.case.upper() == case["name"]:
            test_cases.append({**case, "phase": "prefill"})
if args.mode in ["decode", "all"]:
    for case in DECODE_CASES:
        if args.case is None or args.case.upper() == case["name"]:
            test_cases.append({**case, "phase": "decode"})

print("=" * 80)
print("LLaMA 3.1 405B PagedAttention Baseline Test")
print(f"Model config: num_heads={LLAMA_NUM_HEADS}, head_size={LLAMA_HEAD_SIZE}, block_size={LLAMA_BLOCK_SIZE}")
print(f"Test mode: {args.mode}, dtype: {args.dtype}, kv_cache_dtype: {args.kv_cache_dtype}")
print("=" * 80)

for case in test_cases:
    phase = case["phase"]
    name = case["name"]
    ctx_len = case["ctx_len"]
    bs = case["bs"]
    desc = case["desc"]
    
    print(f"\n{'='*60}")
    print(f"[{phase.upper()}] {name}: ctx_len={ctx_len}, bs={bs} ({desc})")
    print(f"{'='*60}")
    
    for dtype in l_dtype:
        ret = test_paged_attention(
            ctx_len,
            bs,
            LLAMA_NUM_HEADS,
            LLAMA_HEAD_SIZE,
            False,  # use_alibi
            LLAMA_BLOCK_SIZE,
            dtype,
            args.kv_cache_dtype,
            0,  # seed
            args.device,
        )
        ret["phase"] = phase
        ret["case"] = name
        ret["ctx_len"] = ctx_len
        ret["bs"] = bs
        ret["dtype"] = str(dtype)
        df.append(ret)

df = pd.DataFrame(df)
# Reorder columns for better readability
col_order = ["phase", "case", "ctx_len", "bs", "dtype", "aiter_shomy", "aiter_asm"]
df = df[[c for c in col_order if c in df.columns]]
print("\n" + "=" * 80)
aiter.logger.info(f"LLaMA 3.1 405B Baseline Summary:\n{df.to_string(index=False)}")
print("=" * 80)
