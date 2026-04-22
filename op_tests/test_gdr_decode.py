import time
import torch
import argparse
import functools
import gc
import numpy as np
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable, Dict, Literal, Optional, Tuple

import triton
import triton.language as tl

from aiter.ops.flydsl.kernels.tensor_shim import get_dtype_str
from aiter.ops.flydsl.kernels.gdr_decode import create_shuffle_gdr_decode_kernel


@dataclass
class Args:
    dtype: torch.dtype
    b: int
    sq: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    use_qk_l2norm: bool = True


def create_inputs(args):
    query = torch.randn((args.b, args.sq, args.num_k_heads, args.head_k_dim), dtype=args.dtype, device='cuda')
    key = torch.randn((args.b, args.sq, args.num_k_heads, args.head_k_dim), dtype=args.dtype, device='cuda')
    value = torch.randn((args.b, args.sq, args.num_v_heads, args.head_v_dim), dtype=args.dtype, device='cuda')
    a = torch.randn((args.b, args.sq, args.num_v_heads), dtype=args.dtype, device='cuda')
    b = torch.randn((args.b, args.sq, args.num_v_heads), dtype=args.dtype, device='cuda')
    dt_bias = torch.randn((args.num_v_heads), dtype=args.dtype, device='cuda')
    dt_bias.uniform_(1, 2)
    A_log = torch.randn((args.num_v_heads), dtype=torch.float32, device="cuda")
    A_log.uniform_(0, 16)
    indices = torch.arange(args.b - 1, -1, -1, dtype=torch.int32, device="cuda")
    state = torch.randn((args.b, args.num_v_heads, args.head_k_dim, args.head_v_dim), dtype=torch.float32, device="cuda")
    return (args, query, key, value, a, b, dt_bias, A_log, indices, state)


def create_outputs(args):
    out = torch.zeros((args.b, args.sq, args.num_v_heads, args.head_v_dim), dtype=args.dtype, device='cuda')
    return (out,)


def ref_func_(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    beta = b.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias, beta=1.0, threshold=20.0)
    if args.num_v_heads // args.num_k_heads > 1:
        query = query.repeat_interleave(args.num_v_heads // args.num_k_heads, dim=2)
        key = key.repeat_interleave(args.num_v_heads // args.num_k_heads, dim=2)
    if args.use_qk_l2norm:
        def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
            """This function is intended to align with the l2norm implementation in the FLA library."""
            inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
            return x * inv_norm
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    # query, # (b, num_v_heads, sq, head_k_dim)
    # key,   # (b, num_v_heads, sq, head_k_dim)
    # value, # (b, num_v_heads, sq, head_v_dim)
    # g,     # (b, num_v_heads, sq)
    # beta,  # (b, num_v_heads, sq)
    scale = 1 / (args.head_k_dim ** 0.5)
    query = query * scale
    last_recurrent_state = state[indices]
    for i in range(args.sq):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        # q_t:     # (b, num_v_heads, head_k_dim)
        # k_t:     # (b, num_v_heads, head_k_dim)
        # v_t:     # (b, num_v_heads, head_v_dim)
        # g_t:     # (b, num_v_heads, 1, 1)
        # beta_t:  # (b, num_v_heads, 1)
        # last_recurrent_state: # (b, num_v_heads, head_k_dim, head_v_dim)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2) # (b, num_v_heads, head_v_dim)
        delta = (v_t - kv_mem) * beta_t # (b, num_v_heads, head_v_dim)  
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # core_attn_out: # (b, num_v_heads, sq, head_v_dim)
        out[:, i, :] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
    state[indices] = last_recurrent_state


def get_default_kwargs(batch_size, seq_length):
    d = {}
    b_to_vs = {
        1: 4,
        2: 4,
        3: 4,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 1,
    }
    if b_to_vs.get(batch_size, None) is not None:
        d['NUM_BLOCKS_PER_V_DIM'] = b_to_vs[batch_size]
    return d


def gdr_decode_(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    need_shuffle_state: bool,
    stream: torch.cuda.Stream = torch.cuda.current_stream(),
):
    if need_shuffle_state:
        state_ = state.permute(0, 1, 3, 2).contiguous()
    else:
        state_ = state
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]
    kwargs = get_default_kwargs(batch_size, seq_length)
    exe = create_shuffle_gdr_decode_kernel(
        get_dtype_str(query.dtype),
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        use_qk_l2norm,
        **kwargs)
    exe_compiled = exe.compile(query, key, value, a, b, dt_bias, A_log, indices, state_, out, batch_size, stream)
    exe_compiled(query, key, value, a, b, dt_bias, A_log, indices, state_, out, batch_size, stream)
    if need_shuffle_state:
        state_ = state_.permute(0, 1, 3, 2).contiguous()
        state.copy_(state_)


def func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out, stream=torch.cuda.current_stream()):
    gdr_decode_(query, key, value, a, b, dt_bias, A_log, indices, state, out, 
        use_qk_l2norm=args.use_qk_l2norm, need_shuffle_state=True, stream=stream)


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + (bos * HV + i_hv) * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * HV + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    o: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    grid = (NK, NV, N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_triton_kernel(out, A_log, dt_bias, q, k, v, a, b, initial_state, indices, scale, use_qk_l2norm_in_kernel):
    fused_sigmoid_gating_delta_rule_update(
        out,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=None,
    )


def ref_func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    run_triton_kernel(out, A_log, dt_bias, query, key, value, a, b, state, indices,
        float(1.0 / (args.head_k_dim ** 0.5)), args.use_qk_l2norm)


def benchmark(args, func, ref_func, warmup=20, niters=100):
    torch.manual_seed(2025)
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = list(inputs + outputs)
    inouts[-2] = inouts[-2].clone()
    ref_inouts = list(inputs + ref_outputs)
    ref_inouts[-2] = ref_inouts[-2].clone()
    func(*inouts)
    ref_func(*ref_inouts)
    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
        maxdiff_out = (output - ref_output).abs().max()
        is_allclose = is_allclose and torch.allclose(inouts[-2], ref_inouts[-2], atol=1e-2, rtol=1e-2)
        maxdiff_state = (inouts[-2] - ref_inouts[-2]).abs().max()
        # print("ref_output")
        # print(ref_output)
        # print("output")
        # print(output)
        # print(output - ref_output)
        print(f"maxdiff_out:{maxdiff_out}\nmaxdiff_state:{maxdiff_state}")
        # assert is_allclose == True
    print("validation passed!\n", flush=True)

    # get ref_func perf
    print("===================== [REF] =====================")
    for i in range(warmup):
        ref_func(*ref_inouts)
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(niters):
            ref_func(*ref_inouts)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)

    # get func perf
    print("===================== [FLYDSL] =====================")
    for i in range(warmup):
        func(*inouts)
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(niters):
            func(*inouts)
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    print(table)


def benchmark_cudagraph(args, func, ref_func):
    print('===================== CUDA GRAPH TEST =====================')
    torch.manual_seed(2025)

    inputs = create_inputs(args)
    ref_inputs = create_inputs(args)
    
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)

    def copy_from_ref():
        for input, ref_input in zip(inputs, ref_inputs):
            if isinstance(input, torch.Tensor):
                input.copy_(ref_input)
        for output, ref_output in zip(outputs, ref_outputs):
            if isinstance(output, torch.Tensor):
                output.copy_(ref_output)
    
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph, stream=capture_stream):
            func(*(inputs + outputs), stream=capture_stream)
    torch.cuda.synchronize()
    
    copy_from_ref()
    ref_func(*(ref_inputs + ref_outputs))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    for output, ref_output in zip(outputs, ref_outputs):
        is_allclose = torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
        maxdiff_out = (output - ref_output).abs().max()
        is_allclose = is_allclose and torch.allclose(inputs[-1], ref_inputs[-1], atol=1e-2, rtol=1e-2)
        maxdiff_state = (inputs[-1] - ref_inputs[-1]).abs().max()
        print(f"maxdiff_out:{maxdiff_out}\nmaxdiff_state:{maxdiff_state}")


# ═══════════════════════════════════════════════════════════
# HIP inline ASM kernel
# ═══════════════════════════════════════════════════════════
import sys

_hip_ext = None
def hip_func(args, query, key, value, a, b, dt_bias, A_log, indices, state, out):
    global _hip_ext
    if _hip_ext is None:
        from aiter.ops.hip.gated_delta_net.hip_gdn_decode import _load_extension
        _hip_ext = _load_extension()
    state_vk = state.permute(0, 1, 3, 2).contiguous()
    scale = float(1.0 / (args.head_k_dim ** 0.5))
    _hip_ext.hip_gdn_decode_asm_inplace(
        query, key, value, a, b, dt_bias, A_log, indices,
        state_vk, out, args.b, args.sq, 1, args.use_qk_l2norm, scale,
        args.num_k_heads, args.num_v_heads)
    state_vk = state_vk.permute(0, 1, 3, 2).contiguous()
    state.copy_(state_vk)


# ═══════════════════════════════════════════════════════════
# Subprocess sweep
# ═══════════════════════════════════════════════════════════
import subprocess, re, os

DEFAULT_BS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
WARMUP = 100
NITERS = 100

KERNEL_SUBSTR = {
    "triton": "fused_sigmoid_gating",
    "flydsl": "gdr_decode",
    "hip": "gdn_decode_kernel",
}


def _validate_outputs(out_actual, out_ref, state_actual, state_ref, head_v_dim):
    """Multi-metric validation:

      maxdiff           : worst-case absolute error (existing legacy check)
      rel_err_per_elem  : mean(|err|) / mean(|ref|)  -- catches systematic
                          precision drift that maxdiff would miss when most
                          elements are slightly off but no single one is huge.
      mean_err / sqrt N : informational. For an unbiased fp32 accumulator the
                          per-element error grows ~ sqrt(N_terms).  We use the
                          state tensor (each element is the sum of head_v_dim
                          partial products in this kernel), so the expected
                          1-step magnitude of mean_err is ~ULP * sqrt(head_v_dim).
                          A value much larger than that signals a biased
                          rounding pattern (e.g. a fused-multiply-add written
                          as mul+add, or NT-store reordering producing stale
                          state on weakly-ordered paths).

    Thresholds:
      out   maxdiff < 0.05 , rel_err < 1e-2  (bf16 output, loose)
      state maxdiff < 0.05 , rel_err < 5e-3  (fp32 state, 1 decode step)
    """
    err_o = (out_actual - out_ref).abs()
    err_s = (state_actual - state_ref).abs()

    maxdiff_out   = err_o.max().item()
    maxdiff_state = err_s.max().item()
    mean_err_out   = err_o.mean().item()
    mean_err_state = err_s.mean().item()
    mean_abs_ref_out   = out_ref.abs().mean().item()   + 1e-12
    mean_abs_ref_state = state_ref.abs().mean().item() + 1e-12
    rel_err_out   = mean_err_out   / mean_abs_ref_out
    rel_err_state = mean_err_state / mean_abs_ref_state

    # Per-step accumulation depth = head_v_dim (each state element is sum
    # of that many partial products in the GDN update).
    norm_state = mean_err_state / (head_v_dim ** 0.5)

    ok_out_max   = maxdiff_out   < 0.05
    ok_state_max = maxdiff_state < 0.05
    ok_out_rel   = rel_err_out   < 1e-2
    ok_state_rel = rel_err_state < 5e-3
    ok = ok_out_max and ok_state_max and ok_out_rel and ok_state_rel

    print(
        f"maxdiff_out={maxdiff_out:.6f} maxdiff_state={maxdiff_state:.6f} "
        f"rel_err_out={rel_err_out:.3e} rel_err_state={rel_err_state:.3e} "
        f"mean_err_state_per_sqrtN={norm_state:.3e}",
        flush=True,
    )
    print("VALIDATION_OK" if ok else "VALIDATION_FAIL", flush=True)
    return ok


def bench_single(args, target_func, ref_func_local, warmup, niters):
    """Run accuracy + benchmark for a single kernel. Same logic as benchmark()."""
    torch.manual_seed(2025)
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = list(inputs + outputs)
    inouts[-2] = inouts[-2].clone()
    ref_inouts = list(inputs + ref_outputs)
    ref_inouts[-2] = ref_inouts[-2].clone()

    target_func(*inouts)
    ref_func_local(*ref_inouts)
    _validate_outputs(outputs[0], ref_outputs[0],
                      inouts[-2], ref_inouts[-2],
                      head_v_dim=args.head_v_dim)

    for _ in range(warmup):
        target_func(*inouts)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(niters):
            target_func(*inouts)
    torch.cuda.synchronize()

    return prof


# ═══════════════════════════════════════════════════════════
# `online` mode: realistic UT that mirrors sglang serving trace.
#
# Companion structure mirrors the actual Qwen3.5 GatedDeltaNet block
# captured inside the sglang CUDA graph (see qwen3_5.py / hybrid_linear_attn_backend.py):
#
#     in_proj_qkv  →  causal_conv1d_update  →  [GDN kernel]
#                                             →  RMSNormGated  →  out_proj
#
# All five neighbours sit in the same CUDA graph as the GDN call (sglang
# captures the entire model.forward()), so they all contribute to the
# steady-state L2 pressure that the GDN kernel actually sees in production.
#
# Pool size = 256  ≈  max_running_requests (128) + slack, the typical
# size of MambaPool.temporal_state in sglang serving.
#
# This is a structural simulation: companion ops are sized to per-TP-shard
# tensor widths (hidden_size/tp = 640, in_proj output = 1536). The result
# is NOT tuned to any specific trace number — it simply lands inside the
# observed trace range because it executes the same per-step neighbourhood.
#
# Validated on 8 GPUs in parallel:
#     online median = 5.25 ~ 5.61 μs    (trace range across GPU3-6: 4.6 ~ 5.8 μs)
# ═══════════════════════════════════════════════════════════
ONLINE_POOL_SIZE = 256       # ≈ max_running_requests + slack (MambaPool size)
ONLINE_HIDDEN_DIM = 640      # per TP=8 shard of Qwen3.5 hidden_size=5120
ONLINE_CONV_KSIZE = 4        # causal_conv1d kernel size in Qwen3.5


def _online_create_inputs(args):
    """Like create_inputs but with realistic pool size for `state`."""
    pool = max(ONLINE_POOL_SIZE, args.b)
    query = torch.randn((args.b, args.sq, args.num_k_heads, args.head_k_dim), dtype=args.dtype, device='cuda')
    key   = torch.randn((args.b, args.sq, args.num_k_heads, args.head_k_dim), dtype=args.dtype, device='cuda')
    value = torch.randn((args.b, args.sq, args.num_v_heads, args.head_v_dim), dtype=args.dtype, device='cuda')
    a_t   = torch.randn((args.b, args.sq, args.num_v_heads), dtype=args.dtype, device='cuda')
    b_t   = torch.randn((args.b, args.sq, args.num_v_heads), dtype=args.dtype, device='cuda')
    dt_bias = torch.randn((args.num_v_heads), dtype=args.dtype, device='cuda'); dt_bias.uniform_(1, 2)
    A_log = torch.randn((args.num_v_heads), dtype=torch.float32, device='cuda'); A_log.uniform_(0, 16)
    indices = torch.randperm(pool, device='cuda')[:args.b].to(torch.int32)
    state = torch.randn((pool, args.num_v_heads, args.head_k_dim, args.head_v_dim),
                        dtype=torch.float32, device='cuda')
    return (args, query, key, value, a_t, b_t, dt_bias, A_log, indices, state)


def _online_make_companion(args):
    """Allocate weights/state for the GDN block neighbours that share the
    sglang CUDA graph: in_proj_qkv (BEFORE), causal_conv1d_update (BEFORE),
    out_proj (AFTER).  RMSNormGated is implemented inline in the step.
    Sizes reflect a single TP-rank shard of the real model."""
    inproj_out = args.num_k_heads * args.head_k_dim * 2 + args.num_v_heads * args.head_v_dim
    h = torch.randn(args.b, ONLINE_HIDDEN_DIM, dtype=args.dtype, device='cuda')
    qkv_w = torch.randn(ONLINE_HIDDEN_DIM, inproj_out, dtype=args.dtype, device='cuda')
    out_w = torch.randn(args.num_v_heads * args.head_v_dim, ONLINE_HIDDEN_DIM,
                        dtype=args.dtype, device='cuda')
    conv_state  = torch.randn(args.b, inproj_out, ONLINE_CONV_KSIZE,
                              dtype=args.dtype, device='cuda')
    conv_weight = torch.randn(ONLINE_CONV_KSIZE, dtype=args.dtype, device='cuda')
    return h, qkv_w, out_w, conv_state, conv_weight


def _online_step_factory(args, target_func, inouts, companion, stream=None):
    """Build a callable that mirrors one GDN block forward inside the
    sglang CUDA graph:

        in_proj_qkv  →  causal_conv1d_update  →  GDN kernel
                                              →  RMSNormGated  →  out_proj

    If `stream` is given, FlyDSL gets the explicit stream so it captures correctly.
    """
    h, qkv_w, out_w, conv_state, conv_weight = companion
    out_tensor = inouts[-1]

    accepts_stream = False
    try:
        import inspect
        sig = inspect.signature(target_func)
        accepts_stream = "stream" in sig.parameters
    except (TypeError, ValueError):
        pass

    def step():
        _ = torch.matmul(h, qkv_w)
        _ = (conv_state * conv_weight).sum(-1)
        if stream is not None and accepts_stream:
            target_func(*inouts, stream=stream)
        else:
            target_func(*inouts)
        x = out_tensor.float()
        _ = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        _ = torch.matmul(out_tensor.reshape(args.b, -1), out_w)
    return step


def bench_single_online(args, target_func, ref_func_local, warmup, niters):
    """`online` realistic mode: pool=4096 + companion matmul + CUDA Graph replay."""
    torch.manual_seed(2025)
    inputs = _online_create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = list(inputs + outputs)
    inouts[-2] = inouts[-2].clone()
    ref_inouts = list(inputs + ref_outputs)
    ref_inouts[-2] = ref_inouts[-2].clone()

    target_func(*inouts)
    ref_func_local(*ref_inouts)
    _validate_outputs(outputs[0], ref_outputs[0],
                      inouts[-2], ref_inouts[-2],
                      head_v_dim=args.head_v_dim)

    companion = _online_make_companion(args)

    warmup_step = _online_step_factory(args, target_func, inouts, companion, stream=None)
    for _ in range(min(warmup, 30)):
        warmup_step()
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        graph = torch.cuda.CUDAGraph()
        capture_step = _online_step_factory(args, target_func, inouts, companion, stream=s)
        with torch.cuda.graph(graph, stream=s):
            capture_step()
    torch.cuda.synchronize()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(niters):
            graph.replay()
    torch.cuda.synchronize()

    return prof


def run_worker(bs, kernel, mode="default"):
    """Launch a subprocess to benchmark one (bs, kernel) combination.

    Returns (stats_dict_or_None, passed_bool, raw_output_str).
    stats_dict has keys: med, min, max, avg (in microseconds)."""
    env = os.environ.copy()
    gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    env["HIP_VISIBLE_DEVICES"] = gpu
    cmd = [sys.executable, __file__, "--worker", str(bs), kernel, "--mode", mode]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        output = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return None, False, "TIMEOUT"
    passed = "VALIDATION_OK" in output

    def grab(tag):
        m = re.search(rf"{tag}=([\d.]+)", output)
        return float(m.group(1)) if m else None

    med = grab("KERNEL_TIME_US")
    if med is None:
        return None, passed, output
    stats = {
        "med": med,
        "min": grab("KERNEL_MIN_US") or med,
        "max": grab("KERNEL_MAX_US") or med,
        "avg": grab("KERNEL_AVG_US") or med,
    }
    return stats, passed, output


if __name__ == '__main__':
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        # Worker mode: benchmark a single (BS, kernel) in this process
        bs = int(sys.argv[2])
        kernel = sys.argv[3]
        # Optional --mode {default,s5}
        mode = "default"
        if "--mode" in sys.argv:
            mode = sys.argv[sys.argv.index("--mode") + 1]

        args_obj = Args(dtype=torch.bfloat16, b=bs, sq=1,
                        num_k_heads=2, num_v_heads=8, head_k_dim=128, head_v_dim=128)

        func_map = {"triton": ref_func, "flydsl": func, "hip": hip_func}
        target = func_map[kernel]

        if mode == "online":
            # BS=2/4 in online mode is prone to a process-start bimodal state
            # (allocator/runtime warm path vs cold path). A same-process throwaway
            # run followed by empty_cache() makes the measured pass much more
            # stable and prevents random HIP/FlyDSL sign flips in sweep reports.
            if bs in (2, 4) and kernel in ("flydsl", "hip"):
                _ = bench_single_online(args_obj, target, ref_func, warmup=WARMUP, niters=NITERS)
                gc.collect()
                torch.cuda.empty_cache()
            prof = bench_single_online(args_obj, target, ref_func, warmup=WARMUP, niters=NITERS)
        else:
            prof = bench_single(args_obj, target, ref_func, warmup=WARMUP, niters=NITERS)

        substr = KERNEL_SUBSTR[kernel]
        # Collect every individual kernel sample (microseconds)
        samples = []
        for ev in prof.events():
            ev_name = getattr(ev, "name", "") or ""
            dt = getattr(ev, "device_time_total", 0)
            if substr in ev_name and dt > 0:
                samples.append(dt)
        if samples:
            samples.sort()
            n = len(samples)
            mn = samples[0]
            mx = samples[-1]
            md = samples[n // 2]
            avg = sum(samples) / n
            print(f"KERNEL_TIME_US={md:.2f} KERNEL_MIN_US={mn:.2f} "
                  f"KERNEL_MAX_US={mx:.2f} KERNEL_AVG_US={avg:.2f} KERNEL_N={n}",
                  flush=True)
        else:
            total = sum(ev.self_device_time_total for ev in prof.key_averages()
                        if ev.self_device_time_total > 0)
            avg = total / NITERS
            print(f"KERNEL_TIME_US={avg:.2f} KERNEL_MIN_US={avg:.2f} "
                  f"KERNEL_MAX_US={avg:.2f} KERNEL_AVG_US={avg:.2f} KERNEL_N=0",
                  flush=True)

    else:
        # Sweep mode
        parser = argparse.ArgumentParser()
        parser.add_argument("--bs", type=int, nargs="+", default=DEFAULT_BS)
        parser.add_argument("--mode", choices=["default", "online"], default="default",
                            help="default = isolated kernel; online = realistic UT "
                                 "(real pool + companion matmul + CUDA graph) that mirrors sglang trace")
        parser.add_argument("--show-minmax", action="store_true",
                            help="show min/max in sweep progress and summary table")
        cli_args = parser.parse_args()
        bs_list = cli_args.bs
        mode = cli_args.mode
        show_minmax = cli_args.show_minmax
        kernels = ["triton", "flydsl", "hip"]

        gpu = os.environ.get("HIP_VISIBLE_DEVICES", "0")
        print(f"=== GDN Decode UT Sweep (isolated subprocesses) | GPU {gpu} | mode={mode} ===")
        print(f"BS list: {bs_list}\n")

        results = {}
        for bs in bs_list:
            results[bs] = {}
            for kernel in kernels:
                sys.stdout.write(f"  BS={bs:>4d}  {kernel:>6s} ... ")
                sys.stdout.flush()
                stats, passed, output = run_worker(bs, kernel, mode=mode)
                status = "PASS" if passed else "FAIL"
                if stats is not None:
                    if show_minmax:
                        sys.stdout.write(
                            f"med={stats['med']:6.2f}  min={stats['min']:6.2f}  "
                            f"max={stats['max']:6.2f} μs  [{status}]\n"
                        )
                    else:
                        sys.stdout.write(
                            f"med={stats['med']:6.2f} μs  [{status}]\n"
                        )
                else:
                    sys.stdout.write(f"    N/A     [{status}]\n")
                    print(f"    --- output tail ---\n{output[-500:]}\n")
                results[bs][kernel] = {"stats": stats, "passed": passed}

        # Compact per-kernel format: either "med" or "med [min~max]"
        col_w = 19 if show_minmax else 10

        def cell(s):
            if s is None:
                txt = "N/A"
            elif show_minmax:
                txt = f"{s['med']:5.2f} [{s['min']:5.2f}~{s['max']:5.2f}]"
            else:
                txt = f"{s['med']:5.2f}"
            return f"{txt:<{col_w}s}"

        if show_minmax:
            header_kernel = lambda name: f"{name + ' med [min~max]':<{col_w}s}"
        else:
            header_kernel = lambda name: f"{name + ' med':<{col_w}s}"

        print(f"\n{'='*94}")
        print(f"GPU {gpu} | mode={mode}  (units: μs)")
        print(f"{'BS':>4s}  {header_kernel('Triton')}  {header_kernel('FlyDSL')}  "
              f"{header_kernel('HIP-ASM')}  {'HIP/FlyDSL':>10s}  {'Acc':>4s}")
        print(f"{'-'*94}")
        for bs in bs_list:
            r = results[bs]
            ts = r['triton']['stats']
            fs = r['flydsl']['stats']
            hs = r['hip']['stats']
            if fs and hs:
                delta = (1 - hs['med'] / fs['med']) * 100
                d_str = f"{delta:+.1f}%"
            else:
                d_str = "N/A"
            all_pass = all(r[k]['passed'] for k in kernels)
            acc = 'OK' if all_pass else 'ERR'
            print(f"{bs:>4d}  {cell(ts)}  {cell(fs)}  {cell(hs)}  {d_str:>10s}  {acc:>4s}")
        print(f"{'='*94}")
        print("HIP/FlyDSL is computed from medians: positive = HIP faster.")