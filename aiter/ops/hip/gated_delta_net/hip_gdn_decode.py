"""
HIP/ASM GDN decode kernel for sglang.

Drop-in replacement for fused_sigmoid_gating_delta_rule_update (Triton)
in decode mode.  Uses inline-assembly optimized kernel with template-
specialised head dispatch (TP=8 and Full-Heads paths).

State layout: expects [pool, HV, V, K] (VK layout).  The sglang
backend (commit da96f57) transposes once after extend ([K,V]→[V,K])
so the decode path runs without any per-step transpose overhead.

Kernel parameters are specialized for Qwen3.5:
  K_heads=16, V_heads=32, K=128, V=128, bf16.
"""

from typing import Optional

import torch

_ext = None


def _load_extension():
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    import os

    src_dir = os.path.dirname(os.path.abspath(__file__))
    _ext = load(
        name="hip_gdn_decode_ext",
        sources=[
            os.path.join(src_dir, "gdn_decode_ext.cpp"),
            os.path.join(src_dir, "gdn_decode_kernel_hip.hip"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-std=c++17"],
        verbose=False,
    )
    return _ext


def hip_fused_sigmoid_gating_delta_rule_update(
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
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
):
    """VK decode kernel (inline-ASM) — state must already be in [V,K] layout."""
    ext = _load_extension()

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]

    if scale is None:
        scale = K ** -0.5

    N = B * T if cu_seqlens is None else len(cu_seqlens) - 1

    o = torch.empty_like(v)

    dt_bias_bf16 = dt_bias.to(torch.bfloat16) if dt_bias.dtype != torch.bfloat16 else dt_bias

    indices_int32 = (
        initial_state_indices.to(torch.int32)
        if initial_state_indices.dtype != torch.int32
        else initial_state_indices
    )

    batch_size = N
    seq_length = 1 if cu_seqlens is not None else T
    num_k_heads = H
    num_v_heads = HV

    ext.hip_gdn_decode_asm_inplace(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        dt_bias_bf16,
        A_log.contiguous(),
        indices_int32,
        initial_state_source,
        o,
        batch_size,
        seq_length,
        1,  # num_v_blocks (auto-selected by launcher)
        use_qk_l2norm_in_kernel,
        scale,
        num_k_heads,
        num_v_heads,
    )

    return o


def hip_state_transpose_inplace(
    state: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    num_v_heads: int,
):
    """In-place 128x128 state transpose [K,V] <-> [V,K] for selected pool slots."""
    ext = _load_extension()
    indices_int32 = (
        indices.to(torch.int32)
        if indices.dtype != torch.int32
        else indices
    )
    ext.hip_state_transpose(state, indices_int32, batch_size, num_v_heads)
