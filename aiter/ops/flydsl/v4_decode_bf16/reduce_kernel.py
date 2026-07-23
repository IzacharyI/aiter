"""Triton split-K reduce for the FlyDSL bf16 MLA paged-decode kernel.

Combines the ``KV_SPLITS`` partials (m/l/acc) produced by the FlyDSL split
kernel, folds the attention sink, and writes the final output. Vendored here
so the decode path is self-contained inside aiter (no external module load).

2D-tile reduce, grid ``(T, H, ceil(D / D_CHUNK))`` — one CTA per
(token, single-head, D-chunk). The merged ``[KV_SPLITS, D_CHUNK]`` acc load
fits a wave's VGPR and the D-chunked grid widens occupancy at small T, where
the reduce is the latency bottleneck.
"""
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _paged_decode_reduce_kernel(
    m_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    l_partial_ptr,  # [N, KV_SPLITS, H_padded] fp32
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D] fp32
    attn_sink_ptr,  # [H]
    kv_indptr_ptr,  # [N+1] int32
    out_ptr,  # [N, H, D]
    mp_stride_t,
    mp_stride_k,
    mp_stride_h,
    lp_stride_t,
    lp_stride_k,
    lp_stride_h,
    ap_stride_t,
    ap_stride_k,
    ap_stride_h,
    ap_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    log2e,  # = LOG2E, used to convert natural-log sink -> log2 domain
    H: tl.constexpr,
    D: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)

    d_offs = dc * D_CHUNK + tl.arange(0, D_CHUNK)
    k_offs = tl.arange(0, KV_SPLITS)
    d_mask = d_offs < D

    neg_large = -3.4028234663852886e38

    kv_start = tl.load(kv_indptr_ptr + t)
    kv_end = tl.load(kv_indptr_ptr + t + 1)
    kv_len = kv_end - kv_start
    # CTA-level early return for empty tokens (CUDAGraph padding, or any
    # caller-supplied zero-length slice). Split kernel skipped these without
    # writing partials -> partial buffers hold garbage; skipping the whole CTA
    # also halves the reduce cost on mixed-kv batches with many padded tokens.
    if kv_len == 0:
        out_off = t * out_stride_t + h * out_stride_h + d_offs * out_stride_d
        tl.store(
            out_ptr + out_off,
            tl.zeros([D_CHUNK], dtype=out_ptr.dtype.element_ty),
            mask=d_mask,
        )
        return
    tiles_per_segment = tl.cdiv(kv_len, KV_SPLITS * BLOCK_K)
    act_num_segments = tl.cdiv(kv_len, tl.maximum(tiles_per_segment, 1) * BLOCK_K)
    segm_mask = k_offs < act_num_segments

    # 1D loads for (m, l) along splits -- single head h.
    m_p = tl.load(
        m_partial_ptr + t * mp_stride_t + k_offs * mp_stride_k + h * mp_stride_h,
        mask=segm_mask,
        other=neg_large,
    )  # [KV_SPLITS]
    l_p = tl.load(
        l_partial_ptr + t * lp_stride_t + k_offs * lp_stride_k + h * lp_stride_h,
        mask=segm_mask,
        other=0.0,
    )  # [KV_SPLITS]

    # 2D-tile load for acc partials -- the key change vs a strided 3D load.
    a_p = tl.load(
        acc_partial_ptr
        + t * ap_stride_t
        + k_offs[:, None] * ap_stride_k
        + h * ap_stride_h
        + d_offs[None, :] * ap_stride_d,
        mask=segm_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # [KV_SPLITS, D_CHUNK]

    # Combine across splits.
    m_max = tl.max(m_p, axis=0)  # scalar
    alpha_split = tl.exp2(m_p - m_max)  # [KV_SPLITS]
    l_combined = tl.sum(l_p * alpha_split, axis=0)  # scalar
    acc_combined = tl.sum(a_p * alpha_split[:, None], axis=0)  # [D_CHUNK]

    # Fold attn_sink (recomputed across dc -- scalar work, negligible).
    sink_raw = tl.load(attn_sink_ptr + h).to(tl.float32)
    sink = sink_raw * log2e
    m_final = tl.maximum(m_max, sink)
    alpha_kv = tl.exp2(m_max - m_final)
    alpha_sink = tl.exp2(sink - m_final)
    l_final = l_combined * alpha_kv + alpha_sink

    denom = tl.maximum(l_final, 1.0e-30)
    # Direct divide (acc*alpha_kv)/denom, matching the single-CTA reference.
    # A precomputed reciprocal-scaled scalar diverges by ~1 ulp per element
    # across batch shapes under split-K, which can flip MTP-accepted tokens.
    acc_final = acc_combined * alpha_kv
    out = tl.where(l_final > 0.0, acc_final / denom, 0.0)

    tl.store(
        out_ptr + t * out_stride_t + h * out_stride_h + d_offs * out_stride_d,
        out.to(out_ptr.dtype.element_ty),
        mask=d_mask,
    )
