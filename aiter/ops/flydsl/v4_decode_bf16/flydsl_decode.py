"""FlyDSL bf16 MLA paged-decode entry — decode ALWAYS runs the FlyDSL kernel.

No Triton fallback, no env switch, no kernel selection: whoever calls this uses
FlyDSL, unconditionally.

Compilation is cached by ``build_dpb`` (``@lru_cache``) and driven lazily, as in
the other aiter FlyDSL kernels. The one wrinkle: each ``KV_SPLITS`` is a separate
compiled variant and a CUDA-graph capture sweeps several batch sizes that resolve
to different ``KV_SPLITS`` -- and JIT compilation is illegal inside a capture. So
on the first eager call (per ``softmax_scale``) we build every ``KV_SPLITS``
variant up front; by capture time all variants are cache hits (just launches).
The first eager decode (e.g. a framework profile/warm-up step) triggers this
automatically before graphs are captured.
"""
from aiter.ops.flydsl.v4_decode_bf16 import v4_decode_dsplit_dpb as _dpb

_prebuilt_scale = None


def _ensure_built(softmax_scale: float) -> None:
    """Pre-build every KV_SPLITS variant for this scale (idempotent, eager only).

    build_dpb is @lru_cache'd, so repeated calls are cheap; this only forces the
    full set to exist before CUDA-graph capture picks an as-yet-uncompiled one.
    """
    global _prebuilt_scale
    if _prebuilt_scale == float(softmax_scale):
        return
    qk = float(softmax_scale) * _dpb.LOG2E
    for ks in (1, 2, 4, 8, 16, 32, 64, 128):
        _dpb.build_dpb(qk, ks, 8, 1)
    _prebuilt_scale = float(softmax_scale)


def flydsl_decode(q, unified_kv, kv_indices, kv_indptr, attn_sink, softmax_scale):
    _ensure_built(softmax_scale)
    # kv_len for KV_SPLITS/tail selection -- SHAPE-BASED, no GPU->CPU sync.
    # mean gathered length (kv_indices.numel()/T) avoids the per-step .item() host
    # sync AND, unlike a capture-time kvL=63 (which forces KS=1), lets the
    # captured graph pick a proper KV_SPLITS. Safe under CUDA-graph capture.
    T = int(q.shape[0])
    rk = max(kv_indices.numel() // max(T, 1), 64)
    kvL = rk if (rk % 64) else (rk + 1)
    d = {"T": T, "h": int(q.shape[1]), "kv_len": kvL,
         "sm": float(softmax_scale), "q": q, "unified_kv": unified_kv,
         "kv_indices": kv_indices, "kv_indptr": kv_indptr, "attn_sink": attn_sink}
    return _dpb.flydsl_dpb_full(d)
