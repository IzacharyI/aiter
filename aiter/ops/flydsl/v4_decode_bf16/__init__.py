"""FlyDSL bf16 MLA paged-decode kernel (v4 dpb, BK=64).

Self-contained inside aiter: the split kernel is FlyDSL, the split-K reduce is
Triton (``reduce_kernel``), CU count comes from ``aiter.jit.utils.chip_info``.
``flydsl_decode`` runs the FlyDSL decode unconditionally (no Triton fallback,
no kernel selection); compilation is @lru_cache-driven like the other FlyDSL
kernels.
"""
from aiter.ops.flydsl.v4_decode_bf16.v4_decode_dsplit_dpb import (
    build_dpb,
    flydsl_dpb_full,
)
from aiter.ops.flydsl.v4_decode_bf16.flydsl_decode import flydsl_decode

__all__ = ["build_dpb", "flydsl_dpb_full", "flydsl_decode"]
