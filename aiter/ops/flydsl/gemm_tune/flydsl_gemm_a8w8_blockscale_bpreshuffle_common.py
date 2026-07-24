# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL a8w8 *blockscale* bpreshuffle GEMM tuner candidate list.

Sibling of ``flydsl_gemm_a8w8_bpreshuffle_common.py`` but for the DeepSeek-style
128-block-scaled fp8 kernel (``kernels/blockscale_preshuffle_gemm.py``).

The ``kernelInstance.name`` MUST serialize to exactly the format the runtime
dispatcher parses in
``aiter/ops/gemm_op_a8w8.py::_parse_flydsl_blockscale_kernel_name``::

    flydsl_blockscale_bpreshuffle_{tm}x{tn}x{tk}_sbk{sbk}_{csh}x{acp}x{wpe}x{xcd}x{fp}

so that ``name -> parse -> wrapper args`` round-trips to the same config the
tuner measured. Unlike the non-blockscale name, there is no lds_stage / dtype
token (the blockscale kernel is always ping/pong double-buffered).
"""
from dataclasses import dataclass
import math
import os

from aiter.ops.flydsl.utils import get_shared_memory_per_block


def get_gfx():
    """Detect GPU arch: honour GPU_ARCHS env, fall back to chip_info, default gfx950."""
    env = os.environ.get("GPU_ARCHS", "")
    if env and env != "native":
        return env.split(";")[-1].strip()
    try:
        from aiter.jit.utils.chip_info import get_gfx as _get_gfx

        return _get_gfx()
    except Exception:
        return "gfx950"


@dataclass
class kernelInstance:
    tile_m: int
    tile_n: int
    tile_k: int
    scale_block_k: int
    use_cshuffle_epilog: int  # 0 or 1
    use_async_copy: int  # 0 or 1
    waves_per_eu: int  # 0=no hint, 1-4=occupancy limit
    xcd_swizzle: int  # 0=off, >0=XCD grouped rasterization group height
    fused_promote: int  # 0 or 1 (gfx950 only; no effect elsewhere)

    @property
    def name(self) -> str:
        return "_".join(
            [
                "flydsl_blockscale_bpreshuffle",
                "x".join(map(str, [self.tile_m, self.tile_n, self.tile_k])),
                f"sbk{self.scale_block_k}",
                "x".join(
                    map(
                        str,
                        [
                            self.use_cshuffle_epilog,
                            self.use_async_copy,
                            self.waves_per_eu,
                            self.xcd_swizzle,
                            self.fused_promote,
                        ],
                    )
                ),
            ]
        )


def _ki(
    tile_m,
    tile_n,
    tile_k=128,
    scale_block_k=128,
    cshuffle=0,
    async_copy=0,
    waves_per_eu=0,
    xcd_swizzle=0,
    fused_promote=0,
):
    return kernelInstance(
        tile_m,
        tile_n,
        tile_k,
        scale_block_k,
        cshuffle,
        async_copy,
        waves_per_eu,
        xcd_swizzle,
        fused_promote,
    )


# ---------------------------------------------------------------------------
# LDS estimate (mirrors compile_blockscale_preshuffle_gemm ping/pong sizing)
# ---------------------------------------------------------------------------
def _align16(x: int) -> int:
    return (x + 15) // 16 * 16


def blockscale_estimated_lds_bytes(
    tile_m: int, tile_n: int, tile_k: int, use_cshuffle_epilog: int
) -> int:
    """Estimated total LDS bytes: two (ping/pong) buffers of max(A-tile, out/2)."""
    lds_tile_bytes = tile_m * tile_k  # fp8, 1 byte/elem
    lds_out_bytes = 2 * tile_m * tile_n if use_cshuffle_epilog else 0
    buffer_size_bytes = max(lds_tile_bytes, lds_out_bytes // 2)
    return 2 * _align16(buffer_size_bytes)


def kernel_instance_estimated_lds_bytes(ki: kernelInstance) -> int:
    return blockscale_estimated_lds_bytes(
        ki.tile_m, ki.tile_n, ki.tile_k, ki.use_cshuffle_epilog
    )


def max_lds_bytes_for_tune() -> int:
    return get_shared_memory_per_block(fallback_gfx=get_gfx())


# ---------------------------------------------------------------------------
# waves_per_eu ceiling from 16x16 MFMA C-accumulator VGPR pressure
# ---------------------------------------------------------------------------
_MFMA_M = 16
_MFMA_N = 16
_WAVES_PER_WG = 4
_THREADS_PER_TG = _WAVES_PER_WG * 64


def _estimate_max_wpe(tile_m: int, tile_n: int, total_vgpr: int = 512) -> int:
    padded_m = math.ceil(tile_m / _MFMA_M) * _MFMA_M
    padded_n = math.ceil(tile_n / _MFMA_N) * _MFMA_N
    c_per_thread = padded_m * padded_n // _THREADS_PER_TG
    est_per_wave = c_per_thread * 1.5
    return int(total_vgpr / max(est_per_wave, 1))


# ---------------------------------------------------------------------------
# Candidate sweep
# ---------------------------------------------------------------------------
# tile_k is pinned to the DeepSeek scale block (128); the gfx950 mfma_scale path
# consumes k=128 per step, and tile_k must be a multiple of scale_block_k.
_TILE_K = 128
_SCALE_BLOCK_K = 128

# (tile_m, tile_n) base tiles. Trimmed to the set the vendored 0.2.0 kernel
# actually compiles + validates (see validate_flydsl_blockscale_tiles.py).
_TILES_MN = [
    (16, 128),
    (16, 256),
    (32, 128),
    (32, 256),
    (64, 128),
    (64, 256),
    (128, 128),
    (128, 256),
    (256, 128),
    (256, 256),
]

_CSHUFFLE_VALS = (0, 1)
_ASYNC_COPY_VALS = (0, 1)
_WAVES_PER_EU = (0, 1, 2, 3, 4)
_XCD_SWIZZLE_VALS = (0, 8)
_FUSED_PROMOTE_VALS = (0, 1)


def _tile_compiles(tile_m: int, tile_n: int, tile_k: int) -> bool:
    """Static feasibility per compile_blockscale_preshuffle_gemm asserts."""
    total_threads = 256
    if tile_k % 64 != 0:
        return False
    if (tile_m * tile_k) % total_threads != 0:
        return False
    # A per-thread load must be a multiple of 4 bytes.
    if (tile_m * tile_k // total_threads) % 4 != 0:
        return False
    # B is loaded in 16-byte chunks across 256 threads.
    if (tile_n * tile_k) % (total_threads * 16) != 0:
        return False
    return True


def _config_is_valid(tile_m: int, tile_n: int, csh: int, acp: int, fp: int) -> bool:
    """Numerically-validated feasibility (validate_flydsl_blockscale_tiles.py).

    All 10 base tiles pass every (cshuffle, async_copy, fused_promote) combo on
    the reference GEMM EXCEPT ``tile_m <= 16`` with ``async_copy`` enabled, whose
    async A-load path produces garbage/NaN. Everything else is bit-safe.
    """
    if tile_m <= 16 and acp:
        return False
    return True


def _build_kernels_list(total_vgpr: int = 512):
    kl = {}
    idx = 0
    for wpe in _WAVES_PER_EU:
        for csh in _CSHUFFLE_VALS:
            for acp in _ASYNC_COPY_VALS:
                for xcd in _XCD_SWIZZLE_VALS:
                    for fp in _FUSED_PROMOTE_VALS:
                        for tm, tn in _TILES_MN:
                            if not _tile_compiles(tm, tn, _TILE_K):
                                continue
                            if not _config_is_valid(tm, tn, csh, acp, fp):
                                continue
                            if wpe > 0 and wpe > _estimate_max_wpe(tm, tn, total_vgpr):
                                continue
                            kl[idx] = _ki(
                                tm, tn, _TILE_K, _SCALE_BLOCK_K, csh, acp, wpe, xcd, fp
                            )
                            idx += 1
    return kl


kernels_list = _build_kernels_list()

# Fallback configs (known-good, validated during integration).
default_kernels_dict = {
    (-1): _ki(64, 256, 128, 128, 0, 0, 2, 8, 0),
    (-2): _ki(64, 256, 128, 128, 0, 0, 2, 8, 1),
    (-3): _ki(128, 256, 128, 128, 0, 0, 2, 0, 0),
}

# Name-keyed reverse lookup (parity with the CK common), so codegen/round-trip
# checks can map a tuned CSV kernelName back to its kernelInstance.
kernels_by_name = {v.name: v for v in kernels_list.values()}


# ---------------------------------------------------------------------------
# Focused sweep subset (opt-in via env FLYDSL_BS_TUNE_FOCUSED=1)
# ---------------------------------------------------------------------------
# A curated ~75-config cross-section spanning all 10 tiles x a hand-picked flag
# grid (xcd on/off, cshuffle on/off, waves_per_eu {0,2}, fused_promote {0,1},
# async where the A-load path is valid). It keeps the proven 64x256
# cshuffle+async+wpe2+xcd8 winner family AND plain/xcd-only/no-cshuffle baselines
# so the tuner still makes a real per-shape choice, at ~1/8 the tasks of the full
# 608-config sweep. Used to regenerate the tuned CSV quickly on a few GPUs; the
# full sweep remains the default so nothing about the committed capability shrinks.
#
# NOTE: keys are a *strict subset* of kernels_list keys, so kernel ids stay valid
# for run_gemm_a8w8_blockscale_flydsl / getKernelName (which index the full list).
#
# flag signature = (cshuffle, async_copy, waves_per_eu, xcd_swizzle, fused_promote)
_FOCUSED_SIGNATURES_ASYNC = {  # tile_m > 16: async A-load path is valid
    (0, 0, 0, 0, 0),  # plain direct-epilog baseline
    (0, 1, 0, 8, 0),  # + async + xcd
    (0, 1, 2, 8, 0),  # + async + xcd + wpe2
    (0, 1, 2, 8, 1),  # + async + xcd + wpe2 + fused_promote
    (1, 1, 2, 8, 0),  # cshuffle + async + xcd + wpe2  (proven winner family)
    (1, 1, 2, 8, 1),  # cshuffle + async + xcd + wpe2 + fused_promote (proven)
    (1, 1, 0, 8, 0),  # cshuffle + async + xcd (default occupancy)
    (1, 1, 2, 0, 0),  # cshuffle + async + wpe2, no xcd (skinny-N shapes)
}
_FOCUSED_SIGNATURES_SYNC = {  # tile_m <= 16: async invalid (_config_is_valid), sync only
    (0, 0, 0, 0, 0),
    (0, 0, 0, 8, 0),
    (0, 0, 2, 8, 0),
    (0, 0, 2, 8, 1),
    (1, 0, 2, 8, 0),
    (1, 0, 2, 8, 1),
    (1, 0, 0, 8, 0),
    (1, 0, 2, 0, 0),
}


def is_focused_kernel(ki: kernelInstance) -> bool:
    sig = (
        ki.use_cshuffle_epilog,
        ki.use_async_copy,
        ki.waves_per_eu,
        ki.xcd_swizzle,
        ki.fused_promote,
    )
    sigset = _FOCUSED_SIGNATURES_SYNC if ki.tile_m <= 16 else _FOCUSED_SIGNATURES_ASYNC
    return sig in sigset


focused_kernels_list = {
    i: ki for i, ki in kernels_list.items() if is_focused_kernel(ki)
}


def get_tune_kernels_list():
    """Candidate dict for a tuning run.

    Returns the focused subset when ``FLYDSL_BS_TUNE_FOCUSED`` is truthy, else the
    full sweep. Keys are always a subset of ``kernels_list`` keys so ids stay valid
    for the runner and ``getKernelName`` (both index the full list).
    """
    flag = os.environ.get("FLYDSL_BS_TUNE_FOCUSED", "").strip().lower()
    if flag not in ("", "0", "false", "no"):
        return focused_kernels_list
    return kernels_list


# ---------------------------------------------------------------------------
# 8-wave (1024-thread) blockscale ping-pong candidates
# ---------------------------------------------------------------------------
# A separate kernel family (``kernels/fp8_gemm_8wave_blockscale.py``,
# ``compile_fp8_gemm_8w_blockscale`` -> the shipped "cluster" schedule). It wins
# on compute-bound skinny-N shapes (e.g. qkv_proj 4096x2048x7168) where the
# 4-wave ``blockscale_preshuffle`` family loses to CK. It consumes the SAME
# preshuffled B bytes as CK/4-wave -- ``fp8_gemm_utils.preshuffle_b(W)`` is
# byte-identical to ``shuffle_weight(W, layout=(16,16))`` for K%64==0 -- so it is
# a drop-in with no weight re-layout.
#
# The runtime dispatcher parses ``kernelInstance8w.name`` in
# ``aiter/ops/gemm_op_a8w8.py::_parse_flydsl_blockscale_8w_kernel_name``::
#
#     flydsl8w_blockscale_bpreshuffle_{bm}x{bn}_wpe{wpe}_xcd{xcd}
#
# so name -> parse -> wrapper args round-trips to the config the tuner measured.
# ``bn`` (BLOCK_N) is locked to 256 (the scale-block alignment the kernel asserts).
_BLOCK_N_8W = 256
_WAVES_PER_EU_8W = 2  # kernel default; wpe4 spills, wpe hint is a no-op (see handoff)


@dataclass
class kernelInstance8w:
    block_m: int
    block_n: int
    waves_per_eu: int
    use_xcd_remap: int  # 0 or 1

    @property
    def name(self) -> str:
        return (
            "flydsl8w_blockscale_bpreshuffle_"
            f"{self.block_m}x{self.block_n}_"
            f"wpe{self.waves_per_eu}_xcd{self.use_xcd_remap}"
        )


def _build_kernels_list_8w():
    kl = {}
    idx = 0
    for bm in (128, 256):
        for xcd in (0, 1):
            kl[idx] = kernelInstance8w(bm, _BLOCK_N_8W, _WAVES_PER_EU_8W, xcd)
            idx += 1
    return kl


kernels_list_8w = _build_kernels_list_8w()
kernels_by_name_8w = {v.name: v for v in kernels_list_8w.values()}


def get_tune_kernels_list_8w():
    """Candidate dict for the 8-wave blockscale family (small, no focused subset).

    Keys index ``kernels_list_8w`` (a distinct id space from the 4-wave
    ``kernels_list``); the tuner uses a distinct ``flydsl8w`` libtype + runner so
    the two families never collide.
    """
    return kernels_list_8w
