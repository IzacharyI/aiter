# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime config metadata for the gfx950 FlyDSL MXScale GEMM.

Runtime dispatch and tuning share this module without introducing a runtime
dependency on ``gemm_tune``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_DTYPE_SHORT = {
    "fp8": "F8",
    "fp6": "F6",
    "fp4": "F4",
    "bf16": "B16",
    "fp16": "F16",
}
_SHORT_DTYPE = {value: key for key, value in _DTYPE_SHORT.items()}

_COMBOS = (("fp4", "fp4"), ("fp6", "fp4"), ("fp8", "fp8"))
_TILE_M = (32, 64, 96, 128, 256)
_TILE_N = (128, 256, 512)
_TILE_K = (128, 256)
_WAVES_PER_EU = (0, 1, 2, 3, 4)
_XCD_SWIZZLE = (0, 4)


def _a_row_bytes(a_dtype: str, tile_k: int) -> int:
    return tile_k // 2 if a_dtype == "fp4" else tile_k


@dataclass(frozen=True)
class kernelInstance:
    """Stable representation of one serialized MXScale kernel config."""

    tile_m: int
    tile_n: int
    tile_k: int
    a_dtype: str
    b_dtype: str
    out_dtype: str
    waves_per_eu: int
    xcd_swizzle: int = 0
    split_k: int = 1

    @property
    def name(self) -> str:
        return (
            f"flydsl_mxpsh_{self.tile_m}x{self.tile_n}x{self.tile_k}"
            f"_{_DTYPE_SHORT[self.a_dtype]}_{_DTYPE_SHORT[self.b_dtype]}"
            f"_{_DTYPE_SHORT[self.out_dtype]}_w{self.waves_per_eu}"
            f"_x{self.xcd_swizzle}"
        )


_NAME_RE = re.compile(
    r"^flydsl_mxpsh_(\d+)x(\d+)x(\d+)_([A-Z0-9]+)_([A-Z0-9]+)_"
    r"([A-Z0-9]+)_w(\d+)(?:_x(\d+))?$"
)


def parse_kernel_name(name: str):
    """Parse a PR #4254 ``flydsl_mxpsh_*`` kernel name."""
    match = _NAME_RE.fullmatch(str(name).strip())
    if match is None:
        return None
    tm, tn, tk, a_short, b_short, out_short, wpe, xcd = match.groups()
    if any(value not in _SHORT_DTYPE for value in (a_short, b_short, out_short)):
        return None
    return {
        "tile_m": int(tm),
        "tile_n": int(tn),
        "tile_k": int(tk),
        "a_dtype": _SHORT_DTYPE[a_short],
        "b_dtype": _SHORT_DTYPE[b_short],
        "out_dtype": _SHORT_DTYPE[out_short],
        "waves_per_eu": int(wpe),
        "xcd_swizzle": int(xcd) if xcd is not None else 0,
        "split_k": 1,
    }


def estimated_lds_bytes(instance: kernelInstance) -> int:
    """Double-buffered A tile size used by the current kernel."""
    return 2 * instance.tile_m * _a_row_bytes(instance.a_dtype, instance.tile_k)


def _max_lds_bytes() -> int:
    try:
        from .utils import get_shared_memory_per_block

        return int(get_shared_memory_per_block(fallback_gfx="gfx950"))
    except Exception:
        return 160 * 1024


def instance_valid(instance: kernelInstance) -> bool:
    if instance.tile_k not in (128, 256):
        return False
    if instance.tile_m % 32 != 0 or instance.tile_n % 128 != 0:
        return False
    if (instance.tile_m * _a_row_bytes(instance.a_dtype, instance.tile_k)) % 4096:
        return False
    return estimated_lds_bytes(instance) <= _max_lds_bytes()


def fits_shape(instance: kernelInstance, M: int, N: int, K: int) -> bool:
    """Return whether a JIT specialization can execute the logical shape."""
    del M  # M is ragged and guarded by the kernel.
    if K % 128 != 0:
        return False
    return N % instance.tile_n == 0 and K % instance.tile_k == 0


def _build_kernels_list():
    result = {}
    kernel_id = 0
    for a_dtype, b_dtype in _COMBOS:
        for tile_m in _TILE_M:
            for tile_n in _TILE_N:
                for tile_k in _TILE_K:
                    for waves_per_eu in _WAVES_PER_EU:
                        for xcd_swizzle in _XCD_SWIZZLE:
                            instance = kernelInstance(
                                tile_m=tile_m,
                                tile_n=tile_n,
                                tile_k=tile_k,
                                a_dtype=a_dtype,
                                b_dtype=b_dtype,
                                out_dtype="bf16",
                                waves_per_eu=waves_per_eu,
                                xcd_swizzle=xcd_swizzle,
                            )
                            if instance_valid(instance):
                                result[kernel_id] = instance
                                kernel_id += 1
    return result


kernels_list = _build_kernels_list()


def candidates_for(a_dtype: str, b_dtype: str, M: int, N: int, K: int):
    return [
        (kernel_id, instance)
        for kernel_id, instance in kernels_list.items()
        if instance.a_dtype == a_dtype
        and instance.b_dtype == b_dtype
        and fits_shape(instance, M, N, K)
    ]


__all__ = [
    "candidates_for",
    "estimated_lds_bytes",
    "fits_shape",
    "instance_valid",
    "kernelInstance",
    "kernels_list",
    "parse_kernel_name",
]
