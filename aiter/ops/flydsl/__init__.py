# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL.

Kernel compilation and public APIs are only available when a compatible
``flydsl`` package is installed. Use ``is_flydsl_available()`` to check
whether the optional dependency exists before relying on FlyDSL kernels.
"""

from packaging.version import Version

from .utils import is_flydsl_available
from .moe_common import GateMode

_MIN_FLYDSL_VERSION = Version("0.2.0")

__all__ = [
    "is_flydsl_available",
    "GateMode",
]

if is_flydsl_available():
    import flydsl as _flydsl

    installed_flydsl_version = getattr(_flydsl, "__version__", None)
    if installed_flydsl_version is None:
        raise ImportError(
            "`flydsl` is importable but its version cannot be determined."
        )

    _base_version = Version(installed_flydsl_version.split("+")[0])
    if _base_version < _MIN_FLYDSL_VERSION:
        raise ImportError(
            "Unsupported `flydsl` version: "
            f"expected >=`{_MIN_FLYDSL_VERSION}`, "
            f"got `{installed_flydsl_version}`."
        )

    # Compatibility shim for FlyDSL releases whose ``ArithValue`` does not
    # provide ``.ir_value()``. Some kernels call ``.ir_value()`` uniformly on
    # FlyDSL values; for ``ArithValue`` the underlying MLIR value is ``self``.
    try:
        from flydsl.expr.arith import ArithValue as _ArithValue

        if not hasattr(_ArithValue, "ir_value"):

            def _arith_ir_value(self, *, loc=None, ip=None):
                return self

            _ArithValue.ir_value = _arith_ir_value
    except Exception:
        pass

    from .gemm_kernels import (
        flydsl_hgemm,
        flydsl_preshuffle_gemm_a8,
        flydsl_blockscale_preshuffle_gemm_a8,
        flydsl_fp8_gemm_8wave_blockscale_a8,
    )
    from .mxscale_preshuffle_kernels import (
        flydsl_mxscale_preshuffle_gemm,
        clear_mxfp8_b_scale_cache,
        fp32_scale_to_e8m0_exact,
        gemm_mxscale_preshuffle,
        get_mxscale_preshuffle_config,
        prepare_block128_a_scale_e8m0,
        prepare_block32_a_scale_e8m0,
        prepare_block128_b_scale_e8m0,
        prepare_block128_b_scale_e8m0_cached,
        requantize_block128_a_fp8_to_mxfp8,
    )
    from .moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
    from .fmha_kernels import flydsl_flash_attn_func
    from .kernels.qk_norm_rope_quant import flydsl_qk_norm_rope_quant

    try:
        from .mega_moe import MegaMoE, MegaMoeStage1, MegaMoeStage2, Stage1Output
    except ImportError:
        MegaMoE = MegaMoeStage1 = MegaMoeStage2 = Stage1Output = None

    # from .linear_attention_kernels import flydsl_gdr_decode

    __all__ += [
        "flydsl_preshuffle_gemm_a8",
        "flydsl_blockscale_preshuffle_gemm_a8",
        "flydsl_fp8_gemm_8wave_blockscale_a8",
        "flydsl_mxscale_preshuffle_gemm",
        "clear_mxfp8_b_scale_cache",
        "fp32_scale_to_e8m0_exact",
        "gemm_mxscale_preshuffle",
        "get_mxscale_preshuffle_config",
        "prepare_block128_a_scale_e8m0",
        "prepare_block32_a_scale_e8m0",
        "prepare_block128_b_scale_e8m0",
        "prepare_block128_b_scale_e8m0_cached",
        "requantize_block128_a_fp8_to_mxfp8",
        "flydsl_moe_stage1",
        "flydsl_moe_stage2",
        "flydsl_hgemm",
        "flydsl_flash_attn_func",
        "flydsl_qk_norm_rope_quant",
        # "flydsl_gdr_decode",
    ]
    if MegaMoE is not None:
        __all__ += [
            "MegaMoE",
            "MegaMoeStage1",
            "MegaMoeStage2",
            "Stage1Output",
        ]
