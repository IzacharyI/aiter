# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple

import triton
import torch
from aiter.ops.triton._triton_kernels.quant.quant import (
    _static_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_tensor_quant_fp8_i8_kernel,
    _dynamic_per_token_quant_fp8_i8_kernel,
    _dynamic_mxfp4_quant_kernel,
    _mxfp4_quant_op,
    _dynamic_mxfp8_quant_kernel,
    _mxfp8_quant_op,
    _fp8_legacy_to_mxfp8_kernel,
    _dynamic_nvfp4_quant_kernel,
    _nvfp4_quant_op,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import e4m3_dtype

__all__ = [
    "static_per_tensor_quant_fp8_i8",
    "dynamic_per_tensor_quant_fp8_i8",
    "dynamic_per_token_quant_fp8_i8",
    "dynamic_mxfp4_quant",
    "_mxfp4_quant_op",
    "dynamic_mxfp8_quant",
    "fp8_blockscale_to_mxfp8",
    "fp8_legacy_to_mxfp8",
    "_mxfp8_quant_op",
    "dynamic_nvfp4_quant",
    "_nvfp4_quant_op",
]

_MXFP8_QUANT_BLOCK_SIZE = 32
_MXFP8_LEGACY_BLOCK_SIZE = 128


_LOGGER = AiterTritonLogger()


def static_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_in: torch.Tensor
):
    """
    Quantizes tensor using the provided scale to int8 or fp8

    Parameters:
    - qx: Output tensor of same shape as x_in. Must be fp8 or int8 dtype and allocated by the caller
    - x_in: Input tensor of shape (M, N).
    - scale_in: Input Scale tensor of shape (1,) and dtype fp32

    Returns:
    - qx: Quantized output values.
    """
    _LOGGER.info(f"STAIC_PER_TENSOR_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    assert scale_in.numel() == 1  # only single scale value
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_in, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx


def dynamic_per_tensor_quant_fp8_i8(
    qx: torch.Tensor, x_in: torch.Tensor, scale_out: torch.Tensor
):
    """
    Calculate per tensor scale and then uses the scale to quantize input tensor to fp8 or int8

    Parameters:
    - x_in: Input tensor of shape (M, N).
    - qx: Output tensor of same shape as x_in. Must be fp8 or int8 dtype and allocated by the caller
    - scale_out: Output scale tensor of shape (1,), dtype fp32 and allocated by the caller

    Returns:
    - qx: Quantized output values of shape (M, N) with dtype fp8 or int8
    - scale_out: Single scale value of shape (1,)
    """
    _LOGGER.info(f"DYNAMIC_PER_TENSOR_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_tensor_quant_fp8_i8_kernel[grid](
        x_in,
        scale_out,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=(
            torch.finfo(qx.dtype).max
            if torch.is_floating_point(qx)
            else torch.iinfo(qx.dtype).max
        ),
    )

    _static_per_tensor_quant_fp8_i8_kernel[grid](
        qx, x_in, scale_out, cols, x_in.stride(0), NUM_COL_POW2=NUM_COL_POW2
    )

    return qx, scale_out


def dynamic_per_token_quant_fp8_i8(
    qx: torch.Tensor,
    x_in: torch.Tensor,
    scale_out: torch.Tensor,
):
    """
    Quantizes tensor using the provided scale

    Parameters:
    - x_in: Input tensor of shape (M, N).
    - dtype_max: Optional parameter which specifies the max value of the dtype of x_in.
    - qx: Output tensor of same shape as x_in. Must be fp8 dtype and allocated by the caller
    - scale_out: Output scale tensor of shape (M,) dtype fp32 and allocated by the caller

    Returns:
    - qx: Quantized output values.
    - scale_out: Scale tensor of shape (M, )
    """
    _LOGGER.info(f"DYNAMIC_PER_TOKEN_QUANT_FP8_I8: x={tuple(x_in.shape)}")
    rows = x_in.shape[0]
    cols = x_in.shape[1]
    NUM_COL_POW2 = triton.next_power_of_2(cols)
    grid = lambda meta: (rows,)  # noqa: E731
    _dynamic_per_token_quant_fp8_i8_kernel[grid](
        qx,
        scale_out,
        x_in,
        cols,
        x_in.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
        DTYPE_MAX=(
            torch.finfo(qx.dtype).max
            if torch.is_floating_point(qx)
            else torch.iinfo(qx.dtype).max
        ),
    )

    return qx, scale_out


def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    _LOGGER.info(f"DYNAMIC_MXFP4_QUANT: x={tuple(x.shape)}")
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    blockscale_e8m0 = torch.empty(
        ((N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE, M),
        dtype=torch.uint8,
        device=x.device,
    ).T

    # for large N values
    if M <= 32:
        NUM_ITER = 1
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_N = 32
        NUM_WARPS = 1
        NUM_STAGES = 1
    else:
        NUM_ITER = 4
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        NUM_WARPS = 4
        NUM_STAGES = 2

        if N <= 16384:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 128

    # for small N values
    if N <= 1024:
        NUM_ITER = 1
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_SIZE_N = min(256, triton.next_power_of_2(N))
        # BLOCK_SIZE_N needs to be multiple of 32
        BLOCK_SIZE_N = max(32, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N * NUM_ITER),
    )

    _dynamic_mxfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        NUM_ITER=NUM_ITER,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return (x_fp4, blockscale_e8m0)


def dynamic_mxfp8_quant(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
    *,
    pack_scale_a16w4: bool = False,
    block_size_m: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize per 1x32 to FP8 with E8M0 scales.

    ``pack_scale_a16w4=True`` writes the neutral-padded FlyDSL scale layout
    ``(ceil(M/32)*32, ceil((K/32)/8)*8)`` directly.
    """
    assert x.dim() >= 2, f"x must be at least 2D, got {x.dim()}"
    orig_shape = x.shape
    K = orig_shape[-1]
    assert (
        K % _MXFP8_QUANT_BLOCK_SIZE == 0
    ), f"last dim K={K} must be a multiple of {_MXFP8_QUANT_BLOCK_SIZE}"

    x2d = x.reshape(-1, K).contiguous()
    M = x2d.shape[0]
    Ns = K // _MXFP8_QUANT_BLOCK_SIZE  # number of scales per row

    y = torch.empty((M, K), dtype=quant_dtype, device=x.device)
    if pack_scale_a16w4:
        if x.dim() != 2:
            raise ValueError(
                "pack_scale_a16w4=True requires a 2D input, got "
                f"shape {tuple(x.shape)}"
            )
        if quant_dtype != torch.float8_e4m3fn:
            raise TypeError(
                "packed FlyDSL MXFP8 quant requires torch.float8_e4m3fn, "
                f"got {quant_dtype}"
            )
        scale_m_pad = (M + 31) // 32 * 32
        scale_k_pad = (Ns + 7) // 8 * 8
        expected_scale_shape = (scale_m_pad, scale_k_pad)
        if scale is None:
            scale = torch.empty(
                expected_scale_shape, dtype=torch.uint8, device=x.device
            )
        else:
            assert (
                scale.shape == expected_scale_shape
            ), f"scale shape {scale.shape} != {expected_scale_shape}"
            assert scale.dtype == torch.uint8
            assert scale.device == x.device

        BLOCK_SIZE_M = 16 if block_size_m is None else int(block_size_m)
        assert BLOCK_SIZE_M in (1, 2, 4, 8, 16)
        grid = (
            triton.cdiv(scale_m_pad, BLOCK_SIZE_M),
            scale_k_pad,
        )
        # The scale pointer is unused for native quantization.
        _fp8_legacy_to_mxfp8_kernel[grid](
            x2d,
            x2d,
            y,
            scale,
            M,
            K,
            x2d.stride(0),
            x2d.stride(1),
            x2d.stride(0),
            x2d.stride(1),
            y.stride(0),
            y.stride(1),
            scale.stride(0),
            scale.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            QUANT_BLOCK_SIZE=_MXFP8_QUANT_BLOCK_SIZE,
            LEGACY_BLOCK_SIZE=_MXFP8_LEGACY_BLOCK_SIZE,
            SCALE_M_PAD=scale_m_pad,
            SCALE_K_PAD=scale_k_pad,
            PACK_SCALE_A16W4=True,
            APPLY_LEGACY_SCALE=False,
        )
        return y.view(*orig_shape[:-1], K), scale

    if block_size_m is not None:
        raise ValueError("block_size_m is only supported with pack_scale_a16w4=True")
    if scale is None:
        scale = torch.empty((M, Ns), dtype=torch.uint8, device=x.device)
    else:
        assert scale.shape == (M, Ns), f"scale shape {scale.shape} != ({M},{Ns})"
        assert scale.dtype == torch.uint8

    BLOCK_SIZE_N = triton.next_power_of_2(K)
    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _dynamic_mxfp8_quant_kernel[grid](
        x2d,
        y,
        scale,
        M,
        K,
        x2d.stride(0),
        x2d.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=_MXFP8_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )

    y = y.view(*orig_shape[:-1], K)
    s = scale.view(*orig_shape[:-1], Ns)
    return y, s


def fp8_blockscale_to_mxfp8(
    x_fp8: torch.Tensor,
    x_scale_fp32: torch.Tensor,
    y_fp8: Optional[torch.Tensor] = None,
    y_scale: Optional[torch.Tensor] = None,
    *,
    scale_transposed: bool = False,
    pack_scale_a16w4: bool = False,
    block_size_m: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transcode FP8/FP32 block-128 input to per-32 FP8/E8M0.

    The output payload is e4m3fn. ``scale_transposed`` selects the legacy CK
    storage convention; ``pack_scale_a16w4`` writes the final FlyDSL layout.
    """
    assert x_fp8.dim() == 2, f"x must be 2D, got {x_fp8.dim()}"
    assert x_fp8.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ), f"x must be FP8 e4m3fn/e4m3fnuz, got {x_fp8.dtype}"
    assert x_scale_fp32.dtype == torch.float32
    assert x_fp8.device == x_scale_fp32.device

    M, N = x_fp8.shape
    assert N % _MXFP8_QUANT_BLOCK_SIZE == 0
    assert N % _MXFP8_LEGACY_BLOCK_SIZE == 0
    assert x_scale_fp32.shape == (
        M,
        N // _MXFP8_LEGACY_BLOCK_SIZE,
    ), f"x_scale_fp32 shape {x_scale_fp32.shape} != ({M},{N // _MXFP8_LEGACY_BLOCK_SIZE})"

    if scale_transposed:
        # Expose legacy transposed storage as a strided logical view.
        scale_storage = x_scale_fp32.contiguous()
        x_scale_logical = scale_storage.view(N // _MXFP8_LEGACY_BLOCK_SIZE, M).T
    else:
        x_scale_logical = x_scale_fp32

    Ns = N // _MXFP8_QUANT_BLOCK_SIZE
    scale_m_pad = (M + 31) // 32 * 32 if pack_scale_a16w4 else M
    scale_k_pad = (Ns + 7) // 8 * 8 if pack_scale_a16w4 else Ns
    expected_scale_shape = (scale_m_pad, scale_k_pad)
    if y_fp8 is None:
        y_fp8 = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x_fp8.device)
    else:
        assert y_fp8.shape == (M, N)
        assert y_fp8.dtype == torch.float8_e4m3fn
        assert y_fp8.device == x_fp8.device
    if y_scale is None:
        y_scale = torch.empty(
            expected_scale_shape, dtype=torch.uint8, device=x_fp8.device
        )
    else:
        assert y_scale.shape == expected_scale_shape
        assert y_scale.dtype == torch.uint8
        assert y_scale.device == x_fp8.device

    if block_size_m is None:
        # Group rows for larger M to limit the program grid.
        BLOCK_SIZE_M = 16 if pack_scale_a16w4 or M >= 8 else 1
    else:
        BLOCK_SIZE_M = int(block_size_m)
        assert BLOCK_SIZE_M in (1, 2, 4, 8, 16)
    grid = (
        triton.cdiv(scale_m_pad if pack_scale_a16w4 else M, BLOCK_SIZE_M),
        scale_k_pad if pack_scale_a16w4 else Ns,
    )

    _fp8_legacy_to_mxfp8_kernel[grid](
        x_fp8,
        x_scale_logical,
        y_fp8,
        y_scale,
        M,
        N,
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_scale_logical.stride(0),
        x_scale_logical.stride(1),
        y_fp8.stride(0),
        y_fp8.stride(1),
        y_scale.stride(0),
        y_scale.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        QUANT_BLOCK_SIZE=_MXFP8_QUANT_BLOCK_SIZE,
        LEGACY_BLOCK_SIZE=_MXFP8_LEGACY_BLOCK_SIZE,
        SCALE_M_PAD=scale_m_pad,
        SCALE_K_PAD=scale_k_pad,
        PACK_SCALE_A16W4=pack_scale_a16w4,
        APPLY_LEGACY_SCALE=True,
    )

    return y_fp8, y_scale


def fp8_legacy_to_mxfp8(
    x_fnuz: torch.Tensor,
    x_scale_fp32: torch.Tensor,
    y_fn: Optional[torch.Tensor] = None,
    y_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible e4m3fnuz wrapper for
    :func:`fp8_blockscale_to_mxfp8`."""
    return fp8_blockscale_to_mxfp8(
        x_fnuz,
        x_scale_fp32,
        y_fn,
        y_scale,
        scale_transposed=False,
    )


def dynamic_nvfp4_quant(
    x: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
    Returns:
        A tuple of (x_fp4, blockscale_e4m3).
    """
    _LOGGER.info(f"DYNAMIC_NVFP4_QUANT: x={tuple(x.shape)}")
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    NVFP4_QUANT_BLOCK_SIZE = 16
    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    blockscale_e4m3 = torch.empty(
        ((N + NVFP4_QUANT_BLOCK_SIZE - 1) // NVFP4_QUANT_BLOCK_SIZE, M),
        dtype=e4m3_dtype,
        device=x.device,
    ).T

    # for large N values
    if M <= 32:
        NUM_ITER = 1
        BLOCK_SIZE_M = triton.next_power_of_2(M)
        BLOCK_SIZE_N = 32
        NUM_WARPS = 1
        NUM_STAGES = 1
    else:
        NUM_ITER = 4
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        NUM_WARPS = 4
        NUM_STAGES = 2

        if N <= 16384:
            BLOCK_SIZE_M = 32
            BLOCK_SIZE_N = 128

    # for small N values
    if N <= 1024:
        NUM_ITER = 1
        NUM_STAGES = 1
        NUM_WARPS = 4
        BLOCK_SIZE_N = min(256, triton.next_power_of_2(N))
        # BLOCK_SIZE_N needs to be multiple of 32
        BLOCK_SIZE_N = max(32, BLOCK_SIZE_N)
        BLOCK_SIZE_M = min(8, triton.next_power_of_2(M))

    grid = (
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N * NUM_ITER),
    )

    _dynamic_nvfp4_quant_kernel[grid](
        x,
        x_fp4,
        blockscale_e4m3,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e4m3.stride(),
        M=M,
        N=N,
        NVFP4_QUANT_BLOCK_SIZE=NVFP4_QUANT_BLOCK_SIZE,
        NUM_ITER=NUM_ITER,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS,
        waves_per_eu=0,
        num_stages=1,
    )

    return x_fp4, blockscale_e4m3
