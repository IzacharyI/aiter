# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch wrappers and scale preparation for the gfx950 FlyDSL MXScale GEMM.

A is row-major, B is preshuffled, and both per-32 E8M0 scales use A16W4
packing.
"""

from __future__ import annotations

from collections import OrderedDict
import os
import threading
import weakref

import torch

from aiter.ops.flydsl.utils import is_flydsl_available

_OUT_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "fp16"}

_B_SCALE_CACHE_MAX_BYTES = (
    max(0, int(os.getenv("AITER_MXFP8_B_SCALE_CACHE_MB", "512"))) * 1024 * 1024
)
_B_SCALE_CACHE_LOCK = threading.Lock()
_B_SCALE_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
_B_SCALE_CACHE_BYTES = 0


def fp32_scale_to_e8m0_exact(
    scale: torch.Tensor, *, name: str = "scale"
) -> torch.Tensor:
    """Recover exact E8M0 values stored in an FP32 checkpoint tensor.

    Continuous FP32 scales are rejected instead of rounded.
    """
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(scale).__name__}")
    if scale.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {scale.dtype}")

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        raise RuntimeError(
            "torch.float8_e8m0fnu is required for exact FP32-to-E8M0 recovery"
        )

    scale_fp32 = scale.contiguous()
    encoded = scale_fp32.to(e8m0_dtype)
    decoded = encoded.to(torch.float32)
    valid = torch.isfinite(scale_fp32) & (scale_fp32 > 0) & (decoded == scale_fp32)
    message = (
        f"{name} must contain only finite, positive, exactly E8M0-representable "
        "power-of-two values"
    )
    condition = torch.all(valid)
    if hasattr(torch, "_assert_async"):
        # Validate without synchronizing the normal GPU path.
        torch._assert_async(condition, message)
    elif not bool(condition.item()):
        raise ValueError(message)
    return encoded


def requantize_block128_a_fp8_to_mxfp8(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    *,
    scale_transposed: bool = True,
    validate: bool = True,
    use_fused_kernel: bool = True,
    pack_scale_a16w4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize a block-128 FP8/FP32 pair to per-32 MXFP8/E8M0.

    The fused GPU path can consume legacy transposed scales and write the
    neutral-padded A16W4 scale directly.
    """
    if not isinstance(a, torch.Tensor) or not isinstance(a_scale, torch.Tensor):
        raise TypeError("a and a_scale must both be torch.Tensor instances")
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got shape {tuple(a.shape)}")
    if a_scale.ndim != 2:
        raise ValueError(f"a_scale must be 2D, got shape {tuple(a_scale.shape)}")
    if a_scale.dtype != torch.float32:
        raise TypeError(f"a_scale must have dtype torch.float32, got {a_scale.dtype}")
    if a.device != a_scale.device:
        raise ValueError(
            "a and a_scale must share a device, got " f"{a.device} and {a_scale.device}"
        )

    from aiter.utility import dtypes, fp4_utils
    from aiter.utility.mx_types import MxDtypeInt

    if a.dtype != dtypes.fp8:
        raise TypeError(f"a must have the gfx950 FP8 dtype {dtypes.fp8}, got {a.dtype}")

    M, K = int(a.shape[0]), int(a.shape[1])
    if M <= 0 or K <= 0 or K % 128 != 0:
        raise ValueError(
            f"expected M>0 and positive K divisible by 128, got M={M}, K={K}"
        )
    groups = K // 128
    if tuple(a_scale.shape) != (M, groups):
        raise ValueError(
            f"a_scale must have shape {(M, groups)}, got {tuple(a_scale.shape)}"
        )

    scale_storage = a_scale.contiguous()
    logical_scale = (
        scale_storage.view(groups, M).T if scale_transposed else scale_storage
    )
    if validate:
        valid = torch.isfinite(logical_scale) & (logical_scale > 0)
        message = "a_scale must contain only finite, positive FP32 values"
        condition = torch.all(valid)
        if hasattr(torch, "_assert_async"):
            torch._assert_async(condition, message)
        elif not bool(condition.item()):
            raise ValueError(message)

    if a.is_cuda and use_fused_kernel:
        # Triton consumes scale strides without materializing dequantized A.
        from aiter.ops.triton.quant import fp8_blockscale_to_mxfp8

        a_mxfp8, scale_e8m0_u8 = fp8_blockscale_to_mxfp8(
            a,
            a_scale,
            scale_transposed=scale_transposed,
            pack_scale_a16w4=pack_scale_a16w4,
        )
        return a_mxfp8, scale_e8m0_u8.view(dtypes.fp8_e8m0)

    # Torch reference path.
    a_blocks = a.contiguous().view(M, groups, 4, 32).float()
    amax = a_blocks.abs().amax(dim=-1) * logical_scale.unsqueeze(-1)
    scale_e8m0 = fp4_utils.f32_to_mx_e8m0_scale(amax, dtype=MxDtypeInt.FP8_E4M3)
    scale_mxfp8 = fp4_utils.e8m0_to_f32(scale_e8m0).float()
    ratio = torch.where(
        amax > 0,
        logical_scale.unsqueeze(-1) / scale_mxfp8,
        torch.zeros_like(scale_mxfp8),
    )

    a_mxfp8 = (a_blocks * ratio.unsqueeze(-1)).to(a.dtype).view(M, K)
    logical_e8m0 = scale_e8m0.view(M, K // 32)
    if pack_scale_a16w4:
        logical_e8m0 = prepare_block32_a_scale_e8m0(logical_e8m0, M=M, K=K)
    return a_mxfp8, logical_e8m0


def _e8m0_scale_as_u8(
    scale: torch.Tensor, *, name: str, expected_shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.dtype]:
    """Validate a logical E8M0 scale tensor and expose its raw bytes."""
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(scale).__name__}")
    if scale.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(scale.shape)}")
    if tuple(scale.shape) != expected_shape:
        raise ValueError(
            f"{name} must have logical shape {expected_shape}, got {tuple(scale.shape)}"
        )

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scale.dtype != torch.uint8 and (e8m0_dtype is None or scale.dtype != e8m0_dtype):
        raise TypeError(
            f"{name} must contain raw E8M0 bytes (uint8 or float8_e8m0fnu), "
            f"got {scale.dtype}"
        )
    original_dtype = scale.dtype
    return scale.contiguous().view(torch.uint8), original_dtype


def _restore_e8m0_dtype(scale_u8: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return scale_u8 if dtype == torch.uint8 else scale_u8.view(dtype)


def prepare_block128_a_scale_e8m0(
    a_scale: torch.Tensor, *, M: int | None = None, K: int | None = None
) -> torch.Tensor:
    """Expand block-128 A scales and pack the neutral-padded A16W4 layout."""
    if a_scale.ndim != 2:
        raise ValueError(f"a_scale must be 2D, got shape {tuple(a_scale.shape)}")
    M = int(a_scale.shape[0]) if M is None else int(M)
    K = int(a_scale.shape[1]) * 128 if K is None else int(K)
    if M <= 0 or K <= 0 or K % 128 != 0:
        raise ValueError(
            f"expected M>0 and positive K divisible by 128, got M={M}, K={K}"
        )

    scale_u8, original_dtype = _e8m0_scale_as_u8(
        a_scale, name="a_scale", expected_shape=(M, K // 128)
    )
    m_pad = (M + 31) // 32 * 32
    k32 = K // 32
    k32_pad = (k32 + 7) // 8 * 8
    logical_u8 = torch.empty((m_pad, k32_pad), dtype=torch.uint8, device=a_scale.device)
    logical_u8.fill_(0x7F)
    logical_u8[:M, :k32] = scale_u8.repeat_interleave(4, dim=1)

    from aiter.ops.shuffle import shuffle_scale_a16w4

    shuffled_u8 = shuffle_scale_a16w4(logical_u8, 1, False)
    return _restore_e8m0_dtype(shuffled_u8, original_dtype)


def prepare_block32_a_scale_e8m0(
    a_scale: torch.Tensor, *, M: int | None = None, K: int | None = None
) -> torch.Tensor:
    """Pack logical per-32 A scales into the neutral-padded A16W4 layout."""
    if a_scale.ndim != 2:
        raise ValueError(f"a_scale must be 2D, got shape {tuple(a_scale.shape)}")
    M = int(a_scale.shape[0]) if M is None else int(M)
    K = int(a_scale.shape[1]) * 32 if K is None else int(K)
    if M <= 0 or K <= 0 or K % 128 != 0:
        raise ValueError(
            f"expected M>0 and positive K divisible by 128, got M={M}, K={K}"
        )

    scale_u8, original_dtype = _e8m0_scale_as_u8(
        a_scale, name="a_scale", expected_shape=(M, K // 32)
    )
    m_pad = (M + 31) // 32 * 32
    k32 = K // 32
    k32_pad = (k32 + 7) // 8 * 8
    logical_u8 = torch.full(
        (m_pad, k32_pad),
        0x7F,
        dtype=torch.uint8,
        device=a_scale.device,
    )
    logical_u8[:M, :k32] = scale_u8

    from aiter.ops.shuffle import shuffle_scale_a16w4

    shuffled_u8 = shuffle_scale_a16w4(logical_u8, 1, False)
    return _restore_e8m0_dtype(shuffled_u8, original_dtype)


def prepare_block128_b_scale_e8m0(
    b_scale: torch.Tensor, *, N: int, K: int
) -> torch.Tensor:
    """Expand block-128 B scales and pack the neutral-padded A16W4 layout."""
    N, K = int(N), int(K)
    if N <= 0 or K <= 0 or N % 128 != 0 or K % 128 != 0:
        raise ValueError(
            f"expected positive N and K divisible by 128, got N={N}, K={K}"
        )

    scale_u8, original_dtype = _e8m0_scale_as_u8(
        b_scale, name="b_scale", expected_shape=(N // 128, K // 128)
    )
    k32 = K // 32
    k32_pad = (k32 + 7) // 8 * 8
    logical_u8 = torch.empty((N, k32_pad), dtype=torch.uint8, device=b_scale.device)
    logical_u8.fill_(0x7F)
    logical_u8[:, :k32] = scale_u8.repeat_interleave(128, dim=0).repeat_interleave(
        4, dim=1
    )

    from aiter.ops.shuffle import shuffle_scale_a16w4

    shuffled_u8 = shuffle_scale_a16w4(logical_u8, 1, False)
    return _restore_e8m0_dtype(shuffled_u8, original_dtype)


def _tensor_version(tensor: torch.Tensor) -> int | None:
    """Return the mutation version, or ``None`` for inference tensors."""
    try:
        return int(tensor._version)
    except RuntimeError:
        return None


def _current_stream_key(tensor: torch.Tensor) -> int:
    if not tensor.is_cuda:
        return -1
    return int(torch.cuda.current_stream(tensor.device).cuda_stream)


def clear_mxfp8_b_scale_cache() -> None:
    """Release all prepared static B-scale layouts held by the MXFP8 bridge."""
    global _B_SCALE_CACHE_BYTES
    with _B_SCALE_CACHE_LOCK:
        _B_SCALE_CACHE.clear()
        _B_SCALE_CACHE_BYTES = 0


def _drop_b_scale_cache_entry(cache_key: tuple, source_ref) -> None:
    global _B_SCALE_CACHE_BYTES
    with _B_SCALE_CACHE_LOCK:
        entry = _B_SCALE_CACHE.get(cache_key)
        if entry is not None and entry[0] is source_ref:
            _B_SCALE_CACHE_BYTES -= entry[3]
            del _B_SCALE_CACHE[cache_key]


def prepare_block128_b_scale_e8m0_cached(
    b_scale: torch.Tensor, *, N: int, K: int
) -> torch.Tensor:
    """Recover and pack a checkpoint B scale with a bounded weakref LRU.

    The key includes tensor version and stream. CPU or grad tensors bypass the
    cache; ``AITER_MXFP8_B_SCALE_CACHE_MB=0`` disables it.
    """
    if not isinstance(b_scale, torch.Tensor):
        raise TypeError(f"b_scale must be a torch.Tensor, got {type(b_scale).__name__}")
    N, K = int(N), int(K)
    logical_shape = (N // 128, K // 128)
    if tuple(b_scale.shape) != logical_shape:
        raise ValueError(
            f"b_scale must have logical shape {logical_shape}, "
            f"got {tuple(b_scale.shape)}"
        )

    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    is_encoded = b_scale.dtype == torch.uint8 or (
        e8m0_dtype is not None and b_scale.dtype == e8m0_dtype
    )
    if b_scale.dtype != torch.float32 and not is_encoded:
        raise TypeError(
            "MXFP8 B scale must be FP32 checkpoint values or raw E8M0 "
            f"bytes, got {b_scale.dtype}"
        )

    cacheable = (
        _B_SCALE_CACHE_MAX_BYTES > 0 and b_scale.is_cuda and not b_scale.requires_grad
    )
    stream_key = _current_stream_key(b_scale)
    cache_key = (id(b_scale), stream_key, N, K)
    signature = (
        tuple(b_scale.shape),
        tuple(b_scale.stride()),
        b_scale.dtype,
        b_scale.device,
        _tensor_version(b_scale),
    )

    if cacheable:
        with _B_SCALE_CACHE_LOCK:
            entry = _B_SCALE_CACHE.get(cache_key)
            if entry is not None and entry[0]() is b_scale and entry[1] == signature:
                _B_SCALE_CACHE.move_to_end(cache_key)
                return entry[2]

    encoded = (
        fp32_scale_to_e8m0_exact(b_scale, name="b_scale")
        if b_scale.dtype == torch.float32
        else b_scale
    )
    prepared = prepare_block128_b_scale_e8m0(encoded, N=N, K=K)

    if not cacheable:
        return prepared

    prepared_bytes = prepared.numel() * prepared.element_size()
    if prepared_bytes > _B_SCALE_CACHE_MAX_BYTES:
        return prepared

    global _B_SCALE_CACHE_BYTES
    with _B_SCALE_CACHE_LOCK:
        previous = _B_SCALE_CACHE.pop(cache_key, None)
        if previous is not None:
            _B_SCALE_CACHE_BYTES -= previous[3]

        source_ref = weakref.ref(
            b_scale,
            lambda ref, key=cache_key: _drop_b_scale_cache_entry(key, ref),
        )
        _B_SCALE_CACHE[cache_key] = (
            source_ref,
            signature,
            prepared,
            prepared_bytes,
        )
        _B_SCALE_CACHE_BYTES += prepared_bytes
        while _B_SCALE_CACHE and _B_SCALE_CACHE_BYTES > _B_SCALE_CACHE_MAX_BYTES:
            _, evicted = _B_SCALE_CACHE.popitem(last=False)
            _B_SCALE_CACHE_BYTES -= evicted[3]
    return prepared


def flydsl_mxscale_preshuffle_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    Out: torch.Tensor,
    *,
    a_dtype: str,
    b_dtype: str = "fp4",
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    split_k: int = 1,
    stream=None,
) -> torch.Tensor:
    """Run the gfx950 MXScale preshuffle GEMM.

    ``split_k>1`` writes FP32 partials and reduces them into ``Out``.
    """
    if not is_flydsl_available():
        raise RuntimeError(
            "flydsl is not available; cannot run mxscale_preshuffle GEMM"
        )

    from .kernels.mxscale_preshuffle import (
        MXSCALE_DTYPE_FP4,
        MXSCALE_DTYPE_FP6,
        MXSCALE_DTYPE_FP8,
        MXSCALE_OUT_DTYPE_BF16,
        MXSCALE_OUT_DTYPE_FP16,
        launch_gemm,
    )
    from .kernels.tensor_shim import ptr_arg

    # Logical K: fp4 A packs 2 codes/byte (A last dim = K//2); fp6/fp8 A = 1 byte/code.
    if a_dtype not in ("fp4", "fp6", "fp8"):
        raise ValueError(
            f"unsupported a_dtype {a_dtype!r}; expected 'fp4', 'fp6', or 'fp8'"
        )
    if b_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported b_dtype {b_dtype!r}; expected 'fp4' or 'fp8'")

    a_dtype_code = {
        "fp4": MXSCALE_DTYPE_FP4,
        "fp6": MXSCALE_DTYPE_FP6,
        "fp8": MXSCALE_DTYPE_FP8,
    }[a_dtype]
    b_dtype_code = {
        "fp4": MXSCALE_DTYPE_FP4,
        "fp8": MXSCALE_DTYPE_FP8,
    }[b_dtype]

    M = int(A.shape[0])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)
    N = int(Out.shape[-1])
    if N % int(tile_n) != 0:
        raise ValueError(f"N ({N}) is not a multiple of tile_n ({tile_n})")
    if K % int(tile_k) != 0:
        raise ValueError(f"K ({K}) is not a multiple of tile_k ({tile_k})")
    if K % 128 != 0:
        raise ValueError(
            f"K ({K}) must be a multiple of 128 for MXFP microscale; got {K}"
        )
    out_dtype = _OUT_DTYPE_STR.get(Out.dtype)
    if out_dtype is None:
        raise ValueError(
            f"unsupported Out dtype {Out.dtype}; expected bfloat16 or float16"
        )
    out_dtype_code = (
        MXSCALE_OUT_DTYPE_BF16 if out_dtype == "bf16" else MXSCALE_OUT_DTYPE_FP16
    )

    st = stream if stream is not None else torch.cuda.current_stream()

    split_k = int(split_k)
    if split_k > 1:
        # Each split must cover whole tiles and 256-K scale chunks.
        k_per_split = K // split_k
        if K % split_k != 0 or k_per_split % int(tile_k) != 0 or k_per_split % 256 != 0:
            raise ValueError(
                f"illegal split_k={split_k} for K={K}, tile_k={tile_k}: "
                f"K/split_k ({k_per_split}) must be a multiple of tile_k and 256"
            )

    if split_k == 1:
        launch_gemm(
            ptr_arg(Out),
            ptr_arg(A),
            ptr_arg(B),
            ptr_arg(a_scale),
            ptr_arg(b_scale),
            M,
            N,
            st,
            N,
            K,
            int(tile_m),
            int(tile_n),
            int(tile_k),
            a_dtype_code,
            out_dtype_code,
            b_dtype_code,
            1,  # batch
            -1,  # a_row_stride
            -1,  # a_batch_stride
            -1,  # sca_row_stride
            -1,  # sca_batch_stride
            -1,  # c_row_stride
            -1,  # c_batch_stride
            int(waves_per_eu),
            int(xcd_swizzle),
            1,  # k_batch
        )
        return Out

    # split-K: GEMM -> fp32 partial slabs tmp[split_k, M, N] -> fused fp32 reduce -> Out.
    from .kernels.mxscale_preshuffle import launch_splitk_reduce

    tmp = torch.empty((split_k, M, N), dtype=torch.float32, device=A.device)
    launch_gemm(
        ptr_arg(tmp),
        ptr_arg(A),
        ptr_arg(B),
        ptr_arg(a_scale),
        ptr_arg(b_scale),
        M,
        N,
        st,
        N,
        K,
        int(tile_m),
        int(tile_n),
        int(tile_k),
        a_dtype_code,
        out_dtype_code,
        b_dtype_code,
        1,  # batch
        -1,  # a_row_stride
        -1,  # a_batch_stride
        -1,  # sca_row_stride
        -1,  # sca_batch_stride
        -1,  # c_row_stride
        -1,  # c_batch_stride
        int(waves_per_eu),
        int(xcd_swizzle),
        split_k,  # k_batch
    )
    launch_splitk_reduce(
        ptr_arg(tmp),
        ptr_arg(Out),
        (M * N) // 2,  # n_out_dw (2 out elems per dword)
        M * N,  # slab_stride_dw (fp32: 1 dword/elem)
        st,
        split_k,
        out_dtype_code,
    )
    return Out


# ── Tuned dispatch: explicit tile args > shared A8W8 row > heuristic ───────


def get_mxscale_preshuffle_config(M, N, K, a_dtype, b_dtype, tuned_file=None):
    """Return a matching FlyDSL row from the shared A8W8 tuned table."""
    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config

    tf = str(
        tuned_file or AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    )
    config = get_CKGEMM_config(int(M), int(N), int(K), tf)
    if config is None or str(config.get("libtype", "")).lower() != "flydsl":
        return None

    from .mxscale_preshuffle_config import (
        parse_kernel_name,
    )

    parsed = parse_kernel_name(str(config.get("kernelName", "")))
    if parsed is None:
        raise ValueError(
            "invalid FlyDSL MXScale kernelName in A8W8 bpreshuffle tuned "
            f"config: {config.get('kernelName')!r}"
        )
    if parsed["a_dtype"] != str(a_dtype) or parsed["b_dtype"] != str(b_dtype):
        return None
    return config


def _runtime_split_k(value) -> int:
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 1
    parsed = int(value)
    return 1 if parsed <= 0 else parsed


def _heuristic_tile(a_dtype, b_dtype, M, N, K):
    """Pick a legal tile from the catalog when no tune/explicit config is given."""
    from .mxscale_preshuffle_config import candidates_for

    cands = [ki for _, ki in candidates_for(a_dtype, b_dtype, M, N, K)]
    if not cands:
        return None
    target_m = min(max((M + 31) // 32 * 32, 32), 128)
    return max(
        cands,
        key=lambda ki: (
            ki.tile_k,
            ki.tile_n,
            -abs(ki.tile_m - target_m),
            -ki.waves_per_eu,
        ),
    )


def gemm_mxscale_preshuffle(
    A,
    B,
    a_scale,
    b_scale,
    Out,
    *,
    a_dtype,
    b_dtype,
    tile_m=None,
    tile_n=None,
    tile_k=None,
    waves_per_eu=None,
    xcd_swizzle=None,
    split_k=None,
    config=None,
    stream=None,
):
    """Dispatch using explicit tiles, a tuned row, or a legal heuristic."""
    M = int(A.shape[0])
    N = int(Out.shape[-1])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)

    if tile_m is None or tile_n is None or tile_k is None:
        cfg = (
            config
            if config is not None
            else get_mxscale_preshuffle_config(M, N, K, a_dtype, b_dtype)
        )
        if cfg is not None and cfg.get("kernelName"):
            from .mxscale_preshuffle_config import (
                parse_kernel_name,
            )

            p = parse_kernel_name(cfg["kernelName"])
            if p is not None:
                out_dtype = _OUT_DTYPE_STR.get(Out.dtype)
                if p["a_dtype"] != a_dtype or p["b_dtype"] != b_dtype:
                    raise ValueError(
                        "MXScale tuned config dtype mismatch: "
                        f"config={p['a_dtype']}/{p['b_dtype']}, "
                        f"call={a_dtype}/{b_dtype}"
                    )
                if p["out_dtype"] != out_dtype:
                    raise ValueError(
                        "MXScale tuned config output dtype mismatch: "
                        f"config={p['out_dtype']}, call={out_dtype}"
                    )
                tile_m, tile_n, tile_k = p["tile_m"], p["tile_n"], p["tile_k"]
                if waves_per_eu is None:
                    waves_per_eu = p["waves_per_eu"]
                if xcd_swizzle is None:
                    xcd_swizzle = p["xcd_swizzle"]
                if split_k is None:
                    split_k = _runtime_split_k(cfg.get("splitK", p["split_k"]))
        if tile_m is None:
            ki = _heuristic_tile(a_dtype, b_dtype, M, N, K)
            if ki is None:
                raise ValueError(
                    f"no legal tile for M={M} N={N} K={K} "
                    f"{a_dtype}/{b_dtype}; pass tile_m/n/k explicitly"
                )
            tile_m, tile_n, tile_k = ki.tile_m, ki.tile_n, ki.tile_k
            if waves_per_eu is None:
                waves_per_eu = ki.waves_per_eu
            if xcd_swizzle is None:
                xcd_swizzle = ki.xcd_swizzle
            if split_k is None:
                split_k = ki.split_k

    return flydsl_mxscale_preshuffle_gemm(
        A,
        B,
        a_scale,
        b_scale,
        Out,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=(0 if waves_per_eu is None else waves_per_eu),
        xcd_swizzle=(0 if xcd_swizzle is None else xcd_swizzle),
        split_k=(1 if split_k is None else split_k),
        stream=stream,
    )
