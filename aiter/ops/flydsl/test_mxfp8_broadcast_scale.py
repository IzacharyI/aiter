# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for the AITER-only block-128-to-per-32 MXFP8 bridge.

Usage:
    GPU_ARCHS=gfx950 AITER_USE_SYSTEM_TRITON=1 \
      pytest -q aiter/ops/flydsl/test_mxfp8_broadcast_scale.py
"""

from __future__ import annotations

import pytest
import torch

import aiter
import aiter.ops.gemm_op_a8w8 as gemm_ops
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.mxscale_preshuffle_config import (
    kernelInstance,
    parse_kernel_name,
)
from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
    clear_mxfp8_b_scale_cache,
    fp32_scale_to_e8m0_exact,
    get_mxscale_preshuffle_config,
    prepare_block128_b_scale_e8m0_cached,
    prepare_block32_a_scale_e8m0,
    requantize_block128_a_fp8_to_mxfp8,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight
from aiter.ops.triton.quant import dynamic_mxfp8_quant, fp8_legacy_to_mxfp8
from aiter.utility import fp4_utils
from aiter.utility.mx_types import MxDtypeInt

if not torch.cuda.is_available():
    pytest.skip("ROCm not available", allow_module_level=True)
if get_gfx_runtime() != "gfx950":
    pytest.skip("MXFP8 broadcast-scale tests require gfx950", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip("flydsl is not installed", allow_module_level=True)


def _logical_transposed_scale(scale: torch.Tensor) -> torch.Tensor:
    """Undo the physical scale transpose used by the CK bpreshuffle ABI."""
    rows, groups = scale.shape
    return scale.contiguous().view(groups, rows).T.contiguous()


def _make_bpreshuffle_scale_layout(
    logical_scale: torch.Tensor,
    layout: str,
) -> torch.Tensor:
    """Build the legacy metadata or SGLang materialized scale layout."""
    transposed_storage = logical_scale.t().contiguous()
    if layout == "legacy":
        return transposed_storage.view_as(logical_scale)
    if layout == "materialized":
        return transposed_storage.t()
    raise ValueError(f"unsupported test scale layout {layout!r}")


def _flydsl_config() -> dict:
    return {
        "libtype": "flydsl",
        "kernelName": "flydsl_mxpsh_32x128x128_F8_F8_B16_w0_x0",
        "splitK": 0,
    }


_LEGACY_DISPATCH_CASES = [
    pytest.param(
        {"libtype": "ck", "kernelName": "test_ck_kernel", "splitK": 0},
        "gemm_a8w8_blockscale_bpreshuffle_ck",
        False,
        id="ck",
    ),
    pytest.param(
        {
            "libtype": "cktile",
            "kernelName": "test_cktile_kernel",
            "splitK": 0,
        },
        "gemm_a8w8_blockscale_bpreshuffle_cktile",
        False,
        id="cktile",
    ),
    pytest.param(
        {"libtype": "asm", "kernelName": "test_asm_kernel", "splitK": 3},
        "gemm_a8w8_blockscale_bpreshuffle_asm",
        True,
        id="asm",
    ),
    pytest.param(
        {
            "libtype": "flydsl",
            "kernelName": "legacy_flydsl_blockscale_kernel",
            "splitK": 0,
        },
        "gemm_a8w8_blockscale_bpreshuffle_flydsl",
        False,
        id="legacy-flydsl",
    ),
    pytest.param(
        {
            "libtype": "flydsl8w",
            "kernelName": "legacy_flydsl8w_blockscale_kernel",
            "splitK": 0,
        },
        "gemm_a8w8_blockscale_bpreshuffle_flydsl_8w",
        False,
        id="legacy-flydsl8w",
    ),
    pytest.param(
        None,
        "gemm_a8w8_blockscale_bpreshuffle_ck",
        False,
        id="config-miss",
    ),
]


def test_unquantized_input_capability_is_exported_at_package_root():
    assert gemm_ops.GEMM_A8W8_BPRESHUFFLE_SUPPORTS_UNQUANTIZED_INPUT is True
    assert (
        getattr(
            aiter,
            "GEMM_A8W8_BPRESHUFFLE_SUPPORTS_UNQUANTIZED_INPUT",
            False,
        )
        is True
    )


def test_flydsl_winner_config_resolves_launch_knobs():
    expected = {
        "tile_m": 32,
        "tile_n": 128,
        "tile_k": 128,
        "a_dtype": "fp8",
        "b_dtype": "fp8",
        "out_dtype": "bf16",
        "waves_per_eu": 0,
        "xcd_swizzle": 0,
        "split_k": 1,
    }
    assert gemm_ops._resolve_mxscale_flydsl_config(_flydsl_config()) == expected
    assert parse_kernel_name.__module__ == (
        "aiter.ops.flydsl.mxscale_preshuffle_config"
    )
    instance = kernelInstance(**expected)
    assert instance.name == _flydsl_config()["kernelName"]
    assert parse_kernel_name(instance.name) == expected


def test_shared_a8w8_tuned_csv_flydsl_exact_shape_hit_and_miss(tmp_path):
    csv_path = tmp_path / "a8w8_blockscale_bpreshuffle_tuned.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,"
        "kernelName,tflops,bw,errRatio\n"
        f"{get_gfx_runtime()},{get_cu_num()},3,128,384,flydsl,0,0,"
        "1.0,flydsl_mxpsh_32x128x128_F8_F8_B16_w0_x0,0,0,0\n"
    )
    hit = get_mxscale_preshuffle_config(3, 128, 384, "fp8", "fp8", tuned_file=csv_path)
    miss = get_mxscale_preshuffle_config(4, 128, 384, "fp8", "fp8", tuned_file=csv_path)
    assert hit is not None
    assert hit["libtype"] == "flydsl"
    assert miss is None


def test_fp32_scale_to_e8m0_exact_roundtrip_and_rejection():
    src = torch.tensor([[2.0**-12, 1.0, 2.0]], device="cuda", dtype=torch.float32)
    encoded = fp32_scale_to_e8m0_exact(src)
    assert encoded.dtype == torch.float8_e8m0fnu
    assert encoded.view(torch.uint8).cpu().tolist() == [[115, 127, 128]]
    torch.testing.assert_close(encoded.float(), src, rtol=0, atol=0)

    with pytest.raises(RuntimeError, match="exactly E8M0-representable"):
        fp32_scale_to_e8m0_exact(torch.tensor([3.0], device="cpu", dtype=torch.float32))


def test_legacy_fnuz_transcoder_wrapper_remains_compatible():
    torch.manual_seed(20260725)
    M, K = 3, 384
    a_fnuz = (torch.randn((M, K), device="cuda", dtype=torch.float32) * 0.25).to(
        torch.float8_e4m3fnuz
    )
    scale_old = torch.rand((M, K // 128), device="cuda", dtype=torch.float32) + 0.125

    a_new, scale_new_u8 = fp8_legacy_to_mxfp8(a_fnuz, scale_old)
    target = a_fnuz.float() * scale_old.repeat_interleave(128, dim=1)
    expected_scale = fp4_utils.f32_to_mx_e8m0_scale(
        target.view(M, K // 32, 32).abs().amax(dim=-1),
        dtype=MxDtypeInt.FP8_E4M3,
    )
    scale_new = fp4_utils.e8m0_to_f32(expected_scale).float()
    expected_a = (target.view(M, K // 32, 32) / scale_new.unsqueeze(-1)).to(
        torch.float8_e4m3fn
    )

    assert torch.equal(scale_new_u8, expected_scale.view(torch.uint8))
    assert torch.equal(a_new.view(torch.uint8), expected_a.view(M, K).view(torch.uint8))


@pytest.mark.parametrize("M", [3, 20])
def test_native_bf16_mxfp8_quant_writes_final_packed_scale(M):
    torch.manual_seed(20260725)
    K = 384
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25

    q_logical, scale_logical = dynamic_mxfp8_quant(x)
    q_packed, scale_packed = dynamic_mxfp8_quant(x, pack_scale_a16w4=True)
    expected_packed = prepare_block32_a_scale_e8m0(scale_logical, M=M, K=K)

    assert q_packed.dtype == torch.float8_e4m3fn
    assert torch.equal(q_packed.view(torch.uint8), q_logical.view(torch.uint8))
    assert torch.equal(scale_packed, expected_packed.view(torch.uint8))


def test_per_1x128_quant_remains_legacy_fp32():
    torch.manual_seed(20260723)
    M, K = 3, 384
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)

    a_transposed, scale_transposed = quant(
        x, quant_dtype=aiter.dtypes.fp8, transpose_scale=True
    )
    a_row_major, scale_row_major = quant(
        x, quant_dtype=aiter.dtypes.fp8, transpose_scale=False
    )

    assert a_transposed.dtype == aiter.dtypes.fp8
    assert scale_transposed.dtype == torch.float32
    assert scale_transposed.shape == (M, K // 128)
    assert torch.equal(a_transposed.view(torch.uint8), a_row_major.view(torch.uint8))
    torch.testing.assert_close(
        _logical_transposed_scale(scale_transposed),
        scale_row_major,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("M", "K"),
    [
        pytest.param(3, 512, id="general"),
        pytest.param(1, 384, id="single-row"),
        pytest.param(3, 128, id="single-group"),
    ],
)
def test_legacy_and_materialized_scale_layouts_transcode_byte_identically(M, K):
    torch.manual_seed(20260726)
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
    quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    a_old, logical_scale = quant(
        x,
        quant_dtype=aiter.dtypes.fp8,
        transpose_scale=False,
    )
    groups = K // 128
    legacy_scale = _make_bpreshuffle_scale_layout(logical_scale, "legacy")
    materialized_scale = _make_bpreshuffle_scale_layout(logical_scale, "materialized")

    legacy_transposed = gemm_ops._resolve_bpreshuffle_scale_transposed(
        legacy_scale,
        rows=M,
        groups=groups,
    )
    materialized_transposed = gemm_ops._resolve_bpreshuffle_scale_transposed(
        materialized_scale,
        rows=M,
        groups=groups,
    )
    if M > 1 and groups > 1:
        assert legacy_transposed is True
        assert materialized_transposed is False
        assert legacy_scale.stride() == (groups, 1)
        assert materialized_scale.stride() == (1, M)
    else:
        # Singleton dimensions make both layouts contiguous. Their physical
        # transpose is an identity, so either interpretation is equivalent.
        assert legacy_transposed is False
        assert materialized_transposed is False

    legacy_a, legacy_e8m0 = requantize_block128_a_fp8_to_mxfp8(
        a_old,
        legacy_scale,
        scale_transposed=legacy_transposed,
        pack_scale_a16w4=True,
    )
    materialized_a, materialized_e8m0 = requantize_block128_a_fp8_to_mxfp8(
        a_old,
        materialized_scale,
        scale_transposed=materialized_transposed,
        pack_scale_a16w4=True,
    )

    assert torch.equal(legacy_a.view(torch.uint8), materialized_a.view(torch.uint8))
    assert torch.equal(
        legacy_e8m0.view(torch.uint8), materialized_e8m0.view(torch.uint8)
    )


def test_bpreshuffle_scale_layout_rejects_unsupported_stride():
    M, groups = 3, 4
    padded = torch.empty((M, groups * 2), device="cuda", dtype=torch.float32)
    unsupported = padded[:, ::2]

    assert unsupported.shape == (M, groups)
    assert not unsupported.is_contiguous()
    assert not unsupported.t().is_contiguous()
    with pytest.raises(
        ValueError,
        match="unsupported bpreshuffle activation scale layout",
    ):
        gemm_ops._resolve_bpreshuffle_scale_transposed(
            unsupported,
            rows=M,
            groups=groups,
        )


@pytest.mark.parametrize("M", [3, 20])
def test_a_requantizes_each_32_payload_and_scale_as_one_pair(M):
    torch.manual_seed(20260723)
    K = 384
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.5
    quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    a_old, scale_storage = quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=True)
    scale_old = _logical_transposed_scale(scale_storage)

    a_new, scale_e8m0 = requantize_block128_a_fp8_to_mxfp8(
        a_old, scale_storage, scale_transposed=True
    )

    old_blocks = a_old.float().view(M, K // 128, 4, 32)
    old_dequant = old_blocks * scale_old[:, :, None, None]
    expected_scale = fp4_utils.f32_to_mx_e8m0_scale(
        old_dequant.abs().amax(dim=-1),
        dtype=MxDtypeInt.FP8_E4M3,
    ).view(M, K // 32)
    assert scale_e8m0.shape == (M, K // 32)
    assert torch.equal(scale_e8m0.view(torch.uint8), expected_scale.view(torch.uint8))

    scale_new = fp4_utils.e8m0_to_f32(scale_e8m0).float().view(M, K // 128, 4)
    ratio = scale_old.unsqueeze(-1) / scale_new
    expected_a = (old_blocks * ratio.unsqueeze(-1)).to(a_old.dtype)
    assert torch.equal(a_new.view(torch.uint8), expected_a.view(M, K).view(torch.uint8))

    # Requantization adds one FP8 rounding.  Its absolute error is bounded by
    # half of the largest E4M3 bin (16 payload units) times the new scale.
    new_dequant = a_new.view(M, K // 128, 4, 32).float() * scale_new.unsqueeze(-1)
    assert torch.all(
        (new_dequant - old_dequant).abs() <= 16.0 * scale_new.unsqueeze(-1)
    )


def test_per32_scale_pack_does_not_repeat_legacy_block_scale():
    M, K = 3, 384
    k32 = K // 32
    # Deliberately distinct bytes make an accidental repeat_interleave(4)
    # visible after the layout transform.
    logical_u8 = (
        torch.arange(M * k32, device="cuda", dtype=torch.uint8).view(M, k32) + 96
    )
    logical = logical_u8.view(torch.float8_e8m0fnu)
    packed = prepare_block32_a_scale_e8m0(logical, M=M, K=K)

    m_pad = 32
    k32_pad = 16
    padded = torch.full((m_pad, k32_pad), 0x7F, device="cuda", dtype=torch.uint8)
    padded[:M, :k32] = logical_u8
    expected = shuffle_scale_a16w4(padded.view(torch.float8_e8m0fnu), 1, False)
    assert torch.equal(packed.view(torch.uint8), expected.view(torch.uint8))


def test_fused_requant_writes_final_a16w4_scale_layout():
    torch.manual_seed(20260725)
    M, K = 3, 384
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
    quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    a_old, scale_storage = quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=True)

    a_logical, scale_logical = requantize_block128_a_fp8_to_mxfp8(
        a_old, scale_storage, scale_transposed=True
    )
    a_packed, scale_packed = requantize_block128_a_fp8_to_mxfp8(
        a_old,
        scale_storage,
        scale_transposed=True,
        pack_scale_a16w4=True,
    )
    expected_packed = prepare_block32_a_scale_e8m0(scale_logical, M=M, K=K)

    assert torch.equal(a_packed.view(torch.uint8), a_logical.view(torch.uint8))
    assert torch.equal(
        scale_packed.view(torch.uint8), expected_packed.view(torch.uint8)
    )


def test_static_b_scale_preparation_cache_hits_and_tracks_mutation():
    clear_mxfp8_b_scale_cache()
    N, K = 128, 384
    b_scale = torch.tensor(
        [[2.0**-3, 2.0**-2, 2.0**-1]],
        device="cuda",
        dtype=torch.float32,
    )

    first = prepare_block128_b_scale_e8m0_cached(b_scale, N=N, K=K)
    second = prepare_block128_b_scale_e8m0_cached(b_scale, N=N, K=K)
    assert first.data_ptr() == second.data_ptr()

    # A version change must invalidate the cached layout even though the
    # tensor object and storage address remain the same.
    b_scale.mul_(2.0)
    third = prepare_block128_b_scale_e8m0_cached(b_scale, N=N, K=K)
    assert third.data_ptr() != first.data_ptr()
    assert not torch.equal(third.view(torch.uint8), first.view(torch.uint8))
    clear_mxfp8_b_scale_cache()


@pytest.mark.parametrize(("config", "backend_name", "asm_call"), _LEGACY_DISPATCH_CASES)
def test_legacy_dispatch_passes_original_payload_and_scales(
    monkeypatch, config, backend_name, asm_call
):
    M, N, K = 3, 128, 384
    a_q = torch.empty((M, K), device="cuda", dtype=aiter.dtypes.fp8)
    b_q = torch.empty((N, K), device="cuda", dtype=aiter.dtypes.fp8)
    a_scale = torch.empty((M, K // 128), device="cuda", dtype=torch.float32)
    b_scale = torch.empty((N // 128, K // 128), device="cuda", dtype=torch.float32)
    calls = []

    def fake_backend(*args, **kwargs):
        calls.append((args, kwargs))
        return args[2] if asm_call else args[4]

    monkeypatch.setattr(gemm_ops, "get_CKGEMM_config", lambda *args: config)
    monkeypatch.setattr(gemm_ops, "is_flydsl_available", lambda: True)
    monkeypatch.setattr(gemm_ops, backend_name, fake_backend)

    out = gemm_ops.gemm_a8w8_blockscale_bpreshuffle(a_q, b_q, a_scale, b_scale)
    assert out.shape == (M, N)
    assert len(calls) == 1
    args, kwargs = calls[0]
    if asm_call:
        passed_a, passed_b, _, passed_a_scale, passed_b_scale = args[:5]
    else:
        passed_a, passed_b, passed_a_scale, passed_b_scale = args[:4]
    assert passed_a.data_ptr() == a_q.data_ptr()
    assert passed_b.data_ptr() == b_q.data_ptr()
    assert passed_a_scale.data_ptr() == a_scale.data_ptr()
    assert passed_b_scale.data_ptr() == b_scale.data_ptr()
    if config is not None and config["libtype"] in {"ck", "cktile", "asm"}:
        assert kwargs["kernelName"] == config["kernelName"]
    if asm_call:
        assert kwargs["splitK"] == config["splitK"]


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("config", "backend_name", "asm_call"), _LEGACY_DISPATCH_CASES)
def test_unquantized_dispatch_quantizes_once_for_legacy_winner(
    monkeypatch, input_dtype, config, backend_name, asm_call
):
    import aiter.ops.quant as quant_ops

    torch.manual_seed(20260725)
    M, N, K = 3, 128, 384
    x = torch.randn((M, K), device="cuda", dtype=input_dtype)
    b_q = torch.empty((N, K), device="cuda", dtype=aiter.dtypes.fp8)
    b_scale = torch.empty((N // 128, K // 128), device="cuda", dtype=torch.float32)
    expected_a = torch.empty_like(x, dtype=aiter.dtypes.fp8)
    logical_scale = torch.arange(
        1,
        M * (K // 128) + 1,
        device="cuda",
        dtype=torch.float32,
    ).view(M, K // 128)
    quant_calls = []
    backend_calls = []

    def counted_quant(*args, **kwargs):
        quant_calls.append((args, kwargs))
        return expected_a, logical_scale

    def fake_backend(*args, **kwargs):
        backend_calls.append((args, kwargs))
        return args[2] if asm_call else args[4]

    monkeypatch.setattr(gemm_ops, "get_CKGEMM_config", lambda *args: config)
    monkeypatch.setattr(gemm_ops, "is_flydsl_available", lambda: True)
    monkeypatch.setattr(quant_ops, "per_group_quant_hip", counted_quant)
    monkeypatch.setattr(gemm_ops, backend_name, fake_backend)

    out = gemm_ops.gemm_a8w8_blockscale_bpreshuffle(x, b_q, None, b_scale)
    assert out.shape == (M, N)
    assert len(quant_calls) == 1
    assert len(backend_calls) == 1
    quant_args, quant_kwargs = quant_calls[0]
    assert len(quant_args) == 1
    assert quant_args[0] is x
    assert quant_kwargs == {
        "quant_dtype": aiter.dtypes.fp8,
        "group_size": 128,
        "transpose_scale": False,
    }
    backend_args, backend_kwargs = backend_calls[0]
    if asm_call:
        passed_a, _, _, passed_scale, passed_b_scale = backend_args[:5]
    else:
        passed_a, _, passed_scale, passed_b_scale = backend_args[:4]
    assert passed_a.data_ptr() == expected_a.data_ptr()
    assert passed_scale.dtype == torch.float32
    assert tuple(passed_scale.shape) == (M, K // 128)
    assert passed_scale.stride() == (1, M)
    torch.testing.assert_close(passed_scale, logical_scale, rtol=0, atol=0)
    assert passed_b_scale.data_ptr() == b_scale.data_ptr()
    if config is not None and config["libtype"] in {"ck", "cktile", "asm"}:
        assert backend_kwargs["kernelName"] == config["kernelName"]
    if asm_call:
        assert backend_kwargs["splitK"] == config["splitK"]


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
def test_unquantized_dispatch_directly_quantizes_for_flydsl_winner(
    monkeypatch, input_dtype
):
    import aiter.ops.flydsl.mxscale_preshuffle_kernels as mx_kernels
    import aiter.ops.quant as quant_ops
    import aiter.ops.triton.quant as triton_quant

    torch.manual_seed(20260725)
    M, N, K = 3, 128, 384
    x = torch.randn((M, K), device="cuda", dtype=input_dtype) * 0.25
    b_q = (torch.randn((N, K), device="cuda", dtype=torch.float32) * 0.5).to(
        aiter.dtypes.fp8
    )
    b_q_kernel = shuffle_weight(b_q.contiguous(), (16, 16))
    b_scale_fp32 = torch.tensor(
        [[2.0**-2, 2.0**-1, 1.0]], device="cuda", dtype=torch.float32
    )
    lookups = []
    quant_calls = []
    original_quant = triton_quant.dynamic_mxfp8_quant

    def shared_lookup(*args):
        lookups.append(args)
        return _flydsl_config()

    def forbidden_legacy_quant(*args, **kwargs):
        raise AssertionError("legacy per-1x128 quant must not run")

    def forbidden_requant(*args, **kwargs):
        raise AssertionError("native BF16 path must not requantize")

    def counted_mxfp8_quant(*args, **kwargs):
        quant_calls.append((args, kwargs))
        return original_quant(*args, **kwargs)

    monkeypatch.setattr(gemm_ops, "get_CKGEMM_config", shared_lookup)
    monkeypatch.setattr(quant_ops, "per_group_quant_hip", forbidden_legacy_quant)
    monkeypatch.setattr(
        mx_kernels,
        "requantize_block128_a_fp8_to_mxfp8",
        forbidden_requant,
    )
    monkeypatch.setattr(triton_quant, "dynamic_mxfp8_quant", counted_mxfp8_quant)

    out = gemm_ops.gemm_a8w8_blockscale_bpreshuffle(
        x,
        b_q_kernel,
        None,
        b_scale_fp32,
        dtype=torch.bfloat16,
    )
    assert len(lookups) == 1
    assert len(quant_calls) == 1
    assert len(quant_calls[0][0]) == 1
    assert quant_calls[0][0][0] is x
    assert quant_calls[0][1] == {
        "quant_dtype": aiter.dtypes.fp8,
        "pack_scale_a16w4": True,
    }

    a_q, a_scale = original_quant(x)
    a_dequant = a_q.float() * fp4_utils.e8m0_to_f32(a_scale).float().repeat_interleave(
        32, dim=1
    )
    b_dequant = b_q.float() * b_scale_fp32.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    ref = (a_dequant @ b_dequant.T).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("scale_layout", ["legacy", "materialized"])
def test_dispatch_flydsl_requantizes_only_after_winner_selection(
    monkeypatch, scale_layout
):
    torch.manual_seed(20260723)
    M, N, K = 3, 128, 384
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.25
    quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    a_old, logical_scale = quant(
        x,
        quant_dtype=aiter.dtypes.fp8,
        transpose_scale=False,
    )
    a_scale_storage = _make_bpreshuffle_scale_layout(
        logical_scale,
        scale_layout,
    )

    b_q = (torch.randn((N, K), device="cuda", dtype=torch.float32) * 0.5).to(
        aiter.dtypes.fp8
    )
    b_q_kernel = shuffle_weight(b_q.contiguous(), (16, 16))
    # This is the FP32 representation SGLang gets after loading checkpoint
    # E8M0 values through target.copy_(loaded_weight).
    b_scale_fp32 = torch.tensor(
        [[2.0**-2, 2.0**-1, 1.0]], device="cuda", dtype=torch.float32
    )
    lookups = []

    def shared_lookup(*args):
        lookups.append(args)
        return _flydsl_config()

    monkeypatch.setattr(gemm_ops, "get_CKGEMM_config", shared_lookup)

    out = gemm_ops.gemm_a8w8_blockscale_bpreshuffle(
        a_old,
        b_q_kernel,
        a_scale_storage,
        b_scale_fp32,
        dtype=torch.bfloat16,
    )
    assert len(lookups) == 1

    scale_transposed = gemm_ops._resolve_bpreshuffle_scale_transposed(
        a_scale_storage,
        rows=M,
        groups=K // 128,
    )
    a_new, a_scale_e8m0 = requantize_block128_a_fp8_to_mxfp8(
        a_old,
        a_scale_storage,
        scale_transposed=scale_transposed,
    )
    a_dequant = a_new.float() * fp4_utils.e8m0_to_f32(
        a_scale_e8m0
    ).float().repeat_interleave(32, dim=1)
    b_dequant = b_q.float() * b_scale_fp32.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    ref = (a_dequant @ b_dequant.T).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
