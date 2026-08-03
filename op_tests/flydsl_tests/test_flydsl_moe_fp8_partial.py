# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg
from aiter.ops.flydsl.utils import is_flydsl_available

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)
_RUN_LARGE = os.environ.get("AITER_RUN_LARGE_MOE_TESTS", "0") == "1"
_SKIP_FLYDSL = pytest.mark.skipif(not is_flydsl_available(), reason="FlyDSL required")


def _encode_mxfp8_rows(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, model_dim = values.shape
    assert values.dtype == torch.float8_e4m3fn
    assert scales.shape == (rows, model_dim // 8)
    encoded = torch.empty(
        (rows, model_dim + model_dim // 8), dtype=torch.uint8, device=values.device
    )
    encoded[:, :model_dim].copy_(values.view(torch.uint8))
    encoded[:, model_dim:].copy_(scales)
    return encoded


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@_SKIP_GFX950_FLYDSL
def test_mxfp8_partial_reducer(out_dtype):
    from aiter.ops.flydsl.kernels.moe_reduction_fp8 import (
        compile_moe_reduction_fp8,
    )

    torch.manual_seed(17)
    tokens, topk, model_dim = 7, 3, 24
    rows = tokens * topk
    values = (torch.randn(rows, model_dim, device="cuda") * 2).to(torch.float8_e4m3fn)
    scales = torch.randint(
        124, 131, (rows, model_dim // 8), dtype=torch.uint8, device="cuda"
    )
    # Private stage2 FP8-partial contract: E=0 means an all-zero 8-lane group.
    values[0, :8] = 0
    scales[0, 0] = 0
    encoded = _encode_mxfp8_rows(values, scales)
    scale_f32 = torch.exp2(
        (scales.to(torch.int32) - 127).to(torch.float32)
    ).repeat_interleave(8, dim=1)
    expected = (
        (values.float() * scale_f32)
        .view(tokens, topk, model_dim)
        .sum(dim=1)
        .to(out_dtype)
    )
    out = torch.empty((tokens, model_dim), dtype=out_dtype, device="cuda")

    exe = compile_moe_reduction_fp8(
        topk=topk,
        model_dim=model_dim,
        out_dtype_str="bf16" if out_dtype == torch.bfloat16 else "f16",
    )
    _run_compiled(
        exe,
        ptr_arg(encoded),
        ptr_arg(out),
        tokens,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"partial_dtype": "int8"}, "partial_dtype"),
        ({"mode": "atomic"}, "reduce mode"),
        ({"a_dtype": "fp4"}, "A8W4"),
        ({"b_dtype": "fp8"}, "A8W4"),
        ({"out_dtype": "f32"}, "BF16 or FP16"),
        ({"model_dim": 520}, "divisible by 256"),
        ({"model_dim_pad": 64}, "model_dim_pad"),
        ({"tile_n": 128}, "tile_n=256"),
        ({"return_per_slot": True}, "return_per_slot"),
        ({"expert_mask": object()}, "expert_mask"),
        ({"gfx": "gfx942"}, "gfx950"),
    ],
)
def test_validate_fp8_partial_contract(overrides, match):
    from aiter.ops.flydsl.moe_kernels import _validate_stage2_partial_dtype

    kwargs = {
        "partial_dtype": "fp8",
        "mode": "reduce",
        "a_dtype": "fp8",
        "b_dtype": "fp4",
        "out_dtype": "bf16",
        "model_dim": 512,
        "model_dim_pad": 0,
        "tile_n": 256,
        "return_per_slot": False,
        "expert_mask": None,
        "gfx": "gfx950",
    }
    kwargs.update(overrides)
    with pytest.raises((ValueError, RuntimeError, NotImplementedError), match=match):
        _validate_stage2_partial_dtype(**kwargs)


def test_validate_default_partial_is_disabled():
    from aiter.ops.flydsl.moe_kernels import _validate_stage2_partial_dtype

    assert not _validate_stage2_partial_dtype(
        partial_dtype=None,
        mode="atomic",
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        model_dim=510,
        model_dim_pad=0,
        tile_n=128,
        return_per_slot=True,
        expert_mask=object(),
        gfx="gfx942",
    )


@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_a8w4_fp8_partial():
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    from op_tests.flydsl_tests.test_flydsl_moe_a8w4 import (
        _check_close,
        _generate_a8w4_gui_data,
    )

    token, model_dim, inter_dim, experts, topk, block_m = 64, 512, 256, 8, 2, 32
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, experts, topk, block_m, seed=29
    )
    common = {
        "inter_states": data["a2_q"],
        "w2": data["w2_shuf"],
        "sorted_token_ids": data["sorted_ids"],
        "sorted_expert_ids": data["sorted_expert_ids"],
        "num_valid_ids": data["num_valid_ids"],
        "topk": topk,
        "tile_m": 32,
        "tile_n": 256,
        "tile_k": 256,
        "a_dtype": "fp8",
        "b_dtype": "fp4",
        "out_dtype": "bf16",
        "mode": "reduce",
        "w2_scale": data["w2_scale_shuf"],
        "a2_scale": data["a2_scale_sort"],
        "sorted_weights": data["sorted_weights"],
        "inter_dim_pad": data["inter_pad"],
        "model_dim_pad": 0,
    }
    bf16_partial = flydsl_moe_stage2(**common)
    fp8_partial = flydsl_moe_stage2(**common, partial_dtype="fp8")
    torch.cuda.synchronize()

    _check_close(data["ref_stage2"], bf16_partial, "bf16_partial")
    _check_close(
        data["ref_stage2"],
        fp8_partial,
        "fp8_partial",
        atol=1.5,
        rtol=0.08,
        max_err_ratio=0.08,
    )
    _check_close(
        bf16_partial,
        fp8_partial,
        "fp8_vs_bf16_partial",
        atol=1.5,
        rtol=0.08,
        max_err_ratio=0.08,
    )


@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_default_atomic_descriptor_boundary(monkeypatch):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    from op_tests.flydsl_tests.test_flydsl_moe_a8w4 import (
        _check_close,
        _generate_a8w4_gui_data,
    )

    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "0")
    token, model_dim, inter_dim, experts, topk, block_m = 4096, 512, 256, 8, 2, 32
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, experts, topk, block_m, seed=41
    )
    out = flydsl_moe_stage2(
        inter_states=data["a2_q"],
        w2=data["w2_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=data["w2_scale_shuf"],
        a2_scale=data["a2_scale_sort"],
        sorted_weights=data["sorted_weights"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    torch.cuda.synchronize()
    _check_close(data["ref_stage2"], out, "stage2_atomic_descriptor_boundary")


@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_fp8_partial_rejects_out_dtype_mismatch():
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    from op_tests.flydsl_tests.test_flydsl_moe_a8w4 import _generate_a8w4_gui_data

    token, model_dim, inter_dim, experts, topk, block_m = 64, 512, 256, 8, 2, 32
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, experts, topk, block_m, seed=43
    )
    out = torch.empty((token, model_dim), dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="preallocated out dtype"):
        flydsl_moe_stage2(
            inter_states=data["a2_q"],
            w2=data["w2_shuf"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            out=out,
            topk=topk,
            tile_m=32,
            tile_n=256,
            tile_k=256,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="bf16",
            partial_dtype="fp8",
            mode="reduce",
            w2_scale=data["w2_scale_shuf"],
            a2_scale=data["a2_scale_sort"],
            sorted_weights=data["sorted_weights"],
            inter_dim_pad=data["inter_pad"],
            model_dim_pad=0,
        )


@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_fp8_partial_zero_producer_output():
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    from op_tests.flydsl_tests.test_flydsl_moe_a8w4 import _generate_a8w4_gui_data

    token, model_dim, inter_dim, experts, topk, block_m = 64, 512, 256, 8, 2, 32
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, experts, topk, block_m, seed=47
    )
    out = flydsl_moe_stage2(
        inter_states=torch.zeros_like(data["a2_q"]),
        w2=data["w2_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=32,
        tile_n=256,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        partial_dtype="fp8",
        mode="reduce",
        w2_scale=data["w2_scale_shuf"],
        a2_scale=data["a2_scale_sort"],
        sorted_weights=data["sorted_weights"],
        inter_dim_pad=data["inter_pad"],
        model_dim_pad=0,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out) == 0


@pytest.mark.skipif(
    not _RUN_LARGE or get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="set AITER_RUN_LARGE_MOE_TESTS=1 on gfx950",
)
def test_dsv4_m16384_stage2_fp8_partial():
    from aiter import dtypes
    from aiter.fused_moe import fused_topk, moe_sorting
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    from aiter.ops.quant import mxfp4_moe_sort_fwd
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    torch.manual_seed(31)
    token, model_dim, inter_dim = 16384, 7168, 768
    experts, topk, block_m = 384, 6, 128
    routing_input = (
        torch.randn((token, model_dim), dtype=torch.bfloat16, device="cuda") / 4
    )
    score = torch.randn((token, experts), dtype=torch.bfloat16, device="cuda")
    topk_weights, topk_ids = fused_topk(routing_input, score, topk, True)
    del routing_input
    del score
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, torch.bfloat16, block_m
    )

    # Construct stage2 inputs directly in quantized form to keep this a kernel-level
    # contract test without BF16->MX quantization workspace overhead.
    a2_q = (
        torch.randn((token, topk, inter_dim), dtype=torch.float16, device="cuda") / 4
    ).to(dtypes.fp8)
    a2_scale = torch.full(
        (token * topk, inter_dim // 32), 124, dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8_e8m0)
    a2_scale = mxfp4_moe_sort_fwd(
        a2_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        cols=inter_dim,
    )

    # Pack two safe fp4 lanes per byte (exclude nibble values 14/15 to avoid NaN-like
    # payloads), then view as fp4x2 and pair with fixed e8m0 scales.
    lo = torch.randint(
        0, 14, (experts, model_dim, inter_dim // 2), dtype=torch.uint8, device="cuda"
    )
    hi = torch.randint(
        0, 14, (experts, model_dim, inter_dim // 2), dtype=torch.uint8, device="cuda"
    )
    w2_q_unshuffled = (lo | (hi << 4)).view(dtypes.fp4x2)
    del lo
    del hi
    w2_scale_unshuffled = torch.full(
        (experts * model_dim, inter_dim // 32), 123, dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8_e8m0)
    w2_q = shuffle_weight_a16w4(w2_q_unshuffled, 16, False)
    w2_scale = shuffle_scale_a16w4(w2_scale_unshuffled, experts, False)
    del w2_q_unshuffled
    del w2_scale_unshuffled

    kwargs = {
        "inter_states": a2_q,
        "w2": w2_q,
        "sorted_token_ids": sorted_ids,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "topk": topk,
        "tile_m": 64,
        "tile_n": 256,
        "tile_k": 256,
        "a_dtype": "fp8",
        "b_dtype": "fp4",
        "out_dtype": "bf16",
        "mode": "reduce",
        "sort_block_m": 128,
        "w2_scale": w2_scale,
        "a2_scale": a2_scale,
        "sorted_weights": sorted_weights,
        "inter_dim_pad": 0,
        "model_dim_pad": 0,
    }
    baseline = flydsl_moe_stage2(**kwargs)
    actual = flydsl_moe_stage2(**kwargs, partial_dtype="fp8")
    torch.cuda.synchronize()

    assert torch.isfinite(baseline).all()
    baseline_rms = baseline.float().pow(2).mean().sqrt()
    assert baseline_rms.isfinite()
    assert 0.02 < baseline_rms.item() < 0.2
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, baseline, atol=0.5, rtol=0.08)


@_SKIP_FLYDSL
def test_compile_mixed_moe_gemm2_fp8_partial_accumulate_guard():
    from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

    with pytest.raises(ValueError, match="accumulate=False"):
        compile_mixed_moe_gemm2(
            model_dim=512,
            inter_dim=256,
            experts=8,
            topk=2,
            tile_m=32,
            tile_n=256,
            tile_k=256,
            doweight_stage2=True,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="bf16",
            accumulate=True,
            partial_dtype="fp8",
        )


@_SKIP_FLYDSL
def test_compile_mixed_moe_gemm2_fp8_partial_model_dim_pad_guard():
    from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

    with pytest.raises(ValueError, match="model_dim_pad"):
        compile_mixed_moe_gemm2(
            model_dim=512,
            inter_dim=256,
            experts=8,
            topk=2,
            tile_m=32,
            tile_n=256,
            tile_k=256,
            doweight_stage2=True,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="bf16",
            accumulate=False,
            model_dim_pad=64,
            partial_dtype="fp8",
        )


@_SKIP_FLYDSL
def test_compile_mixed_moe_gemm2_fp8_partial_gfx_guard(monkeypatch):
    import aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage as mm2

    monkeypatch.setattr(mm2, "get_hip_arch", lambda: "gfx942")
    with pytest.raises(RuntimeError, match="gfx950"):
        mm2.compile_mixed_moe_gemm2(
            # Use a unique compile key to avoid process-global cache_clear side effects.
            model_dim=768,
            inter_dim=256,
            experts=8,
            topk=2,
            tile_m=32,
            tile_n=256,
            tile_k=256,
            doweight_stage2=True,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="bf16",
            accumulate=False,
            partial_dtype="fp8",
        )


@_SKIP_FLYDSL
@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [
        ("bf16", "fp4"),  # a16w4 path
        ("bf16", "int4"),  # int4_bf16 path
    ],
)
def test_compile_wrapper_rejects_partial_dtype_non_a8w4(a_dtype, b_dtype):
    from aiter.ops.flydsl.moe_kernels import compile_flydsl_moe_stage2

    with pytest.raises(ValueError, match="supports only A8W4"):
        compile_flydsl_moe_stage2(
            model_dim=512,
            inter_dim=256,
            experts=8,
            topk=2,
            tile_m=32,
            tile_n=256,
            tile_k=256,
            doweight_stage2=True,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype="bf16",
            partial_dtype="fp8",
            accumulate=False,
        )


@_SKIP_FLYDSL
def test_compile_wrapper_rejects_unknown_partial_dtype():
    from aiter.ops.flydsl.moe_kernels import compile_flydsl_moe_stage2

    with pytest.raises(ValueError, match="partial_dtype"):
        compile_flydsl_moe_stage2(
            model_dim=512,
            inter_dim=256,
            experts=8,
            topk=2,
            tile_m=32,
            tile_n=256,
            tile_k=256,
            doweight_stage2=True,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="bf16",
            partial_dtype="int8",
            accumulate=False,
        )


@_SKIP_GFX950_FLYDSL
def test_flydsl_stage2_fp8_partial_rejects_atomic_before_rewrite(monkeypatch):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2
    from op_tests.flydsl_tests.test_flydsl_moe_a8w4 import _generate_a8w4_gui_data

    token, model_dim, inter_dim, experts, topk, block_m = 64, 512, 256, 8, 2, 32
    data = _generate_a8w4_gui_data(
        token, model_dim, inter_dim, experts, topk, block_m, seed=53
    )
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    with pytest.raises(ValueError, match="reduce mode"):
        flydsl_moe_stage2(
            inter_states=data["a2_q"],
            w2=data["w2_shuf"],
            sorted_token_ids=data["sorted_ids"],
            sorted_expert_ids=data["sorted_expert_ids"],
            num_valid_ids=data["num_valid_ids"],
            topk=topk,
            tile_m=32,
            tile_n=256,
            tile_k=256,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="bf16",
            partial_dtype="fp8",
            mode="atomic",
            w2_scale=data["w2_scale_shuf"],
            a2_scale=data["a2_scale_sort"],
            sorted_weights=data["sorted_weights"],
            inter_dim_pad=data["inter_pad"],
            model_dim_pad=0,
        )
