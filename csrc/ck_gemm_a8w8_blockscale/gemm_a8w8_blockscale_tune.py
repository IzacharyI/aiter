# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

import aiter
from aiter import dtypes
from aiter.jit.core import (
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE,
    AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
    get_asm_dir,
)
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner
from aiter.ops.shuffle import shuffle_weight
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

sys.path.insert(0, str(Path(__file__).parent.parent))
from ck_gemm_a8w8_blockscale_bpreshuffle.gemm_a8w8_blockscale_bpreshuffle_common import (
    kernels_list as candidate_kernels_bpreshuffle_dict,
)
from gemm_a8w8_blockscale_instance import candidate_kernels_dict

# cktile
from gemm_a8w8_blockscale_cktile_instance import (
    candidate_kernels_cktile_dict,
    BLOCK_PER_CU_MAX,
)

# flydsl (blockscale bpreshuffle) — guarded so the tuner still works when the
# FlyDSL candidate module or runtime is unavailable.
try:
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_bpreshuffle_common import (
        kernel_instance_estimated_lds_bytes,
        kernels_list as kernels_list_flydsl_blockscale,
        get_tune_kernels_list as get_flydsl_blockscale_tune_kernels,
        max_lds_bytes_for_tune,
        kernels_list_8w as kernels_list_flydsl_blockscale_8w,
        get_tune_kernels_list_8w as get_flydsl_blockscale_8w_tune_kernels,
    )
except ImportError:
    print(
        "[FlyDSL] flydsl_gemm_a8w8_blockscale_bpreshuffle_common.py not found, "
        "flydsl blockscale tuning disabled"
    )
    kernels_list_flydsl_blockscale = {}
    kernels_list_flydsl_blockscale_8w = {}

    def kernel_instance_estimated_lds_bytes(_ki):
        return 0

    def max_lds_bytes_for_tune():
        return 1 << 30

    def get_flydsl_blockscale_tune_kernels():
        return {}

    def get_flydsl_blockscale_8w_tune_kernels():
        return {}


from aiter.ops.flydsl.utils import is_flydsl_available

if is_flydsl_available():
    from aiter.ops.flydsl.gemm_kernels import flydsl_blockscale_preshuffle_gemm_a8

    try:
        from aiter.ops.flydsl.gemm_kernels import (
            flydsl_fp8_gemm_8wave_blockscale_a8,
        )
    except ImportError:
        pass

block_shape = (128, 128)


def get_valid_asm_splitK_list(K: int, max_splitK: int, tile_k: int = 128):
    """Filter splitK values to only those that produce valid TileK-aligned partitions."""
    valid = []
    for sk in range(1, max_splitK + 1):
        k_per_split = (K + sk - 1) // sk
        k_per_split_aligned = ((k_per_split + tile_k - 1) // tile_k) * tile_k
        actual_ksplit = (K + k_per_split_aligned - 1) // k_per_split_aligned
        if actual_ksplit == sk:
            valid.append(sk)
    return valid if valid else [1]


def _get_padded_m(M: int) -> int:
    """Rounded-up M used to validate flydsl tile_m divisibility (matches dispatch)."""
    if M <= 256:
        return (M + 15) // 16 * 16
    elif M <= 1024:
        return (M + 31) // 32 * 32
    elif M <= 4096:
        return (M + 63) // 64 * 64
    else:
        return (M + 127) // 128 * 128


"""
a8w8_blockscale_gemm tuning for ck, ck_tile and asm
"""


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    """
    Run the reference GEMM operation using PyTorch.
    """

    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    x = x.to(x_scale.dtype).view(
        m, k // block_shape[1], block_shape[1]
    ) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))

    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_gemm_a8w8_blockscale_cktile(
    x, weight, x_scale, w_scale, out, kernel_id, splitK, preshuffleB
):
    """
    Run gemm a8w8 blockscale tuned kernel for ck_tile type.
    """

    if preshuffleB:
        return aiter.gemm_a8w8_blockscale_bpreshuffle_cktile_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )
    else:
        return aiter.gemm_a8w8_blockscale_cktile_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )


def run_gemm_a8w8_blockscale(
    x, weight, x_scale, w_scale, out, kernel_id, splitK, preshuffleB
):
    """
    Run gemm a8w8 blockscale tuned kernel for ck type.
    """

    if preshuffleB:
        return aiter.gemm_a8w8_blockscale_bpreshuffle_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )
    else:
        return aiter.gemm_a8w8_blockscale_tune(
            x, weight, x_scale, w_scale, out, kernel_id, splitK
        )


def run_gemm_a8w8_blockscale_asm(
    x,
    weight,
    x_scale,
    w_scale,
    out,
    zero_bias_buf,
    kernel_name,
    splitK=1,
    preshuffleB=True,
):
    """
    Run gemm a8w8 blockscale tuned kernel for asm type.
    """

    return aiter.gemm_a8w8_blockscale_bpreshuffle_asm(
        x,
        weight,
        out,
        x_scale,
        w_scale,
        None,
        splitK,
        kernel_name,
        preshuffleB,
        zero_bias_buf,
    )


def run_gemm_a8w8_blockscale_flydsl(x, weight_shuffle, x_scale_t, w_scale, out, kernel_id):
    """
    Run gemm a8w8 blockscale tuned kernel for flydsl type (preshuffleB only).

    kernel_id -> kernelInstance whose fields are the exact wrapper args, so the
    tuned kernelName round-trips back to this same launch config at dispatch.
    """

    ki = kernels_list_flydsl_blockscale[kernel_id]
    flydsl_blockscale_preshuffle_gemm_a8(
        x,
        weight_shuffle,
        x_scale_t,
        w_scale,
        out,
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
        ki.scale_block_k,
        ki.use_cshuffle_epilog,
        ki.use_async_copy,
        ki.waves_per_eu,
        ki.xcd_swizzle,
        ki.fused_promote,
    )
    return out


def run_gemm_a8w8_blockscale_flydsl_8w(
    x, weight_shuffle, x_scale_t, w_scale, out, kernel_id
):
    """Run the 8-wave blockscale ping-pong kernel (flydsl8w libtype).

    ``kernel_id`` indexes ``kernels_list_flydsl_blockscale_8w`` (a distinct id
    space from the 4-wave list). The kernelInstance8w fields are the exact wrapper
    args, so the tuned kernelName round-trips back to this launch config at
    dispatch. B is the SAME ``shuffle_weight(16,16)`` bytes as the 4-wave/CK path.
    """

    ki = kernels_list_flydsl_blockscale_8w[kernel_id]
    flydsl_fp8_gemm_8wave_blockscale_a8(
        x,
        weight_shuffle,
        x_scale_t,
        w_scale,
        out,
        block_m=ki.block_m,
        block_n=ki.block_n,
        waves_per_eu=ki.waves_per_eu,
        use_xcd_remap=bool(ki.use_xcd_remap),
    )
    return out


def generate_data(m, n, k, seed, device="cuda"):
    """
    Generate random data for testing the gemm a8w8 blockscale kernel.
    """

    torch.manual_seed(seed)
    block_shape_n, block_shape_k = block_shape
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    zero_bias = torch.zeros((1, n), dtype=torch.float32, device=device)
    return {
        "x": x,
        "weight": weight,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "out": out,
        "weight_shuffle": weight_shuffle,
        "x_scale_t": x_scale_t,
        "zero_bias": zero_bias,
    }


def generate_production_data(m, n, k, seed, dtype=dtypes.bf16, device="cuda"):
    """Build the production boundary used for backend E2E comparison.

    This deliberately matches ``tune_mxscale_preshuffle``'s production data:
    activation A remains BF16, B is an already-quantized/preshuffled checkpoint
    payload, and the checkpoint block-128 B scale contains exact powers of two.
    The production dispatcher therefore owns the single activation quantization
    step selected by the tuned backend.
    """
    if n % 128 != 0 or k % 128 != 0:
        raise ValueError(
            f"production blockscale E2E requires N/K divisible by 128, got "
            f"N={n}, K={k}"
        )

    torch.manual_seed(seed)
    a_bf16 = torch.randn((m, k), dtype=dtype, device=device) * 0.25
    b_q = (
        torch.randn((n, k), dtype=torch.float32, device=device) * 0.25
    ).to(dtypes.fp8)
    b_kernel = shuffle_weight(b_q.contiguous(), layout=(16, 16))
    exponents = torch.randint(
        low=-6,
        high=2,
        size=(n // 128, k // 128),
        device=device,
    )
    b_scale = torch.pow(2.0, exponents.float())
    return {
        "A": a_bf16,
        "B": b_kernel,
        "B_logical": b_q,
        "a_scale": None,
        "b_scale": b_scale,
    }


def calculate_error_metrics(actual, reference, *, atol=1e-2, rtol=1e-2):
    """Return threshold ratio plus scale-independent and absolute metrics."""
    actual_f32 = actual.to(torch.float32)
    reference_f32 = reference.to(torch.float32)
    abs_delta = (actual_f32 - reference_f32).abs()
    mismatch = abs_delta > (atol + rtol * reference_f32.abs())
    reference_l2 = torch.linalg.vector_norm(reference_f32).item()
    delta_l2 = torch.linalg.vector_norm(abs_delta).item()
    return {
        "err_ratio": float(mismatch.count_nonzero().item() / actual.numel()),
        "rel_l2": float(delta_l2 / reference_l2 if reference_l2 else delta_l2),
        "mean_abs": float(abs_delta.mean().item()),
        "max_abs": float(abs_delta.max().item()),
    }


def measure_single_call(function, *arguments):
    """Measure one synchronized production call on device and host clocks."""
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    output = function(*arguments)
    end_event.record()
    end_event.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1_000_000.0
    device_us = start_event.elapsed_time(end_event) * 1_000.0
    return output, float(device_us), float(wall_us)


def measure_steady_wall(function, arguments, iterations):
    """Measure amortized synchronized wall time while keeping static B fixed."""
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = function(*arguments)
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1_000_000.0 / iterations
    return output, float(wall_us)


class GemmA8W8BlockScaleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE}",
        "untune_file": "aiter/configs/a8w8_blockscale_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",  # for both results
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    }

    def __init__(self, name, keys, resultList, description=""):
        """
        Initialize the Gemm A8W8 BlockScale Tuner.
        """

        super().__init__(name, keys, resultList, description)

    def run(self, args, fast_mode=False):
        if getattr(args, "production_e2e", False) and not args.run_config:
            self.parser.error("--production-e2e requires --run_config")
        if getattr(args, "production_e2e", False) and not args.preshuffle:
            self.parser.error("--production-e2e requires --preshuffle")
        if int(getattr(args, "max_shapes", 0)) < 0:
            self.parser.error("--max-shapes must be non-negative")
        if getattr(args, "preshuffle", False):
            self.ARG_DEFAULTS["config_env_name"] = (
                "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"
            )
            self.ARG_DEFAULTS["tune_file"] = (
                f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}"
            )
        return super().run(args, fast_mode)

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op.get_CKGEMM_config.cache_clear()
        _op._CKGEMM_CONFIG_CACHE.clear()
        _op._CKGEMM_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        """
        Setup specific arguments for the tuner.
        """

        self.parser.add_argument(
            "--libtype",
            type=str,
            default="all",
            choices=[
                "ck",
                "cktile",
                "asm",
                "flydsl",
                "legacy",
                "all",
                "both",
            ],
            required=False,
            help=(
                "A8W8 blockscale backend to tune: ck, cktile, asm, flydsl, "
                "both (ck+cktile), legacy (ck+cktile+asm), or all (also "
                "includes FlyDSL)"
            ),
        )

        self.parser.add_argument(
            "--preshuffle",
            action="store_true",
            help="Enable B-matrix preshuffle for CK gemm a8w8 blockscale",
        )

        self.parser.add_argument(
            "--blockPerCu",
            nargs="+",
            type=int,
            default=list(range(1, BLOCK_PER_CU_MAX + 1)),
            help="List of BlockPerCu values to tune (CKTile only)",
        )

        self.parser.add_argument(
            "--production_e2e",
            "--production-e2e",
            action="store_true",
            help=(
                "With --run_config --preshuffle, benchmark the full production "
                "BF16 A -> per-1x128 FP8 quant -> dispatcher -> CK/CKTile/ASM "
                "GEMM path and report both native-quantized and BF16-reference "
                "accuracy. Without this flag, preserve the historical "
                "pre-quantized run_config behavior."
            ),
        )
        self.parser.add_argument(
            "--run_config_output",
            "--run-config-output",
            default="",
            help=(
                "Optional resumable CSV checkpoint written by "
                "--production-e2e --run_config."
            ),
        )
        self.parser.add_argument(
            "--num_rotate_args",
            "--num-rotate-args",
            type=int,
            default=1,
            help=(
                "Number of GPU argument copies used by production E2E "
                "perftest; use 1 to keep checkpoint B identity/static."
            ),
        )
        self.parser.add_argument(
            "--max_shapes",
            "--max-shapes",
            type=int,
            default=0,
            help=(
                "Process at most this many pending production-E2E shapes "
                "(0 means all)."
            ),
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        """
        Calculate performance metrics based on results.
        """

        _info, time, _err_ratio = results
        if time == self.INVALID_TIME or time == self.INF_TIME:
            return 0, 0
        return super().calculate(results, bpes=(1, 1, 2))

    def getKernelName(self, kernelId, libType="ck", preshuffleB=False):
        """
        Get the kernel name based on the kernel ID for different types.
        """
        if libType == "ck":
            kernel_list = (
                candidate_kernels_bpreshuffle_dict
                if preshuffleB
                else candidate_kernels_dict
            )
        elif libType == "cktile":
            # kernel_list = candidate_kernels_bpreshuffle_cktile_dict if preshuffleB else candidate_kernels_cktile_dict
            kernel_list = candidate_kernels_cktile_dict
        elif libType == "flydsl":
            if kernelId not in kernels_list_flydsl_blockscale:
                return None
            return kernels_list_flydsl_blockscale[kernelId].name
        elif libType == "flydsl8w":
            if kernelId not in kernels_list_flydsl_blockscale_8w:
                return None
            return kernels_list_flydsl_blockscale_8w[kernelId].name
        else:
            return None

        if kernelId >= len(kernel_list) or kernelId < 0:
            return None
        return kernel_list[kernelId].name

    def get_asm_kernels(self, file, preshuffleB):
        if not os.path.exists(file):
            print(f"ASM kernel list file not exist: {file}")
            return {}

        df = pd.read_csv(file)
        asm_df = (
            df[df["bpreshuffle"] == int(preshuffleB)]
            .reset_index(drop=True)
            .sort_values(by=["tile_m", "tile_n", "splitK"])
        )
        kernel_dict = (
            asm_df.groupby(["tile_m", "tile_n", "splitK"])["knl_name"]
            .apply(list)
            .to_dict()
        )
        return kernel_dict

    def get_gemm_a8w8_blockscale_cktile_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
        block_per_cu,
        run_kwargs,
    ):
        gfx, cu_num, M, N, K = info_keys
        # kernel_list = candidate_kernels_bpreshuffle_cktile_dict if preshuffleB else candidate_kernels_cktile_dict
        kernel_list = {
            k: v
            for k, v in candidate_kernels_cktile_dict.items()
            if v.BlockPerCu in block_per_cu
        }
        gemm_keys = (
            ["x", "weight_shuffle", "x_scale_t", "w_scale", "out"]
            if preshuffleB
            else ["x", "weight", "x_scale", "w_scale", "out"]
        )
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks_cktile = []
        for i, kernel in kernel_list.items():
            if not get_gfx().startswith("gfx95"):
                if (kernel.M_Warp * kernel.N_Warp * kernel.K_Warp == 8) or (
                    kernel.K_Warp_Tile > 64  # gfx942 not support
                ):
                    continue

            maxsplitK = (
                0
                if preshuffleB
                else (
                    aiter.compute_gemm_SplitK(
                        M,
                        N,
                        K,
                        kernel.M_Tile,
                        kernel.N_Tile,
                        kernel.K_Tile,
                    )
                    if useSplitK
                    else 0
                )
            )
            for splitK in range(maxsplitK + 1):
                info = (info_keys, i, splitK, "", "cktile", preshuffleB)
                tasks_cktile.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a8w8_blockscale_cktile,
                        (
                            gemm_keys,
                            i,
                            splitK,
                            preshuffleB,
                        ),
                        dict(run_kwargs),
                        run_torch,
                        (
                            ref_keys,
                            None,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                    )
                )
        return tasks_cktile

    def get_gemm_a8w8_blockscale_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
        run_kwargs,
    ):
        gfx, cu_num, M, N, K = info_keys
        kernel_list = (
            candidate_kernels_bpreshuffle_dict
            if preshuffleB
            else candidate_kernels_dict
        )
        kernels_num = len(kernel_list)
        gemm_keys = (
            ["x", "weight_shuffle", "x_scale_t", "w_scale", "out"]
            if preshuffleB
            else ["x", "weight", "x_scale", "w_scale", "out"]
        )
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks_ck = []
        for i in range(kernels_num):
            kernel = kernel_list[i]
            maxsplitK = (
                0
                if preshuffleB
                else (
                    aiter.compute_gemm_SplitK(
                        M,
                        N,
                        K,
                        kernel.MPerBLOCK,
                        kernel.NPerBLOCK,
                        kernel.KPerBLOCK,
                    )
                    if useSplitK
                    else 0
                )
            )
            for splitK in range(maxsplitK + 1):
                info = (info_keys, i, splitK, "", "ck", preshuffleB)
                tasks_ck.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed),
                        run_gemm_a8w8_blockscale,
                        (
                            gemm_keys,
                            i,
                            splitK,
                            preshuffleB,
                        ),
                        dict(run_kwargs),
                        run_torch,
                        (
                            ref_keys,
                            None,
                            dtypes.bf16,
                        ),
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                    )
                )
        return tasks_ck

    def run_production_e2e(self, args):
        """Benchmark legacy blockscale winners from the real BF16 boundary.

        The timed operation is the public production dispatcher, not a tune
        wrapper.  A native reference independently applies the same per-1x128
        FP8+FP32 activation format; a second reference keeps A in BF16.  This
        separates backend/dispatcher correctness from intrinsic quantization
        loss using the same protocol as the MXFP8 FlyDSL production report.
        """
        if not args.preshuffle:
            raise ValueError("--production-e2e currently requires --preshuffle")
        if not args.run_config:
            raise ValueError("--production-e2e requires --run_config")
        if int(args.num_rotate_args) != 1:
            raise ValueError(
                "production E2E requires --num-rotate-args 1 so static B "
                "retains its identity across calls"
            )
        if int(args.max_shapes) < 0:
            raise ValueError("--max-shapes must be non-negative")

        from aiter.jit import core as jit_core
        from aiter.ops import gemm_op_a8w8 as gemm_op
        from aiter.ops.quant import per_group_quant_hip
        from aiter.test_common import run_perftest

        production_op = gemm_op.gemm_a8w8_blockscale_bpreshuffle
        tuned_file = (
            jit_core.AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
        )
        checkpoint_path = str(args.run_config_output or "").strip()
        checkpoint_columns = [
            "gfx",
            "cu_num",
            "M",
            "N",
            "K",
            "libtype",
            "kernelId",
            "splitK",
            "kernel_only_us",
            "warmup",
            "iters",
            "num_rotate_args",
            "allowed_error",
            "cold_device_us",
            "cold_wall_us",
            "e2e_us",
            "e2e_wall_us",
            "native_err_ratio",
            "native_rel_l2",
            "native_mean_abs",
            "native_max_abs",
            "bf16_err_ratio",
            "bf16_rel_l2",
            "bf16_mean_abs",
            "bf16_max_abs",
            "status",
        ]
        resume_columns = [
            "gfx",
            "cu_num",
            "M",
            "N",
            "K",
            "libtype",
            "kernelId",
            "splitK",
            "warmup",
            "iters",
            "num_rotate_args",
            "allowed_error",
        ]

        checkpoint = pd.DataFrame(columns=checkpoint_columns)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = pd.read_csv(checkpoint_path)
            missing = sorted(set(checkpoint_columns) - set(checkpoint.columns))
            if missing:
                raise ValueError(
                    f"run-config checkpoint {checkpoint_path!r} is missing "
                    f"columns: {missing}"
                )
            checkpoint = checkpoint[checkpoint_columns]

        def config_value(config, name, default):
            if config is None:
                return default
            value = config.get(name, default)
            return default if pd.isna(value) else value

        def normalized_int(config, name, default):
            return int(config_value(config, name, default))

        def resume_key(
            *, gfx, cu_num, M, N, K, libtype, kernel_id, split_k, allowed_error
        ):
            return (
                str(gfx),
                int(cu_num),
                int(M),
                int(N),
                int(K),
                str(libtype),
                int(kernel_id),
                int(split_k),
                int(args.warmup),
                int(args.iters),
                int(args.num_rotate_args),
                float(allowed_error),
            )

        completed = set()
        if not checkpoint.empty:
            passed = checkpoint[checkpoint["status"] == "ok"]
            for row in passed.itertuples(index=False):
                completed.add(
                    (
                        str(row.gfx),
                        int(row.cu_num),
                        int(row.M),
                        int(row.N),
                        int(row.K),
                        str(row.libtype),
                        int(row.kernelId),
                        int(row.splitK),
                        int(row.warmup),
                        int(row.iters),
                        int(row.num_rotate_args),
                        float(row.allowed_error),
                    )
                )

        def persist(record):
            nonlocal checkpoint
            if not checkpoint_path:
                return
            checkpoint = pd.concat(
                [checkpoint, pd.DataFrame([record], columns=checkpoint_columns)],
                ignore_index=True,
            )
            checkpoint = checkpoint.drop_duplicates(
                subset=resume_columns,
                keep="last",
            )
            directory = os.path.dirname(os.path.abspath(checkpoint_path))
            os.makedirs(directory, exist_ok=True)
            temporary = os.path.join(
                directory,
                f".{os.path.basename(checkpoint_path)}.{os.getpid()}.tmp",
            )
            checkpoint.to_csv(temporary, index=False)
            os.replace(temporary, checkpoint_path)

        results = []
        pending_processed = 0
        gfx = self.get_gfx()
        cu_num = self.get_cu_num()
        for i in range(len(self.untunedf)):
            shape_row = self.untunedf.iloc[i]
            M, N, K = (
                int(shape_row["M"]),
                int(shape_row["N"]),
                int(shape_row["K"]),
            )
            shape = f"({M}, {N}, {K})"
            config = gemm_op.get_CKGEMM_config(M, N, K, tuned_file)
            libtype = str(config_value(config, "libtype", "missing")).lower()
            kernel_id = normalized_int(config, "kernelId", -1)
            split_k = normalized_int(config, "splitK", 0)
            kernel_only_us = float(config_value(config, "us", float("nan")))
            allowed_error, allowed_error_desc = (
                self._get_run_config_err_ratio_limit(config, args)
            )
            key = resume_key(
                gfx=gfx,
                cu_num=cu_num,
                M=M,
                N=N,
                K=K,
                libtype=libtype,
                kernel_id=kernel_id,
                split_k=split_k,
                allowed_error=allowed_error,
            )
            if key in completed:
                continue
            if args.max_shapes and pending_processed >= int(args.max_shapes):
                break
            pending_processed += 1

            record = {
                "gfx": str(gfx),
                "cu_num": int(cu_num),
                "M": M,
                "N": N,
                "K": K,
                "libtype": libtype,
                "kernelId": kernel_id,
                "splitK": split_k,
                "kernel_only_us": kernel_only_us,
                "warmup": int(args.warmup),
                "iters": int(args.iters),
                "num_rotate_args": int(args.num_rotate_args),
                "allowed_error": float(allowed_error),
                "cold_device_us": -1.0,
                "cold_wall_us": -1.0,
                "e2e_us": -1.0,
                "e2e_wall_us": -1.0,
                "native_err_ratio": float("nan"),
                "native_rel_l2": float("nan"),
                "native_mean_abs": float("nan"),
                "native_max_abs": float("nan"),
                "bf16_err_ratio": float("nan"),
                "bf16_rel_l2": float("nan"),
                "bf16_mean_abs": float("nan"),
                "bf16_max_abs": float("nan"),
                "status": "not-run",
            }
            try:
                if config is None:
                    raise ValueError(
                        f"no tuned config for M={M}, N={N}, K={K} in "
                        f"{tuned_file}"
                    )
                if libtype not in {"ck", "cktile", "asm"}:
                    raise ValueError(
                        "legacy production E2E requires a CK/CKTile/ASM "
                        f"winner, got libtype={libtype!r} for {shape}"
                    )

                data = generate_production_data(M, N, K, 0)
                production_args = (
                    data["A"],
                    data["B"],
                    None,
                    data["b_scale"],
                )

                # Compile and initialize first. The measured cold call retains
                # compiled kernels but includes config lookup, activation quant,
                # dispatcher overhead, and the selected GEMM backend.
                production_op(*production_args)
                torch.cuda.synchronize()
                self._clear_op_caches()
                _, cold_device_us, cold_wall_us = measure_single_call(
                    production_op, *production_args
                )

                out, e2e_us = run_perftest(
                    production_op,
                    *production_args,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    num_rotate_args=args.num_rotate_args,
                )
                out, e2e_wall_us = measure_steady_wall(
                    production_op,
                    production_args,
                    args.iters,
                )

                # Independent legacy-native reference: quantize the original A
                # once to per-1x128 E4M3+FP32, undo the CK scale transpose as a
                # view, and GEMM against the same dequantized checkpoint B.
                a_native_q, a_native_scale_t = per_group_quant_hip(
                    data["A"],
                    quant_dtype=dtypes.fp8,
                    group_size=128,
                    transpose_scale=True,
                )
                groups = K // 128
                a_native_scale = a_native_scale_t.contiguous().view(groups, M).T
                a_native_deq = (
                    a_native_q.float().view(M, groups, 128)
                    * a_native_scale.unsqueeze(-1)
                ).view(M, K)
                b_deq = data["B_logical"].float() * data[
                    "b_scale"
                ].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                native_reference = (a_native_deq @ b_deq.T).to(dtypes.bf16)
                native_metrics = calculate_error_metrics(
                    out.to(dtypes.bf16), native_reference
                )
                del (
                    a_native_q,
                    a_native_scale_t,
                    a_native_scale,
                    a_native_deq,
                    native_reference,
                )

                bf16_reference = (data["A"].float() @ b_deq.T).to(dtypes.bf16)
                bf16_metrics = calculate_error_metrics(
                    out.to(dtypes.bf16), bf16_reference
                )
                del b_deq, bf16_reference

                status = (
                    "ok"
                    if native_metrics["err_ratio"] <= allowed_error
                    else "mismatch:native_err_ratio="
                    f"{native_metrics['err_ratio']:.6g}"
                    f"(>{allowed_error_desc})"
                )
                record.update(
                    {
                        "cold_device_us": cold_device_us,
                        "cold_wall_us": cold_wall_us,
                        "e2e_us": float(e2e_us),
                        "e2e_wall_us": e2e_wall_us,
                        "native_err_ratio": native_metrics["err_ratio"],
                        "native_rel_l2": native_metrics["rel_l2"],
                        "native_mean_abs": native_metrics["mean_abs"],
                        "native_max_abs": native_metrics["max_abs"],
                        "bf16_err_ratio": bf16_metrics["err_ratio"],
                        "bf16_rel_l2": bf16_metrics["rel_l2"],
                        "bf16_mean_abs": bf16_metrics["mean_abs"],
                        "bf16_max_abs": bf16_metrics["max_abs"],
                        "status": status,
                    }
                )
                results.append(
                    {
                        "shape": shape,
                        "kernel_us": kernel_only_us,
                        "e2e_us": float(e2e_us),
                        "status": status,
                    }
                )
                print(
                    f"[production-e2e] {shape} backend={libtype} "
                    f"kernel={kernel_only_us:.4f}us steady_device={e2e_us:.4f}us "
                    f"wall={e2e_wall_us:.4f}us "
                    f"native_err={native_metrics['err_ratio']:.6%} "
                    f"bf16_rel_l2={bf16_metrics['rel_l2']:.6%}",
                    flush=True,
                )
            except Exception as error:
                status = f"error:{error}"
                record["status"] = status
                results.append(
                    {
                        "shape": shape,
                        "kernel_us": kernel_only_us,
                        "e2e_us": -1,
                        "status": status,
                    }
                )
            finally:
                persist(record)
                torch.cuda.empty_cache()
        return results

    def run_config(self, args):
        if getattr(args, "production_e2e", False):
            return self.run_production_e2e(args)

        from aiter.ops.gemm_op_a8w8 import (
            gemm_a8w8_blockscale,
            gemm_a8w8_blockscale_bpreshuffle,
        )
        from aiter.test_common import run_perftest, checkAllclose

        is_preshuffle = args.preshuffle
        untunedf = self.untunedf
        run_kwargs = {
            "num_warmup": args.warmup,
            "num_iters": args.iters,
        }
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            shape_str = f"({M}, {N}, {K})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                gd = generate_data(M, N, K, 0)
                x, weight, x_scale, w_scale, out = (
                    gd["x"],
                    gd["weight"],
                    gd["x_scale"],
                    gd["w_scale"],
                    gd["out"],
                )
                weight_shuffle, x_scale_t = gd["weight_shuffle"], gd["x_scale_t"]
                if is_preshuffle:
                    out, us = run_perftest(
                        gemm_a8w8_blockscale_bpreshuffle,
                        x,
                        weight_shuffle,
                        x_scale_t,
                        w_scale,
                        **run_kwargs,
                    )
                else:
                    out, us = run_perftest(
                        gemm_a8w8_blockscale,
                        x,
                        weight,
                        x_scale,
                        w_scale,
                        **run_kwargs,
                    )
                ref = run_torch(x, weight, x_scale, w_scale)
                err_ratio = checkAllclose(out, ref, msg=f"run_config {shape_str}")
                status = (
                    "ok"
                    if err_ratio <= allowed_err_ratio
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_err_ratio_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def get_gemm_a8w8_blockscale_asm_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
        run_kwargs,
    ):
        gfx, cu_num, M, N, K = info_keys
        asm_kernel_list_csv = (
            f"{get_asm_dir()}/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv"
        )
        asm_kernels = self.get_asm_kernels(asm_kernel_list_csv, preshuffleB)
        if not asm_kernels:
            return []

        gemm_asm_keys = (
            ["x", "weight_shuffle", "x_scale_t", "w_scale", "out", "zero_bias"]
            if preshuffleB
            else ["x", "weight", "x_scale", "w_scale", "out", "zero_bias"]
        )
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks_asm = []
        asm_kernel_id = 0
        for key, kernel_names in asm_kernels.items():
            _tile_m, tile_n, splitk_supported = key
            # Respect ASM kernel tile constraints from the config CSV.
            if N % tile_n != 0:
                continue
            splitK_list = (
                get_valid_asm_splitK_list(K, 8)
                if useSplitK and int(splitk_supported) == 1
                else [1]
            )
            for kernel_name in kernel_names:
                for splitK in splitK_list:
                    info = (
                        info_keys,
                        asm_kernel_id,
                        splitK,
                        kernel_name,
                        "asm",
                        preshuffleB,
                    )
                    tasks_asm.append(
                        (
                            info,
                            generate_data,
                            (M, N, K, seed),
                            run_gemm_a8w8_blockscale_asm,
                            (
                                gemm_asm_keys,
                                kernel_name,
                                splitK,
                                preshuffleB,
                            ),
                            dict(run_kwargs),
                            run_torch,
                            (
                                ref_keys,
                                None,
                                dtypes.bf16,
                            ),
                            {},
                            None,
                            1e-2,
                            0.01,
                            None,
                            None,
                            ("out",),
                        )
                    )
                    asm_kernel_id += 1
        return tasks_asm

    def get_gemm_a8w8_blockscale_flydsl_tune_task(
        self,
        info_keys,
        seed,
        preshuffleB,
        run_kwargs,
    ):
        """Build FlyDSL blockscale-bpreshuffle candidate tasks for one shape.

        FlyDSL blockscale is preshuffleB-only and targets gfx95x (scaled-MFMA +
        fused_promote). Candidates are filtered per shape by LDS budget, tile
        divisibility, CTA count and the i32 index guard so only launchable,
        dispatch-round-trippable configs reach the profiler.
        """

        gfx, cu_num, M, N, K = info_keys

        # FlyDSL blockscale kernel requires a preshuffled B and the gfx95x
        # scaled-MFMA path; skip otherwise so ck/cktile/asm still tune normally.
        if not preshuffleB:
            return []
        if not str(gfx).startswith("gfx95"):
            return []
        if (not kernels_list_flydsl_blockscale) or (
            "flydsl_blockscale_preshuffle_gemm_a8" not in globals()
        ):
            return []

        # Guard against i32 element indexing overflow inside the kernel
        # (M*N must stay < 2^31); dispatch falls back to ck/cktile/asm here.
        if int(M) * int(N) >= (1 << 31):
            return []

        gemm_flydsl_keys = ["x", "weight_shuffle", "x_scale_t", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks = []
        lds_limit = max_lds_bytes_for_tune()
        padded_m = _get_padded_m(M)
        min_ctas = max(4, min(16, N // 64))
        # Full 608-config sweep by default; FLYDSL_BS_TUNE_FOCUSED=1 selects the
        # curated ~75-config subset (ids are a subset of the full list, so the
        # runner/getKernelName lookups below stay valid).
        tune_kernels = get_flydsl_blockscale_tune_kernels()
        for i in sorted(tune_kernels.keys()):
            ki = tune_kernels[i]
            if kernel_instance_estimated_lds_bytes(ki) > lds_limit:
                continue
            if N % ki.tile_n != 0 or K % ki.tile_k != 0:
                continue
            if K % ki.scale_block_k != 0:
                continue
            if padded_m % ki.tile_m != 0:
                continue
            num_ctas = ((M + ki.tile_m - 1) // ki.tile_m) * (N // ki.tile_n)
            if num_ctas < min_ctas:
                continue
            if M >= 8192 and ki.tile_m < 64:
                continue
            if M >= 4096 and ki.tile_m < 32:
                continue
            if M >= 2048 and ki.tile_m == 16 and ki.tile_n <= 128:
                continue
            # XCD workgroup swizzle needs enough workgroups to be meaningful;
            # skip xcd>0 on shapes with <64 CTAs (matches non-blockscale flydsl).
            if getattr(ki, "xcd_swizzle", 0) > 0 and num_ctas < 64:
                continue
            kernel_name = ki.name
            info = (info_keys, i, 0, kernel_name, "flydsl", preshuffleB)
            tasks.append(
                (
                    info,
                    generate_data,
                    (M, N, K, seed),
                    run_gemm_a8w8_blockscale_flydsl,
                    (
                        gemm_flydsl_keys,
                        i,
                    ),
                    dict(run_kwargs),
                    run_torch,
                    (
                        ref_keys,
                        None,
                        dtypes.bf16,
                    ),
                    {},
                    None,
                    1e-2,
                    0.01,
                    None,
                    None,
                    ("out",),
                )
            )
        return tasks

    def get_gemm_a8w8_blockscale_flydsl_8w_tune_task(
        self,
        info_keys,
        seed,
        preshuffleB,
        run_kwargs,
    ):
        """Build 8-wave blockscale-bpreshuffle candidate tasks for one shape.

        Same family as the 4-wave builder (preshuffleB-only, gfx95x, i32-guarded)
        but for ``compile_fp8_gemm_8w_blockscale``. Candidates carry the distinct
        ``flydsl8w`` libtype + kernelName so the runtime dispatcher routes them to
        the 8-wave wrapper. B is the SAME ``shuffle_weight(16,16)`` bytes.
        """

        gfx, cu_num, M, N, K = info_keys

        if not preshuffleB:
            return []
        if not str(gfx).startswith("gfx95"):
            return []
        if (not kernels_list_flydsl_blockscale_8w) or (
            "flydsl_fp8_gemm_8wave_blockscale_a8" not in globals()
        ):
            return []
        # Guard against i32 element indexing overflow (M*N must stay < 2^31);
        # dispatch falls back to ck/cktile/asm here.
        if int(M) * int(N) >= (1 << 31):
            return []

        gemm_flydsl_keys = ["x", "weight_shuffle", "x_scale_t", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks = []
        padded_m = _get_padded_m(M)
        min_ctas = max(4, min(16, N // 64))
        tune_kernels = get_flydsl_blockscale_8w_tune_kernels()
        for i in sorted(tune_kernels.keys()):
            ki = tune_kernels[i]
            # BLOCK_N is locked to 256; scale block K is 128 (== preshuffle_b K%64).
            if N % ki.block_n != 0 or K % 128 != 0:
                continue
            if padded_m % ki.block_m != 0:
                continue
            num_ctas = ((M + ki.block_m - 1) // ki.block_m) * (N // ki.block_n)
            if num_ctas < min_ctas:
                continue
            # XCD workgroup swizzle needs enough workgroups to be meaningful.
            if ki.use_xcd_remap and num_ctas < 64:
                continue
            kernel_name = ki.name
            info = (info_keys, i, 0, kernel_name, "flydsl8w", preshuffleB)
            tasks.append(
                (
                    info,
                    generate_data,
                    (M, N, K, seed),
                    run_gemm_a8w8_blockscale_flydsl_8w,
                    (
                        gemm_flydsl_keys,
                        i,
                    ),
                    dict(run_kwargs),
                    run_torch,
                    (
                        ref_keys,
                        None,
                        dtypes.bf16,
                    ),
                    {},
                    None,
                    1e-2,
                    0.01,
                    None,
                    None,
                    ("out",),
                )
            )
        return tasks

    def tune(
        self,
        untunedf,
        tunedf,
        args,
    ):
        useSplitK = args.splitK
        mp_num = args.mp
        isPreshuffleB = args.preshuffle
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        block_per_cu = args.blockPerCu
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()
        run_kwargs = {
            "num_warmup": args.warmup,
            "num_iters": args.iters,
        }
        task = []
        tasks_data = []  # [(kernel_nums, datas)]
        seed = 0
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            prev_task_count = len(task)
            info_keys = (gfx, cu_num, M, N, K)
            lib = args.libtype
            if lib in ("ck", "both", "legacy", "all"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
                    )
                )
            if lib in ("cktile", "both", "legacy", "all"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_cktile_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        isPreshuffleB,
                        block_per_cu,
                        run_kwargs,
                    )
                )
            if lib in ("asm", "legacy", "all"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_asm_tune_task(
                        info_keys,
                        useSplitK,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
                    )
                )
            if lib in ("flydsl", "all"):
                task.extend(
                    self.get_gemm_a8w8_blockscale_flydsl_tune_task(
                        info_keys,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
                    )
                )
                task.extend(
                    self.get_gemm_a8w8_blockscale_flydsl_8w_tune_task(
                        info_keys,
                        seed,
                        isPreshuffleB,
                        run_kwargs,
                    )
                )
            shape_kernel_nums = len(task) - prev_task_count

            # A shape may yield zero candidates (e.g. flydsl-only tuning where a
            # small shape fails every feasibility/min-CTA filter). Skip it so the
            # shape_grouped in_datas count stays equal to the task-group count.
            if shape_kernel_nums > 0:
                tasks_data.append((shape_kernel_nums, ()))
        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_data,
                mp_num,
                False,
                shape_grouped,
                errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
            )
        return ret

    def result_to_df(self, results):
        """
        post-process the tuning results into a DataFrame.
        """

        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype, preshuffleB = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME or time == self.INF_TIME
                else (
                    self.getKernelName(kernelId, libtype, preshuffleB)
                    if kernelName == ""
                    else kernelName
                )
            )
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))

            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} "
                    f"{kernelName} splitK={splitK}, {time}us, err_ratio={err_ratio}, "
                    f"tflops={tflops} TFLOPS, bw={bw} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "errRatio": [err_ratio],
                    "tflops": [tflops],
                    "bw": [bw],
                }
            )
            temp = pd.DataFrame(key_dict)
            if resultdf.empty:
                resultdf = temp
            else:
                resultdf = pd.concat([resultdf, temp], ignore_index=True)
        return resultdf


if __name__ == "__main__":
    key = ["gfx", "cu_num", "M", "N", "K"]
    resultList = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    tuner = GemmA8W8BlockScaleTuner(
        "GemmA8W8BlockScaleTuner",
        key,
        resultList,
        description="Tune a8w8 blockscale GEMM (CK, CKTile, ASM backends)",
    )

    args = tuner.parse_args()
    tuner.run(args, False)
