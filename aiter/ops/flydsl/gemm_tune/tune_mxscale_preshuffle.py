# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune the gfx950 FlyDSL MXScale preshuffle GEMM.

This tuner selects the best FlyDSL JIT specialization for each input shape and
writes ``libtype=flydsl`` rows into the existing A8W8 blockscale bpreshuffle
tuned table. It does not compare FlyDSL with CK/CK-Tile/ASM; tune only shapes
whose shared-table winner should be replaced by FlyDSL.

Example::

    HIP_VISIBLE_DEVICES=0 python -m \
      aiter.ops.flydsl.gemm_tune.tune_mxscale_preshuffle \
      -i /path/to/shapes.csv \
      -o /path/to/a8w8_blockscale_bpreshuffle_tuned_gemm.csv \
      --shape_grouped --warmup 20 --iters 100
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE
from aiter.ops.flydsl.mxscale_preshuffle_config import (
    candidates_for,
    kernels_list,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import (
    per_1x32_f4_quant,
    per_1x32_f8_scale_f8_quant,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight
from aiter.utility import fp4_utils
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

if is_flydsl_available():
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
        flydsl_mxscale_preshuffle_gemm,
    )


def _quant(x_f, dtype):
    if dtype == "fp8":
        return per_1x32_f8_scale_f8_quant(
            x_f,
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
        )
    if dtype == "fp4":
        return per_1x32_f4_quant(x_f, quant_dtype=dtypes.fp4x2)
    raise ValueError(
        "tuning supports fp4/fp8 operands only "
        f"(AITER has no fp6 quantizer), got {dtype!r}"
    )


def _dequant(codes, scale, dtype, rows):
    scale_f32 = fp4_utils.e8m0_to_f32(
        scale[:rows].repeat_interleave(32, dim=1)
    )
    if dtype == "fp8":
        return codes.float() * scale_f32
    return fp4_utils.mxfp4_to_f32(codes) * scale_f32


def generate_data(
    m,
    n,
    k,
    seed,
    a_dtype,
    b_dtype,
    dtype=dtypes.bf16,
    device="cuda",
):
    torch.manual_seed(seed)
    padded_m = (m + 31) // 32 * 32
    padded_n = (n + 31) // 32 * 32
    a_f = torch.zeros((padded_m, k), dtype=torch.float32, device=device)
    b_f = torch.zeros((padded_n, k), dtype=torch.float32, device=device)
    a_f[:m] = torch.randn((m, k), device=device)
    b_f[:n] = torch.randn((n, k), device=device)

    a_q, a_scale_logical = _quant(a_f, a_dtype)
    b_q, b_scale_logical = _quant(b_f, b_dtype)
    a_codes = a_q[:m]
    b_codes = b_q[:n]
    b_shuffled = shuffle_weight(b_codes, layout=(16, 16))
    a_scale = shuffle_scale_a16w4(a_scale_logical, 1, False)
    b_scale = shuffle_scale_a16w4(b_scale_logical, 1, False)
    return {
        "A": a_codes,
        "B": b_shuffled,
        "a_scale": a_scale,
        "b_scale": b_scale,
        "out": torch.empty((m, n), dtype=dtype, device=device),
        "a_deq": _dequant(a_codes, a_scale_logical, a_dtype, m),
        "b_deq": _dequant(b_codes, b_scale_logical, b_dtype, n),
    }


def generate_production_a8w8_data(
    m,
    n,
    k,
    seed,
    dtype=dtypes.bf16,
    device="cuda",
):
    """Build the inputs at the SGLang -> AITER production boundary.

    The activation intentionally remains BF16 and has no scale.  This makes
    ``gemm_a8w8_blockscale_bpreshuffle`` query the tuned backend first and,
    for a FlyDSL winner, quantize BF16 directly to native per-1x32 MXFP8.
    Passing a pre-quantized FP8 activation here would benchmark the legacy
    compatibility requantization path instead of the production path.
    """
    if n % 128 != 0 or k % 128 != 0:
        raise ValueError(
            f"A8W8 MXFP8 bridge requires N/K divisible by 128, got N={n}, K={k}"
    )
    torch.manual_seed(seed)
    a_bf16 = torch.randn((m, k), dtype=dtype, device=device) * 0.25

    b_q = (
        torch.randn((n, k), dtype=torch.float32, device=device) * 0.25
    ).to(dtypes.fp8)
    b_kernel = shuffle_weight(b_q.contiguous(), layout=(16, 16))

    # SGLang exposes checkpoint E8M0 scales as FP32. Powers of two reproduce
    # that exact, lossless E8M0 -> FP32 load conversion.
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


def calculate_error_metrics(
    actual,
    reference,
    *,
    atol=1e-2,
    rtol=1e-2,
):
    """Return thresholded ratio, relative L2, mean abs, and max abs error."""
    actual_f32 = actual.to(torch.float32)
    reference_f32 = reference.to(torch.float32)
    abs_delta = (actual_f32 - reference_f32).abs()
    mismatch = abs_delta > (atol + rtol * reference_f32.abs())
    error_ratio = mismatch.count_nonzero().item() / actual.numel()
    reference_l2 = torch.linalg.vector_norm(reference_f32).item()
    delta_l2 = torch.linalg.vector_norm(abs_delta).item()
    relative_l2 = delta_l2 / reference_l2 if reference_l2 else delta_l2
    return {
        "err_ratio": float(error_ratio),
        "rel_l2": float(relative_l2),
        "mean_abs": float(abs_delta.mean().item()),
        "max_abs": float(abs_delta.max().item()),
    }


def measure_single_call(function, *arguments):
    """Measure one synchronized call with GPU events and host wall-clock."""
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
    """Measure amortized synchronized wall time while keeping B static."""
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = function(*arguments)
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - wall_start) * 1_000_000.0 / iterations
    return output, float(wall_us)


def run_gemm_flydsl(
    A,
    B,
    a_scale,
    b_scale,
    out,
    kernel_id,
    a_dtype,
    b_dtype,
    split_k,
):
    instance = kernels_list[kernel_id]
    return flydsl_mxscale_preshuffle_gemm(
        A,
        B,
        a_scale,
        b_scale,
        out,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        tile_m=instance.tile_m,
        tile_n=instance.tile_n,
        tile_k=instance.tile_k,
        waves_per_eu=instance.waves_per_eu,
        xcd_swizzle=instance.xcd_swizzle,
        split_k=split_k,
    )


def run_torch(a_deq, b_deq, dtype=dtypes.bf16):
    return (a_deq @ b_deq.T).to(dtype)


class MxscalePreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
        "untune_file": "aiter/configs/a8w8_blockscale_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    }

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as op

        op.get_CKGEMM_config.cache_clear()
        op._CKGEMM_CONFIG_CACHE.clear()
        op._CKGEMM_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--max_shapes",
            "--max-shapes",
            type=int,
            default=0,
            help=(
                "Process at most this many currently-untuned shapes, then exit "
                "cleanly (0 means no limit). This makes long tuning jobs safe "
                "to resume across short execution sessions."
            ),
        )
        self.parser.add_argument(
            "--run_config_output",
            "--run-config-output",
            default="",
            help=(
                "Optional CSV checkpoint for production --run_config. "
                "Successful rows are skipped on resume and each newly tested "
                "shape is persisted atomically."
            ),
        )
        self.parser.add_argument(
            "--candidate_profile",
            "--candidate-profile",
            default="",
            help="Restrict tuning to candidates selected from an earlier profile CSV.",
        )
        self.parser.add_argument(
            "--shortlist_topk",
            "--shortlist-topk",
            type=int,
            default=0,
            help="With --candidate-profile, retain at least the fastest N candidates per shape.",
        )
        self.parser.add_argument(
            "--shortlist_ratio",
            "--shortlist-ratio",
            type=float,
            default=0.0,
            help="Also retain candidates within this fractional latency of the profile winner (0.10 = 10%%).",
        )
        self.parser.add_argument(
            "--num_rotate_args",
            "--num-rotate-args",
            type=int,
            default=0,
            help="Number of GPU argument copies used by perftest; 1 disables automatic cache-sized rotation.",
        )

    def pre_process(self, args):
        super().pre_process(args)
        if args.max_shapes < 0:
            raise ValueError("--max-shapes must be non-negative")
        if args.max_shapes and not args.run_config:
            self.untunedf = self.untunedf.iloc[: args.max_shapes].reset_index(
                drop=True
            )

    def _prepare_candidate_shortlist(self, args, gfx, cu_num):
        profile_path = str(args.candidate_profile or "").strip()
        if not profile_path:
            self._candidate_shortlist = None
            return

        if args.shortlist_topk < 0:
            raise ValueError("--shortlist-topk must be non-negative")
        if args.shortlist_ratio < 0:
            raise ValueError("--shortlist-ratio must be non-negative")
        if args.shortlist_topk == 0 and args.shortlist_ratio == 0:
            raise ValueError(
                "--candidate-profile requires --shortlist-topk > 0 or "
                "--shortlist-ratio > 0"
            )
        if args.num_rotate_args < 0:
            raise ValueError("--num-rotate-args must be non-negative")

        cache_key = (
            profile_path,
            int(args.shortlist_topk),
            float(args.shortlist_ratio),
            str(gfx),
            int(cu_num),
        )
        if getattr(self, "_candidate_shortlist_cache_key", None) == cache_key:
            return

        profile = pd.read_csv(profile_path)
        required = {"M", "N", "K", "kernelId", "splitK", "us", "errRatio"}
        missing = sorted(required - set(profile.columns))
        if missing:
            raise ValueError(
                f"candidate profile {profile_path!r} is missing columns: {missing}"
            )

        profile = profile.copy()
        for column in ("M", "N", "K", "kernelId", "splitK", "us", "errRatio"):
            profile[column] = pd.to_numeric(profile[column], errors="coerce")
        if "gfx" in profile.columns:
            profile = profile[profile["gfx"].astype(str) == str(gfx)]
        if "cu_num" in profile.columns:
            profile = profile[profile["cu_num"] == int(cu_num)]
        if "libtype" in profile.columns:
            profile = profile[profile["libtype"].astype(str) == "flydsl"]

        valid = (
            profile["us"].notna()
            & np.isfinite(profile["us"])
            & (profile["us"] > 0)
            & profile["errRatio"].notna()
            & (profile["errRatio"] <= float(args.errRatio))
            & profile[["M", "N", "K", "kernelId", "splitK"]]
            .notna()
            .all(axis=1)
        )
        profile = profile[valid].copy()
        if profile.empty:
            raise ValueError(
                f"candidate profile {profile_path!r} has no valid FlyDSL rows "
                f"for gfx={gfx}, cu_num={cu_num}"
            )

        shape_keys = ["M", "N", "K"]
        candidate_keys = shape_keys + ["kernelId", "splitK"]
        profile = (
            profile.sort_values("us")
            .drop_duplicates(candidate_keys, keep="first")
            .reset_index(drop=True)
        )
        grouped = profile.groupby(shape_keys, sort=False)
        profile["_rank"] = grouped["us"].rank(method="first")
        profile["_min_us"] = grouped["us"].transform("min")
        keep = profile["us"] <= profile["_min_us"] * (
            1.0 + float(args.shortlist_ratio)
        )
        if args.shortlist_topk > 0:
            keep |= profile["_rank"] <= int(args.shortlist_topk)
        selected = profile[keep]

        shortlist = {}
        for row in selected.itertuples(index=False):
            shape = (int(row.M), int(row.N), int(row.K))
            shortlist.setdefault(shape, set()).add(
                (int(row.kernelId), int(row.splitK))
            )

        self._candidate_shortlist = shortlist
        self._candidate_shortlist_cache_key = cache_key
        counts = [len(value) for value in shortlist.values()]
        print(
            "[mxscale] candidate shortlist: "
            f"{sum(counts)} candidates across {len(counts)} shapes "
            f"(min={min(counts)}, max={max(counts)}, "
            f"topk={args.shortlist_topk}, ratio={args.shortlist_ratio:g})",
            flush=True,
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="flydsl"):
        del libtype
        instance = kernels_list.get(kernelId)
        return instance.name if instance is not None else None

    def get_flydsl_mxscale_tune_task(self, info_keys, seed, args):
        gfx, cu_num, M, N, K = info_keys
        del gfx, cu_num
        a_dtype = b_dtype = "fp8"
        if (
            not is_flydsl_available()
            or "flydsl_mxscale_preshuffle_gemm" not in globals()
        ):
            return []

        gemm_keys = ["A", "B", "a_scale", "b_scale", "out"]
        ref_keys = ["a_deq", "b_deq"]
        shortlist = getattr(self, "_candidate_shortlist", None)
        allowed_candidates = None
        if shortlist is not None:
            shape = (M, N, K)
            if shape not in shortlist:
                raise ValueError(
                    "candidate profile has no shortlist for "
                    f"M={M}, N={N}, K={K}"
                )
            allowed_candidates = shortlist[shape]
        tasks = []
        for kernel_id, instance in candidates_for(a_dtype, b_dtype, M, N, K):
            split_candidates = [1]
            if args.splitK:
                split_k = 2
                while split_k <= 8 and K // split_k >= 256:
                    k_per_split = K // split_k
                    if (
                        K % split_k == 0
                        and k_per_split % instance.tile_k == 0
                        and k_per_split % 256 == 0
                    ):
                        split_candidates.append(split_k)
                    split_k *= 2
            for split_k in split_candidates:
                csv_split_k = 0 if split_k == 1 else split_k
                if (
                    allowed_candidates is not None
                    and (kernel_id, csv_split_k) not in allowed_candidates
                ):
                    continue
                info = (
                    info_keys,
                    kernel_id,
                    csv_split_k,
                    instance.name,
                    "flydsl",
                )
                tasks.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, a_dtype, b_dtype),
                        run_gemm_flydsl,
                        (
                            gemm_keys,
                            kernel_id,
                            a_dtype,
                            b_dtype,
                            split_k,
                        ),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                            "num_rotate_args": args.num_rotate_args,
                        },
                        run_torch,
                        (ref_keys, dtypes.bf16),
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

    def tune(self, untunedf, tunedf, args):
        del tunedf
        tasks = []
        tasks_data = []
        seed = 0
        gfx = self.get_gfx()
        cu_num = self.get_cu_num()
        self._prepare_candidate_shortlist(args, gfx, cu_num)
        for index in range(len(untunedf)):
            M = int(untunedf.loc[index, "M"])
            N = int(untunedf.loc[index, "N"])
            K = int(untunedf.loc[index, "K"])
            seed += 1
            info_keys = (gfx, cu_num, M, N, K)
            shape_tasks = self.get_flydsl_mxscale_tune_task(
                info_keys, seed, args
            )
            if not shape_tasks:
                print(
                    "[mxscale] skip shape with no legal candidate: "
                    f"M={M} N={N} K={K} fp8/fp8"
                )
                continue
            tasks.extend(shape_tasks)
            tasks_data.append((len(shape_tasks), ()))

        if not tasks:
            return []
        return mp_tuner(
            tasks,
            tasks_data,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def result_to_df(self, results):
        result = pd.DataFrame(columns=self.columns)
        for item in results:
            info, time_us, err_ratio = item
            keys, kernel_id, split_k, kernel_name, libtype = info
            if time_us in (self.INVALID_TIME, self.INF_TIME):
                kernel_name = "None"
            elif not kernel_name:
                kernel_name = self.getKernelName(kernel_id, libtype)
            tflops, bandwidth = self.calculate(item)
            row = dict(zip(self.keys, keys))
            row.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernel_id],
                    "splitK": [split_k],
                    "us": [time_us],
                    "kernelName": [kernel_name],
                    "errRatio": [err_ratio],
                    "tflops": [tflops],
                    "bw": [bandwidth],
                }
            )
            frame = pd.DataFrame(row)
            result = (
                frame
                if result.empty
                else pd.concat([result, frame], ignore_index=True)
            )
        return result

    def run_config(self, args):
        from aiter.test_common import run_perftest
        from aiter.ops import gemm_op_a8w8 as gemm_op
        from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
            clear_mxfp8_b_scale_cache,
        )
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle
        from aiter.ops.triton.quant import dynamic_mxfp8_quant

        if int(args.num_rotate_args) != 1:
            raise ValueError(
                "production --run_config requires --num-rotate-args 1: "
                "A is dynamic, but B and its checkpoint scale are static and "
                "must retain their identity so the prepared B-scale cache hits"
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

        def config_value(row, name, default):
            value = row[name] if name in row.index else default
            return default if pd.isna(value) else value

        def resume_key(row, allowed_error):
            return (
                str(config_value(row, "gfx", self.get_gfx())),
                int(config_value(row, "cu_num", self.get_cu_num())),
                int(row["M"]),
                int(row["N"]),
                int(row["K"]),
                str(config_value(row, "libtype", "flydsl")),
                int(config_value(row, "kernelId", -1)),
                int(config_value(row, "splitK", 0)),
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

        pending = []
        for _, row in self.untunedf.iterrows():
            allowed_error, allowed_error_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            if resume_key(row, allowed_error) not in completed:
                pending.append((row, allowed_error, allowed_error_desc))
        if args.max_shapes:
            pending = pending[: args.max_shapes]

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
        for row, allowed_error, allowed_error_desc in pending:
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            shape = f"({M}, {N}, {K})"
            record = {
                "gfx": str(config_value(row, "gfx", self.get_gfx())),
                "cu_num": int(config_value(row, "cu_num", self.get_cu_num())),
                "M": M,
                "N": N,
                "K": K,
                "libtype": str(config_value(row, "libtype", "flydsl")),
                "kernelId": int(config_value(row, "kernelId", -1)),
                "splitK": int(config_value(row, "splitK", 0)),
                "kernel_only_us": float(config_value(row, "us", float("nan"))),
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
                data = generate_production_a8w8_data(M, N, K, 0)
                production_args = (
                    data["A"],
                    data["B"],
                    None,
                    data["b_scale"],
                )

                # Compile/initialize first, then deliberately drop only the
                # prepared static-B cache.  The next measured call represents
                # a first weight use without mixing JIT compilation into it.
                gemm_a8w8_blockscale_bpreshuffle(*production_args)
                torch.cuda.synchronize()
                clear_mxfp8_b_scale_cache()
                gemm_op.get_CKGEMM_config.cache_clear()
                gemm_op._CKGEMM_CONFIG_CACHE.clear()
                _, cold_device_us, cold_wall_us = measure_single_call(
                    gemm_a8w8_blockscale_bpreshuffle,
                    *production_args,
                )

                out, time_us = run_perftest(
                    gemm_a8w8_blockscale_bpreshuffle,
                    *production_args,
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                    num_rotate_args=args.num_rotate_args,
                )
                out, wall_us = measure_steady_wall(
                    gemm_a8w8_blockscale_bpreshuffle,
                    production_args,
                    args.iters,
                )

                # Independent logical-scale quantization is the native MXFP8
                # reference.  It uses the non-packed quantizer, so this check
                # validates the packed production quantizer, scale addressing,
                # dispatcher, B-scale preparation, and GEMM together.
                a_native_q, a_native_scale = dynamic_mxfp8_quant(
                    data["A"],
                    quant_dtype=dtypes.fp8,
                )
                a_native_deq = a_native_q.float() * fp4_utils.e8m0_to_f32(
                    a_native_scale
                ).repeat_interleave(32, dim=1)
                b_deq = data["B_logical"].float() * data[
                    "b_scale"
                ].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                native_reference = (a_native_deq @ b_deq.T).to(dtypes.bf16)
                native_metrics = calculate_error_metrics(
                    out.to(dtypes.bf16),
                    native_reference,
                )
                del a_native_q, a_native_scale, a_native_deq, native_reference

                # This second reference starts from the BF16 activation that
                # arrived from SGLang and the already-quantized checkpoint
                # weight.  Its delta therefore isolates activation MXFP8
                # quantization error from kernel/dispatcher implementation
                # error.
                bf16_reference = (
                    data["A"].float() @ b_deq.T
                ).to(dtypes.bf16)
                bf16_metrics = calculate_error_metrics(
                    out.to(dtypes.bf16),
                    bf16_reference,
                )
                del b_deq, bf16_reference

                status = (
                    "ok"
                    if native_metrics["err_ratio"] <= allowed_error
                    else "mismatch:native_err_ratio="
                    f"{native_metrics['err_ratio']:.6g}"
                    f"(>{allowed_error_desc})"
                )
                results.append(
                    {
                        "shape": shape,
                        "kernel_us": record["kernel_only_us"],
                        "e2e_us": time_us,
                        "status": status,
                    }
                )
                record.update(
                    {
                        "cold_device_us": cold_device_us,
                        "cold_wall_us": cold_wall_us,
                        "e2e_us": float(time_us),
                        "e2e_wall_us": wall_us,
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
                print(
                    f"[run-config] {shape} kernel={record['kernel_only_us']:.4f}us "
                    f"steady_device={time_us:.4f}us wall={wall_us:.4f}us "
                    f"native_err={native_metrics['err_ratio']:.6%} "
                    f"bf16_err={bf16_metrics['err_ratio']:.6%}",
                    flush=True,
                )
            except Exception as error:
                status = f"error:{error}"
                results.append(
                    {
                        "shape": shape,
                        "kernel_us": record["kernel_only_us"],
                        "e2e_us": -1,
                        "status": status,
                    }
                )
                record["status"] = status
            finally:
                persist(record)
                clear_mxfp8_b_scale_cache()
                torch.cuda.empty_cache()
        return results


if __name__ == "__main__":
    tuner = MxscalePreShuffleTuner(
        "MxscalePreShuffleTuner",
        key=["gfx", "cu_num", "M", "N", "K"],
        resultList=[
            "libtype",
            "kernelId",
            "splitK",
            "us",
            "kernelName",
            "tflops",
            "bw",
            "errRatio",
        ],
        description="Tune FlyDSL MXScale preshuffle GEMM",
    )
    tuner.run(tuner.parse_args(), False)
