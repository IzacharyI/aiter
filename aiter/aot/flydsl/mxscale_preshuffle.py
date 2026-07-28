# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Precompile tuned FlyDSL MXScale kernels into the runtime cache.

This is optional: without it, the first matching runtime call JIT-compiles the
specialization and subsequent calls reuse the cache.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, List

from aiter.aot.flydsl.common import compile_only_env
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.flydsl.mxscale_preshuffle_config import (
    parse_kernel_name,
)


def _default_csvs() -> List[str]:
    return [
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    ]


def _runtime_split_k(value) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 1
    parsed = int(value)
    return 1 if parsed <= 0 else parsed


def _compile_to_cache(
    M,
    N,
    K,
    tile_m,
    tile_n,
    tile_k,
    a_dtype,
    b_dtype,
    out_dtype,
    waves_per_eu,
    xcd_swizzle,
    split_k,
):
    import torch

    from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
        flydsl_mxscale_preshuffle_gemm,
    )

    a_bytes = K // 2 if a_dtype == "fp4" else K
    b_bytes = K // 2 if b_dtype == "fp4" else K
    output_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale_k = ((K // 32 + 7) // 8) * 8
    with compile_only_env():
        A = torch.zeros((M, a_bytes), dtype=torch.uint8, device=device)
        B = torch.zeros((N, b_bytes), dtype=torch.uint8, device=device)
        a_scale = torch.zeros(
            (((M + 31) // 32) * 32, scale_k),
            dtype=torch.uint8,
            device=device,
        )
        b_scale = torch.zeros(
            (N, scale_k), dtype=torch.uint8, device=device
        )
        output = torch.zeros((M, N), dtype=output_dtype, device=device)
        flydsl_mxscale_preshuffle_gemm(
            A,
            B,
            a_scale,
            b_scale,
            output,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            waves_per_eu=waves_per_eu,
            xcd_swizzle=xcd_swizzle,
            split_k=split_k,
        )


def parse_csv(csv_path: str) -> List[Dict]:
    with open(csv_path, newline="") as file:
        return list(csv.DictReader(file))


def compile_one_config(row: Dict) -> bool:
    parsed = parse_kernel_name(row.get("kernelName") or "")
    if parsed is None:
        return False
    _compile_to_cache(
        M=int(row["M"]),
        N=int(row["N"]),
        K=int(row["K"]),
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        waves_per_eu=parsed["waves_per_eu"],
        xcd_swizzle=parsed["xcd_swizzle"],
        split_k=_runtime_split_k(row.get("splitK")),
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        nargs="+",
        default=_default_csvs(),
        help="tuned CSV file(s) to precompile",
    )
    args = parser.parse_args()

    total = 0
    compiled = 0
    for csv_path in args.csv:
        if not os.path.exists(csv_path):
            print(f"[aot.mxscale] skip missing CSV: {csv_path}", flush=True)
            continue
        for row in parse_csv(csv_path):
            total += 1
            try:
                if compile_one_config(row):
                    compiled += 1
                    print(
                        f"[aot.mxscale] compiled {row.get('kernelName')} "
                        f"(M={row.get('M')} N={row.get('N')} K={row.get('K')})",
                        flush=True,
                    )
            except Exception as error:
                first_line = str(error).splitlines()[0][:120]
                print(
                    f"[aot.mxscale] FAILED {row.get('kernelName')}: "
                    f"{first_line}",
                    flush=True,
                )
    print(f"[aot.mxscale] compiled {compiled}/{total} configs", flush=True)


if __name__ == "__main__":
    main()
