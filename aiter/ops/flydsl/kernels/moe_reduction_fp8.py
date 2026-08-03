# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from aiter.ops.flydsl.kernels import buffer_ops, vector


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            block = if_op.then_block
            if (not block.operations) or not isinstance(
                block.operations[-1], scf.YieldOp
            ):
                scf.YieldOp([])


@functools.lru_cache(maxsize=1024)
def compile_moe_reduction_fp8(
    *,
    topk: int,
    model_dim: int,
    out_dtype_str: str,
):
    """Compile MXFP8 row reducer for rows encoded as ``[N fp8 bytes | N/8 e8m0]``.

    This private transport format reserves ``E=0`` as an all-zero 8-value group
    sentinel (non-standard vs OCP E8M0). Non-zero groups use the biased exponent
    payload emitted by the stage2 producer.

    ``model_dim`` only needs to be divisible by 8 for scale grouping; standalone
    row-byte alignment is intentionally not tightened beyond that on gfx950.
    """
    arch = get_hip_arch()
    if not str(arch).startswith("gfx950"):
        raise RuntimeError(f"MXFP8 partial reduction requires gfx950, got {arch}")
    if model_dim % 8 != 0:
        raise ValueError(f"model_dim must be divisible by 8, got {model_dim}")
    if out_dtype_str not in ("bf16", "f16"):
        raise ValueError(
            f"out_dtype_str must be 'bf16' or 'f16', got {out_dtype_str!r}"
        )
    ir.ShapedType.get_dynamic_size()

    BLOCK_SIZE = 256
    VEC_WIDTH = 8
    fp8_row_bytes_in = model_dim + model_dim // 8
    elem_bytes_c = 2

    def elem_type():
        ty = T.bf16 if out_dtype_str == "bf16" else T.f16
        return ty() if callable(ty) else ty

    module_name = f"moe_reduction_fp8_kernel_{out_dtype_str}_topk{topk}_md{model_dim}"

    @flyc.kernel(name=module_name)
    def moe_reduction_fp8_kernel(
        X: fx.Pointer,
        Y: fx.Pointer,
        i32_m_tokens: fx.Int32,
    ):
        m_tokens = fx.Index(i32_m_tokens)
        c_model_dim = fx.Index(model_dim)
        c_row_bytes_in = fx.Index(fp8_row_bytes_in)
        c_scale_base = fx.Index(model_dim)

        def _ptr_buffer_resource_off(ptr, num_records_bytes, byte_off_i64=None):
            addr = fx.ptrtoint(ptr)
            addr_i64 = arith.index_cast(T.i64, addr)
            if byte_off_i64 is not None:
                addr_i64 = addr_i64 + byte_off_i64
            return buffer_ops.create_buffer_resource_from_addr(
                addr_i64, num_records_bytes=num_records_bytes
            )

        token_idx = gpu.block_id("x")
        tile_idx = gpu.block_id("y")
        tid = gpu.thread_id("x")

        x_slab_nbytes = fx.Index(topk) * c_row_bytes_in
        y_slab_nbytes = c_model_dim * fx.Index(elem_bytes_c)
        x_base_off_i64 = fx.Int64(token_idx * x_slab_nbytes)
        y_base_off_i64 = fx.Int64(token_idx * y_slab_nbytes)

        y_rsrc = _ptr_buffer_resource_off(Y, fx.Int64(y_slab_nbytes), y_base_off_i64)

        tok_ok = token_idx < m_tokens
        _if_tok = scf.IfOp(tok_ok)
        with _if_then(_if_tok):
            c_tile_cols = fx.Index(BLOCK_SIZE * VEC_WIDTH)
            c_vecw = fx.Index(VEC_WIDTH)
            col_base = tile_idx * c_tile_cols + tid * c_vecw
            col_ok = col_base < c_model_dim
            _if_col = scf.IfOp(col_ok)
            with _if_then(_if_col):
                i32 = T.i32
                i8 = T.i8
                f32 = T.f32
                vec2_f32 = T.vec(2, f32)

                acc = [arith.constant(0.0, type=f32) for _ in range(VEC_WIDTH)]
                scale_col = col_base // c_vecw
                for k in range_constexpr(topk):
                    row_base = fx.Index(k) * c_row_bytes_in
                    # Fold each top-k row byte base into the descriptor so
                    # load/store offsets remain row-local within that row.
                    row_rsrc = _ptr_buffer_resource_off(
                        X,
                        fx.Int64(c_row_bytes_in),
                        x_base_off_i64 + fx.Int64(row_base),
                    )
                    val_i32_off = fx.Int32(col_base // fx.Index(4))
                    w_v = buffer_ops.buffer_load(
                        row_rsrc, val_i32_off, vec_width=2, dtype=i32
                    )
                    w01 = vector.extract(w_v, static_position=[0], dynamic_position=[])
                    w23 = vector.extract(w_v, static_position=[1], dynamic_position=[])

                    scale_off_i32 = fx.Int32(c_scale_base + scale_col)
                    scale_i8 = buffer_ops.buffer_load(
                        row_rsrc, scale_off_i32, vec_width=1, dtype=i8
                    )
                    scale_f32 = (arith.extui(T.i32, scale_i8) << fx.Int32(23)).bitcast(
                        T.f32
                    )

                    pairs = [
                        rocdl.cvt_pk_f32_fp8(vec2_f32, w01, False),
                        rocdl.cvt_pk_f32_fp8(vec2_f32, w01, True),
                        rocdl.cvt_pk_f32_fp8(vec2_f32, w23, False),
                        rocdl.cvt_pk_f32_fp8(vec2_f32, w23, True),
                    ]
                    for pi in range_constexpr(4):
                        v0 = vector.extract(
                            pairs[pi], static_position=[0], dynamic_position=[]
                        )
                        v1 = vector.extract(
                            pairs[pi], static_position=[1], dynamic_position=[]
                        )
                        acc[2 * pi] = acc[2 * pi] + v0 * scale_f32
                        acc[2 * pi + 1] = acc[2 * pi + 1] + v1 * scale_f32

                out_vec_ty = T.vec(VEC_WIDTH, elem_type())
                out_elems = [acc[i].truncf(elem_type()) for i in range(VEC_WIDTH)]
                out_vec = vector.from_elements(out_vec_ty, out_elems)
                buffer_ops.buffer_store(out_vec, y_rsrc, fx.Int32(col_base))

    tile_size = BLOCK_SIZE * VEC_WIDTH
    gy_static = (model_dim + tile_size - 1) // tile_size

    @flyc.jit
    def launch_moe_reduction_fp8(
        X: fx.Pointer,
        Y: fx.Pointer,
        i32_m_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        gx = fx.Index(i32_m_tokens)
        moe_reduction_fp8_kernel(X, Y, i32_m_tokens).launch(
            grid=(gx, gy_static, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_reduction_fp8
