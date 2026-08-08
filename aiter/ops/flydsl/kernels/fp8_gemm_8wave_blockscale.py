# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave FP8 GEMM with **per-block** scaling for AMD CDNA4 (gfx950).

Ping-pong schedule derived from HipKittens FP8_8wave (see ``fp8_gemm_8wave.py``),
but the row-wise epilog scale is replaced by DeepSeek-style a8w8 blockscale:

    x_scale : [scale_k, M]        (per-token, per-128-K-block; transposed layout)
    w_scale : [scale_n, scale_k]  (per-(128-N-block, 128-K-block); row-major)

Because the scale changes every 128 K elements (== BLOCK_K), we cannot fold it
into a single epilog multiply. Instead we use two-level accumulation ("promotion"):
per K-block, MFMA into a *zeroed* fragment (identity hardware scale) and then
``global += block_partial * (x_scale[m,kb] * w_scale[nb,kb])`` via ``math.fma``.

Fragment -> (m, n) mapping (matches ``StoreC``), for group c_XY
(X = row-half from a_cur0/a_cur1, Y = col-half from b_cur0/b_cur1),
accumulator (ti, tj), lane, vec element i:

    m = block_m*BLOCK_M + X*LDS_BLOCK_M + wave_m*(N_TILES_A*16) + ti*16 + (lane//16)*4 + i
    n = block_n*BLOCK_N + Y*LDS_BLOCK_N + wave_n*(N_TILES_B*16) + tj*16 + (lane%16)

Since the per-group N span is exactly LDS_BLOCK_N (== 128 == scale_block_n),
``nb = n // 128 = block_n*(BLOCK_N//128) + Y`` is constant across a group, so each
group needs a single w_scale scalar per K-block plus one x_scale vec4 per ti.

PERFORMANCE FINDING (MI355X gfx950, MLP_down M=16384 N=7168 K=768)
------------------------------------------------------------------
This port is *correct* (cos == 1.0) but does **not** beat the 4-wave
blockscale+xcd baseline (~174 us hot) on this shape. Apples-to-apples
(same timer/box), best config per tile:

    tile      hot us   MFMA/GUI util   note
    128x256    ~184     28%            fits (108 VGPR), but no 8-wave win
    256x256    ~900      5%            register-bound: MFMA units 95% idle

Root cause (rocprofv3 hardware counters): the 8-wave latency-hiding win only
materialises at the 256x256 tile (rowscale 256x256 = ~144 us, 1.2x over
baseline; rowscale 128x256 already loses at ~183 us). But two-level blockscale
needs the MFMA output ``blk`` live *simultaneously* with the promoted
accumulator ``c_frag`` during promote. At 256x256, ``c_frag`` alone (4 groups x
8 accums x vec4 f32 = 128 regs) consumes the entire 128-VGPR/wave budget that a
512-thread (2-waves/SIMD) workgroup allows, leaving no headroom to keep ``blk``
live and overlap the next MFMA. The compiler serialises MFMA->promote->MFMA
(+ a small 444 B spill), collapsing MFMA utilisation to ~5%. It is NOT
LDS/rematerialisation bound (LDS insts == rowscale, waitLDS is lower).

This is an architectural catch-22: the win needs the big tile, but the big tile
has no register room for blockscale's extra promote state. Register-relief levers
were tried and **empirically rejected** (rocprofv3-verified):

  * fp16/bf16 ``c_frag`` storage: compiler no-op -- it keeps the accumulator in
    f32 across the tight loop (narrowing optimised away), VGPR stayed 128, spill
    stayed 444 B, timing unchanged (~874 us).
  * AGPR-resident ``c_frag`` (manual ``v_accvgpr_read/write`` per K-block): you
    cannot keep a VALU-updated accumulator in AGPR -- the backend coalesces it
    back to VGPR (Accum_VGPR_Count stayed 0), so the forced accvgpr round-trips
    only add overhead: ~2185 us (2.4x *worse*), still VGPR=128, still spilling.
  * AGPR-resident ``blk``: the SSA MFMA already places ``blk`` in AGPR, so this
    is a no-op. (The MFMA hardware scale is E8M0 microscale, not f32, so the
    per-128-K f32 blockscale cannot be folded into the MFMA either.)

XCD grouped rasterisation (``use_xcd_remap``, on by default, ported from the
4-wave kernel) recovers only ~2% *cold* here and nothing hot (the win shape is
L2-resident once warm). N-subtiling to 256x128x2 would halve ``c_frag`` but also
removes the tile-size amortisation that makes 256x256 win in the first place.
The 4-wave 64x256 blockscale+xcd kernel remains the recommended production path
for this shape.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import math as math_dialect
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from .fp8_gemm_4wave import _xcd_swizzle
from .fp8_gemm_utils import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    ceildiv,
    compute_global_swizzle,
    divmod,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    wait_barrier,
)

SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128


class StoreCPlain:
    """Epilog store for blockscale (no scale multiply -- scaling done in loop)."""

    def __init__(self, C, c_rows, c_cols, c_idx_fn, n_tiles_a, n_tiles_b):
        self.c_rows = c_rows
        self.c_cols = c_cols
        self.lane_id = fx.thread_idx.x % 64
        self.c_idx_fn = c_idx_fn
        self.n_tiles_a = n_tiles_a
        self.n_tiles_b = n_tiles_b
        c_nbytes = c_rows * c_cols * 2  # BFloat16 = 2 bytes
        gC = fx.rocdl.make_buffer_tensor(C, max_size=False, num_records_bytes=c_nbytes)
        self.c_div = fx.logical_divide(gC, fx.make_layout(1, 1))
        self.out_atom_1 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        self.reg_bf16_1 = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.BFloat16)

    def _store_bf16(self, value_bf16, c_index):
        fx.memref_store_vec(Vec.filled(1, value_bf16, fx.BFloat16), self.reg_bf16_1)
        fx.copy(self.out_atom_1, self.reg_bf16_1, fx.slice(self.c_div, (None, fx.Int32(c_index))))

    def store(self, c_frag, base_row, base_col):
        for ti in range_constexpr(self.n_tiles_a):
            row = base_row + ti * 16 + (self.lane_id // 16) * 4
            for tj in range_constexpr(self.n_tiles_b):
                col = base_col + tj * 16 + self.lane_id % 16
                col_valid = col < self.c_cols
                oob = fx.Int32(self.c_rows * self.c_cols)
                vec_f32 = Vec(c_frag[self.c_idx_fn(ti, tj)])
                for i in range_constexpr(4):
                    scaled = vec_f32[i].to(fx.BFloat16)
                    c_index = (row + i) * self.c_cols + col
                    self._store_bf16(scaled, arith.select(col_valid, c_index, oob))


def compile_fp8_gemm_8w_blockscale(*, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, b_preshuffled: bool = False, waves_per_eu: int = 2, use_xcd_remap: bool = True, promote_sched: int = 8):
    BLOCK_K = 128

    assert BLOCK_M >= 64 and BLOCK_N >= 128 and BLOCK_M % 64 == 0 and BLOCK_N % 128 == 0
    assert K % BLOCK_K == 0
    assert BLOCK_K == SCALE_BLOCK_K, "this port assumes BLOCK_K == scale_block_k == 128 (kb == k)"

    K_ITERS = K // BLOCK_K
    assert K_ITERS >= 2

    scale_k = K // SCALE_BLOCK_K

    num_threads = 256 if BLOCK_N == 128 else 512
    num_waves = num_threads // 64
    waves_n = 4 if BLOCK_N == 256 else 2
    waves_m = num_waves // waves_n

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    assert LDS_BLOCK_N <= SCALE_BLOCK_N

    N_TILES_A = LDS_BLOCK_M // waves_m // 16
    N_TILES_B = BLOCK_N // (waves_n * 2 * 16)
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    N_LDS_STEPS_A = max(1, a_lds_size // (num_waves * 1024))
    N_LDS_STEPS_B = max(1, b_lds_size // (num_waves * 1024))
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    NB_PER_BLOCK = BLOCK_N // SCALE_BLOCK_N
    WAVE_M_OFF = N_TILES_A * 16

    @fx.struct
    class SharedStorage:
        A_lds_cur_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_cur_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_0: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        A_lds_next_1: fx.Array[fx.Float8E4M3FN, a_lds_size, 16]
        B_lds_cur_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_cur_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_0: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]
        B_lds_next_1: fx.Array[fx.Float8E4M3FN, b_lds_size, 16]

    @flyc.kernel(known_block_size=[num_threads, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        scale_a: fx.Tensor,
        scale_b: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        c_M = fx.Index(c_m)
        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_cur0 = lds.A_lds_cur_0
        a_cur1 = lds.A_lds_cur_1
        a_next0 = lds.A_lds_next_0
        a_next1 = lds.A_lds_next_1
        b_cur0 = lds.B_lds_cur_0
        b_cur1 = lds.B_lds_cur_1
        b_next0 = lds.B_lds_next_0
        b_next1 = lds.B_lds_next_1

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n
        if const_expr(use_xcd_remap):
            block_m, block_n = _xcd_swizzle(ceildiv(c_m, BLOCK_M), n_blocks)
        else:
            block_m, block_n = divmod(fx.block_idx.x, n_blocks)

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)
        zero_c = [mfma.zero_value] * N_ACCUMS

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        store_c = StoreCPlain(C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B)

        # ── Blockscale buffer resources + promotion helpers ───────────────────
        # Two-level accumulation: MFMA a K-block into a zeroed fragment (identity
        # hardware scale), then promote  global += block_partial * (x_scale * w_scale).
        # Scales are *preloaded* at the top of each K-iteration so the global-load
        # latency overlaps with the four MFMA groups (loading per-group right before
        # the FMA stalls the pipeline -> ~2x slower).
        sa_nbytes = scale_k * c_M * 4  # [scale_k, M] f32
        scale_a_rsrc = buffer_ops.create_buffer_resource(scale_a, max_size=False, num_records_bytes=sa_nbytes)
        scale_b_rsrc = buffer_ops.create_buffer_resource(scale_b, max_size=True)

        lane_row_off = (lane_id // 16) * 4
        nb0 = block_n * NB_PER_BLOCK  # w_scale N-block for col-half Y=0
        nb1 = nb0 + (1 if BLOCK_N == 256 else 0)
        xrow0 = block_m * BLOCK_M + wave_m * WAVE_M_OFF + lane_row_off  # X=0 row base
        xrow1 = xrow0 + LDS_BLOCK_M                                     # X=1 row base

        def preload_scales(kb):
            """Load x_scale (per row-half, per ti vec4) + w_scale (per col-half) for K-block kb."""
            w0 = fx.Float32(
                buffer_ops.buffer_load(
                    scale_b_rsrc, nb0 * scale_k + kb, vec_width=1, dtype=T.f32
                )
            )
            if const_expr(BLOCK_N == 128):
                w1 = w0
            else:
                w1 = fx.Float32(
                    buffer_ops.buffer_load(
                        scale_b_rsrc,
                        nb1 * scale_k + kb,
                        vec_width=1,
                        dtype=T.f32,
                    )
                )
            base = kb * c_M
            xs0 = [
                Vec(buffer_ops.buffer_load(scale_a_rsrc, base + xrow0 + ti * 16, vec_width=4, dtype=T.f32)).bitcast(fx.Float32)
                for ti in range_constexpr(N_TILES_A)
            ]
            xs1 = [
                Vec(buffer_ops.buffer_load(scale_a_rsrc, base + xrow1 + ti * 16, vec_width=4, dtype=T.f32)).bitcast(fx.Float32)
                for ti in range_constexpr(N_TILES_A)
            ]
            return xs0, xs1, w0, w1

        def promote(blk, c_frag, xs, w):
            """global += blk * (x_scale * w_scale) for one group."""
            out = list(c_frag)
            for ti in range_constexpr(N_TILES_A):
                comb = xs[ti] * w
                for tj in range_constexpr(N_TILES_B):
                    idx = ti * N_TILES_B + tj
                    out[idx] = math_dialect.fma(blk[idx], comb, c_frag[idx])
            return out

        acc_init = fx.full(4, 0.0, fx.Float32)
        c00_frag = [acc_init] * N_ACCUMS
        c01_frag = [acc_init] * N_ACCUMS
        c10_frag = [acc_init] * N_ACCUMS
        c11_frag = [acc_init] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        if wave_m == 1:
            rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            xs0, xs1, w0, w1 = preload_scales(k)
            b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            rocdl.s_barrier()

            # Cluster-promote schedule (rocprofv3-tuned): issue all four MFMA groups
            # back-to-back and cluster every promote() AFTER the last MFMA, so no
            # promote sits before the barrier-synced groups B2/B3. A promote wedged
            # there idles the MFMA unit (measured 45.8% util vs rowscale's 66.9%);
            # clustering lifts it to 47.9% (-3% cycles). Three of the four promotes
            # then overlap MFMA(c11)'s compute shadow (retired blocks); only c11's is
            # exposed. Keeps all 4 blks live (+8 VGPR) but occupancy is LDS-bound so
            # there is no spill/occupancy loss. Only the next iteration's B1 sees any
            # promote. (Superseded the earlier per-group "delayed promotion".)
            c00_blk = mfma.call(a0_frag, b0_frag, zero_c)

            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            rocdl.s_barrier()

            c01_blk = mfma.call(a0_frag, b1_frag, zero_c)

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            rocdl.s_barrier()

            c10_blk = mfma.call(a1_frag, b0_frag, zero_c)

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            c11_blk = mfma.call(a1_frag, b1_frag, zero_c)
            if const_expr(BLOCK_N == 128 and promote_sched > 0):
                rocdl.sched_mfma(promote_sched)
            c00_frag = promote(c00_blk, c00_frag, xs0, w0)
            c01_frag = promote(c01_blk, c01_frag, xs0, w1)
            c10_frag = promote(c10_blk, c10_frag, xs1, w0)
            c11_frag = promote(c11_blk, c11_frag, xs1, w1)
            if const_expr(BLOCK_N == 128 and promote_sched > 0):
                rocdl.sched_barrier(0)

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        xs0, xs1, w0, w1 = preload_scales(k)
        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
        a0_frag = a_s2r.load(a_cur0)
        rocdl.s_barrier()

        c00_blk = mfma.call(a0_frag, b0_frag, zero_c)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_blk = mfma.call(a0_frag, b1_frag, zero_c)

        a1_frag = a_s2r.load(a_cur1)
        a_g2s.load(a_next1, A1_gl_offset + (K_ITERS - 1) * BLOCK_K)
        rocdl.s_barrier()

        c10_blk = mfma.call(a1_frag, b0_frag, zero_c)

        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c11_blk = mfma.call(a1_frag, b1_frag, zero_c)
        c00_frag = promote(c00_blk, c00_frag, xs0, w0)
        c01_frag = promote(c01_blk, c01_frag, xs0, w1)
        c10_frag = promote(c10_blk, c10_frag, xs1, w0)
        c11_frag = promote(c11_blk, c11_frag, xs1, w1)

        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        xs0, xs1, w0, w1 = preload_scales(k)
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        c00_blk = mfma.call(a0_frag, b0_frag, zero_c)

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        rocdl.s_barrier()

        c01_blk = mfma.call(a0_frag, b1_frag, zero_c)

        a1_frag = a_s2r.load(a_cur1)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_blk = mfma.call(a1_frag, b0_frag, zero_c)
        c11_blk = mfma.call(a1_frag, b1_frag, zero_c)
        c00_frag = promote(c00_blk, c00_frag, xs0, w0)
        c01_frag = promote(c01_blk, c01_frag, xs0, w1)
        c10_frag = promote(c10_blk, c10_frag, xs1, w0)
        c11_frag = promote(c11_blk, c11_frag, xs1, w1)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Scale and store back to gmem.
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        store_c.store(c00_frag, base_row + 0, base_col + 0)
        store_c.store(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        store_c.store(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        store_c.store(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        scale_a: fx.Tensor,
        scale_b: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            scale_a,
            scale_b,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu,
                "rocdl.flat_work_group_size": (
                    "256,256" if num_threads == 256 else "512,512"
                ),
            },
        ).launch(
            grid=(grid_x, 1, 1), block=(num_threads, 1, 1), stream=stream
        )

    return launch_gemm
