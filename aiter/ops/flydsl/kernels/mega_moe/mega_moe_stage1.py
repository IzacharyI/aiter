# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Fused stage1 with low-ID dispatch producers and oversubscribed FP8xFP4 grouped-GEMM1 consumers."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .dispatch import (
    DispatchSlot,
    emit_direct_fixed_slot_finalize,
    emit_direct_fixed_slot_payload,
    emit_dispatch_group,
    emit_dispatch_payload,
    emit_dispatch_plan,
)
from .gemm1 import _LdsF32View, build_fused_gemm1
from .gemm_util import _buffer_load, _buffer_store, _make_buffer, _make_buffer_from_addr

_SC0_CACHE = 1
_BUFFER_OFFSET_ABI_BYTES = 1 << 32
# gfx95x buffer cache bits: bit0=sc0, bit1=nt, bit4=sc1. ``sc0 sc1`` is a
# system-scope (write-through) store; see ``SiluQuantEpilogue.out_cache_modifier``.
_G1_OUT_WT = 1 | 16


def ceildiv(a, b):
    return (a + b - 1) // b


class _Slab:
    """Adapt an LDS array to the ``.buf`` shape ``emit_stage2_body`` expects.

    FlyDSL allows one ``SharedAllocator`` per kernel, so the megakernel cannot let the
    Stage2 emitter allocate its own slab; it hands over the GEMM1 pool instead.
    """

    __slots__ = ("buf",)

    def __init__(self, buf):
        self.buf = buf


def _use_direct_fixed_slot(
    enabled, npes, experts_per_rank, max_tokens_per_rank, cap, tile_m
):
    if not enabled or tile_m <= 0 or max_tokens_per_rank <= 0:
        return False
    required_cap = ((npes * max_tokens_per_rank + tile_m - 1) // tile_m) * tile_m
    return npes == 8 and experts_per_rank == 48 and cap == required_cap


def _validate_dispatch_capacity(
    batch_size,
    npes,
    experts_per_rank,
    topk,
    tile_m,
    row_bytes,
    output_row_bytes,
    use_tile_resource,
):
    max_rows = npes * batch_size * topk + experts_per_rank * tile_m
    if not use_tile_resource and max_rows * row_bytes >= _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError(
            "MegaMoE v2 stage1 payload exceeds the 32-bit buffer-resource ABI"
        )
    if (
        not use_tile_resource
        and max_rows * output_row_bytes >= _BUFFER_OFFSET_ABI_BYTES
    ):
        raise ValueError(
            "MegaMoE v2 stage1 output exceeds the 32-bit buffer-resource ABI"
        )


# fmt: off
@functools.cache
def compile_mega_moe_stage1(
    *, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int, fuse_topk: int,
    fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, fixed_slot_dispatch: bool, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32, b_nt: int = -1,
    work_shards: int | None = None, external_grouping: bool | None = None,
    external_counting: bool | None = None, payload_chunk_rows: int = 0, payload_tile_ready: bool = False,
    swiglu_limit: float = 0.0,
    fused_stage2=None,
    fused_g2_pref: int = 0,
    fused_g2_chunk: int = 4,
    fused_s2_nw8: bool = False,
    fused_g2_skew: tuple = (5, 4),
    fused_diag_nopub: bool = False,
):
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(f"MegaMoE v2 stage1 requires CDNA4 (gfx95x), got {arch or 'unknown'}")
    NUM_WAVES = int(num_waves)
    assert NUM_WAVES > 1, "planner needs one communication wave and at least one grouping wave"
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % NUM_WAVES == 0
    n_per_wave = tile_n // NUM_WAVES
    assert (2 * inter_dim) % tile_n == 0, "2*inter_dim must tile evenly by tile_n"
    N_TILES = (2 * inter_dim) // tile_n
    GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    assert grid_mult in GRID_MULT_VALUES, "grid_mult out of range"
    grid_epoch_slot = GRID_MULT_VALUES.index(grid_mult)
    dispatch_blocks = int(num_dispatch_cu)
    payload_chunk_rows = int(payload_chunk_rows)
    assert 0 < dispatch_blocks < num_cu, "num_dispatch_cu must be in [1, num_cu)"
    assert dispatch_blocks % fuse_npes == 0, "num_dispatch_cu must be divisible by fuse_npes"
    if payload_chunk_rows:
        assert not fixed_slot_dispatch and payload_chunk_rows % sort_block_m == 0
    assert not payload_tile_ready or payload_chunk_rows > 0
    planner_blocks = 1
    # Keep the fused grid on an exact CU multiple instead of appending control/producer CTAs as a tail.
    grid_x = num_cu * grid_mult - planner_blocks - dispatch_blocks
    assert grid_x > 0, "consumer grid must remain positive"
    launch_grid_x = planner_blocks + dispatch_blocks + grid_x
    assert launch_grid_x <= num_cu * 33 + 1
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0

    TILE_K_BYTES = tile_k // 2
    assert TILE_K_BYTES % 128 == 0
    A_K_STEP_BYTES = tile_k
    assert A_K_STEP_BYTES == 256, "MegaMoE v2 GEMM1 requires tile_k=256"
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64
    WORK_SHARDS = 4 if work_shards is None and int(fuse_mtpr) >= 8192 else 8
    if work_shards is not None:
        WORK_SHARDS = int(work_shards)
    assert WORK_SHARDS in (1, 2, 4, 8)

    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    cs_size = sort_block_m * cs_tile_n
    lds_pool_bytes = max(2 * a_lds_size, cs_size * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)

    # ---- fused GEMM2 role (megakernel) ----------------------------------------
    # ``fused_stage2`` turns this kernel into the single persistent kernel: the same
    # 512-thread blocks that drain the GEMM1 work queue also drain a GEMM2 queue,
    # taking a GEMM2 tile as soon as its producing m-tile has published. GEMM2 keeps
    # its native 4-wave geometry and runs as ``NUM_WAVES // 4`` lockstep half-blocks
    # inside one block (see handoff section 14, "block-shape decision").
    S2 = None
    if fused_stage2 is not None:
        from .mega_moe_stage2 import (
            derive_stage2_emit_constants,
            make_stage2_body_emitter,
        )

        s2_kw = dict(fused_stage2)
        s2_kw.setdefault("SBM", sort_block_m)
        # Native 8-wave GEMM2 tile: instead of two lockstep 4-wave half-blocks, widen
        # BN by the wave ratio and let all NUM_WAVES waves cooperate on one tile. Each
        # wave still owns 64 N-columns, so the MFMA fragment shape and accumulator
        # VGPR count are unchanged; what goes away is the block-wide barrier coupling
        # between the halves (measured at +0.53 ms for halves=2 in Stage2 standalone).
        _s2_nw = NUM_WAVES if fused_s2_nw8 else 4
        if fused_s2_nw8:
            s2_kw["NW"] = _s2_nw
            s2_kw["BN"] = int(s2_kw.get("BN", 256)) * (NUM_WAVES // 4)
        # One unit per call: the persistent tile loop is replaced by this kernel's
        # own work queue, so the body is emitted in its single-tile form.
        s2_kw["persist"] = False
        s2_kw["g2_spart"] = 0
        s2_consts, _s2_name = derive_stage2_emit_constants(**s2_kw)
        G2_CHUNK = max(1, int(fused_g2_chunk))
        S2_HALVES = 1 if fused_s2_nw8 else NUM_WAVES // 4
        assert NUM_WAVES % 4 == 0, "fused GEMM2 needs a wave count that is a multiple of 4"
        S2_BM, S2_BN = s2_consts["BM"], s2_consts["BN"]
        S2_NUM_N = s2_consts["N_OUT"] // S2_BN
        assert S2_NUM_N % S2_HALVES == 0, (
            f"fused GEMM2 halves={S2_HALVES} must divide num_n_blocks={S2_NUM_N}"
        )
        assert sort_block_m % S2_BM == 0
        S2_SLAB = s2_consts["lds_ready_off"] + (
            s2_consts["npes"] * 8 + 16 if s2_consts["publish_tok_ready"] else 0
        )
        # The GEMM2 slab aliases the GEMM1 pool: a block runs one unit at a time and
        # the work-loop barrier separates a unit's LDS reads from the next unit's
        # writes, so the two roles never hold LDS at the same moment.
        lds_pool_bytes = max(lds_pool_bytes, S2_SLAB * S2_HALVES)
        emit_stage2_body = make_stage2_body_emitter(**s2_consts)
        S2 = {
            "halves": S2_HALVES, "BM": S2_BM, "BN": S2_BN, "num_n": S2_NUM_N,
            "nw8": bool(fused_s2_nw8),
            "slab": S2_SLAB, "emit": emit_stage2_body,
            "inter": s2_consts["INTER_MAX"], "hidden": s2_consts["N_OUT"],
        }

    fz_npes, fz_epr, fz_k = int(fuse_npes), int(experts_per_rank), int(fuse_topk)
    fz_cap, fz_mtpr, fz_rank = int(fuse_cap), int(fuse_mtpr), int(rank)
    if fz_npes * fz_mtpr > 1 << 24:
        raise ValueError("MegaMoE v2 source-token encoding exceeds 24 bits")
    if fz_k > 1 << 8:
        raise ValueError("MegaMoE v2 top-k slot encoding exceeds 8 bits")
    if external_grouping is None:
        external_grouping = fz_mtpr >= 2048 and fz_npes == 8 and fz_epr == 48
    if external_counting is None:
        external_counting = external_grouping and fz_mtpr >= 8192
    assert not external_counting or external_grouping
    fz_tile_m = int(sort_block_m)
    assert fz_cap % fz_tile_m == 0, f"fuse_cap({fz_cap}) % tile_m({fz_tile_m}) != 0"
    direct_fixed_slot = _use_direct_fixed_slot(
        fixed_slot_dispatch, fz_npes, fz_epr, fz_mtpr, fz_cap, fz_tile_m
    )
    fz_total_experts = fz_npes * fz_epr
    # Small batches stream B; large batches cache it across M tiles.
    b_cache_modifier = int(b_nt) if int(b_nt) >= 0 else (3 if fz_mtpr <= 512 else 0)
    fz_n_i32, fz_nbytes = model_dim // 4, model_dim
    fz_scale_bytes = int(fuse_scale_dim)
    fz_scale_n_i32 = (fz_scale_bytes + 3) // 4 if fz_scale_bytes > 0 else 0
    if direct_fixed_slot and fz_scale_n_i32 > 64:
        raise ValueError("direct fixed-slot dispatch supports at most 64 packed scale columns")
    fz_enable_scales = fz_scale_bytes > 0
    fz_safe_end_i32 = (fz_n_i32 // 512) * 512
    _validate_dispatch_capacity(
        fz_mtpr, fz_npes, fz_epr, fz_k, fz_tile_m, fz_nbytes, inter_dim, use_tile_resource
    )

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    dispatch_path = "fixedslot" if fixed_slot_dispatch else "compact"
    swiglu_suffix = "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    kernel_name = (
        f"megamoe_stage1_{dispatch_path}_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_w{NUM_WAVES}_gm{grid_mult}"
        f"_dcu{dispatch_blocks}_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}"
        f"aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}_bnt{b_cache_modifier}_ws{WORK_SHARDS}"
        f"_pc{payload_chunk_rows}"
        f"_ptr{int(payload_tile_ready)}"
        f"{swiglu_suffix}"
        + ("" if S2 is None else f"_g2h{S2['halves']}m{S2['BM']}n{S2['BN']}p{int(fused_g2_pref)}c{G2_CHUNK}"
           f"s{fused_g2_skew[0]}_{fused_g2_skew[1]}{'_nopub' if fused_diag_nopub else ''}")
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        s2_aq: fx.Int64, s2_ascale: fx.Int64, s2_bq: fx.Int64, s2_bscale: fx.Int64,
        s2_eids: fx.Int64, s2_cumsum: fx.Int64, s2_metiles: fx.Int64, s2_stids: fx.Int64,
        s2_sweights: fx.Int64, s2_trb: fx.Int64, s2_p2p: fx.Int64, s2_ctr: fx.Int64,
        i32_s2_maxmb: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        disp_rsrc = _make_buffer_from_addr(addr_disp, fx.Int64)
        parity_rsrc = _make_buffer_from_addr(addr_parity, fx.Int32)
        expected_rsrc = _make_buffer_from_addr(addr_expected, fx.Int32)

        def _disp_ptr(slot):
            return _buffer_load(disp_rsrc, fx.Int32(int(slot)), fx.Int64)

        a_entry_count = _disp_ptr(DispatchSlot.ENTRY_COUNT)
        a_epoch_gate = _disp_ptr(DispatchSlot.EPOCH_GATE)
        a_pair_order_ready = _disp_ptr(DispatchSlot.PAIR_ORDER_READY)
        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        a_work_tail = _disp_ptr(DispatchSlot.WORK_TAIL)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        a_payload_blocks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
        a_payload_chunks_per_destination = _disp_ptr(DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION)
        a_launch_ready = _disp_ptr(DispatchSlot.LAUNCH_READY)
        p_launch_ready = _disp_ptr(DispatchSlot.P2P_LAUNCH_READY)
        a_payload_ready_rows = _disp_ptr(DispatchSlot.PAYLOAD_READY_ROWS)

        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if tid == fx.Int32(0):
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(a_entry_count + fx.Int64(grid_epoch_slot * 8), fx.Int64(1))
            )
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        generation = ticket64 // fx.Int64(launch_grid_x)
        ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
        gate_addr = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        compact_owner = ticket == fx.Int32(0)
        compact_producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(dispatch_blocks))
        producer_slot = ticket - fx.Int32(1)

        if compact_owner:
            next_parity_lane = fx.Int32(0)
            launch_epoch_lane = fx.Int32(0)
            if tid == fx.Int32(0):
                old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
                next_parity_lane = old_parity ^ fx.Int32(1)
                previous_expected = _buffer_load(expected_rsrc, next_parity_lane, fx.Int32)
                next_expected = previous_expected + fx.Int32(fz_npes)
                _buffer_store(expected_rsrc, next_parity_lane, next_expected, fx.Int32)
                launch_epoch_lane = (
                    (next_expected // fx.Int32(fz_npes)) * fx.Int32(2) - next_parity_lane
                )
            next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
            launch_epoch = fx.Int32(fx.rocdl.readfirstlane(T.i32, launch_epoch_lane))
            if const_expr(payload_tile_ready):
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(a_payload_ready_rows, fx.Int32(0), fx.Int32(fz_tile_m))
                    comm_ops.fence_system_release()
                fx.barrier()
            if tid < fx.Int32(fz_npes):
                peer = (tid + fx.Int32(fz_rank)) % fx.Int32(fz_npes)
                comm_ops.fence_system_release()
                launch_ready_table = _make_buffer_from_addr(p_launch_ready, fx.Int64)
                remote_launch_ready = _buffer_load(launch_ready_table, peer, fx.Int64)
                comm_ops.store_i32_system(remote_launch_ready, fx.Int32(fz_rank), launch_epoch)
                mori_shmem.int32_wait_until_greater_than(
                    a_launch_ready + fx.Int64(peer) * fx.Int64(4), launch_epoch - fx.Int32(1)
                )
                comm_ops.fence_system_acquire()
            if tid == fx.Int32(0):
                work_head_rsrc = _make_buffer_from_addr(a_work_head, fx.Int32)
                for shard in range_constexpr(8):
                    _buffer_store(work_head_rsrc, fx.Int32(shard * 16), fx.Int32(0), fx.Int32)
                _buffer_store(_make_buffer_from_addr(a_work_tail, fx.Int32), fx.Int32(0), fx.Int32(0), fx.Int32)
                if const_expr(external_grouping or direct_fixed_slot):
                    group_done_rsrc = _make_buffer_from_addr(a_group_done, fx.Int32)
                    for destination in range_constexpr(fz_npes if direct_fixed_slot else 1):
                        _buffer_store(group_done_rsrc, fx.Int32(destination), fx.Int32(0), fx.Int32)
            fx.barrier()
            if tid == fx.Int32(0):
                fx.rocdl.s_waitcnt(0)
                comm_ops.fence_agent_release()
                _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
                fx.rocdl.s_waitcnt(0)
                comm_ops.fence_agent_release()
                comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
            fx.rocdl.s_waitcnt(0)
            fx.barrier()
        else:
            if tid == fx.Int32(0):
                mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)
                comm_ops.fence_agent_acquire()
            fx.barrier()

        payload_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE)
        payload_expected = _buffer_load(expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE)

        if const_expr(S2 is not None):
            # Clear the per-m-tile GEMM1 completion counters before PLAN_READY is
            # published. Every producer and consumer of these counters waits on
            # PLAN_READY first, so this one block's clear is ordered ahead of all of
            # them without a grid barrier. Worst-case tile count is used because the
            # real one is not known until the plan the owner is about to emit.
            if compact_owner:
                _ctr_rsrc = _make_buffer_from_addr(s2_ctr, fx.Int32)
                _max_tiles = (i32_s2_maxmb * fx.Int32(S2["BM"]) + fx.Int32(sort_block_m - 1)) // fx.Int32(sort_block_m)
                for _c in range(tid, _max_tiles, fx.Int32(TOTAL_THREADS)):
                    _buffer_store(_ctr_rsrc, fx.Int32(_c), fx.Int32(0), fx.Int32)
                if tid < fx.Int32(WORK_SHARDS):
                    # The sharded GEMM2 work heads live past the worst-case tile
                    # range, one per 64-byte line so they do not contend.
                    _buffer_store(
                        _ctr_rsrc, i32_s2_maxmb + tid * fx.Int32(16), fx.Int32(0), fx.Int32
                    )
                fx.rocdl.s_waitcnt(0)
                comm_ops.fence_system_release()
                fx.barrier()

        if compact_owner:  # noqa: SIM102 - keep the device and compile-time branches separate.
            if const_expr(not direct_fixed_slot):
                emit_dispatch_plan(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr,
                    fz_rank=fz_rank, fz_tile_m=fz_tile_m, fz_total_experts=fz_total_experts, addr_disp=addr_disp,
                    i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, parity=payload_parity,
                    expected=payload_expected, external_grouping=external_grouping,
                    external_counting=external_counting,
                    dispatch_blocks=dispatch_blocks, payload_chunk_rows=payload_chunk_rows,
                    payload_tile_ready=payload_tile_ready,
                )

        if compact_producer:
            if const_expr(direct_fixed_slot):
                emit_direct_fixed_slot_payload(
                    num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_cap=fz_cap,
                    fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes,
                    fz_n_i32=fz_n_i32,
                    fz_scale_n_i32=fz_scale_n_i32, fz_enable_scales=fz_enable_scales, addr_disp=addr_disp,
                    addr_in_tok=addr_in_tok, addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                    i32_cur_tok=i32_cur_tok, dispatch_blocks=dispatch_blocks, producer_slot=producer_slot,
                    parity=payload_parity, expected=payload_expected,
                )
            else:
                if const_expr(external_grouping):
                    emit_dispatch_group(
                        num_waves=NUM_WAVES, fz_k=fz_k, fz_total_experts=fz_total_experts, addr_disp=addr_disp,
                        i32_cur_tok=i32_cur_tok, addr_in_idx=addr_in_idx, dispatch_blocks=dispatch_blocks,
                        producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                        external_counting=external_counting, adaptive_grouping=payload_tile_ready,
                    )
                else:
                    if tid == fx.Int32(0):
                        mori_shmem.int32_wait_until_equals(
                            a_pair_order_ready + fx.Int64(payload_parity) * fx.Int64(4), payload_expected)
                        comm_ops.fence_agent_acquire()
                    fx.barrier()
                producers_per_destination = fx.Int32(dispatch_blocks // fz_npes)
                chunks_per_destination = fx.Int32(1)
                if const_expr(payload_chunk_rows > 0):
                    chunks_per_destination = fx.Int32(
                        (fz_mtpr + payload_chunk_rows - 1) // payload_chunk_rows
                    )
                payload_active = fx.Int32(0) == fx.Int32(0)
                if const_expr(payload_tile_ready and dispatch_blocks > 32):
                    producer_destination = producer_slot % fx.Int32(fz_npes)
                    producer_round = producer_slot // fx.Int32(fz_npes)
                    producers_per_destination = _buffer_load(
                        _make_buffer_from_addr(a_payload_blocks_per_destination, fx.Int32),
                        producer_destination, fx.Int32,
                    )
                    chunks_per_destination = _buffer_load(
                        _make_buffer_from_addr(a_payload_chunks_per_destination, fx.Int32),
                        producer_destination, fx.Int32,
                    )
                    payload_active = producer_round < producers_per_destination
                if payload_active:
                    emit_dispatch_payload(
                        num_waves=NUM_WAVES, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                        fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes, fz_n_i32=fz_n_i32,
                        fz_safe_end_i32=fz_safe_end_i32, fz_scale_n_i32=fz_scale_n_i32,
                        fz_enable_scales=fz_enable_scales, addr_disp=addr_disp, addr_in_tok=addr_in_tok,
                        addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc, dispatch_blocks=dispatch_blocks,
                        producer_slot=producer_slot, parity=payload_parity, expected=payload_expected,
                        producers_per_destination=producers_per_destination, payload_chunk_rows=payload_chunk_rows,
                        chunks_per_destination=chunks_per_destination, payload_tile_ready=payload_tile_ready,
                    )
        if const_expr(direct_fixed_slot):
            if compact_owner:
                emit_direct_fixed_slot_finalize(
                    fz_npes=fz_npes, fz_epr=fz_epr, fz_cap=fz_cap, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                    fz_tile_m=fz_tile_m, n_tiles=N_TILES, addr_disp=addr_disp, parity=payload_parity,
                    expected=payload_expected,
                )
        else:
            payload_table = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.P2P_PAYLOAD_READY)), fx.Int64)
            addr_payload_ready = _buffer_load(
                _make_buffer_from_addr(payload_table, fx.Int64), fx.Int32(fz_rank), fx.Int64
            )
            addr_tile_ready = _disp_ptr(DispatchSlot.TILE_READY)
            addr_tile_expected = _disp_ptr(DispatchSlot.TILE_EXPECTED)
        wave_id = fx.thread_idx.x // 64

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        nv_rsrc = _make_buffer(num_valid_ids, fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_nbytes = tokens * fx.Int32(inter_dim)
            out_rsrc = _make_buffer(out, fx.Int16, max_size=False, num_records_bytes=out_nbytes)
        os_rsrc = _make_buffer(out_scale, fx.Int8, max_size=False, num_records_bytes=os_nbytes)

        expert_of_flat, _do_scheduled_tile = build_fused_gemm1(
            x_tensor=x, w_rsrc=w_rsrc,
            sw_rsrc=sw_rsrc, sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc,
            trb_rsrc=trb_rsrc, expert_rsrc=expert_rsrc, out_tensor=out,
            a_buf=a_buf, a_scale_lds=a_scale_lds, c_tile=c_tile,
            model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
            tile_n=tile_n, num_waves=NUM_WAVES, n_per_wave=n_per_wave, wave_id=wave_id,
            m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_k_step_bytes=A_K_STEP_BYTES,
            total_threads=TOTAL_THREADS, k_iters=K_ITERS, a_lds_i32=a_lds_i32,
            n_tiles=N_TILES, expert_offset=fz_rank * fz_epr, b_cache_modifier=b_cache_modifier,
            swizzle_a=swizzle_a, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
            async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
            swiglu_limit=swiglu_limit,
            out_cache_modifier=(0 if S2 is None else _G1_OUT_WT),
        )

        if tid == fx.Int32(0):
            local_plan_ready = _buffer_load(disp_rsrc, fx.Int32(int(DispatchSlot.PLAN_READY)), fx.Int64)
            ready_index = payload_parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            mori_shmem.int32_wait_until_equals(
                local_plan_ready + fx.Int64(ready_index) * fx.Int64(4), payload_expected)
            comm_ops.fence_agent_acquire()
        fx.barrier()

        num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        total_work = num_m_tiles * fx.Int32(N_TILES)

        def _wait_tile_payload(flat):
            if const_expr(payload_tile_ready):
                tile_index = flat // fx.Int32(N_TILES)
                expected_tiles = _buffer_load(
                    _make_buffer_from_addr(addr_tile_expected, fx.Int32), tile_index, fx.Int32
                )
                mori_shmem.int32_wait_until_equals(
                    addr_tile_ready + fx.Int64(tile_index) * fx.Int64(4), expected_tiles
                )
            else:
                pe = expert_of_flat(flat)
                pe_index = payload_parity * fx.Int32(fz_epr) + pe
                mori_shmem.int32_wait_until_equals(
                    addr_payload_ready + fx.Int64(pe_index) * fx.Int64(4), payload_expected
                )

        def _peek_tile_payload(flat):
            """Non-blocking twin of ``_wait_tile_payload``.

            Returns a device predicate. The megakernel uses it to turn a dispatch
            payload wait into GEMM2 work instead of a stall: the whole point of the
            fusion is that a block with nothing to compute for GEMM1 *yet* still has
            GEMM2 tiles it could be computing. ``atomic_add(addr, 0)`` is the
            non-destructive relaxed read; like ``_wait_tile_payload`` it does not
            invalidate L2, so the consumer still needs its acquire fence.
            """
            if const_expr(direct_fixed_slot):
                return fx.Int32(0) == fx.Int32(0)
            if const_expr(payload_tile_ready):
                tile_index = flat // fx.Int32(N_TILES)
                expected_tiles = _buffer_load(
                    _make_buffer_from_addr(addr_tile_expected, fx.Int32), tile_index, fx.Int32
                )
                cur = fx.Int32(
                    comm_ops.atomic_add_agent(
                        addr_tile_ready + fx.Int64(tile_index) * fx.Int64(4), fx.Int32(0)
                    )
                )
                return cur == expected_tiles
            pe = expert_of_flat(flat)
            pe_index = payload_parity * fx.Int32(fz_epr) + pe
            cur = fx.Int32(
                comm_ops.atomic_add_agent(
                    addr_payload_ready + fx.Int64(pe_index) * fx.Int64(4), fx.Int32(0)
                )
            )
            return cur == payload_expected

        # Control CTAs join the work pool after dispatch.
        consumer_active = fx.Int32(1) == fx.Int32(1)
        work_scratch = fx.recast_iter(fx.Int32, a_buf.ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
        work_shard = ticket & fx.Int32(WORK_SHARDS - 1)
        if const_expr(S2 is None):
            while consumer_active:
                if tid == fx.Int32(0):
                    local_work = fx.Int32(
                        comm_ops.atomic_add_agent(
                            a_work_head + fx.Int64(work_shard) * fx.Int64(64), fx.Int32(1)
                        )
                    )
                    work = work_shard + local_work * fx.Int32(WORK_SHARDS)
                    fx.ptr_store(Vec.from_elements([work], fx.Int32), work_scratch)
                fx.barrier()
                work = Vec(work_scratch_view.load())[0]
                if tid == fx.Int32(0):
                    has_work = (work < total_work).select(fx.Int32(1), fx.Int32(0))
                    if has_work != fx.Int32(0):  # noqa: SIM102 - device vs compile-time branches
                        if const_expr(not direct_fixed_slot):
                            _wait_tile_payload(work)
                    fx.ptr_store(Vec.from_elements([has_work], fx.Int32), work_scratch)
                fx.barrier()
                has_work = Vec(work_scratch_view.load())[0]
                if has_work != fx.Int32(0):
                    if const_expr(not direct_fixed_slot):
                        comm_ops.fence_system_acquire()
                    _do_scheduled_tile(work)
                consumer_active = has_work != fx.Int32(0)
        else:
            # ---- megakernel: one work loop draining GEMM1 and GEMM2 -------------
            # kind 0 = both queues drained (exit), 1 = GEMM1 unit, 2 = GEMM2 pair.
            #
            # Deadlock freedom: only blocks with ``ticket % fused_g2_pref == 0`` may
            # jump the queue for a ready GEMM2 pair, so with ``fused_g2_pref >= 2`` at
            # least half of the blocks always keep draining GEMM1 and every GEMM2
            # dependency is eventually satisfied. A block that has already committed
            # to a GEMM2 pair (because it lost the peek race) blocks on the readiness
            # wait, which is monotone and therefore bounded.
            #
            # The GEMM2 queue head is sharded exactly like the GEMM1 one and handed
            # out in chunks of ``G2_CHUNK`` consecutive pairs. A single unsharded
            # head serialized ~11k agent atomics on one cache line and cost more
            # than the GEMM2 math itself; shards x chunk cuts that by
            # ``WORK_SHARDS * G2_CHUNK``.
            g2_ctr_i64 = s2_ctr
            g2_head = (
                s2_ctr
                + fx.Int64(i32_s2_maxmb) * fx.Int64(4)
                + fx.Int64(work_shard) * fx.Int64(64)
            )
            s2_num_n = fx.Int32(S2["num_n"])
            s2_halves = fx.Int32(S2["halves"])
            total_m_blocks2 = ceildiv(num_valid, fx.Int32(S2["BM"]))
            total_g2_pairs = total_m_blocks2 * s2_num_n // s2_halves
            n_tiles_i32 = fx.Int32(N_TILES)
            # Opportunistic GEMM2 is only worth its cost when there is a bubble to
            # fill. Measured at tokens=8192: on rank-mixed-skew, pref=6 is 5.464 and
            # pref=0 (no preemption at all) is 5.972; on uniform the order reverses,
            # pref=0 is 4.536 and pref=6 is 4.834. So the overlap machinery buys
            # 0.51 ms of hidden wait on a skewed route and costs 0.30 ms on a balanced
            # one, and a single compile-time constant cannot serve both.
            #
            # The imbalance that matters is *per expert inside a rank*, not per rank:
            # on rank-mixed-skew every rank receives roughly the balanced row count
            # (measured -- a rank-level test never fires below a 1.0 threshold), but
            # one local expert owns a disproportionate share of the tiles and its
            # tail is what the peers end up waiting on. The dispatch plan already
            # reduces exactly that number: ``MAX_EXPERT_TILES`` is
            # ``max_e ceildiv(count_e, tile_m)`` over this rank's experts, while the
            # mean is ``num_valid / tile_m / experts_per_rank``. So the hot-rank test
            # is ``max_tiles * epr * tile_m  >  num_valid * num/den``.
            #
            # The predicate is derived from data identical for every block in the
            # rank, so blocks never disagree, and if it comes out false the kernel
            # degrades to the drain-then-GEMM2 order, which is independently correct
            # and deadlock-free.
            _g2_bal_num, _g2_bal_den = fused_g2_skew
            _g2_metiles = _buffer_load(
                _make_buffer_from_addr(s2_metiles, fx.Int32), fx.Int32(0), fx.Int32
            )
            _g2_skewed = (
                _g2_metiles * fx.Int32(fz_epr * fz_tile_m * _g2_bal_den)
                > num_valid * fx.Int32(_g2_bal_num)
            )
            g2_pref = const_expr(fused_g2_pref > 0) and (
                (ticket % fx.Int32(max(1, fused_g2_pref))) == fx.Int32(0)
            ) and _g2_skewed
            # The chunk exists to amortize the claim atomic on the *preemption* path,
            # so it is worth its coarseness only when that path is live. Handing out
            # 16 pairs at a time when nothing preempts just serializes: at 512 tokens
            # there are ~2.7k pairs and a 2048-block grid, so chunk=16 leaves ~168
            # blocks doing 16 tiles each while the rest idle. Measured on the
            # balanced routes, chunk 1 vs 16: 512 uniform 0.6826 vs 0.8599, 512 skew
            # 0.7706 vs 0.8523, 2048 uniform 1.5536 vs 1.8464. On 8192 skew, where
            # preemption is live, the order reverses hard: 5.4863 vs 6.5808.
            g2_chunk = fx.Int32(G2_CHUNK)
            total_g2_claims = ceildiv(total_g2_pairs, g2_chunk)
            _s2_slab = _Slab(a_buf)
            kind_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
            unit_scratch = fx.recast_iter(
                fx.Int32, fx.add_offset(a_buf.ptr, fx.make_int_tuple(64))
            )
            unit_view = fx.make_view(unit_scratch, fx.make_layout(1, 1))
            s2_bm_i32 = fx.Int32(S2["BM"])
            sbm_i32 = fx.Int32(sort_block_m)

            def _g2_tile_of(pair):
                # pair -> m_block -> the SBM-aligned Stage1 tile that produced it.
                return (((pair * s2_halves) // s2_num_n) * s2_bm_i32) // sbm_i32

            def _g2_cid(local):
                # Per-shard counter value -> global claim id. Strided exactly like the
                # GEMM1 queue; without the stride every shard would replay claims
                # 0,1,2,... and each GEMM2 pair would be computed ``WORK_SHARDS``
                # times.
                return work_shard + local * fx.Int32(WORK_SHARDS)

            def _g2_base(claim):
                # claim id -> first pair of the chunk it owns.
                return claim * g2_chunk

            # Chunk state carried by the work loop itself rather than by a nested
            # loop around the GEMM2 body: a nested ``while`` around that body made
            # the whole GEMM2 live range loop-carried and the register allocator
            # spilled it, costing ~2.6x. ``g2_pend`` counts pairs still owed from the
            # current claim; ``g2_next`` is the next of them.
            g2_pend = fx.Int32(0)
            g2_next = fx.Int32(0)

            while consumer_active:
                # Separate the previous unit's LDS reads (the GEMM2 epilogue has no
                # trailing barrier) from this iteration's scratch write.
                fx.barrier()
                kind = fx.Int32(0)
                unit = fx.Int32(0)
                # 0. Pairs still owed from the current chunk. ``g2_pend``/``g2_next``
                #    are carried identically by *every* thread, so this path costs
                #    neither an atomic nor the LDS broadcast round trip below; the
                #    branch stays block-uniform, which is what makes the barrier in
                #    the else-arm legal.
                if g2_pend > fx.Int32(0):
                    kind = fx.Int32(2)
                    unit = g2_next
                    g2_pend = g2_pend - fx.Int32(1)
                    g2_next = g2_next + fx.Int32(1)
                else:
                    if tid == fx.Int32(0):
                        # 1. Opportunistic GEMM2: peek the head and claim only if the
                        #    chunk it points at is already produced, so this path does not
                        #    block in the common case.
                        if const_expr(fused_g2_pref > 0):
                            if g2_pref and kind == fx.Int32(0):
                                peek = fx.Int32(comm_ops.atomic_add_agent(g2_head, fx.Int32(0)))
                                if _g2_cid(peek) < total_g2_claims:
                                    # Readiness of the chunk's last in-range pair covers
                                    # the whole chunk (``_g2_tile_of`` is monotone). The
                                    # clamp matters: a pair past ``total_g2_pairs`` maps
                                    # to an m-tile nothing ever publishes.
                                    _pl = _g2_base(_g2_cid(peek)) + g2_chunk - fx.Int32(1)
                                    if _pl >= total_g2_pairs:
                                        _pl = total_g2_pairs - fx.Int32(1)
                                    done = fx.Int32(
                                        comm_ops.atomic_add_agent(
                                            g2_ctr_i64 + fx.Int64(_g2_tile_of(_pl)) * fx.Int64(4),
                                            fx.Int32(0),
                                        )
                                    )
                                    if done >= n_tiles_i32:
                                        got = fx.Int32(
                                            comm_ops.atomic_add_agent(g2_head, fx.Int32(1))
                                        )
                                        if _g2_cid(got) < total_g2_claims:
                                            kind = fx.Int32(2)
                                            unit = _g2_base(_g2_cid(got))
                                            _cl = unit + g2_chunk - fx.Int32(1)
                                            if _cl >= total_g2_pairs:
                                                _cl = total_g2_pairs - fx.Int32(1)
                                            # ``got`` is normally ``peek``, whose readiness
                                            # was just proven, so this wait returns at
                                            # once. It is still needed: another block can
                                            # take ``peek`` in between, leaving this block
                                            # with a chunk nobody checked.
                                            mori_shmem.int32_wait_until_greater_than(
                                                g2_ctr_i64
                                                + fx.Int64(_g2_tile_of(_cl)) * fx.Int64(4),
                                                n_tiles_i32 - fx.Int32(1),
                                            )
                        # 2. GEMM1 queue.
                        if kind == fx.Int32(0):
                            local_work = fx.Int32(
                                comm_ops.atomic_add_agent(
                                    a_work_head + fx.Int64(work_shard) * fx.Int64(64), fx.Int32(1)
                                )
                            )
                            # ``_wu`` and not ``w``: ``w`` is the weight memref kernel
                            # argument, and rebinding it inside a device ``if`` makes it a
                            # carried variable of the region with a mismatched type.
                            _wu = work_shard + local_work * fx.Int32(WORK_SHARDS)
                            if _wu < total_work:
                                kind = fx.Int32(1)
                                unit = _wu
                        # 3. GEMM1 drained: fall through to GEMM2 and wait for readiness.
                        if kind == fx.Int32(0):
                            got = fx.Int32(comm_ops.atomic_add_agent(g2_head, fx.Int32(1)))
                            if _g2_cid(got) < total_g2_claims:
                                kind = fx.Int32(2)
                                unit = _g2_base(_g2_cid(got))
                                _cl = unit + g2_chunk - fx.Int32(1)
                                if _cl >= total_g2_pairs:
                                    _cl = total_g2_pairs - fx.Int32(1)
                                # One wait for the whole chunk: ``_g2_tile_of`` is monotone
                                # in the pair index, so the chunk's last pair dominates.
                                mori_shmem.int32_wait_until_greater_than(
                                    g2_ctr_i64 + fx.Int64(_g2_tile_of(_cl)) * fx.Int64(4),
                                    n_tiles_i32 - fx.Int32(1),
                                )
                        if kind == fx.Int32(1):  # noqa: SIM102 - device vs compile-time branches
                            if const_expr(not direct_fixed_slot):
                                _wait_tile_payload(unit)
                        fx.ptr_store(Vec.from_elements([kind], fx.Int32), work_scratch)
                        fx.ptr_store(Vec.from_elements([unit], fx.Int32), unit_scratch)
                    fx.barrier()
                    kind = Vec(kind_view.load())[0]
                    unit = Vec(unit_view.load())[0]
                    if kind == fx.Int32(2):
                        # Re-derive the chunk state from the broadcast base so that all
                        # threads agree without a second broadcast.
                        g2_next = unit + fx.Int32(1)
                        g2_pend = g2_chunk - fx.Int32(1)
                        if unit + g2_pend >= total_g2_pairs:
                            g2_pend = total_g2_pairs - unit - fx.Int32(1)
                if kind == fx.Int32(1):
                    if const_expr(not direct_fixed_slot):
                        comm_ops.fence_system_acquire()
                    _do_scheduled_tile(unit)
                    # Publish this m-tile's progress. The activation and scale stores
                    # are write-through, so draining vmcnt is the whole release; the
                    # atomic itself is the ordering point the consumer waits on.
                    #
                    # DIAGNOSTIC ONLY: fused_diag_nopub compiles the release out. The
                    # kernel is then racy by construction -- it exists to price the
                    # per-tile drain, never to be shipped.
                    if const_expr(not fused_diag_nopub):
                        fx.rocdl.s_waitcnt(0)
                        fx.barrier()
                        if tid == fx.Int32(0):
                            comm_ops.atomic_add_system(
                                g2_ctr_i64 + fx.Int64(unit // n_tiles_i32) * fx.Int64(4),
                                fx.Int32(1),
                            )
                    else:
                        if tid == fx.Int32(0):
                            comm_ops.atomic_add_system(
                                g2_ctr_i64 + fx.Int64(unit // n_tiles_i32) * fx.Int64(4),
                                fx.Int32(1),
                            )
                if kind == fx.Int32(2):
                    # halves>1: two lockstep 4-wave half-blocks, each on its own tile
                    # and its own LDS slab. nw8: all waves cooperate on one wide tile,
                    # so there is no half index and the slab offset is zero.
                    if const_expr(S2["nw8"]):
                        _half = fx.Int32(0)
                        _s2_tx = tid
                    else:
                        _half = fx.rocdl.readfirstlane(T.i32, tid // fx.Int32(256))
                        _s2_tx = tid % fx.Int32(256)
                    _s2_bx = unit * s2_halves + _half
                    S2["emit"](
                            tx_i32=_s2_tx, bx_i32=_s2_bx,
                            lane=_s2_tx % fx.Int32(64),
                            wave=fx.rocdl.readfirstlane(T.i32, _s2_tx // fx.Int32(64)),
                            arg_aq=s2_aq, arg_ascale=s2_ascale, arg_bq=s2_bq, arg_bscale=s2_bscale,
                            arg_eids=s2_eids, arg_cumsum=s2_cumsum, arg_max_expert_tiles=s2_metiles,
                            arg_stids=s2_stids, arg_sweights=s2_sweights, arg_trb=s2_trb,
                            arg_p2p_comb_inp=s2_p2p, i32_max_m_blocks=i32_s2_maxmb,
                            i32_inter=fx.Int32(S2["inter"]), i32_hidden=fx.Int32(S2["hidden"]),
                            i32_kpad=fx.Int32(0), i32_npad=fx.Int32(0),
                        lds_slab=_s2_slab, lds_byte_off=_half * fx.Int32(S2["slab"]),
                    )
                consumer_active = kind != fx.Int32(0)

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, num_valid_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        s2_aq: fx.Int64, s2_ascale: fx.Int64, s2_bq: fx.Int64, s2_bscale: fx.Int64,
        s2_eids: fx.Int64, s2_cumsum: fx.Int64, s2_metiles: fx.Int64, s2_stids: fx.Int64,
        s2_sweights: fx.Int64, s2_trb: fx.Int64, s2_p2p: fx.Int64, s2_ctr: fx.Int64,
        i32_s2_maxmb: fx.Int32, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale, tokens,
            addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected,
            s2_aq, s2_ascale, s2_bq, s2_bscale, s2_eids, s2_cumsum, s2_metiles, s2_stids,
            s2_sweights, s2_trb, s2_p2p, s2_ctr, i32_s2_maxmb,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(launch_grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch


def run_mega_moe_stage1(out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
    tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    addr_parity, addr_expected, stream, *, model_dim, inter_dim, rank, experts_per_rank, fuse_npes,
    fuse_topk, fuse_cap, fuse_mtpr, fuse_scale_dim, fixed_slot_dispatch, num_cu,
    sort_block_m=32, tile_n=256, tile_k=256, num_waves=4, grid_mult=4, pipe_weights=True,
    mfma_amajor=False, swizzle_a=True, async_a_copy=False, num_dispatch_cu=32,
    use_tile_resource=True, waves_per_eu_hint=2,
    b_nt=-1, work_shards=None, external_grouping=None, external_counting=None,
    payload_chunk_rows=0, payload_tile_ready=False, swiglu_limit=0.0,
    fused_stage2=None, fused_g2_pref=0, fused_g2_chunk=4, fused_s2_nw8=False,
    fused_g2_skew=(5, 4), fused_diag_nopub=False, fused_args=None):
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, fixed_slot_dispatch=fixed_slot_dispatch,
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu, num_dispatch_cu=num_dispatch_cu,
        b_nt=b_nt, work_shards=work_shards, external_grouping=external_grouping,
        external_counting=external_counting, payload_chunk_rows=payload_chunk_rows,
        payload_tile_ready=payload_tile_ready,
        swiglu_limit=swiglu_limit,
        # ``compile_mega_moe_stage1`` is ``functools.cache``d, so the Stage2 spec has
        # to arrive as a hashable value; it is turned back into a dict inside.
        fused_stage2=(
            None if fused_stage2 is None else tuple(sorted(dict(fused_stage2).items()))
        ),
        fused_g2_pref=fused_g2_pref, fused_g2_chunk=fused_g2_chunk, fused_s2_nw8=fused_s2_nw8,
        fused_g2_skew=fused_g2_skew, fused_diag_nopub=fused_diag_nopub,
    )
    # 12 Stage2 pointers + ``max_m_blocks``; zeros on the unfused path, where the
    # kernel never dereferences them.
    fa = tuple(fused_args) if fused_args else (fx.Int64(0),) * 12 + (fx.Int32(0),)
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, num_valid_ids, out_scale,
        tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        addr_parity, addr_expected, *fa, stream,
    )
# fmt: on
