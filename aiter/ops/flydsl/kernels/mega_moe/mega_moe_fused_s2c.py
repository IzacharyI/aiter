# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, I001
"""Single-grid fusion of Stage2 (GEMM2 + P2P scatter) and combine.

MegaMoE v2 runs Stage2 and combine as two launches, ordered only by stream
semantics. This module emits both into one kernel and replaces that implicit
ordering with an explicit in-kernel readiness edge, which is the prerequisite
for making combine start per token instead of after every rank finishes.

The grid is split by role. Blocks ``[0, s2_total_blocks)`` run the Stage2 tile
loop exactly as the standalone kernel does; blocks ``[s2_total_blocks, ...)``
run combine. Both roles come from the shared emitters (``make_stage2_body_emitter``,
``emit_combine_barrier_and_reduce``) so there is one copy of each body.

Deadlock freedom rests on AMD dispatching workgroups in increasing block index:
every Stage2-role block is dispatched before the first combine-role block, and
no Stage2 block ever waits on a combine block, so the Stage2 role always drains
and the combine blocks always become co-resident (required by their grid-wide
barrier on ``combine_bar``).

This still preserves combine's all-rank barrier, so a rank's combine work cannot
start before its own Stage2 work finishes. The win here is one launch, not
overlap; it exists as the infrastructure the per-token readiness step needs.

It is opt-in and off by default because, measured, it is *slower*: rank-max
stage2+combine over the four route guards moves 0.2510 -> 0.2674 ms (512
uniform), 0.2920 -> 0.3242 (512 skew), 2.0777 -> 2.2334 (8192 uniform) and
2.5729 -> 2.6490 (8192 skew). Per-kernel traces put the cause squarely on the
readiness edge, not on the merge: the Stage2 role alone runs at the standalone
kernel's speed and the combine role costs about what the standalone combine
kernel does even at 4 waves, but publishing completion from every Stage2 block
adds ~30-60 us that the kernel boundary used to provide for free. Only per-token
readiness, which lets combine start inside Stage2's shadow, can pay that back.

``AITER_MEGAMOE_FUSE_TOK_READY=1`` is that per-token variant, and measured, it
does *not* pay it back: rank-max stage2+combine is 0.2698 ms at 512 uniform and
0.3561 at 512 skew against 0.2332/0.2942 for the two-launch path with
write-through P2P, and 2.6712 vs 2.5518 at 8192 skew. Two structural reasons,
both worth carrying into the next milestone. At 512 there is nothing to overlap:
``tiles_per_slot == 1``, so a persistent block finishes its single m-tile and the
readiness edge can only add cost. At 8192 there are ~7 m-tiles per slot, but
``n_block = bx % num_n_blocks`` is fixed for a block's whole life, so every one
of a tile's column stripes completes at nearly the same time and the arrival
counters go from 0 to topk in a burst -- the progressive readiness the design
wanted does not exist at this granularity. On top of that the variant is not yet
live: at 8192 it still wedges intermittently after a handful of iterations. It
therefore stays opt-in and default-off, and it must not be enabled until that is
resolved; what it is good for today is the evidence above.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T

from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    create_buffer_resource_from_addr,
)

from ..communication_ops_utils import (
    atomic_add_agent,
    atomic_add_system,
    atomic_add_global_at,
    fence_release,
    fence_system_acquire,
    fence_system_release,
    store_i32_system,
)
from ..flydsl_dispatch_combine_intranode_kernel import (
    _combine_transport_spec,
    emit_combine_barrier_and_reduce,
)
from ..tensor_shim import _run_compiled
from .mega_moe_stage2 import (
    derive_stage2_emit_constants,
    make_stage2_body_emitter,
)

# Both roles share one block shape, so combine runs at Stage2's 256 threads
# (4 waves) rather than the 16 waves the standalone combine kernel is tuned for.
_FUSED_BLOCK_THREADS = 256
_FUSED_WARPS_PER_BLOCK = _FUSED_BLOCK_THREADS // 64

# Stage2 completion is counted into ``_S2_DONE_SHARDS`` separate i32 slots rather
# than one. A single counter puts every Stage2 block's atomic on one cache line,
# contended across all 8 XCDs; sharding by block index spreads it and measured
# most of the readiness edge's cost away. 64 is a measured optimum, not a guess:
# 16 shards leave too much contention and 256 make the consumer's poll loop
# contend with the producers' atomics, and both are ~15% slower than 64. Slot ``_S2_DONE_SHARDS`` is the combine
# arrival count, used to pick the block that clears the counters for next launch,
# and slot ``_S2_DONE_SHARDS + 1`` is the single go flag every combine block polls.
_S2_DONE_SHARDS = 64
_S2_ACK_SLOT = _S2_DONE_SHARDS
_S2_GO_SLOT = _S2_DONE_SHARDS + 1
# Launch counter for the per-token path's epoch barrier. It has to be private to
# this kernel: ``addr_xdb_flag`` is the dispatch/combine op's shared cross-device
# barrier flag and is stepped by other kernels too, so deriving an arrival target
# from it would ask for a count that never arrives.
_S2_EPOCH_SLOT = _S2_DONE_SHARDS + 2
_S2_RESET_SLOTS = _S2_DONE_SHARDS + 2
FUSED_S2C_WORKSPACE_I32 = _S2_DONE_SHARDS + 3

# Attribution knob, analysis only -- the kernel produces wrong output when it is
# set. ``nocomb`` keeps the readiness edge but drops the combine work; ``nosig``
# drops the readiness edge too, leaving the bare Stage2 role. Together they split
# the fused kernel's time three ways, which is how the fence-scope and counter-
# sharding costs below were measured.
_DIAG = os.environ.get("AITER_MEGAMOE_FUSE_DIAG", "")

# Per-token-readiness diagnostic, analysis only: publish the arrival counters but
# neither wait on them nor clear them, so the host can read the per-token counts
# back after a launch. Output is wrong under this knob (combine reduces whatever
# has landed); it exists to tell an undershooting counter from an overshooting one.
_TOKRDY_DEBUG = os.environ.get("AITER_MEGAMOE_TOKRDY_DEBUG") == "1"

# Bisection knob, analysis only: keep the per-token waits and the end-of-kernel
# reset but drop the inter-iteration epoch barrier. Back-to-back iterations are
# then unsynchronised and may read stale payload, so this is not a correctness
# configuration; it exists to tell a hang in the barrier from a hang in the
# per-token wait.
_TOKRDY_NOEPOCH = os.environ.get("AITER_MEGAMOE_TOKRDY_NOEPOCH") == "1"


# fmt: off
def compile_mega_moe_fused_s2c(*, s2_total_blocks: int, comb_block_num: int,
    combine_hidden_elem_size: int, combine_max_recv: int, combine_data_type,
    model_dim: int, inter_dim: int, experts: int, topk: int, rank: int, npes: int,
    max_tok: int, recv_cap: int | None = None, comb_inp_nbytes: int | None = None, BM: int = 32,
    BN: int = 256, BK: int = 256, use_nt: bool = True, HIDDEN_MAX: int = 8192, INTER_MAX: int = 8192,
    a_dtype: str = "fp8", SBM: int | None = None, persist: bool = False, cu_num: int = 0,
    has_pad: bool = False, g2_bhoist=None, g2_ascale_pf=None, g2_spart=None,
    persist_strided: bool = False, g2_bf16_lds: bool = False, p2p_quant_type: str = "none",
    fixed_slot_dispatch: bool = False, skew_cu: int = 0, analysis_no_p2p_payload: bool = False,
    per_token_ready: bool = False):
# fmt: on
    """Compile the single-grid Stage2+combine kernel and return its launcher.

    ``s2_total_blocks`` is the Stage2 grid the standalone launcher would have used
    (``grid_blocks * model_dim // BN``); it is a compile-time constant here because
    the role split and the Stage2 completion target both depend on it. The combine
    half is fixed to the fused-GEMM2 contract: ``skip_stage1``, no weights, no
    zero-copy, no std-MoE.
    """
    if not persist:
        raise ValueError(
            "fused Stage2+combine requires persist=True so the Stage2 grid is a "
            "compile-time constant"
        )
    if not 0 < comb_block_num <= cu_num:
        raise ValueError(
            f"comb_block_num={comb_block_num} must be in (0, cu_num={cu_num}]; the "
            "combine role runs a grid-wide barrier and must be co-resident"
        )
    consts, s2_name = derive_stage2_emit_constants(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts, topk=topk, rank=rank,
        npes=npes, max_tok=max_tok, recv_cap=recv_cap, comb_inp_nbytes=comb_inp_nbytes,
        BM=BM, BN=BN, BK=BK, use_nt=use_nt, HIDDEN_MAX=HIDDEN_MAX, INTER_MAX=INTER_MAX,
        a_dtype=a_dtype, SBM=SBM, persist=persist, cu_num=cu_num, has_pad=has_pad,
        g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf, g2_spart=g2_spart,
        persist_strided=persist_strided, g2_bf16_lds=g2_bf16_lds,
        p2p_quant_type=p2p_quant_type, fixed_slot_dispatch=fixed_slot_dispatch,
        skew_cu=skew_cu, analysis_no_p2p_payload=analysis_no_p2p_payload,
        publish_tok_ready=per_token_ready,
    )
    BN = consts["BN"]
    emit_stage2_body = make_stage2_body_emitter(**consts)

    blockwise_fp8_transport = p2p_quant_type == "fp8_blockwise_1x32"
    _spec = _combine_transport_spec(
        npes=npes,
        experts_per_token=topk,
        hidden_dim=model_dim,
        hidden_elem_size=combine_hidden_elem_size,
        max_tok_per_rank=max_tok,
        data_type=combine_data_type,
        enable_weights=False,
        fp8_direct_cast=False,
        blockwise_fp8_transport=blockwise_fp8_transport,
        max_recv=combine_max_recv,
    )
    # FlyDSL allows one SharedAllocator per kernel, so both roles share one slab.
    # The roles never run in the same block, so the fields could overlap; they are
    # kept disjoint because the combine half is only npes*16 bytes.
    s2_lds_bytes = consts["lds_ready_off"] + (npes * 8 + 16 if per_token_ready else 0)

    @fx.struct
    class FusedSharedStorage:
        buf: fx.Array[Int8, s2_lds_bytes, 16]
        p2p_bases: fx.Array[fx.Int64, npes, 16]
        ack: fx.Array[fx.Int32, 4, 16]

    # The per-token arrival array is allocated with a tail past ``max_tok``; its
    # first tail element is the epoch-barrier counter (see the barrier below).
    _EPOCH_CTR_OFF = int(max_tok) * 4
    S2_TOTAL = int(s2_total_blocks)
    _SHARD_BASE = S2_TOTAL // _S2_DONE_SHARDS
    _SHARD_REM = S2_TOTAL % _S2_DONE_SHARDS
    COMB_BLOCKS = int(comb_block_num)
    WPB = _FUSED_WARPS_PER_BLOCK
    kernel_name = f"{s2_name}__fusedcomb_b{COMB_BLOCKS}_s{S2_TOTAL}"

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[_FUSED_BLOCK_THREADS, 1, 1])
    def fused_s2c(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_max_expert_tiles: fx.Int64, arg_stids: fx.Int64,
        arg_sweights: fx.Int64, arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64,
        addr_shmem_tok: fx.Int64, addr_out_shmem_tok: fx.Int64, addr_shmem_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64, addr_inp_tok_map: fx.Int64, addr_comb_bar: fx.Int64,
        addr_inp_total_recv: fx.Int64, addr_p2p_xdb_mem: fx.Int64, addr_out_shmem_wts: fx.Int64,
        addr_inp_disp_wts: fx.Int64, addr_s2_done: fx.Int64,
        arg_p2p_tok_ready: fx.Int64, addr_tok_ready: fx.Int64, arg_mtile_ctr: fx.Int64,
        i32_max_m_blocks: fx.Int32, i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32,
        i32_npad: fx.Int32, cur_rank_num_token: fx.Int32):
    # fmt: on
        tx_i32 = fx.thread_idx.x
        bx_i32 = fx.block_idx.x
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        _lds = fx.SharedAllocator().allocate(FusedSharedStorage).peek()

        if bx_i32 < fx.Int32(S2_TOTAL):
            emit_stage2_body(
                tx_i32=tx_i32, bx_i32=bx_i32, lane=lane, wave=wave,
                arg_aq=arg_aq, arg_ascale=arg_ascale, arg_bq=arg_bq, arg_bscale=arg_bscale,
                arg_eids=arg_eids, arg_cumsum=arg_cumsum,
                arg_max_expert_tiles=arg_max_expert_tiles, arg_stids=arg_stids,
                arg_sweights=arg_sweights, arg_trb=arg_trb, arg_p2p_comb_inp=arg_p2p_comb_inp,
                i32_max_m_blocks=i32_max_m_blocks, i32_inter=i32_inter, i32_hidden=i32_hidden,
                i32_kpad=i32_kpad, i32_npad=i32_npad, lds_slab=_lds,
                arg_p2p_tok_ready=arg_p2p_tok_ready, arg_mtile_ctr=arg_mtile_ctr,
            )
            # Publish this block's completion.
            #
            # The fence scope here is performance-critical. MI355X has 8 XCDs with
            # a private L2 each, so *any* agent- or system-scope release lowers to
            # ``buffer_wbl2`` -- a full L2 writeback. Paying that once per Stage2
            # block (13440 of them at bs=512) measured +0.35 ms on top of a 0.22 ms
            # role; a workgroup-scope release is just ``s_waitcnt vmcnt(0)``, which
            # is all this edge needs: it guarantees the block's stores have been
            # acknowledged by its XCD's L2 before the counter moves. Getting those
            # lines out of the per-XCD L2s then costs one writeback per combine
            # block instead of one per Stage2 block (see below).
            if _DIAG != "nosig" and not per_token_ready:
                # Every wave runs the fence: ``s_waitcnt vmcnt(0)`` is per-wave, and
                # ``s_barrier`` does not wait on memory, so fencing only in wave 0
                # would leave the other three waves' payload stores unordered.
                fence_release(fx.rocdl.SyncScope.WorkgroupOneAs)
                fx.barrier()
                if tx_i32 == 0:
                    s2_shard = bx_i32 & fx.Int32(_S2_DONE_SHARDS - 1)
                    atomic_add_agent(
                        addr_s2_done + fx.Int64(s2_shard) * fx.Int64(4), fx.Int32(1)
                    )
        else:
            cb = bx_i32 - fx.Int32(S2_TOTAL)
            tid = tx_i32
            clane = tid & fx.Int32(63)
            cwarp = tid >> fx.Int32(6)
            global_warp_id = cb * fx.Int32(WPB) + cwarp
            grid_thread_id = cb * fx.Int32(_FUSED_BLOCK_THREADS) + tid

            # Local readiness edge, replacing the Stage2->combine kernel boundary.
            #
            # Only combine block 0 watches the shards -- lane k of its first wave
            # takes shard k, and the shard targets partition S2_TOTAL. It then
            # raises a single go flag that the other blocks poll. Letting all
            # blocks poll all shards is measurably worse: the poll loop's
            # uncached loads contend with the Stage2 blocks' completion atomics
            # on the very lines they are updating.
            if _DIAG != "nosig" and not per_token_ready:
                if cb == fx.Int32(0):
                    if tid < fx.Int32(_S2_DONE_SHARDS):
                        shard_target = fx.Int32(_SHARD_BASE) + (
                            tid < fx.Int32(_SHARD_REM)
                        ).select(fx.Int32(1), fx.Int32(0))
                        mori_shmem.int32_wait_until_equals(
                            addr_s2_done + fx.Int64(tid) * fx.Int64(4), shard_target
                        )
                    fx.barrier()
                    if tid == fx.Int32(0):
                        store_i32_system(addr_s2_done, _S2_GO_SLOT, fx.Int32(1))
                else:
                    if tid == fx.Int32(0):
                        mori_shmem.int32_wait_until_equals(
                            addr_s2_done + fx.Int64(_S2_GO_SLOT * 4), fx.Int32(1)
                        )
            fx.barrier()
            # wait_until_equals uses a relaxed system load that does not invalidate
            # L2, so the acquire is required before reading anything it guards.
            # Per-token readiness has nothing to acquire here -- it waits per token
            # and reads the payload system-scope -- and an acquire would invalidate
            # the whole L2 for nothing.
            if const_expr(not per_token_ready):
                fence_system_acquire()

            # The system-scope release for the whole Stage2 half. In the two-kernel
            # path this came free from Stage2's kernel-end release; here it must be
            # explicit, and it must precede both the local reduction and the xdb
            # flag store that tells peers their rows are ready.
            #
            # One block is *not* enough: ``buffer_wbl2`` writes back the issuing
            # XCD's L2 only. Every combine block issues it, and blocks are dealt
            # round-robin across the 8 XCDs, so all eight L2s are written back
            # before the grid-wide ``comb_bar`` barrier inside the emitter below.
            # Write-through payload stores (per-token readiness) are already
            # system-visible, so there is nothing to write back.
            if const_expr(not per_token_ready):
                if tid == fx.Int32(0):
                    fence_system_release()
                fx.barrier()

            # Arrive-and-clear: every combine block has passed the wait by the time
            # the last one arrives, so that block resets both counters for the next
            # launch (stream ordering keeps the next Stage2 out of the way).
            if const_expr(not per_token_ready):
                if tid == fx.Int32(0):
                    prev_ack = atomic_add_global_at(
                        addr_s2_done + fx.Int64(_S2_ACK_SLOT * 4), fx.Int32(1)
                    )
                    if prev_ack == fx.Int32(COMB_BLOCKS - 1):
                        for _slot in range_constexpr(_S2_RESET_SLOTS):
                            store_i32_system(addr_s2_done, _slot, fx.Int32(0))

            # DIAG-ONLY: attribute fused-kernel time between the two roles.
            if _DIAG not in ("nocomb", "nosig"):
                def _maybe_load(rsrc, offset, vld_flag, **kwargs):
                    raw = buffer_load(rsrc, offset, **kwargs)
                    return vld_flag.select(raw, 0)

                _r_trecv = create_buffer_resource_from_addr(addr_inp_total_recv)
                _r_xdb_flag = create_buffer_resource_from_addr(addr_xdb_flag)
                _r_comb_bar = create_buffer_resource_from_addr(addr_comb_bar)
                _r_p2p_comb = create_buffer_resource_from_addr(arg_p2p_comb_inp)
                _r_p2p_xdb = create_buffer_resource_from_addr(addr_p2p_xdb_mem)
                _rsrc_tok_map = create_buffer_resource_from_addr(addr_inp_tok_map)

                xdb_cur_flag = buffer_load(_r_xdb_flag, 0, vec_width=1, dtype=T.i64)
                if const_expr(per_token_ready and not _TOKRDY_DEBUG):
                    _r_p2p_tok_ready = create_buffer_resource_from_addr(
                        arg_p2p_tok_ready
                    )

                _lds_p2p_bases = _lds.p2p_bases.view(fx.make_layout(npes, 1))
                if clane < fx.Int32(npes):
                    p2p_base_addr = buffer_load(_r_p2p_comb, clane, vec_width=1, dtype=T.i64)
                    fx.memref_store(p2p_base_addr, _lds_p2p_bases, clane)
                fx.barrier()

                emit_combine_barrier_and_reduce(
                    _spec,
                    rank=rank,
                    npes=npes,
                    experts_per_token=topk,
                    max_tok_per_rank=max_tok,
                    block_num=COMB_BLOCKS,
                    zero_copy=False,
                    skip_stage1=True,
                    enable_weights=False,
                    blockwise_fp8_transport=blockwise_fp8_transport,
                    analysis_wait_timing=False,
                    tid=tid,
                    bid=cb,
                    lane=clane,
                    warp=cwarp,
                    grid_thread_id=grid_thread_id,
                    global_warp_id=global_warp_id,
                    global_warp_num=COMB_BLOCKS * WPB,
                    addr_shmem_tok=addr_shmem_tok,
                    addr_out_shmem_tok=addr_out_shmem_tok,
                    addr_shmem_xdb_mem=addr_shmem_xdb_mem,
                    addr_xdb_flag=addr_xdb_flag,
                    addr_comb_bar=addr_comb_bar,
                    addr_shmem_wts=fx.Int64(0),
                    addr_out_shmem_wts=addr_out_shmem_wts,
                    addr_inp_disp_wts=addr_inp_disp_wts,
                    cur_rank_num_token=cur_rank_num_token,
                    xdb_cur_flag=xdb_cur_flag,
                    r_comb_bar=_r_comb_bar,
                    r_trecv=_r_trecv,
                    r_p2p_xdb=_r_p2p_xdb,
                    rsrc_tok_map=_rsrc_tok_map,
                    lds_p2p_bases=_lds_p2p_bases,
                    lds_p2p_wt_bases=None,
                    maybe_load=_maybe_load,
                    tok_ready_addr=(addr_tok_ready
                        if (per_token_ready and not _TOKRDY_DEBUG) else None),
                    tok_ready_expected=topk,
                )

                # Per-token readiness clears *after* the reduction, not before it:
                # the counters are what Stage 3 waits on. The last combine block to
                # arrive is by definition the one that finds every other block done
                # reducing, so it can reset both the arrival counters and the
                # per-m-tile counters for the next launch without anyone waiting.
                if const_expr(per_token_ready and not _TOKRDY_DEBUG):
                    _lds_ack = _lds.ack.view(fx.make_layout(4, 1))
                    if tid == fx.Int32(0):
                        prev_ack = atomic_add_global_at(
                            addr_s2_done + fx.Int64(_S2_ACK_SLOT * 4), fx.Int32(1)
                        )
                        fx.memref_store(prev_ack, _lds_ack, 0)
                    fx.barrier()
                    if fx.memref_load(_lds_ack, 0) == fx.Int32(COMB_BLOCKS - 1):
                        if tid == fx.Int32(0):
                            store_i32_system(
                                addr_s2_done, _S2_ACK_SLOT, fx.Int32(0)
                            )
                        for _t in range(tid, cur_rank_num_token,
                                        fx.Int32(_FUSED_BLOCK_THREADS)):
                            store_i32_system(addr_tok_ready, _t, fx.Int32(0))
                        # The m-tile counters clear themselves in Stage2 (the closing
                        # block subtracts what it counted), so only the arrival
                        # counters are swept here.
                        #
                        # Inter-iteration epoch barrier. Per-token readiness removed
                        # the pre-reduce peer wait, and with it the only thing that
                        # kept the ranks in step: nothing otherwise stops a fast rank
                        # from starting the next launch's Stage2 and incrementing our
                        # arrival counters -- or overwriting the single-buffered
                        # payload -- while we are still reducing this one. Publishing
                        # the epoch flag *after* the reset and waiting for every peer's
                        # arrival before exiting restores that: a peer can only leave
                        # this kernel once we have finished reducing and cleared, so
                        # its next Stage2 cannot race us. The wait sits after all the
                        # useful work, so unlike the two-launch barrier it does not
                        # serialise the reduction behind the slowest peer's Stage2.
                        #
                        # The arrival is a remote *atomic increment* of a monotone
                        # counter, not a per-peer flag store, for two reasons. An
                        # ordinary store would sit dirty in this XCD's L2 with no
                        # kernel boundary left to write it back -- the two-launch
                        # barrier gets that flush for free because its store is
                        # followed by the whole reduction and then kernel end, and
                        # this one is followed only by a spin -- whereas an atomic
                        # goes to the coherence point by construction. And a monotone
                        # counter cannot be missed: an equality wait on a flag slot
                        # deadlocks if the peer overwrites it with the next epoch's
                        # value before we sample it, while ``>= npes*epoch`` is
                        # satisfied by every later value too.
                        fence_system_release()
                        if tid < fx.Int32(npes):
                            _peer_tr = buffer_load(
                                _r_p2p_tok_ready, tid, vec_width=1, dtype=T.i64
                            )
                            atomic_add_system(
                                _peer_tr + fx.Int64(_EPOCH_CTR_OFF), fx.Int32(1)
                            )
                        if tid == fx.Int32(0):
                            _epoch = atomic_add_global_at(
                                addr_s2_done + fx.Int64(_S2_EPOCH_SLOT * 4),
                                fx.Int32(1),
                            )
                            if const_expr(not _TOKRDY_NOEPOCH):
                                mori_shmem.int32_wait_until_greater_than(
                                    addr_tok_ready + fx.Int64(_EPOCH_CTR_OFF),
                                    fx.Int32(npes) * (_epoch + fx.Int32(1))
                                    - fx.Int32(1),
                                )

    # fmt: off
    @flyc.jit
    def launch(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_max_expert_tiles: fx.Int64, arg_stids: fx.Int64,
        arg_sweights: fx.Int64, arg_trb: fx.Int64, arg_p2p_comb_inp: fx.Int64,
        addr_shmem_tok: fx.Int64, addr_out_shmem_tok: fx.Int64, addr_shmem_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64, addr_inp_tok_map: fx.Int64, addr_comb_bar: fx.Int64,
        addr_inp_total_recv: fx.Int64, addr_p2p_xdb_mem: fx.Int64, addr_out_shmem_wts: fx.Int64,
        addr_inp_disp_wts: fx.Int64, addr_s2_done: fx.Int64,
        arg_p2p_tok_ready: fx.Int64, addr_tok_ready: fx.Int64, arg_mtile_ctr: fx.Int64,
        i32_max_m_blocks: fx.Int32, i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_kpad: fx.Int32,
        i32_npad: fx.Int32, cur_rank_num_token: fx.Int32, stream: fx.Stream):
    # fmt: on
        fused_s2c(
            arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_cumsum, arg_max_expert_tiles,
            arg_stids, arg_sweights, arg_trb, arg_p2p_comb_inp, addr_shmem_tok,
            addr_out_shmem_tok, addr_shmem_xdb_mem, addr_xdb_flag, addr_inp_tok_map,
            addr_comb_bar, addr_inp_total_recv, addr_p2p_xdb_mem, addr_out_shmem_wts,
            addr_inp_disp_wts, addr_s2_done, arg_p2p_tok_ready, addr_tok_ready,
            arg_mtile_ctr, i32_max_m_blocks, i32_inter, i32_hidden,
            i32_kpad, i32_npad, cur_rank_num_token,
        ).launch(
            grid=(S2_TOTAL + COMB_BLOCKS, 1, 1),
            block=(_FUSED_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


_FUSED_LAUNCH_CACHE = {}


def get_fused_s2c_launch(**compile_kw):
    """Get-or-compile a fused Stage2+combine launcher for a full param set."""
    key = tuple(sorted((k, str(v)) for k, v in compile_kw.items()))
    launch = _FUSED_LAUNCH_CACHE.get(key)
    if launch is None:
        launch = compile_mega_moe_fused_s2c(**compile_kw)
        _FUSED_LAUNCH_CACHE[key] = launch
    return launch


def run_mega_moe_fused_s2c(launch, args, stream):
    """Launch a compiled fused Stage2+combine kernel."""
    _run_compiled(launch, *args, stream)
