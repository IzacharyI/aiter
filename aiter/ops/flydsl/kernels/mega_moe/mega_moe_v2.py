# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MegaMoE v2 fused dispatch, GEMM1, GEMM2, and combine implementation."""

import os
from dataclasses import replace

import flydsl.expr as fx
import mori.shmem as ms
import torch

from ..flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from .dispatch import DISPATCH_TABLE_SIZE, DispatchSlot
from .mega_moe_config import (
    FIXED_SLOT_MAX_MTPR,
    MegaMoEConfig,
    Stage1Config,
    select_mega_moe_config,
)
from .quant import per_1x32_mx_quant

__all__ = ["MegaMoEV2"]


def _group_done_slots(world_size: int) -> int:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    return world_size


class MegaMoEV2:
    """Fused dispatch, GEMM1, GEMM2, and combine with one in-flight launch per instance."""

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int, experts: int, topk: int,
        quant: str, w1: torch.Tensor, w1_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor,
        max_tok_per_rank: int, mega_scheme: str = "fixedslot", swiglu_limit: float = 0.0):
    # fmt: on
        if quant != "a8w4":
            raise ValueError("MegaMoEV2 currently supports quant='a8w4' only")
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        if max_tok_per_rank <= 0 or max_tok_per_rank & (max_tok_per_rank - 1):
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = int(experts // world_size)
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.swiglu_limit = float(swiglu_limit)
        if self.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be non-negative")
        self.dev = torch.device("cuda", rank)
        self.max_recv = self.world_size * self.mtpr
        compact = self.mtpr > FIXED_SLOT_MAX_MTPR
        capacity_tile_m = 128 if compact else 32
        self._s1_fixed_slot = not compact
        self._s1_scale_dim = self.model_dim // 32
        # fmt: off
        self.comb_cfg = FlyDSLDispatchCombineConfig(rank=self.rank, world_size=self.world_size,
            hidden_dim=self.model_dim, max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=self.topk, combine_dtype=torch.bfloat16,
            dispatch_dtype=torch.float8_e4m3fn, scale_dim=self._s1_scale_dim, scale_type_size=1,
            enable_std_moe=False, enable_group_major=True, gm_unit_size=capacity_tile_m,
            gm_scheme=mega_scheme, gm_compact=compact, max_total_recv_tokens=self.world_size)
        # fmt: on
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self._build_fused_stage1(w1, w1_scale)
        self._build_fused_stage2()

    def _build_fused_stage1(self, w1, w1_scale):
        from .mega_moe_stage1 import run_mega_moe_stage1

        self.sort_block_m = 32
        self._s1_w1 = w1.contiguous().view(torch.uint8)
        self._s1_w1_scale = w1_scale.contiguous().view(torch.uint8)
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        # Payload capacity follows the largest SBM; metadata covers the smallest candidate.
        metadata_blocks = (op.num_valid_max + self.sort_block_m - 1) // self.sort_block_m
        if metadata_blocks > op.max_blocks:
            op.max_blocks = metadata_blocks
            op.sorted_expert_ids = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
            op.tile_row_base = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
        self._s1_nvm = op.num_valid_max
        self._s1_cap = op.ll_cap
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_num_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._allocate_dispatch_workspace(op, metadata_blocks)
        self._s1_mega = run_mega_moe_stage1

        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]

        inter_dim = self.inter_dim
        a2rows = self._s1_nvm
        self._s1_out = torch.zeros((a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        prows = ((a2rows + 255) // 256) * 256
        pcols = (((inter_dim // 32) + 7) // 8) * 8
        self._s1_osd = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        self._build_v2_disp_table()

    def _allocate_dispatch_workspace(self, op, metadata_blocks):
        total_experts = self.world_size * self.epr
        workspace = {
            "local_hist": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "local_cursor": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "pair_order": torch.empty(self.mtpr * self.topk, dtype=torch.int32, device=self.dev),
            "pair_base": torch.empty(total_experts, dtype=torch.int32, device=self.dev),
            "pair_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "entry_count": torch.zeros(10, dtype=torch.int64, device=self.dev),
            "epoch_gate": torch.zeros(10, dtype=torch.int32, device=self.dev),
            "pair_order_ready": torch.zeros(2, dtype=torch.int32, device=self.dev),
            "work_head": torch.zeros(8 * 16, dtype=torch.int32, device=self.dev),
            "work_tail": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "expert_tile_end": torch.empty(self.epr, dtype=torch.int32, device=self.dev),
            "max_expert_tiles": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "payload_chunk_done": torch.zeros(total_experts, dtype=torch.int32, device=self.dev),
            "tile_expected": torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev),
            "active_payload_blocks": torch.zeros(1, dtype=torch.int32, device=self.dev),
            "payload_blocks_per_destination": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
            "payload_chunks_per_destination": torch.zeros(self.world_size, dtype=torch.int32, device=self.dev),
            # Fixed-slot dispatch indexes one completion counter per destination rank.
            "group_done": torch.zeros(
                _group_done_slots(self.world_size),
                dtype=torch.int32,
                device=self.dev,
            ),
        }
        workspace["bigcnt"] = op._sym((self.world_size * self.epr,), torch.int32)
        workspace["count_done"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["my_base"] = op._sym((total_experts,), torch.int32)
        workspace["plan_ready"] = op._sym((2 * self.world_size,), torch.int32)
        workspace["payload_ready"] = op._sym((2 * self.epr,), torch.int32)
        workspace["launch_ready"] = op._sym((self.world_size,), torch.int32)
        workspace["tile_ready"] = op._sym((metadata_blocks,), torch.int32)
        workspace["payload_ready_rows"] = op._sym((1,), torch.int32)
        ms.shmem_barrier_all()
        workspace["p2p_bigcnt"] = op._p2p_table(workspace["bigcnt"])
        workspace["p2p_count_done"] = op._p2p_table(workspace["count_done"])
        workspace["p2p_my_base"] = op._p2p_table(workspace["my_base"])
        workspace["p2p_plan_ready"] = op._p2p_table(workspace["plan_ready"])
        workspace["p2p_payload_ready"] = op._p2p_table(workspace["payload_ready"])
        workspace["p2p_launch_ready"] = op._p2p_table(workspace["launch_ready"])
        workspace["p2p_tile_ready"] = op._p2p_table(workspace["tile_ready"])
        workspace["p2p_payload_ready_rows"] = op._p2p_table(workspace["payload_ready_rows"])
        self._s1_dispatch_workspace = workspace

    def _build_v2_disp_table(self):
        op = self._s1_op
        workspace = self._s1_dispatch_workspace
        table = [0] * DISPATCH_TABLE_SIZE
        table[DispatchSlot.PAIR_BASE] = workspace["pair_base"].data_ptr()
        table[DispatchSlot.P2P_TOKEN] = op.p2p_rx_em.data_ptr()
        table[DispatchSlot.P2P_SCALE] = op.p2p_scale_em.data_ptr()
        table[DispatchSlot.P2P_WEIGHT] = op.p2p_wts_em.data_ptr()
        table[DispatchSlot.P2P_SRCMAP] = op.p2p_srcmap_em.data_ptr()
        table[DispatchSlot.SORTED_EXPERT] = op.sorted_expert_ids.data_ptr()
        table[DispatchSlot.TILE_ROW_BASE] = op.tile_row_base.data_ptr()
        table[DispatchSlot.NUM_VALID] = op.num_valid.data_ptr()
        table[DispatchSlot.SRCMAP] = op.srcmap_em.data_ptr()
        table[DispatchSlot.LOCAL_HIST] = workspace["local_hist"].data_ptr()
        table[DispatchSlot.COUNT_MATRIX] = workspace["bigcnt"].data_ptr()
        table[DispatchSlot.P2P_COUNT_MATRIX] = workspace["p2p_bigcnt"].data_ptr()
        table[DispatchSlot.COUNT_DONE] = workspace["count_done"].data_ptr()
        table[DispatchSlot.P2P_COUNT_DONE] = workspace["p2p_count_done"].data_ptr()
        table[DispatchSlot.TASK_ROW_BASE] = workspace["my_base"].data_ptr()
        table[DispatchSlot.LOCAL_CURSOR] = workspace["local_cursor"].data_ptr()
        table[DispatchSlot.P2P_PAYLOAD_READY] = workspace["p2p_payload_ready"].data_ptr()
        table[DispatchSlot.PAIR_ORDER] = workspace["pair_order"].data_ptr()
        table[DispatchSlot.P2P_TASK_ROW_BASE] = workspace["p2p_my_base"].data_ptr()
        table[DispatchSlot.P2P_PLAN_READY] = workspace["p2p_plan_ready"].data_ptr()
        table[DispatchSlot.PLAN_READY] = workspace["plan_ready"].data_ptr()
        table[DispatchSlot.PAIR_READY] = workspace["pair_ready"].data_ptr()
        table[DispatchSlot.ENTRY_COUNT] = workspace["entry_count"].data_ptr()
        table[DispatchSlot.EPOCH_GATE] = workspace["epoch_gate"].data_ptr()
        table[DispatchSlot.PAIR_ORDER_READY] = workspace["pair_order_ready"].data_ptr()
        table[DispatchSlot.WORK_HEAD] = workspace["work_head"].data_ptr()
        table[DispatchSlot.WORK_TAIL] = workspace["work_tail"].data_ptr()
        table[DispatchSlot.EXPERT_TILE_END] = workspace["expert_tile_end"].data_ptr()
        table[DispatchSlot.GROUP_DONE] = workspace["group_done"].data_ptr()
        table[DispatchSlot.RUNNING] = op.running.data_ptr()
        table[DispatchSlot.P2P_RUNNING] = op.p2p_running.data_ptr()
        table[DispatchSlot.LAUNCH_READY] = workspace["launch_ready"].data_ptr()
        table[DispatchSlot.P2P_LAUNCH_READY] = workspace["p2p_launch_ready"].data_ptr()
        table[DispatchSlot.MAX_EXPERT_TILES] = workspace["max_expert_tiles"].data_ptr()
        table[DispatchSlot.PAYLOAD_CHUNK_DONE] = workspace["payload_chunk_done"].data_ptr()
        table[DispatchSlot.TILE_READY] = workspace["tile_ready"].data_ptr()
        table[DispatchSlot.P2P_TILE_READY] = workspace["p2p_tile_ready"].data_ptr()
        table[DispatchSlot.TILE_EXPECTED] = workspace["tile_expected"].data_ptr()
        table[DispatchSlot.ACTIVE_PAYLOAD_BLOCKS] = workspace["active_payload_blocks"].data_ptr()
        table[DispatchSlot.PAYLOAD_READY_ROWS] = workspace["payload_ready_rows"].data_ptr()
        table[DispatchSlot.P2P_PAYLOAD_READY_ROWS] = workspace["p2p_payload_ready_rows"].data_ptr()
        table[DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION] = workspace[
            "payload_blocks_per_destination"
        ].data_ptr()
        table[DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION] = workspace[
            "payload_chunks_per_destination"
        ].data_ptr()
        self._s1_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)

    def _select_config(self, tokens: int) -> MegaMoEConfig:
        config = select_mega_moe_config(
            tokens,
            self.mtpr,
            experts_per_rank=self.epr,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
        )
        # Stage1 tile-geometry override. Full fusion needs Stage1 and Stage2 to agree
        # on a block size (Stage2's kernel is fixed at 256 threads, Stage1 at >=2048
        # tokens picks 8 waves = 512), so the fused path has to run Stage1 at some
        # 4-wave shape. These knobs are how that shape gets chosen by measurement
        # rather than by assertion; ``block_m`` follows because Stage2's block_m must
        # divide ``sort_block_m``.
        s1_over = {
            k: int(v)
            for k, v in (
                ("sort_block_m", os.environ.get("AITER_MEGAMOE_S1_SBM")),
                ("tile_n", os.environ.get("AITER_MEGAMOE_S1_TILEN")),
                ("num_waves", os.environ.get("AITER_MEGAMOE_S1_WAVES")),
            )
            if v
        }
        if s1_over:
            stage1 = replace(config.stage1, **s1_over)
            stage2 = config.stage2
            if stage1.sort_block_m % stage2.block_m:
                stage2 = replace(stage2, block_m=stage1.sort_block_m)
            config = replace(config, stage1=stage1, stage2=stage2)
        s2_bm = os.environ.get("AITER_MEGAMOE_S2_BM")
        if s2_bm:
            config = replace(config, stage2=replace(config.stage2, block_m=int(s2_bm)))
        # A/B knob for the opt-in single-grid Stage2+combine path. No route table
        # entry sets fuse_combine yet; this is how the fused kernel gets measured.
        if os.environ.get("AITER_MEGAMOE_FUSE_S2C") == "1" and config.stage2.persist:
            config = replace(
                config, stage2=replace(config.stage2, fuse_combine=True)
            )
        self._active_config = config
        return config

    def _fused_all_kwargs(self, config: MegaMoEConfig):
        """Stage2 compile spec + kernel arguments for the fused megakernel path."""
        op = self._s1_op
        comb_op = self.comb_op
        stage2 = config.stage2
        invariants = dict(self._g2_invariants_by_quant[config.p2p_quant])
        invariants.pop("cu_num")
        s2_spec = dict(
            invariants, BM=stage2.block_m, BN=stage2.block_n, BK=stage2.block_k,
            use_nt=stage2.use_nt, g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch, g2_bf16_lds=stage2.bf16_lds,
            analysis_no_p2p_payload=stage2.analysis_no_p2p_payload,
            SBM=config.stage1.sort_block_m, cu_num=0,
        )
        max_m_blocks = (self._s1_nvm + stage2.block_m - 1) // stage2.block_m
        args = (
            fx.Int64(self._s1_out.view(-1).data_ptr()), fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()), fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()), fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["max_expert_tiles"].data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()), fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()), comb_op._fx_p2p_comb_inp,
            self._fx_fused_mtile_ctr, fx.Int32(max_m_blocks),
        )
        self._g2_active_block_m = stage2.block_m
        return s2_spec, args

    def _run_fused_stage1(self, x, wts, scales, topk_ids, stream=None, config: Stage1Config | None = None,
                          fused_config: "MegaMoEConfig | None" = None):
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        cur_tok = int(x.shape[0])
        if cur_tok > self.mtpr:
            raise ValueError(f"run_tokens={cur_tok} > max_tok_per_rank={self.mtpr}")
        if x.dtype != torch.float8_e4m3fn or not x.is_contiguous():
            raise ValueError("x must be contiguous float8_e4m3fn")
        if tuple(x.shape) != (cur_tok, self.model_dim):
            raise ValueError(f"x must have shape ({cur_tok}, {self.model_dim})")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if tuple(wts.shape) != (cur_tok, self.topk):
            raise ValueError(f"wts must have shape ({cur_tok}, {self.topk})")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if tuple(topk_ids.shape) != (cur_tok, self.topk):
            raise ValueError(f"topk_ids must have shape ({cur_tok}, {self.topk})")
        if not scales.is_contiguous():
            raise ValueError("scales must be contiguous")
        if config is None:
            config = self._select_config(cur_tok).stage1
        op = self._s1_op
        _fused_kw = {}
        _s2_nw8 = os.environ.get("AITER_MEGAMOE_FUSE_S2_NW8", "1") == "1"
        if fused_config is not None:
            _s2_spec, _s2_args = self._fused_all_kwargs(fused_config)
            _fused_kw = {
                "fused_stage2": _s2_spec, "fused_args": _s2_args,
                # nw8 (native 8-wave GEMM2 tile) is the default: it removes the
                # halves=2 lockstep and is worth -0.48 ms. It halves the GEMM2 unit
                # count, so it wants a larger pref stride than the halves=2 path did
                # (tuned: 6/16 -> 5.464 vs 5.531 scattered; 2/16 -> 5.689).
                "fused_s2_nw8": _s2_nw8,
                "fused_g2_pref": int(
                    os.environ.get("AITER_MEGAMOE_FUSE_G2_PREF", "6" if _s2_nw8 else "2")
                ),
                # The chunk amortizes the claim atomic, but it is only affordable
                # when there are far more GEMM2 pairs than blocks. At 512 tokens
                # there are ~2.7k pairs for a ~2048-block grid, so handing out 16 at
                # a time leaves ~168 blocks doing 16 tiles each while the rest idle.
                # Measured chunk 1 vs 16: 512 uniform 0.6826/0.8599, 512 skew
                # 0.7706/0.8523, 2048 uniform 1.5536/1.8464, but 8192 skew reverses
                # to 5.4863/6.5808 once the preemption path is live. Keyed on the
                # bucket, not on the device skew predicate: that predicate is
                # size-blind (tile-ceiling padding makes it fire on 512-uniform too).
                "fused_g2_chunk": int(
                    os.environ.get(
                        "AITER_MEGAMOE_FUSE_G2_CHUNK", "16" if cur_tok >= 4096 else "1"
                    )
                ),
                # num/den of the "this rank is hot" test: enable opportunistic GEMM2
                # only when received rows exceed num/den of the balanced expectation.
                "fused_diag_nopub": os.environ.get("AITER_MEGAMOE_FUSE_DIAG_NOPUB", "0") == "1",
                "fused_g2_skew": (
                    int(os.environ.get("AITER_MEGAMOE_FUSE_G2_SKEW_NUM", "5")),
                    int(os.environ.get("AITER_MEGAMOE_FUSE_G2_SKEW_DEN", "4")),
                ),
            }
        # fmt: off
        self._s1_mega(
            self._s1_out, self._s1_rx, self._s1_w1, self._s1_scale_i32, self._s1_w1_scale,
            op.tile_row_base, op.sorted_expert_ids, op.num_valid, self._s1_osd, fx.Int32(self._s1_nvm),
            fx.Int64(self._s1_disp.data_ptr()), fx.Int32(cur_tok), fx.Int64(x.data_ptr()),
            fx.Int64(topk_ids.data_ptr()), fx.Int64(wts.data_ptr()), fx.Int64(scales.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()), fx.Int64(self._s1_epoch_expected.data_ptr()),
            stream, model_dim=self.model_dim, inter_dim=self.inter_dim, rank=self.rank,
            experts_per_rank=self.epr, fuse_npes=self.world_size, fuse_topk=self.topk,
            fuse_cap=self._s1_cap, fuse_mtpr=self.mtpr, fuse_scale_dim=self._s1_scale_dim,
            fixed_slot_dispatch=self._s1_fixed_slot, num_cu=self._s1_num_cu,
            sort_block_m=config.sort_block_m, tile_n=config.tile_n, tile_k=config.tile_k,
            num_waves=config.num_waves, grid_mult=config.grid_mult, pipe_weights=config.pipe_weights,
            mfma_amajor=config.mfma_amajor, swizzle_a=config.swizzle_a,
            async_a_copy=config.async_a_copy, num_dispatch_cu=config.num_dispatch_cu,
            use_tile_resource=config.use_tile_resource,
            waves_per_eu_hint=config.waves_per_eu_hint, b_nt=config.b_nt,
            work_shards=config.work_shards, external_grouping=config.external_grouping,
            external_counting=config.external_counting, payload_chunk_rows=config.payload_chunk_rows,
            payload_tile_ready=config.payload_tile_ready,
            swiglu_limit=self.swiglu_limit,
            **_fused_kw)
        # fmt: on
        self._s1_active_tile_m = config.sort_block_m
        return self._s1_active_tile_m

    def quantize(self, x_bf16):
        return per_1x32_mx_quant(x_bf16, quant_mode="fp8")

    def _run_joint(self, x, scales, wts, topk_ids, run_tokens, stream, slice_output):
        config = self._select_config(run_tokens)
        if config.stage1.num_waves % 4 == 0 and os.environ.get("AITER_MEGAMOE_FUSE_ALL") == "1":
            # Megakernel: Stage1 and Stage2 in one launch. Stage2's tiles are drawn
            # from a second queue inside Stage1's own persistent work loop, gated on
            # per-m-tile GEMM1 completion, so GEMM2 (and its P2P traffic) overlaps
            # the tail of GEMM1 instead of waiting for a kernel boundary.
            self._run_fused_stage1(
                x, wts, scales, topk_ids, stream=stream, config=config.stage1,
                fused_config=config,
            )
            # ``combine_no_stage1`` returns the same ``(out_tok, ...)`` shape as the
            # scattered path, so unwrap it exactly like ``_run_stage2`` does.
            ret = self.comb_op.combine_no_stage1(
                self._g2_combine_placeholder, None, None, cur_tok=run_tokens,
                enable_weights=False, stage2_p2p_quant=config.p2p_quant,
            )
            out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
            if out_tok is None:
                cfg = self.comb_cfg
                out_tok = (
                    self.comb_op.shmem_comb_out_tok.view(torch.int8)[
                        : self.mtpr * cfg.combine_token_bytes
                    ]
                    .view(cfg.combine_dtype)
                    .view(self.mtpr, cfg.combine_token_view_dim)
                )
            return out_tok[:run_tokens] if slice_output else out_tok
        self._run_fused_stage1(x, wts, scales, topk_ids, stream=stream, config=config.stage1)
        return self._run_stage2(run_tokens, stream, slice_output, config)

    def _run_stage2(self, run_tokens, stream, slice_output, config: MegaMoEConfig):
        ret = self._run_fused_stage2(run_tokens, config, stream)
        out_tok = ret[0] if isinstance(ret, (tuple, list)) else ret
        if out_tok is None:
            cfg = self.comb_cfg
            out_tok = (
                self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.combine_token_bytes]
                .view(cfg.combine_dtype)
                .view(self.mtpr, cfg.combine_token_view_dim)
            )
        return out_tok[:run_tokens] if slice_output else out_tok

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        x_q, scales = self.quantize(x_bf16)
        return self._run_joint(x_q, scales, wts, topk_ids, run_tokens, stream, slice_output)

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_q.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        return self._run_joint(x_q, scales, wts, topk_ids, run_tokens, stream, slice_output)

    forward_bf16 = forward
    __call__ = forward

    def _build_fused_stage2(self):
        from .mega_moe_stage2 import run_mega_moe_stage2

        FlyDSLDispatchCombineIntraNodeOp._ENABLE_COMBINE_NO_STAGE1 = True
        comb_cfg = self.comb_cfg
        dev = torch.device("cuda", comb_cfg.rank)
        k = comb_cfg.num_experts_per_token
        cu_num = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._g2v2_inter = int(self.inter_dim)
        self._g2v2_hidden = int(comb_cfg.hidden_dim)
        self._g2_run = run_mega_moe_stage2
        self._g2_invariants_by_quant = {}
        for p2p_quant in ("none", "fp8_blockwise_1x32"):
            p2p_row_nbytes = (
                int(comb_cfg.hidden_dim) + int(comb_cfg.hidden_dim) // 32
                if p2p_quant == "fp8_blockwise_1x32"
                else int(comb_cfg.hidden_dim) * 2
            )
            self._g2_invariants_by_quant[p2p_quant] = {
                "model_dim": int(comb_cfg.hidden_dim), "inter_dim": int(self.inter_dim),
                "experts": int(comb_cfg.num_experts_per_rank), "topk": int(k), "rank": int(comb_cfg.rank),
                "npes": int(comb_cfg.world_size), "max_tok": int(comb_cfg.max_num_inp_token_per_rank),
                "recv_cap": int(self.max_recv),
                "comb_inp_nbytes": int(comb_cfg.max_num_inp_token_per_rank) * int(k) * p2p_row_nbytes,
                "HIDDEN_MAX": int(comb_cfg.hidden_dim), "INTER_MAX": int(self.inter_dim), "cu_num": int(cu_num),
                "p2p_quant_type": p2p_quant, "fixed_slot_dispatch": bool(self._s1_fixed_slot),
            }
        self._g2_combine_placeholder = torch.empty(
            1, comb_cfg.hidden_dim, dtype=comb_cfg.combine_dtype, device=dev
        )
        # [0] Stage2 block completion count, [1] combine block arrival count. The
        # fused kernel clears both before it returns, so one buffer serves every
        # launch; zeroed once here.
        from .mega_moe_fused_s2c import FUSED_S2C_WORKSPACE_I32

        self._fused_s2c_ws = torch.zeros(
            FUSED_S2C_WORKSPACE_I32, dtype=torch.int32, device=dev
        )
        self._fx_fused_s2c_ws = fx.Int64(self._fused_s2c_ws.data_ptr())
        # Per-m-tile completion counters for the per-token-readiness variant: the
        # Stage2 block that observes the last of an m-tile's ``num_n_blocks``
        # column stripes is the one that publishes that tile's rows to the peers.
        # Sized by the worst case ``BM == 1``; the fused kernel clears it before
        # returning, so one buffer serves every launch.
        self._fused_mtile_ctr = torch.zeros(
            max(1, self._s1_nvm), dtype=torch.int32, device=dev
        )
        self._fx_fused_mtile_ctr = fx.Int64(self._fused_mtile_ctr.data_ptr())
        self._g2_cu_num = int(cu_num)

    def _run_fused_s2c(self, run_tokens, config: MegaMoEConfig, s_fx):
        """Opt-in single-grid Stage2+combine (mega_moe_fused_s2c)."""
        from .mega_moe_fused_s2c import get_fused_s2c_launch, run_mega_moe_fused_s2c

        comb_op = self.comb_op
        op = self._s1_op
        stage2 = config.stage2
        invariants = dict(self._g2_invariants_by_quant[config.p2p_quant])
        cu_num = invariants.pop("cu_num")
        launch_cu_num = min(cu_num, stage2.persist_cu) if stage2.persist_cu > 0 else cu_num
        num_n_blocks = self._g2v2_hidden // stage2.block_n
        # Sweep knob for the fused combine role's block count. Its tuned standalone
        # geometry (128 blocks x 16 waves) is unavailable here because both roles
        # must share Stage2's 4-wave block, so the block count has to be re-tuned.
        _bn_env = os.environ.get("AITER_MEGAMOE_FUSE_COMB_BN")
        # Per-token readiness (M1.5): Stage2 publishes per-destination-token arrival
        # counts as each m-tile closes and combine waits per token, instead of the
        # all-rank Stage2-complete handshake. Only progressive at token counts where
        # a persistent block owns more than one m-tile, so it stays opt-in.
        _tok_ready = os.environ.get("AITER_MEGAMOE_FUSE_TOK_READY") == "1"
        comb_ptrs, comb_meta = comb_op.fused_s2c_combine_context(
            run_tokens, comb_block_num=int(_bn_env) if _bn_env else None
        )
        launch = get_fused_s2c_launch(
            s2_total_blocks=launch_cu_num * num_n_blocks,
            comb_block_num=comb_meta["comb_block_num"],
            combine_hidden_elem_size=comb_meta["combine_hidden_elem_size"],
            combine_max_recv=comb_meta["combine_max_recv"],
            combine_data_type=comb_meta["combine_data_type"],
            BM=stage2.block_m, SBM=config.stage1.sort_block_m, BN=stage2.block_n,
            BK=stage2.block_k, use_nt=stage2.use_nt, g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch, g2_spart=stage2.spatial_partition,
            persist=True, cu_num=launch_cu_num, persist_strided=stage2.persist_strided,
            skew_cu=stage2.skew_cu, g2_bf16_lds=stage2.bf16_lds,
            analysis_no_p2p_payload=stage2.analysis_no_p2p_payload,
            per_token_ready=_tok_ready, **invariants,
        )
        max_m_blocks = (self._s1_nvm + stage2.block_m - 1) // stage2.block_m
        # fmt: off
        run_mega_moe_fused_s2c(launch, (
            fx.Int64(self._s1_out.view(-1).data_ptr()), fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()), fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()), fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["max_expert_tiles"].data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()), fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()), comb_op._fx_p2p_comb_inp,
            *comb_ptrs, self._fx_fused_s2c_ws,
            comb_op._fx_p2p_tok_ready, comb_op._fx_tok_ready, self._fx_fused_mtile_ctr,
            fx.Int32(max_m_blocks), fx.Int32(self._g2v2_inter), fx.Int32(self._g2v2_hidden),
            fx.Int32(0), fx.Int32(0), fx.Int32(comb_meta["cur_tok"]),
        ), s_fx)
        # fmt: on
        self._g2_active_block_m = stage2.block_m
        return comb_op.fused_s2c_output()

    def _run_fused_stage2(self, run_tokens, config: MegaMoEConfig, stream=None):
        comb_op = self.comb_op
        op = self._s1_op
        if stream is None:
            stream = torch.cuda.current_stream()
        s_fx = fx.Stream(stream.cuda_stream)
        stage2 = config.stage2
        p2p_quant = config.p2p_quant
        invariants = self._g2_invariants_by_quant[p2p_quant]
        if stage2.fuse_combine:
            return self._run_fused_s2c(run_tokens, config, s_fx)
        # fmt: off
        self._g2_run(
            fx.Int64(self._s1_out.view(-1).data_ptr()), fx.Int64(self._s1_osd.data_ptr()),
            fx.Int64(self.w2.data_ptr()), fx.Int64(self.w2_scale.data_ptr()),
            fx.Int64(op.sorted_expert_ids.data_ptr()), fx.Int64(op.num_valid.data_ptr()),
            fx.Int64(self._s1_dispatch_workspace["max_expert_tiles"].data_ptr()),
            fx.Int64(op.srcmap_em.data_ptr()), fx.Int64(op.wts_em.data_ptr()),
            fx.Int64(op.tile_row_base.data_ptr()), comb_op._fx_p2p_comb_inp, self._s1_nvm,
            self._g2v2_inter, self._g2v2_hidden, s_fx, BM=stage2.block_m,
            SBM=config.stage1.sort_block_m, BN=stage2.block_n, BK=stage2.block_k,
            use_nt=stage2.use_nt, g2_bhoist=stage2.b_hoist,
            g2_ascale_pf=stage2.ascale_prefetch, g2_spart=stage2.spatial_partition,
            persist=stage2.persist, persist_cu=stage2.persist_cu,
            persist_strided=stage2.persist_strided, skew_cu=stage2.skew_cu,
            g2_bf16_lds=stage2.bf16_lds,
            analysis_no_p2p_payload=stage2.analysis_no_p2p_payload, **invariants)
        # fmt: on
        self._g2_active_block_m = stage2.block_m
        return comb_op.combine_no_stage1(
            self._g2_combine_placeholder, None, None, cur_tok=run_tokens, enable_weights=False,
            stage2_p2p_quant=p2p_quant,
        )
