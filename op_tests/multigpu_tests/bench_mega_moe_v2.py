# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare Mori EP and MegaMoEV2 with the same v4_pro A8W4 CUDA Graph workload."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MORI_EP_LAUNCH_CONFIG_MODE", "AUTO")
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

import mori
import mori.shmem as ms
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

MODEL_DIM = 7168
INTER_DIM = 3072
EXPERTS = 384
TOPK = 6
SWIGLU_LIMIT = 10.0

PERF_GUARD_MIN_SPEEDUP = {
    (512, "uniform"): 140.0,
    (512, "rank-mixed-skew"): 110.0,
    (8192, "uniform"): 50.0,
    (8192, "rank-mixed-skew"): 40.0,
}


class _AmdSmiXgmiSampler:
    """Read firmware XGMI accumulators without making AMD-SMI a hard dependency."""

    def __init__(self):
        os.environ["AMDSMI_GPU_METRICS_CACHE_MS"] = "0"
        amd_smi_path = Path("/opt/rocm/share/amd_smi")
        if amd_smi_path.exists() and str(amd_smi_path) not in sys.path:
            sys.path.insert(0, str(amd_smi_path))
        import amdsmi

        self._amdsmi = amdsmi
        amdsmi.amdsmi_init()
        self._closed = False
        self.version = dict(amdsmi.amdsmi_get_lib_version())
        torch_bus_to_index = {
            int(torch.cuda.get_device_properties(index).pci_bus_id): index
            for index in range(torch.cuda.device_count())
        }
        self._handles = []
        for amd_smi_index, handle in enumerate(amdsmi.amdsmi_get_processor_handles()):
            bdf = amdsmi.amdsmi_get_gpu_device_bdf(handle)
            bus = int(bdf.split(":")[1], 16)
            if bus not in torch_bus_to_index:
                raise RuntimeError(f"AMD-SMI BDF {bdf} is not visible to Torch")
            self._handles.append(
                {
                    "handle": handle,
                    "amd_smi_index": amd_smi_index,
                    "torch_index": torch_bus_to_index[bus],
                    "bdf": bdf,
                }
            )
        if len(self._handles) != torch.cuda.device_count():
            raise RuntimeError(
                f"AMD-SMI exposed {len(self._handles)} GPUs, "
                f"Torch exposed {torch.cuda.device_count()}"
            )

    def snapshot(self):
        snapshot = []
        for entry in self._handles:
            metrics = self._amdsmi.amdsmi_get_gpu_metrics_info(entry["handle"])
            header = self._amdsmi.amdsmi_get_gpu_metrics_header_info(entry["handle"])
            snapshot.append(
                {
                    "amd_smi_index": entry["amd_smi_index"],
                    "torch_index": entry["torch_index"],
                    "bdf": entry["bdf"],
                    "header": header,
                    "xgmi_read_data_acc_kb": [
                        int(value) for value in metrics["xgmi_read_data_acc"]
                    ],
                    "xgmi_write_data_acc_kb": [
                        int(value) for value in metrics["xgmi_write_data_acc"]
                    ],
                    "xgmi_link_status": [
                        str(value) for value in metrics["xgmi_link_status"]
                    ],
                }
            )
        return sorted(snapshot, key=lambda entry: entry["torch_index"])

    def close(self):
        if not self._closed:
            self._amdsmi.amdsmi_shut_down()
            self._closed = True


def _counter_delta(before, after):
    if after >= before:
        return after - before
    return (1 << 64) - before + after


def _xgmi_delta(before, after):
    before_by_bdf = {entry["bdf"]: entry for entry in before}
    endpoints = []
    for end in after:
        start = before_by_bdf[end["bdf"]]
        read_delta = [
            _counter_delta(old, new)
            for old, new in zip(
                start["xgmi_read_data_acc_kb"],
                end["xgmi_read_data_acc_kb"],
                strict=True,
            )
        ]
        write_delta = [
            _counter_delta(old, new)
            for old, new in zip(
                start["xgmi_write_data_acc_kb"],
                end["xgmi_write_data_acc_kb"],
                strict=True,
            )
        ]
        endpoints.append(
            {
                "torch_index": end["torch_index"],
                "amd_smi_index": end["amd_smi_index"],
                "bdf": end["bdf"],
                "read_delta_kb_by_link": read_delta,
                "write_delta_kb_by_link": write_delta,
                "read_delta_kb": sum(read_delta),
                "write_delta_kb": sum(write_delta),
                "endpoint_total_delta_kb": sum(read_delta) + sum(write_delta),
            }
        )
    endpoint_sum_kb = sum(entry["endpoint_total_delta_kb"] for entry in endpoints)
    return {
        "endpoints": endpoints,
        "endpoint_sum_kb": endpoint_sum_kb,
        "paired_endpoint_normalized_kb": endpoint_sum_kb / 2.0,
    }


def _measure_xgmi(graph, replays, rank):
    sampler = None
    initialization_error = None
    if rank == 0:
        try:
            sampler = _AmdSmiXgmiSampler()
        except Exception as error:  # propagate before peers enter the measurement
            initialization_error = f"{type(error).__name__}: {error}"
    error_box = [initialization_error]
    dist.broadcast_object_list(error_box, src=0)
    if error_box[0]:
        raise RuntimeError(f"AMD-SMI XGMI sampler initialization failed: {error_box[0]}")

    torch.cuda.synchronize()
    dist.barrier()
    before = sampler.snapshot() if rank == 0 else None
    dist.barrier()
    started = time.monotonic()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    workload_wall_s = time.monotonic() - started
    after = sampler.snapshot() if rank == 0 else None
    elapsed_box = [workload_wall_s if rank == 0 else None]
    dist.broadcast_object_list(elapsed_box, src=0)
    workload_wall_s = float(elapsed_box[0])

    dist.barrier()
    idle_before = sampler.snapshot() if rank == 0 else None
    dist.barrier()
    time.sleep(workload_wall_s)
    dist.barrier()
    idle_after = sampler.snapshot() if rank == 0 else None
    dist.barrier()

    if rank != 0:
        return None
    try:
        workload_delta = _xgmi_delta(before, after)
        idle_delta = _xgmi_delta(idle_before, idle_after)
        net_kb = max(
            workload_delta["paired_endpoint_normalized_kb"]
            - idle_delta["paired_endpoint_normalized_kb"],
            0.0,
        )
        return {
            "schema_version": "mega-moe-v2-amdsmi-xgmi-v1",
            "collector": "AMD-SMI raw GPU metrics API",
            "tool_version": sampler.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command_argv": list(sys.argv),
            "counter_semantics": {
                "fields": "xgmi_read_data_acc + xgmi_write_data_acc",
                "raw_unit": "KB as defined by AMD-SMI GPU metrics content revision 9",
                "paired_endpoint_normalized": (
                    "sum of endpoint read+write deltas divided by two to avoid "
                    "counting the mirrored link endpoints twice"
                ),
                "not_wire_bytes": (
                    "firmware accumulator traffic includes fabric amplification "
                    "and is not a protocol packet/CRC/retry byte counter"
                ),
            },
            "replays": replays,
            "workload_wall_s": workload_wall_s,
            "raw_before": before,
            "raw_after": after,
            "workload_delta": workload_delta,
            "idle_baseline": {
                "duration_s": workload_wall_s,
                "raw_before": idle_before,
                "raw_after": idle_after,
                "delta": idle_delta,
            },
            "idle_subtracted_paired_endpoint_kb": net_kb,
            "idle_subtracted_paired_endpoint_bytes": net_kb * 1024.0,
        }
    finally:
        sampler.close()


def _percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _collect_combine_wait(graph, combine_op, replays, rank, world):
    records = []
    for replay in range(replays):
        combine_op.reset_analysis_wait_timing()
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        ticks = [
            int(value)
            for value in combine_op.get_analysis_wait_timing().cpu().tolist()
        ]
        gathered = [None] * world
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "block_wait_realtime_ticks": ticks,
            },
        )
        if rank == 0:
            records.append({"replay": replay, "ranks": gathered})
    if rank != 0:
        return None
    rank_summaries = []
    for rank_id in range(world):
        values = [
            tick
            for record in records
            for rank_record in record["ranks"]
            if rank_record["rank"] == rank_id
            for tick in rank_record["block_wait_realtime_ticks"]
        ]
        rank_summaries.append(
            {
                "rank": rank_id,
                "samples": len(values),
                "mean_ticks": statistics.fmean(values),
                "p50_ticks": _percentile(values, 0.50),
                "p95_ticks": _percentile(values, 0.95),
                "max_ticks": max(values),
                "mean_us": statistics.fmean(values) * 0.01,
                "p95_us": _percentile(values, 0.95) * 0.01,
                "max_us": max(values) * 0.01,
            }
        )
    return {
        "schema_version": "mega-moe-v2-combine-wait-timing-v1",
        "status": "complete_for_instrumented_combine_peer_wait_scope",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timer": {
            "instruction": "s_memrealtime",
            "frequency_hz": 100_000_000,
            "tick_ns": 10,
        },
        "replays": replays,
        "world_size": world,
        "rank_summaries": rank_summaries,
        "rank_max_of_max_us": max(entry["max_us"] for entry in rank_summaries),
        "rank_max_of_p95_us": max(entry["p95_us"] for entry in rank_summaries),
        "records": records,
        "scope": (
            "one wave-level timer group per Combine block around the eight peer "
            "uint64_wait_until_equals calls and acquire fences"
        ),
        "scope_warning": (
            "Two scalar timer reads and one rank-local store perturb short waits. "
            "The elapsed value is the wave reconvergence time for all peers, not "
            "eight independently attributable peer durations."
        ),
    }


def setup_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def barrier():
    torch.cuda.synchronize()
    ms.shmem_barrier_all()


def make_inputs(tokens, rank, world, model_dim, experts, topk, route, hot_bias, device):
    local_experts = experts // world
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=generator
    )
    if route == "hot-rank0":
        scores[:, :local_experts] += hot_bias
    values, ids = torch.topk(scores, topk, dim=-1)
    if route == "local-only":
        values, local_ids = torch.topk(
            scores[:, rank * local_experts : (rank + 1) * local_experts],
            topk,
            dim=-1,
        )
        ids = local_ids + rank * local_experts
    if route == "all-remote":
        token_ids = torch.arange(tokens, device=device).view(-1, 1)
        slots = torch.arange(topk, device=device).view(1, -1)
        destination = (rank + 1 + slots) % world
        local_ids = (token_ids + slots) % local_experts
        ids = destination * local_experts + local_ids
        values = torch.zeros(
            (tokens, topk), dtype=torch.float32, device=device
        )
    if route in ("rank-balanced-hot", "rank-balanced-last", "rank-mixed-skew"):
        destination_scores = torch.rand(
            (tokens, world), device=device, generator=generator
        )
        destination = torch.topk(destination_scores, topk, dim=-1).indices
        if route == "rank-balanced-last":
            hot = torch.ones_like(destination, dtype=torch.bool)
        elif route == "rank-mixed-skew":
            hot = destination < world // 2
        else:
            hot = (
                torch.rand((tokens, topk), device=device, generator=generator)
                < hot_bias
            )
        cold_expert = torch.randint(
            1, local_experts, (tokens, topk), device=device, generator=generator
        )
        hot_expert = local_experts - 1 if route == "rank-balanced-last" else 0
        ids = destination * local_experts + torch.where(hot, hot_expert, cold_expert)
        values = torch.randn(
            (tokens, topk), dtype=torch.float32, device=device, generator=generator
        )
    return (
        x.contiguous(),
        values.softmax(dim=-1).contiguous(),
        ids.to(torch.int32).contiguous(),
    )


def make_weights(local_experts, model_dim, inter_dim, rank, device):
    generator = torch.Generator(device=device).manual_seed(9000 + rank)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1 = torch.randn(
        (local_experts, 2 * inter_dim, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1.mul_(model_dim**-0.25)
    w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    del w1
    w1_q = w1_q.view(local_experts, 2 * inter_dim, model_dim // 2)
    w1_q = shuffle_weight_a16w4(w1_q, 16, True).contiguous()
    w1_scale = shuffle_scale_a16w4(w1_scale, local_experts, True).contiguous()

    w2 = torch.randn(
        (local_experts, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(inter_dim**-0.25)
    w2_q, w2_scale = quantize(w2, quant_dtype=dtypes.fp4x2)
    del w2
    w2_q = w2_q.view(local_experts, model_dim, inter_dim // 2)
    w2_q = shuffle_weight_a16w4(w2_q, 16, False).contiguous()
    w2_scale = shuffle_scale_a16w4(w2_scale, local_experts, False).contiguous()
    torch.cuda.empty_cache()
    return w1_q, w1_scale, w2_q, w2_scale


def capture(body):
    barrier()
    body()
    barrier()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        body()
    for _ in range(5):
        graph.replay()
    barrier()
    return graph


def time_graph(graph, iters, device):
    barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters
    mean = torch.tensor(local_ms, dtype=torch.float64, device=device)
    maximum = mean.clone()
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return (
        float(mean.item() / dist.get_world_size()),
        float(maximum.item()),
        float(local_ms),
    )


def profile_graph(graph, name, rank, out_dir, replays=3):
    barrier()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        dist.barrier()
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path / f"{name}_rank{rank}.json"))
    barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--rank-tokens", default="")
    parser.add_argument("--config-tokens", type=int, default=0)
    parser.add_argument("--mtpr", type=int, default=8192)
    parser.add_argument("--model-dim", type=int, default=MODEL_DIM)
    parser.add_argument("--inter-dim", type=int, default=INTER_DIM)
    parser.add_argument("--experts", type=int, default=EXPERTS)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--route",
        choices=(
            "uniform",
            "local-only",
            "all-remote",
            "hot-rank0",
            "rank-balanced-hot",
            "rank-balanced-last",
            "rank-mixed-skew",
        ),
        default="uniform",
    )
    parser.add_argument("--hot-bias", type=float, default=0.6)
    parser.add_argument("--stage2-strided", action="store_true")
    parser.add_argument("--stage2-persist-cu", type=int, default=0)
    parser.add_argument("--stage2-skew-cu", type=int, default=0)
    parser.add_argument("--disable-stage2-skew", action="store_true")
    parser.add_argument(
        "--p2p-quant",
        choices=("default", "none", "fp8_blockwise_1x32"),
        default="default",
    )
    parser.add_argument(
        "--analysis-no-p2p-payload",
        action="store_true",
        help="analysis only: keep Stage2 scatter instructions but force all payload stores OOB",
    )
    parser.add_argument("--stage1-payload-chunk-rows", type=int, default=0)
    parser.add_argument("--stage1-tile-ready", action="store_true")
    parser.add_argument("--disable-stage1-tile-ready", action="store_true")
    parser.add_argument("--stage1-internal-grouping", action="store_true")
    parser.add_argument("--stage1-work-shards", type=int, default=0)
    parser.add_argument("--stage1-dispatch-cu", type=int, default=0)
    parser.add_argument("--stage1-grid-mult", type=int, default=0)
    parser.add_argument("--stage1-b-nt", type=int, default=-1)
    parser.add_argument("--stage1-tile-resource", action="store_true")
    parser.add_argument("--combine-block-num", type=int, default=0)
    parser.add_argument("--combine-warp-num", type=int, default=0)
    parser.add_argument("--check-variant", action="store_true")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--xgmi-output", default="")
    parser.add_argument("--xgmi-replays", type=int, default=0)
    parser.add_argument("--combine-wait-output", default="")
    parser.add_argument("--combine-wait-replays", type=int, default=0)
    parser.add_argument("--mega-only", action="store_true")
    parser.add_argument("--prequant", action="store_true")
    parser.add_argument("--perf-guard", action="store_true")
    args = parser.parse_args()
    if bool(args.xgmi_output) != bool(args.xgmi_replays):
        raise ValueError("--xgmi-output and --xgmi-replays must be passed together")
    if args.xgmi_replays < 0:
        raise ValueError("--xgmi-replays must be positive")
    if args.xgmi_output and not args.mega_only:
        raise ValueError("--xgmi-output requires --mega-only")
    if bool(args.combine_wait_output) != bool(args.combine_wait_replays):
        raise ValueError(
            "--combine-wait-output and --combine-wait-replays must be passed together"
        )
    if args.combine_wait_replays < 0:
        raise ValueError("--combine-wait-replays must be positive")
    if args.combine_wait_output and not args.mega_only:
        raise ValueError("--combine-wait-output requires --mega-only")

    rank, world, device = setup_dist()
    if world != 8:
        raise ValueError("This comparison requires eight ranks")
    if args.experts % world:
        raise ValueError(f"experts={args.experts} must be divisible by world={world}")
    rank_tokens = [int(value) for value in args.rank_tokens.split(",") if value]
    if rank_tokens and len(rank_tokens) != world:
        raise ValueError(f"--rank-tokens requires {world} comma-separated values")
    tokens = rank_tokens[rank] if rank_tokens else args.tokens
    local_experts = args.experts // world
    x, route_weights, ids = make_inputs(
        tokens,
        rank,
        world,
        args.model_dim,
        args.experts,
        args.topk,
        args.route,
        args.hot_bias,
        device,
    )
    route_counts = torch.zeros(world, dtype=torch.int64, device=device)
    route_counts.scatter_add_(
        0,
        ids.flatten().to(torch.int64) // local_experts,
        torch.ones_like(ids.flatten(), dtype=torch.int64),
    )
    local_route_counts = route_counts.clone()
    dist.all_reduce(route_counts, op=dist.ReduceOp.SUM)
    expert_counts = torch.bincount(
        ids.flatten().to(torch.int64), minlength=args.experts
    )
    local_expert_counts = expert_counts.clone()
    dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM)
    w1, w1_scale, w2, w2_scale = make_weights(
        local_experts, args.model_dim, args.inter_dim, rank, device
    )

    mega = MegaMoEV2(
        rank=rank,
        world_size=world,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        quant="a8w4",
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        max_tok_per_rank=args.mtpr,
        swiglu_limit=SWIGLU_LIMIT,
    )
    mega.comb_op.set_analysis_wait_timing(bool(args.combine_wait_output))
    if bool(args.combine_block_num) != bool(args.combine_warp_num):
        raise ValueError(
            "--combine-block-num and --combine-warp-num must be passed together"
        )
    combine_variant = bool(args.combine_block_num)
    if combine_variant:
        mega.comb_cfg.combine_block_num = args.combine_block_num
        mega.comb_cfg.combine_warp_num_per_block = args.combine_warp_num
    default_select_config = mega._select_config
    variant_select_config = None
    if (
        args.stage2_strided
        or args.stage2_persist_cu
        or args.stage2_skew_cu
        or args.disable_stage2_skew
        or args.p2p_quant != "default"
        or args.analysis_no_p2p_payload
        or args.stage1_payload_chunk_rows
        or args.stage1_tile_ready
        or args.disable_stage1_tile_ready
        or args.stage1_internal_grouping
        or args.stage1_work_shards
        or args.stage1_dispatch_cu
        or args.stage1_grid_mult
        or args.stage1_b_nt >= 0
        or args.stage1_tile_resource
        or args.config_tokens
    ):

        def select_strided_config(tokens):
            config = default_select_config(args.config_tokens or tokens)
            stage1 = config.stage1
            stage2 = config.stage2
            if args.stage1_payload_chunk_rows:
                stage1 = replace(
                    stage1, payload_chunk_rows=args.stage1_payload_chunk_rows
                )
            if args.stage1_tile_ready:
                stage1 = replace(stage1, payload_tile_ready=True)
            if args.disable_stage1_tile_ready:
                stage1 = replace(stage1, payload_tile_ready=False)
            if args.stage1_internal_grouping:
                stage1 = replace(
                    stage1, external_grouping=False, external_counting=False
                )
            if args.stage1_work_shards:
                stage1 = replace(stage1, work_shards=args.stage1_work_shards)
            if args.stage1_dispatch_cu:
                stage1 = replace(stage1, num_dispatch_cu=args.stage1_dispatch_cu)
            if args.stage1_grid_mult:
                stage1 = replace(stage1, grid_mult=args.stage1_grid_mult)
            if args.stage1_b_nt >= 0:
                stage1 = replace(stage1, b_nt=args.stage1_b_nt)
            if args.stage1_tile_resource:
                stage1 = replace(stage1, use_tile_resource=True)
            if (
                args.stage2_strided
                or args.stage2_persist_cu
                or args.stage2_skew_cu
                or args.disable_stage2_skew
                or args.analysis_no_p2p_payload
            ):
                stage2 = replace(
                    stage2,
                    persist_strided=args.stage2_strided,
                    persist_cu=args.stage2_persist_cu or stage2.persist_cu,
                    skew_cu=(
                        0
                        if args.disable_stage2_skew
                        else args.stage2_skew_cu or stage2.skew_cu
                    ),
                    analysis_no_p2p_payload=args.analysis_no_p2p_payload,
                )
            if args.p2p_quant != "default":
                config = replace(config, p2p_quant=args.p2p_quant)
            config = replace(
                config,
                stage1=stage1,
                stage2=stage2,
            )
            mega._active_config = config
            return config

        variant_select_config = select_strided_config
        mega._select_config = variant_select_config

    mori_cfg = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world,
        hidden_dim=args.model_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=torch.bfloat16.itemsize,
        max_num_inp_token_per_rank=args.mtpr,
        num_experts_per_rank=local_experts,
        num_experts_per_token=args.topk,
        warp_num_per_block=16,
        block_num=128,
        gpu_per_node=world,
    )
    mori_op = mori.ops.EpDispatchCombineOp(mori_cfg)
    expert_mask = torch.zeros(args.experts, dtype=torch.int32, device=device)
    expert_mask[rank * local_experts : (rank + 1) * local_experts] = 1
    holders = {}

    def mori_body():
        dispatched, recv_weights, _, recv_ids, recv_tokens = mori_op.dispatch(
            x, route_weights, None, ids
        )
        local_out = fused_moe(
            dispatched,
            w1,
            w2,
            recv_weights,
            recv_ids,
            expert_mask,
            quant_type=aiter.QuantType.per_1x32,
            num_local_tokens=recv_tokens,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            dtype=torch.bfloat16,
            swiglu_limit=SWIGLU_LIMIT,
            gate_mode=GateMode.INTERLEAVE.value,
        )
        holders["mori"] = mori_op.combine(local_out, None, ids)[0]

    prequant_x, prequant_scale = mega.quantize(x)

    def mega_body():
        holders["mega"] = (
            mega.forward_prequant(prequant_x, prequant_scale, route_weights, ids)
            if args.prequant
            else mega(x, route_weights, ids)
        )

    mori_graph = None if args.mega_only else capture(mori_body)
    print(f"[STEP] rank={rank} mori-capture-done", flush=True)
    mega_graph = capture(mega_body)
    print(f"[STEP] rank={rank} mega-capture-done", flush=True)
    mori_ms = (
        (float("nan"), float("nan"))
        if mori_graph is None
        else time_graph(mori_graph, args.iters, device)
    )
    mega_ms = time_graph(mega_graph, args.iters, device)

    x_q, x_scale = prequant_x, prequant_scale

    def mega_stage1():
        mega._run_fused_stage1(x_q, route_weights, x_scale, ids)

    stage1_graph = capture(mega_stage1)
    print(f"[STEP] rank={rank} stage1-capture-done", flush=True)
    mega_stage1()
    barrier()

    def mega_stage2():
        holders["stage2"] = mega._run_stage2(tokens, None, True, mega._active_config)

    stage2_graph = capture(mega_stage2)
    print(f"[STEP] rank={rank} stage2-capture-done", flush=True)
    stage1_ms = time_graph(stage1_graph, args.iters, device)
    mega_stage1()
    barrier()
    stage2_ms = time_graph(stage2_graph, args.iters, device)
    xgmi_evidence = (
        _measure_xgmi(mega_graph, args.xgmi_replays, rank)
        if args.xgmi_output
        else None
    )
    combine_wait_evidence = (
        _collect_combine_wait(
            mega_graph,
            mega.comb_op,
            args.combine_wait_replays,
            rank,
            world,
        )
        if args.combine_wait_output
        else None
    )
    measurement_config = mega._active_config

    rel_l2 = None
    if args.check_variant:
        if (
            variant_select_config is None
            and not combine_variant
            and not args.combine_wait_output
        ):
            raise ValueError("--check-variant requires a Stage1, Stage2, or combine variant")
        mega._select_config = default_select_config
        if args.combine_wait_output:
            mega.comb_op.set_analysis_wait_timing(False)
        if combine_variant:
            mega.comb_cfg.combine_block_num = None
            mega.comb_cfg.combine_warp_num_per_block = None
        reference = mega(x, route_weights, ids).clone()
        barrier()
        mega._select_config = variant_select_config or default_select_config
        if args.combine_wait_output:
            mega.comb_op.set_analysis_wait_timing(True)
        if combine_variant:
            mega.comb_cfg.combine_block_num = args.combine_block_num
            mega.comb_cfg.combine_warp_num_per_block = args.combine_warp_num
        candidate = mega(x, route_weights, ids).clone()
        barrier()
        rel_l2 = (
            candidate.float() - reference.float()
        ).norm() / reference.float().norm()
        dist.all_reduce(rel_l2, op=dist.ReduceOp.MAX)

    if args.profile_dir:
        if mori_graph is not None:
            profile_graph(mori_graph, f"mori_{args.route}", rank, args.profile_dir)
        profile_graph(mega_graph, f"mega_{args.route}", rank, args.profile_dir)
    speedup = (mori_ms[1] / mega_ms[1] - 1.0) * 100.0
    guard_floor = None
    if args.perf_guard:
        if args.mega_only or rank_tokens or args.mtpr != 8192:
            raise ValueError(
                "--perf-guard requires Mori, equal rank tokens, and mtpr=8192"
            )
        if (args.model_dim, args.inter_dim, args.experts, args.topk) != (
            MODEL_DIM,
            INTER_DIM,
            EXPERTS,
            TOPK,
        ):
            raise ValueError("--perf-guard requires the v4_pro shape")
        guard_floor = PERF_GUARD_MIN_SPEEDUP.get((args.tokens, args.route))
        if guard_floor is None:
            raise ValueError(
                f"no performance guard for tokens={args.tokens}, route={args.route}"
            )
    guard_pass = guard_floor is None or speedup >= guard_floor
    local_record = {
        "rank": rank,
        "tokens": tokens,
        "route_counts_by_destination": local_route_counts.tolist(),
        "expert_counts": local_expert_counts.tolist(),
        "timing_ms": {
            "e2e": mega_ms[2],
            "stage1": stage1_ms[2],
            "stage2_combine": stage2_ms[2],
        },
    }
    rank_records = [None] * world
    dist.all_gather_object(rank_records, local_record)
    if rank == 0:
        if combine_wait_evidence is not None:
            combine_wait_evidence["workload"] = {
                "route": args.route,
                "tokens_per_rank": tokens,
                "world_size": world,
                "topk": args.topk,
                "stage2_p2p_quant": measurement_config.p2p_quant,
                "analysis_no_p2p_payload": bool(args.analysis_no_p2p_payload),
                "instrumented_e2e_rank_max_ms": mega_ms[1],
                "instrumented_stage2_combine_rank_max_ms": stage2_ms[1],
            }
            combine_wait_evidence["command_argv"] = list(sys.argv)
            wait_output = Path(args.combine_wait_output)
            wait_output.parent.mkdir(parents=True, exist_ok=True)
            wait_output.write_text(
                json.dumps(
                    combine_wait_evidence,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        if xgmi_evidence is not None:
            remote_rows = sum(
                int(value)
                for source in rank_records
                for destination, value in enumerate(
                    source["route_counts_by_destination"]
                )
                if destination != int(source["rank"])
            )
            stage1_row_bytes = args.model_dim + args.model_dim // 32
            stage2_row_bytes = (
                args.model_dim + args.model_dim // 32
                if measurement_config.p2p_quant == "fp8_blockwise_1x32"
                else args.model_dim * 2
            )
            stage1_payload_bytes_per_replay = remote_rows * stage1_row_bytes
            stage2_payload_bytes_per_replay = (
                0
                if args.analysis_no_p2p_payload
                else remote_rows * stage2_row_bytes
            )
            metadata_bytes_per_replay = remote_rows * 8
            useful_bytes_per_replay = (
                stage1_payload_bytes_per_replay
                + stage2_payload_bytes_per_replay
                + metadata_bytes_per_replay
            )
            counter_bytes_per_replay = (
                xgmi_evidence["idle_subtracted_paired_endpoint_bytes"]
                / args.xgmi_replays
            )
            xgmi_evidence["workload"] = {
                "route": args.route,
                "tokens_per_rank": tokens,
                "world_size": world,
                "topk": args.topk,
                "remote_route_rows_per_replay": remote_rows,
                "stage1_row_bytes": stage1_row_bytes,
                "stage2_row_bytes": stage2_row_bytes,
                "stage2_p2p_quant": measurement_config.p2p_quant,
                "analysis_no_p2p_payload": bool(args.analysis_no_p2p_payload),
                "logical_stage1_payload_bytes_per_replay": (
                    stage1_payload_bytes_per_replay
                ),
                "logical_stage2_payload_bytes_per_replay": (
                    stage2_payload_bytes_per_replay
                ),
                "logical_route_metadata_bytes_per_replay": metadata_bytes_per_replay,
                "logical_useful_bytes_per_replay": useful_bytes_per_replay,
            }
            xgmi_evidence["derived"] = {
                "idle_subtracted_paired_endpoint_bytes_per_replay": (
                    counter_bytes_per_replay
                ),
                "counter_to_logical_useful_amplification": (
                    counter_bytes_per_replay / useful_bytes_per_replay
                    if useful_bytes_per_replay
                    else None
                ),
            }
            xgmi_output = Path(args.xgmi_output)
            xgmi_output.parent.mkdir(parents=True, exist_ok=True)
            xgmi_output.write_text(
                json.dumps(
                    xgmi_evidence,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        print(f"[ROUTES] per-destination-rank={route_counts.tolist()}", flush=True)
        print(
            f"[EXPERTS] active={(expert_counts > 0).sum().item()} max_routes={expert_counts.max().item()} "
            f"mean_routes={expert_counts.float().mean().item():.1f}",
            flush=True,
        )
        if rel_l2 is not None:
            print(
                f"[ACCURACY] variant_vs_default_rel_l2={rel_l2.item():.6e}", flush=True
            )
        print(
            f"[RESULT] route={args.route} hot_bias={args.hot_bias} tokens={tokens} "
            f"rank_tokens={rank_tokens or 'same'} mtpr={args.mtpr} "
            f"shape={args.model_dim}x{args.inter_dim} epr={local_experts} topk={args.topk} "
            f"mori_e2e={mori_ms[0]:.4f}/{mori_ms[1]:.4f}ms "
            f"mega_e2e={mega_ms[0]:.4f}/{mega_ms[1]:.4f}ms speedup={speedup:.2f}% "
            f"stage1={stage1_ms[0]:.4f}/{stage1_ms[1]:.4f}ms "
            f"stage2_combine={stage2_ms[0]:.4f}/{stage2_ms[1]:.4f}ms rank-mean/max",
            flush=True,
        )
        if guard_floor is not None:
            status = "PASS" if guard_pass else "FAIL"
            print(
                f"[PERF-GUARD] {status} speedup={speedup:.2f}% minimum={guard_floor:.2f}%",
                flush=True,
            )
        if args.json_output:
            output = Path(args.json_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            variant_parts = []
            if args.stage1_payload_chunk_rows:
                variant_parts.append(f"pc{args.stage1_payload_chunk_rows}")
            if args.p2p_quant != "default":
                variant_parts.append(
                    "p2p_" + args.p2p_quant.replace("fp8_blockwise_1x32", "fp8")
                )
            if args.analysis_no_p2p_payload:
                variant_parts.append("no_p2p_payload")
            variant_name = "_".join(variant_parts) or "default"
            base_case_id = f"t{tokens}_{args.route.replace('-', '_')}"
            case_id = (
                f"{base_case_id}_{variant_name}"
                if variant_name != "default"
                else base_case_id
            )
            payload = {
                "schema_version": "mega-moe-v2-route-benchmark-v1",
                "record_type": "run",
                "status": "pass" if guard_pass else "fail",
                "metadata": {
                    "world_size": world,
                    "iters": args.iters,
                    "network": "v4_pro",
                },
                "cases": [
                    {
                        "case_id": case_id,
                        "network": "v4_pro",
                        "tokens_per_rank": tokens,
                        "world_size": world,
                        "route": args.route,
                        "comparison_group": f"v4_pro_t{tokens}_ep{world}",
                        "variant": {
                            "name": variant_name,
                            "stage1_payload_chunk_rows": args.stage1_payload_chunk_rows,
                            "p2p_quant": args.p2p_quant,
                            "analysis_no_p2p_payload": bool(
                                args.analysis_no_p2p_payload
                            ),
                            "prequant": bool(args.prequant),
                        },
                        "correctness": {
                            "variant_vs_default_rel_l2": (
                                float(rel_l2.item()) if rel_l2 is not None else None
                            )
                        },
                        "ranks": rank_records,
                        "route_summary": {
                            "per_destination_rank": route_counts.tolist(),
                            "active_experts": int((expert_counts > 0).sum().item()),
                            "expert_max_routes": int(expert_counts.max().item()),
                            "expert_mean_routes": float(expert_counts.float().mean().item()),
                            "per_expert_routes": expert_counts.tolist(),
                        },
                    }
                ],
            }
            output.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    ms.shmem_finalize()
    dist.destroy_process_group()
    if not guard_pass:
        raise AssertionError(f"speedup {speedup:.2f}% is below {guard_floor:.2f}%")


if __name__ == "__main__":
    main()
