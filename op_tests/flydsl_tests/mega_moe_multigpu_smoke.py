#!/usr/bin/env python3
"""8-GPU MegaMoE smoke/perf validation.

Run with:
  AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=$PWD MORI_SHMEM_HEAP_SIZE=8G \
    torchrun --standalone --nproc_per_node=8 op_tests/flydsl_tests/mega_moe_multigpu_smoke.py
"""

from __future__ import annotations

import argparse
import os
import sys

import mori.shmem as ms
import torch
import torch.distributed as dist

from aiter.ops.flydsl.mega_moe import MegaMoE as AiterMegaMoE
from aiter.ops.quant import per_1x32_mx_quant_hip
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import dtypes, fp4_utils


NETWORKS = {
    "r1_v3": {"model_dim": 7168, "inter_dim": 2048, "experts": 256, "topk": 8},
    "v4_flash": {"model_dim": 4096, "inter_dim": 2048, "experts": 256, "topk": 6},
    "v4_pro": {"model_dim": 7168, "inter_dim": 3072, "experts": 384, "topk": 6},
}


def _setup_dist() -> tuple[int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", device_id=dev)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, dev


def _all_max(dev: torch.device, val: float) -> float:
    t = torch.tensor([float(val)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _all_min(dev: torch.device, val: float) -> float:
    t = torch.tensor([float(val)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return float(t.item())


def _all_mean(dev: torch.device, val: float) -> float:
    t = torch.tensor([float(val)], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / float(dist.get_world_size())


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float()
    bf = b.float()
    denom = torch.sum(bf * bf)
    if float(denom.item()) == 0.0:
        return float(torch.linalg.vector_norm(af - bf).item())
    return float(torch.sqrt(torch.sum((af - bf) ** 2) / denom).item())


def _make_local_weights(
    *,
    dev: torch.device,
    rank: int,
    model_dim: int,
    inter_dim: int,
    epr: int,
    gate_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Rank-specific but deterministic local expert weights. Both fused and
    # baseline paths consume the same preshuffled local buffers.
    gen = torch.Generator(device=dev)
    gen.manual_seed(1000 + rank)
    scale = float(model_dim) ** -0.25

    w1_bf16 = (torch.randn((epr, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32, generator=gen) * scale).to(
        torch.bfloat16
    )
    w1_fp4, w1_scale = per_1x32_mx_quant_hip(
        w1_bf16.view(-1, model_dim).contiguous(), quant_dtype=dtypes.fp4x2
    )
    if gate_mode == "interleave":
        w1 = fp4_utils.shuffle_weight_w4(
            w1_fp4.view(epr, 2 * inter_dim, model_dim // 2),
            NLane=16,
            gate_up=True,
            moe_gemm=True,
        ).view(torch.uint8).contiguous()
        w1_scale = fp4_utils.shuffle_scale_w4(
            w1_scale.view(epr * 2 * inter_dim, model_dim // 32),
            experts_cnt=epr,
            gate_up=True,
        ).view(torch.uint8).contiguous()
    else:
        w1 = shuffle_weight(w1_fp4.view(dtypes.fp4x2)).view(torch.uint8).contiguous()
        w1_scale = fp4_utils.e8m0_shuffle(w1_scale.view(torch.uint8)).view(torch.uint8).contiguous()

    w2_bf16 = (torch.randn((epr, model_dim, inter_dim), device=dev, dtype=torch.float32, generator=gen) * scale).to(
        torch.bfloat16
    )
    w2_fp4, w2_scale = per_1x32_mx_quant_hip(
        w2_bf16.view(-1, inter_dim).contiguous(), quant_dtype=dtypes.fp4x2
    )
    w2 = shuffle_weight(w2_fp4.view(dtypes.fp4x2)).view(torch.uint8).contiguous()
    w2_scale = fp4_utils.e8m0_shuffle(w2_scale.view(torch.uint8)).view(torch.uint8).contiguous()
    return w1, w1_scale, w2, w2_scale


def _time_ms(body, *, iters: int, use_cudagraph: bool) -> float:
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    for _ in range(2):
        body()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    graph = None
    if use_cudagraph:
        capture_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            body()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if graph is None:
            body()
        else:
            graph.replay()
    end.record()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    return start.elapsed_time(end) / max(1, iters)


def _next_power_of_two(v: int) -> int:
    return 1 << (max(1, int(v)) - 1).bit_length()


def _make_cross_rank_topk(
    *, tokens: int, topk: int, rank: int, world: int, epr: int, dev: torch.device
) -> torch.Tensor:
    rows = []
    for t in range(tokens):
        row = []
        for k in range(topk):
            dest_rank = (rank + 1 + k) % world
            local_expert = (t + k) % epr
            row.append(dest_rank * epr + local_expert)
        rows.append(row)
    return torch.tensor(rows, device=dev, dtype=torch.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--network", choices=tuple(NETWORKS), default="v4_pro")
    parser.add_argument("--max-tok-per-rank", type=int, default=0)
    parser.add_argument("--rel-threshold", type=float, default=3.0e-1)
    parser.add_argument("--source-flydsl-root", type=str, default="/home/ywx/megamoev1/FlyDSL")
    parser.add_argument("--gate-mode", choices=("separated", "interleave"), default="interleave")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--no-cudagraph", action="store_true")
    args = parser.parse_args()

    rank, world, dev = _setup_dist()
    if world != 8:
        raise SystemExit(f"this smoke expects WORLD_SIZE=8, got {world}")

    net = NETWORKS[str(args.network)]
    model_dim = int(net["model_dim"])
    inter_dim = int(net["inter_dim"])
    experts = int(net["experts"])
    topk = int(net["topk"])
    if experts % world != 0:
        raise SystemExit(f"experts={experts} must divide world={world}")
    epr = experts // world
    tokens = int(args.tokens)
    mtpr = int(args.max_tok_per_rank) if int(args.max_tok_per_rank) > 0 else _next_power_of_two(tokens)
    gate_mode = str(args.gate_mode)

    torch.manual_seed(2026 + rank)
    x = (torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * (float(model_dim) ** -0.25)).to(
        torch.bfloat16
    )
    wts = torch.full((tokens, topk), 1.0 / float(topk), device=dev, dtype=torch.float32)
    # Force all top-k choices to remote ranks while keeping expert ids distinct.
    topk_ids = _make_cross_rank_topk(tokens=tokens, topk=topk, rank=rank, world=world, epr=epr, dev=dev)
    w1, w1_scale, w2, w2_scale = _make_local_weights(
        dev=dev, rank=rank, model_dim=model_dim, inter_dim=inter_dim, epr=epr, gate_mode=gate_mode
    )

    common = dict(
        rank=rank,
        world_size=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant="a8w4",
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        max_tok_per_rank=mtpr,
        network=str(args.network),
        gate_mode=gate_mode,
    )
    fused = AiterMegaMoE(**common, enable_fused_stage1=True, enable_fused_stage2=True)
    sys.path.insert(0, args.source_flydsl_root)
    from kernels.moe.mega_moe import MegaMoE as SourceMegaMoE  # noqa: PLC0415

    baseline = SourceMegaMoE(**common, enable_fused_stage1=True, enable_fused_stage2=True)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    xq, scales = fused.quantize(x)
    out_fused = fused.forward_prequant(xq, scales, wts, topk_ids)
    out_base = baseline.forward_prequant(xq, scales, wts, topk_ids)
    torch.cuda.synchronize()
    ms.shmem_barrier_all()

    rel = _rel_l2(out_fused, out_base)
    finite = bool(torch.isfinite(out_fused.float()).all().item() and torch.isfinite(out_base.float()).all().item())
    aiter_ll_count = int(fused.stage1.op.ll_count.sum().item())
    source_ll_count = int(baseline.stage1.op.ll_count.sum().item())
    aiter_num_valid = int(fused.stage1.output.num_valid_ids[0].item())
    source_num_valid = int(baseline.stage1.output.num_valid_ids[0].item())
    max_abs = float(out_fused.float().abs().max().item())

    max_rel = _all_max(dev, rel)
    min_aiter_ll_count = int(_all_min(dev, float(aiter_ll_count)))
    min_source_ll_count = int(_all_min(dev, float(source_ll_count)))
    min_aiter_num_valid = int(_all_min(dev, float(aiter_num_valid)))
    min_source_num_valid = int(_all_min(dev, float(source_num_valid)))
    max_output_abs = _all_max(dev, max_abs)
    all_finite = bool(_all_min(dev, 1.0 if finite else 0.0))

    use_cudagraph = not bool(args.no_cudagraph)
    fused_ms = _time_ms(
        lambda: fused.forward_prequant(xq, scales, wts, topk_ids),
        iters=int(args.iters),
        use_cudagraph=use_cudagraph,
    )
    base_ms = _time_ms(
        lambda: baseline.forward_prequant(xq, scales, wts, topk_ids),
        iters=int(args.iters),
        use_cudagraph=use_cudagraph,
    )
    fused_ms_mean = _all_mean(dev, fused_ms)
    base_ms_mean = _all_mean(dev, base_ms)
    fused_ms_max = _all_max(dev, fused_ms)
    base_ms_max = _all_max(dev, base_ms)

    breakdown = None
    if args.breakdown:
        xq_a, sc_a = fused.quantize(x)
        xq_s, sc_s = baseline.quantize(x)
        s1_a = fused.stage1.forward(xq_a, wts, sc_a, topk_ids)
        s1_s = baseline.stage1.forward(xq_s, wts, sc_s, topk_ids)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

        q_a = _time_ms(lambda: fused.quantize(x), iters=int(args.iters), use_cudagraph=False)
        q_s = _time_ms(lambda: baseline.quantize(x), iters=int(args.iters), use_cudagraph=False)
        s1_ms_a = _time_ms(
            lambda: fused.stage1.forward(xq_a, wts, sc_a, topk_ids),
            iters=int(args.iters),
            use_cudagraph=use_cudagraph,
        )
        s1_ms_s = _time_ms(
            lambda: baseline.stage1.forward(xq_s, wts, sc_s, topk_ids),
            iters=int(args.iters),
            use_cudagraph=use_cudagraph,
        )

        s1_a = fused.stage1.forward(xq_a, wts, sc_a, topk_ids)
        s1_s = baseline.stage1.forward(xq_s, wts, sc_s, topk_ids)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        s2_ms_a = _time_ms(
            lambda: fused._run_stage2(s1_a, tokens, None, True),
            iters=int(args.iters),
            use_cudagraph=use_cudagraph,
        )
        s2_ms_s = _time_ms(
            lambda: baseline._run_stage2(s1_s, tokens, None, True),
            iters=int(args.iters),
            use_cudagraph=use_cudagraph,
        )

        breakdown = {
            "quant_aiter": _all_mean(dev, q_a),
            "quant_source": _all_mean(dev, q_s),
            "stage1_aiter": _all_mean(dev, s1_ms_a),
            "stage1_source": _all_mean(dev, s1_ms_s),
            "stage2_aiter": _all_mean(dev, s2_ms_a),
            "stage2_source": _all_mean(dev, s2_ms_s),
        }

    passed = (
        all_finite
        and min_aiter_ll_count > 0
        and min_source_ll_count > 0
        and min_aiter_num_valid > 0
        and min_source_num_valid > 0
        and max_output_abs > 0.0
        and max_rel <= float(args.rel_threshold)
    )
    if rank == 0:
        print(
            f"[MegaMoE-8GPU] {'PASS' if passed else 'FAIL'} network={args.network} tokens={tokens} "
            f"mtpr={mtpr} experts={experts} epr={epr} topk={topk} quant=a8w4 gate={gate_mode}",
            flush=True,
        )
        print(
            f"  correctness: max_relL2(aiter_vs_source_fused)={max_rel:.3e} "
            f"threshold={float(args.rel_threshold):.3e} finite={all_finite}",
            flush=True,
        )
        print(
            f"  p2p/stage1: min_ll_count aiter={min_aiter_ll_count} source={min_source_ll_count}; "
            f"min_num_valid aiter={min_aiter_num_valid} source={min_source_num_valid}; "
            f"max_abs={max_output_abs:.3e}",
            flush=True,
        )
        print(
            f"  perf ms/rank-mean ({'cudagraph' if use_cudagraph else 'cuda-event'}): "
            f"aiter={fused_ms_mean:.4f} source={base_ms_mean:.4f} "
            f"speedup={base_ms_mean / fused_ms_mean if fused_ms_mean > 0 else -1:.3f}",
            flush=True,
        )
        print(
            f"  perf ms/rank-max:  aiter={fused_ms_max:.4f} source={base_ms_max:.4f}",
            flush=True,
        )
        if breakdown is not None:
            print(
                "  breakdown ms/rank-mean: "
                f"quant aiter={breakdown['quant_aiter']:.4f} source={breakdown['quant_source']:.4f}; "
                f"stage1 aiter={breakdown['stage1_aiter']:.4f} source={breakdown['stage1_source']:.4f}; "
                f"stage2 aiter={breakdown['stage2_aiter']:.4f} source={breakdown['stage2_source']:.4f}",
                flush=True,
            )

    try:
        ms.shmem_finalize()
    finally:
        dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
