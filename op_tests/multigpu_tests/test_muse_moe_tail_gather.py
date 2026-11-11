# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness / performance of aiter.ops.muse_moe_tail_gather (Muse Spark MoE unpool tail, TP4, gfx950).

Every rank holds random FP32 split-K partials of its own column slab. The gathered BF16 output must equal, bit for
bit, each rank's slab summed over the splits in order and rounded once to BF16 (-0.0 written as +0.0). A CUDA
graph is replayed with alternating inputs to check the per-row ring across calls.

    python op_tests/multigpu_tests/test_muse_moe_tail_gather.py            # correctness
    python op_tests/multigpu_tests/test_muse_moe_tail_gather.py --perf     # + per-call latency (graph, cold L2)
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

NR, COLS, SK = 4, 8192, 4
ROWS = (1, 8, 16, 24, 32, 48, 64)
BOUNDS = {
    "even": [0, 2048, 4096, 6144, 8192],
    "uneven": [0, 1536, 4096, 5632, 8192],
}


def _expected(part, bounds, rank):
    acc = part[0].clone()
    for s in range(1, part.shape[0]):
        acc += part[s]
    width = bounds[rank + 1] - bounds[rank]
    slab = acc[:, :width].to(torch.bfloat16)
    bits = slab.view(torch.int16)
    slab = torch.where(bits == -32768, torch.zeros_like(bits), bits).view(
        torch.bfloat16
    )
    slabs = [None] * NR
    dist.all_gather_object(slabs, slab.cpu())
    return torch.cat(slabs, dim=1).to(part.device)


def _inputs(m, sw, salt, rank, dev):
    g = torch.Generator(device=dev).manual_seed(1234 + 97 * salt + 13 * m + rank)
    part = torch.randn(SK, m, sw, device=dev, generator=g)
    part[:, :, :8] = 0.0  # exact zeros (and -0.0 below) must arrive as +0.0
    part[:, :, :4] = -0.0  # sums to -0.0
    return part.contiguous()


def _cold_l2_us(fn, dev, calls=50, rounds=5):
    """Per-call latency (median, min, max over rounds, rank max) with L2 and MALL flushed before every call."""
    flush = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    graphs = []
    for with_op in (False, True):
        fn()
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(calls):
                flush.zero_()
                if with_op:
                    fn()
        graphs.append(g)
    per_round = []
    for _ in range(rounds):
        t = []
        for g in graphs:
            g.replay()
            torch.cuda.synchronize()
            dist.barrier()
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            t.append(e0.elapsed_time(e1))
        us = torch.tensor([(t[1] - t[0]) / calls * 1000], device=dev)
        dist.all_reduce(us, op=dist.ReduceOp.MAX)
        per_round.append(us.item())
    per_round.sort()
    return per_round[len(per_round) // 2], per_round[0], per_round[-1]


def worker(rank, port, args, results):
    from aiter.ops.muse_moe_tail_gather import MuseMoETailGather

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=NR,
        device_id=dev,
    )
    cpu = dist.new_group(backend="gloo")
    op = MuseMoETailGather(cpu, dev, max_rows=64, split_k=SK)
    rows = []
    for name, b in BOUNDS.items():
        bounds = torch.tensor(b, dtype=torch.int32, device=dev)
        sw = max(b[i + 1] - b[i] for i in range(NR))
        for m in ROWS:
            ins = [_inputs(m, sw, salt, rank, dev) for salt in (0, 1)]
            refs = [_expected(p, b, rank) for p in ins]
            ok = all(
                torch.equal(op(p, bounds).view(torch.int16), r.view(torch.int16))
                for p, r in zip(ins, refs)
            )
            static = ins[0].clone()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gout = op(static, bounds)
            for i in range(16):
                static.copy_(ins[i % 2])
                graph.replay()
                torch.cuda.synchronize()
                ok &= torch.equal(gout.view(torch.int16), refs[i % 2].view(torch.int16))
            perf = (float("nan"),) * 3
            if args.perf:
                static.copy_(ins[0])
                perf = _cold_l2_us(lambda: op(static, bounds), dev)
            flag = torch.tensor([0.0 if ok else 1.0], device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            rows.append((f"{name}_m{m}", flag.item() == 0.0, *perf))
            del graph
    if rank == 0:
        results.extend(rows)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true")
    args = ap.parse_args()
    assert torch.cuda.device_count() >= NR, "needs 4 GPUs"
    port = 29500 + os.getpid() % 1000
    with mp.Manager() as mgr:
        results = mgr.list()
        mp.spawn(worker, args=(port, args, results), nprocs=NR, join=True)
        rows = list(results)
    print(f"{'case':<14}{'bit-exact':>10}{'us (cold L2)':>14}{'range':>16}")
    for name, ok, us, lo, hi in rows:
        print(f"{name:<14}{int(ok):>10}{us:>14.1f}{f'{lo:.1f}-{hi:.1f}':>16}")
    assert rows and all(
        r[1] for r in rows
    ), "muse_moe_tail_gather: correctness check failed"


if __name__ == "__main__":
    main()
