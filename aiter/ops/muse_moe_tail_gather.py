# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Muse Spark MoE unpool tail: all-gather of per-rank column slabs over xGMI (TP4, gfx950).

Each of the 4 ranks computes split-K FP32 partials of its own column slab of an [M, 8192] output. One launch
(``csrc/kernels/muse_moe_tail_gather.hip``) sums the partials in split order, rounds once to BF16, writes the slab
and pushes it to the peers' IPC inboxes, then fills the peers' slabs from this rank's inbox. The data is its own
flag (BF16 -0.0 sentinel; senders write +0.0 for -0.0), so no fence and no separate flag are needed, and the call
is CUDA-graph replayable. Every rank must issue the calls of one instance in the same order.

``sentinel_inbox`` gives a peer-mapped inbox filled with the same sentinel, for producers that push BF16 rows
with the same protocol from another kernel.

The code object is compiled with ``hipcc --genco -O3`` into the aiter JIT directory.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile

import torch
import torch.distributed as dist

from aiter.jit.core import AITER_CSRC_DIR, get_user_jit_dir
from aiter.jit.utils.chip_info import get_gfx

_SRC = os.path.join(AITER_CSRC_DIR, "kernels", "muse_moe_tail_gather.hip")

NR = 4  # TP ranks
OUT_COLS = 8192
NTHR = 256

_P = ctypes.c_void_p


class _TArgs(ctypes.Structure):
    _fields_ = [
        ("part", _P),
        ("out", _P),
        ("bounds", _P),
        ("inbox", _P * NR),
        ("row_state", _P),
        ("rank", ctypes.c_int32),
        ("rows", ctypes.c_int32),
        ("sw", ctypes.c_int32),
        ("max_rows", ctypes.c_uint32),
    ]


class _FillArgs(ctypes.Structure):
    _fields_ = [("p", _P), ("n16", ctypes.c_uint64)]


class _IpcHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


_hip = None


def _hip_lib():
    global _hip
    if _hip is None:
        h = ctypes.CDLL("libamdhip64.so")
        h.hipModuleLoad.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        h.hipModuleGetFunction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        h.hipModuleLaunchKernel.argtypes = (
            [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        )
        h.hipMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        h.hipIpcGetMemHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        h.hipIpcOpenMemHandle.argtypes = [ctypes.c_void_p, _IpcHandle, ctypes.c_uint]
        h.hipGetErrorString.restype = ctypes.c_char_p
        _hip = h
    return _hip


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(
            f"muse_moe_tail_gather: {what} failed: {rc} "
            f"({_hip_lib().hipGetErrorString(rc).decode()})"
        )


def _require_gfx950():
    if get_gfx() != "gfx950":
        raise NotImplementedError(
            f"muse_moe_tail_gather supports gfx950 only, not {get_gfx()}"
        )


def _build(device: torch.device, flags: tuple):
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    src = open(_SRC, "rb").read()
    key = hashlib.sha256(src + repr((arch, flags)).encode()).hexdigest()[:20]
    cdir = os.path.join(get_user_jit_dir(), "build", "muse_moe_tail_gather")
    os.makedirs(cdir, exist_ok=True)
    co = os.path.join(cdir, f"muse_moe_tail_gather_{key}.co")
    if not os.path.exists(co):
        hipcc = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "bin", "hipcc")
        fd, tmp = tempfile.mkstemp(dir=cdir, suffix=".co")
        os.close(fd)
        cmd = [hipcc, "--genco", f"--offload-arch={arch}", "-O3", *flags]
        r = subprocess.run(cmd + ["-o", tmp, _SRC], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"hipcc failed for {_SRC}:\n{r.stderr[-4000:]}")
        os.replace(tmp, co)  # atomic: concurrent ranks may build the same key
    module = ctypes.c_void_p()
    _check(_hip_lib().hipModuleLoad(ctypes.byref(module), co.encode()), "hipModuleLoad")
    return module


def _fn(module, name: str) -> ctypes.c_void_p:
    f = ctypes.c_void_p()
    _check(
        _hip_lib().hipModuleGetFunction(ctypes.byref(f), module, name.encode()), name
    )
    return f


def _launch(fn, grid: int, args, stream: int) -> None:
    size = ctypes.c_size_t(ctypes.sizeof(args))
    extra = (ctypes.c_void_p * 5)(
        1, ctypes.addressof(args), 2, ctypes.addressof(size), 3
    )
    _check(
        _hip_lib().hipModuleLaunchKernel(
            fn, grid, 1, 1, NTHR, 1, 1, 0, ctypes.c_void_p(stream), None,
            ctypes.cast(extra, ctypes.c_void_p),
        ),
        "hipModuleLaunchKernel",
    )  # fmt: skip


def _sentinel_inbox(cpu_group, device: torch.device, nbytes: int, module) -> list:
    rank = dist.get_rank(group=cpu_group)
    h = _hip_lib()
    ptr = ctypes.c_void_p()
    _check(h.hipMalloc(ctypes.byref(ptr), nbytes), "hipMalloc")
    handle = _IpcHandle()
    _check(h.hipIpcGetMemHandle(ctypes.byref(handle), ptr), "hipIpcGetMemHandle")
    handles = [None] * NR
    dist.all_gather_object(
        handles, ctypes.string_at(ctypes.addressof(handle), 64), group=cpu_group
    )
    peers = []
    for i, raw in enumerate(handles):
        if i == rank:
            peers.append(ptr.value)
            continue
        peer, ph = ctypes.c_void_p(), _IpcHandle()
        ctypes.memmove(ctypes.addressof(ph), raw, 64)
        _check(
            h.hipIpcOpenMemHandle(ctypes.byref(peer), ph, 0x1), "hipIpcOpenMemHandle"
        )
        peers.append(peer.value)
    stream = torch.cuda.current_stream(device).cuda_stream
    _launch(
        _fn(module, "muse_tail_fill"), 1024, _FillArgs(ptr.value, nbytes // 16), stream
    )
    torch.cuda.synchronize(device)
    dist.barrier(group=cpu_group)
    return peers


def sentinel_inbox(cpu_group, device: torch.device, nbytes: int) -> list:
    """Allocate this rank's ``nbytes`` inbox filled with the BF16 -0.0 sentinel and map the peers' inboxes
    (collective over the 4-rank ``cpu_group``). Returns the 4 device addresses, indexed by rank.
    """
    _require_gfx950()
    assert dist.get_world_size(group=cpu_group) == NR
    with torch.cuda.device(device):
        return _sentinel_inbox(cpu_group, device, nbytes, _build(device, ()))


class MuseMoETailGather:
    """Column-slab all-gather of an [M, 8192] BF16 output for one call site on one rank.

    Args:
        cpu_group: gloo group of the 4 TP ranks (IPC handle exchange at construction only).
        device: this rank's device.
        max_rows: largest M of a call.
        split_k: number of FP32 partials per slab element.
        blocks_per_row: thread blocks per output row (1, 2 or 4).
    """

    def __init__(
        self,
        cpu_group,
        device: torch.device,
        max_rows: int = 64,
        split_k: int = 4,
        blocks_per_row: int = 4,
    ):
        _require_gfx950()
        assert dist.get_world_size(group=cpu_group) == NR
        assert blocks_per_row in (1, 2, 4)
        self.rank = dist.get_rank(group=cpu_group)
        self.max_rows, self.split_k, self.nb = (
            int(max_rows),
            int(split_k),
            int(blocks_per_row),
        )
        flags = (f"-DSK={self.split_k}", f"-DNB={self.nb}")
        with torch.cuda.device(device):
            self._module = _build(device, flags)
            self._fn = _fn(self._module, "muse_tail_gather_tp4")
            nbytes = 3 * NR * self.max_rows * OUT_COLS * 2
            self._inbox = (_P * NR)(
                *_sentinel_inbox(cpu_group, device, nbytes, self._module)
            )
        self.row_state = torch.zeros(
            self.max_rows * self.nb, dtype=torch.int32, device=device
        )

    def __call__(self, part: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        """Gather the slabs.

        Args:
            part: fp32 [split_k, M, SW] split-K partials of this rank's slab, columns
                ``[bounds[rank], bounds[rank + 1])``; SW is the widest slab.
            bounds: int32 [5] column split of the 8192 output columns (multiples of 8), identical on all ranks.

        Returns:
            bf16 [M, 8192]: every rank's slab, each summed over the splits in order and rounded once.
        """
        sk, m, sw = part.shape
        assert sk == self.split_k and 0 < m <= self.max_rows
        assert part.dtype == torch.float32 and part.is_contiguous()
        assert bounds.dtype == torch.int32 and bounds.numel() == NR + 1
        out = torch.empty((m, OUT_COLS), dtype=torch.bfloat16, device=part.device)
        args = _TArgs(
            part.data_ptr(), out.data_ptr(), bounds.data_ptr(), self._inbox,
            self.row_state.data_ptr(), self.rank, m, sw, self.max_rows,
        )  # fmt: skip
        _launch(
            self._fn,
            m * self.nb,
            args,
            torch.cuda.current_stream(part.device).cuda_stream,
        )
        return out
