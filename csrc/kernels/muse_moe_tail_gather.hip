// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Muse Spark MoE unpool tail: column-slab all-gather over xGMI (gfx950, TP4 IPC peers).
//
// Each rank computes y[:, c0:c1) = h @ Wu[c0:c1).T for its unpool column slab as FP32 split-K partials
// [SK, M, SW]; this kernel (NB 256-thread blocks per row, 8-column units) sums the partials
// in split order, rounds once to BF16, writes them to `out` and pushes them to the three peers, then fills the
// peers' units of the row from this rank's inbox. The data is its own flag
// (BF16 -0.0 sentinel, senders map -0.0 to +0.0), sc0|sc1 write-through stores, volatile system-scope loads,
// 3-slot ring keyed by a per-row call counter; row-call c resets slot (c + 2) % 3 of its own inbox.
#include <hip/hip_runtime.h>
#include <cstdint>
#define NR 4
#define OUTW 8192
#define ROW_BYTES (OUTW * 2)
#define UNITS (OUTW / 8)
#ifndef NB
#define NB 1
#endif
// NB blocks per row, each owning OUTW / NB contiguous columns (UPT 8-column units per thread).
#define UPT (UNITS / 256 / NB)
#define SENT2 0x80008000u
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
typedef __amdgpu_buffer_rsrc_t rsrc_t;
#define CPOL_SYS 17
#define CPOL_SYS_VOL (17 | (1u << 31))
#ifndef SK
#define SK 4
#endif

struct TArgs {
    const float* part;      // [SK, M, SW] fp32 partials of this rank's slab
    uint4* out;             // [M, 8192] bf16
    const int32_t* bounds;  // [NR + 1] unpool column split
    uint8_t* inbox[NR];
    int32_t* row_state;     // [max_rows * NB]
    int32_t rank;
    int32_t rows;
    int32_t sw;
    uint32_t max_rows;
};

__device__ __forceinline__ rsrc_t make_rsrc(const void* base) {
    return __builtin_amdgcn_make_buffer_rsrc((void*)base, 0, 0x7fffffff, 0x00020000);
}
__device__ __forceinline__ void bst_x4(rsrc_t r, uint32_t off, uint4 v) {
    u32x4 t = {v.x, v.y, v.z, v.w};
    __builtin_amdgcn_raw_buffer_store_b128(t, r, off, 0, CPOL_SYS);
}
__device__ __forceinline__ uint4 bld_x4(rsrc_t r, uint32_t off) {
    u32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, off, 0, CPOL_SYS_VOL);
    asm volatile("" ::: "memory");
    return make_uint4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ uint32_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ uint32_t nnz(uint32_t b16) { return b16 == 0x8000u ? 0u : b16; }
__device__ __forceinline__ bool has_sentinel(uint4 v) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    bool hit = false;
#pragma unroll
    for (int i = 0; i < 4; ++i) hit |= ((w[i] & 0xffffu) == 0x8000u) | ((w[i] >> 16) == 0x8000u);
    return hit;
}
__device__ __forceinline__ uint32_t rec_off(const TArgs& a, uint32_t slot, int sender, int row) {
    return ((slot * NR + sender) * a.max_rows + row) * ROW_BYTES;
}

extern "C" __global__ void __launch_bounds__(256) muse_tail_gather_tp4(TArgs a) {
    const int r = blockIdx.x / NB, sub = blockIdx.x % NB, tid = threadIdx.x;
    const uint32_t call = (uint32_t)a.row_state[blockIdx.x];
    const uint32_t slot = call % 3u, prev = (call + 2u) % 3u;
    const size_t mn = (size_t)a.rows * a.sw;
    int b[NR + 1];
#pragma unroll
    for (int i = 0; i <= NR; ++i) b[i] = a.bounds[i];
    const int c0 = b[a.rank];
    const float* prow = a.part + (size_t)r * a.sw;
    const rsrc_t my = make_rsrc(a.inbox[a.rank]);
    uint4* orow = a.out + (size_t)r * UNITS;

    int col[UPT], own[UPT];
    float4 pv[UPT][SK][2];
    uint4 t[UPT];
#pragma unroll
    for (int i = 0; i < UPT; ++i) {
        col[i] = sub * (OUTW / NB) + (tid + i * 256) * 8;
        own[i] = (col[i] >= b[1]) + (col[i] >= b[2]) + (col[i] >= b[3]);
        if (own[i] == a.rank) {
            const float* src = prow + (col[i] - c0);
#pragma unroll
            for (int s = 0; s < SK; ++s) {
                pv[i][s][0] = ((const float4*)(src + s * mn))[0];
                pv[i][s][1] = ((const float4*)(src + s * mn))[1];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < UPT; ++i)
        if (own[i] != a.rank) t[i] = bld_x4(my, rec_off(a, slot, own[i], r) + col[i] * 2);
#pragma unroll
    for (int i = 0; i < UPT; ++i) {
        if (own[i] != a.rank) continue;
        float acc[8] = {pv[i][0][0].x, pv[i][0][0].y, pv[i][0][0].z, pv[i][0][0].w,
                        pv[i][0][1].x, pv[i][0][1].y, pv[i][0][1].z, pv[i][0][1].w};
#pragma unroll
        for (int s = 1; s < SK; ++s) {
            acc[0] += pv[i][s][0].x; acc[1] += pv[i][s][0].y; acc[2] += pv[i][s][0].z; acc[3] += pv[i][s][0].w;
            acc[4] += pv[i][s][1].x; acc[5] += pv[i][s][1].y; acc[6] += pv[i][s][1].z; acc[7] += pv[i][s][1].w;
        }
        uint32_t w[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) w[k] = nnz(f2bf(acc[2 * k])) | (nnz(f2bf(acc[2 * k + 1])) << 16);
        const uint4 v = make_uint4(w[0], w[1], w[2], w[3]);
#pragma unroll
        for (int k = 1; k < NR; ++k)
            bst_x4(make_rsrc(a.inbox[(a.rank + k) % NR]), rec_off(a, slot, a.rank, r) + col[i] * 2, v);
        orow[col[i] / 8] = v;
    }
#pragma unroll
    for (int i = 0; i < UPT; ++i) {
        if (own[i] == a.rank) continue;
        const uint32_t off = rec_off(a, slot, own[i], r) + col[i] * 2;
        while (has_sentinel(t[i])) t[i] = bld_x4(my, off);
        orow[col[i] / 8] = t[i];
    }
    const uint4 sent = make_uint4(SENT2, SENT2, SENT2, SENT2);
#pragma unroll
    for (int k = 1; k < NR; ++k) {
        const uint32_t base = rec_off(a, prev, (a.rank + k) % NR, r) + sub * (ROW_BYTES / NB);
        for (int p = tid; p < ROW_BYTES / NB / 16; p += 256) bst_x4(my, base + p * 16, sent);
    }
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    if (tid == 0) a.row_state[blockIdx.x] = (int32_t)(call + 1u);
}

struct FillArgs { uint4* p; uint64_t n16; };
extern "C" __global__ void muse_tail_fill(FillArgs a) {
    const uint4 s = make_uint4(SENT2, SENT2, SENT2, SENT2);
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < a.n16; i += (uint64_t)gridDim.x * blockDim.x)
        a.p[i] = s;
}
