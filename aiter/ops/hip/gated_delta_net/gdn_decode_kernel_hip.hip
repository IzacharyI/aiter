#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>

constexpr int HEAD_K_DIM = 128;
constexpr int HEAD_V_DIM = 128;
constexpr int NUM_K_HEADS = 16;
constexpr int NUM_V_HEADS = 32;

constexpr int WARP_SIZE = 64;
constexpr int NUM_WARPS = 4;
constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;
constexpr int WARP_K_THREADS = 8;
constexpr int WARP_V_THREADS = WARP_SIZE / WARP_K_THREADS;
constexpr int VEC_K = 4;
constexpr int WARP_TILE_K = WARP_K_THREADS * VEC_K;
constexpr int K_ITERS = HEAD_K_DIM / WARP_TILE_K;

__device__ __forceinline__ float as_float(int v) {
    union { int i; float f; } u; u.i = v; return u.f;
}
__device__ __forceinline__ int as_int(float v) {
    union { float f; int i; } u; u.f = v; return u.i;
}
__device__ __forceinline__ float shfl_xor(float val, int offset) {
    int lane = threadIdx.x % WARP_SIZE;
    return as_float(__builtin_amdgcn_ds_bpermute((lane ^ offset) << 2, as_int(val)));
}
__device__ __forceinline__ float shfl_idx(float val, int src) {
    return as_float(__builtin_amdgcn_ds_bpermute(src << 2, as_int(val)));
}
__device__ __forceinline__ float reduce_sum_k(float val) {
    val += shfl_xor(val, 1);
    val += shfl_xor(val, 2);
    val += shfl_xor(val, 4);
    return val;
}

// ============================================================
// Inline ASM reduce helpers: guaranteed 8-way ds_bpermute
// interleaving with precise s_waitcnt
// ============================================================
__device__ __forceinline__ void reduce_8way_asm(
    float& v0, float& v1, float& v2, float& v3,
    float& v4, float& v5, float& v6, float& v7,
    int a1, int a2, int a4)
{
    float t0, t1, t2, t3, t4, t5, t6, t7;
    asm volatile(
        "ds_bpermute_b32 %0, %16, %8\n"
        "ds_bpermute_b32 %1, %16, %9\n"
        "ds_bpermute_b32 %2, %16, %10\n"
        "ds_bpermute_b32 %3, %16, %11\n"
        "ds_bpermute_b32 %4, %16, %12\n"
        "ds_bpermute_b32 %5, %16, %13\n"
        "ds_bpermute_b32 %6, %16, %14\n"
        "ds_bpermute_b32 %7, %16, %15\n"
        "s_waitcnt lgkmcnt(7)\n" "v_add_f32 %8, %8, %0\n"
        "s_waitcnt lgkmcnt(6)\n" "v_add_f32 %9, %9, %1\n"
        "s_waitcnt lgkmcnt(5)\n" "v_add_f32 %10, %10, %2\n"
        "s_waitcnt lgkmcnt(4)\n" "v_add_f32 %11, %11, %3\n"
        "s_waitcnt lgkmcnt(3)\n" "v_add_f32 %12, %12, %4\n"
        "s_waitcnt lgkmcnt(2)\n" "v_add_f32 %13, %13, %5\n"
        "s_waitcnt lgkmcnt(1)\n" "v_add_f32 %14, %14, %6\n"
        "s_waitcnt lgkmcnt(0)\n" "v_add_f32 %15, %15, %7\n"
        "ds_bpermute_b32 %0, %17, %8\n"
        "ds_bpermute_b32 %1, %17, %9\n"
        "ds_bpermute_b32 %2, %17, %10\n"
        "ds_bpermute_b32 %3, %17, %11\n"
        "ds_bpermute_b32 %4, %17, %12\n"
        "ds_bpermute_b32 %5, %17, %13\n"
        "ds_bpermute_b32 %6, %17, %14\n"
        "ds_bpermute_b32 %7, %17, %15\n"
        "s_waitcnt lgkmcnt(7)\n" "v_add_f32 %8, %8, %0\n"
        "s_waitcnt lgkmcnt(6)\n" "v_add_f32 %9, %9, %1\n"
        "s_waitcnt lgkmcnt(5)\n" "v_add_f32 %10, %10, %2\n"
        "s_waitcnt lgkmcnt(4)\n" "v_add_f32 %11, %11, %3\n"
        "s_waitcnt lgkmcnt(3)\n" "v_add_f32 %12, %12, %4\n"
        "s_waitcnt lgkmcnt(2)\n" "v_add_f32 %13, %13, %5\n"
        "s_waitcnt lgkmcnt(1)\n" "v_add_f32 %14, %14, %6\n"
        "s_waitcnt lgkmcnt(0)\n" "v_add_f32 %15, %15, %7\n"
        "ds_bpermute_b32 %0, %18, %8\n"
        "ds_bpermute_b32 %1, %18, %9\n"
        "ds_bpermute_b32 %2, %18, %10\n"
        "ds_bpermute_b32 %3, %18, %11\n"
        "ds_bpermute_b32 %4, %18, %12\n"
        "ds_bpermute_b32 %5, %18, %13\n"
        "ds_bpermute_b32 %6, %18, %14\n"
        "ds_bpermute_b32 %7, %18, %15\n"
        "s_waitcnt lgkmcnt(7)\n" "v_add_f32 %8, %8, %0\n"
        "s_waitcnt lgkmcnt(6)\n" "v_add_f32 %9, %9, %1\n"
        "s_waitcnt lgkmcnt(5)\n" "v_add_f32 %10, %10, %2\n"
        "s_waitcnt lgkmcnt(4)\n" "v_add_f32 %11, %11, %3\n"
        "s_waitcnt lgkmcnt(3)\n" "v_add_f32 %12, %12, %4\n"
        "s_waitcnt lgkmcnt(2)\n" "v_add_f32 %13, %13, %5\n"
        "s_waitcnt lgkmcnt(1)\n" "v_add_f32 %14, %14, %6\n"
        "s_waitcnt lgkmcnt(0)\n" "v_add_f32 %15, %15, %7\n"
        : "=&v"(t0), "=&v"(t1), "=&v"(t2), "=&v"(t3),
          "=&v"(t4), "=&v"(t5), "=&v"(t6), "=&v"(t7),
          "+v"(v0), "+v"(v1), "+v"(v2), "+v"(v3),
          "+v"(v4), "+v"(v5), "+v"(v6), "+v"(v7)
        : "v"(a1), "v"(a2), "v"(a4)
    );
}

__device__ __forceinline__ void reduce_2way_asm(
    float& v0, float& v1, int a1, int a2, int a4)
{
    float t0, t1;
    asm volatile(
        "ds_bpermute_b32 %0, %4, %2\n"
        "ds_bpermute_b32 %1, %4, %3\n"
        "s_waitcnt lgkmcnt(1)\n" "v_add_f32 %2, %2, %0\n"
        "s_waitcnt lgkmcnt(0)\n" "v_add_f32 %3, %3, %1\n"
        "ds_bpermute_b32 %0, %5, %2\n"
        "ds_bpermute_b32 %1, %5, %3\n"
        "s_waitcnt lgkmcnt(1)\n" "v_add_f32 %2, %2, %0\n"
        "s_waitcnt lgkmcnt(0)\n" "v_add_f32 %3, %3, %1\n"
        "ds_bpermute_b32 %0, %6, %2\n"
        "ds_bpermute_b32 %1, %6, %3\n"
        "s_waitcnt lgkmcnt(1)\n" "v_add_f32 %2, %2, %0\n"
        "s_waitcnt lgkmcnt(0)\n" "v_add_f32 %3, %3, %1\n"
        : "=&v"(t0), "=&v"(t1), "+v"(v0), "+v"(v1)
        : "v"(a1), "v"(a2), "v"(a4)
    );
}

__device__ __forceinline__ void shfl_idx_4way_asm(
    float& d0, float& d1, float& d2, float& d3,
    float s0, float s1, float s2, float s3, int addr)
{
    asm volatile(
        "ds_bpermute_b32 %0, %8, %4\n"
        "ds_bpermute_b32 %1, %8, %5\n"
        "ds_bpermute_b32 %2, %8, %6\n"
        "ds_bpermute_b32 %3, %8, %7\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(d0), "=v"(d1), "=v"(d2), "=v"(d3)
        : "v"(s0), "v"(s1), "v"(s2), "v"(s3), "v"(addr)
    );
}

// ============================================================
// HIP kernel: q+k in registers, auto-dispatch regqk/hybrid
// ============================================================
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 2)
void gdn_decode_kernel_hip(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NUM_V_HEADS;
    const int hv_idx = b_hv % NUM_V_HEADS;
    const int hk_idx = hv_idx / (NUM_V_HEADS / NUM_K_HEADS);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NUM_V_HEADS * shs + hv_idx * shs;

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = __bfloat162float(dt_bias[hv_idx]);

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * NUM_V_HEADS + sq * NUM_V_HEADS + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = 1.0f / (1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * NUM_K_HEADS * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * NUM_K_HEADS * HEAD_K_DIM + hk_idx * HEAD_K_DIM;

        float q_reg[K_ITERS][VEC_K], k_reg[K_ITERS][VEC_K];
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                q_reg[ki][j] = __bfloat162float(qb[ko + j]);
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
            }
        }

        if constexpr (USE_QK_L2NORM) {
            float sq_q = 0, sq_k = 0;
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    sq_q += q_reg[ki][j] * q_reg[ki][j];
                    sq_k += k_reg[ki][j] * k_reg[ki][j];
                }
            sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
            float qs = scale * __frsqrt_rn(sq_q + 1e-6f), ks = __frsqrt_rn(sq_k + 1e-6f);
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) { q_reg[ki][j] *= qs; k_reg[ki][j] *= ks; }
        } else {
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) q_reg[ki][j] *= scale;
        }

        float dot_kq = 0;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) dot_kq += k_reg[ki][j] * q_reg[ki][j];
        dot_kq = reduce_sum_k(dot_kq);

        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * NUM_V_HEADS * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            int gv = global_v_base + vi * WARP_GROUP_TILE_V;
            float rv = __bfloat162float(vb[gv]);
            float shk = 0, shq = 0;
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    shk += S[vi][ki][j] * k_reg[ki][j];
                    shq += S[vi][ki][j] * q_reg[ki][j];
                }
            shk = reduce_sum_k(shk) * exp_g;
            shq = reduce_sum_k(shq) * exp_g;
            float vn = (rv - shk) * beta;
            vn = shfl_idx(vn, (lane_id / WARP_K_THREADS) * WARP_K_THREADS);
            float ov = shq + vn * dot_kq;
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + k_reg[ki][j] * vn;
            if (k_tid == 0)
                output[(int64_t)(batch_idx * seq_length + sq) * NUM_V_HEADS * HEAD_V_DIM + hv_idx * HEAD_V_DIM + gv] = __float2bfloat16(ov);
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// ============================================================
// (removed: gdn_decode_kernel_asm — inferior to iasm variant)
// ============================================================

// ============================================================
// PLACEHOLDER_ASM_REMOVED
// ============================================================
#if 0
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
void gdn_decode_kernel_asm_REMOVED(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NUM_V_HEADS;
    const int hv_idx = b_hv % NUM_V_HEADS;
    const int hk_idx = hv_idx / (NUM_V_HEADS / NUM_K_HEADS);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NUM_V_HEADS * shs + hv_idx * shs;

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = __bfloat162float(dt_bias[hv_idx]);

    // k_reg persists across both phases
    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * NUM_V_HEADS + sq * NUM_V_HEADS + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = 1.0f / (1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * NUM_K_HEADS * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * NUM_K_HEADS * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * NUM_V_HEADS * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        // Load k into registers (persists into Phase B)
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        // Intermediate results: dot_kq, per-vi dot products
        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        // ===== Phase A: q_reg scoped, compute all dot products =====
        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
                float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
                float ks = __frsqrt_rn(sq_k + 1e-6f);
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                int gv = global_v_base + vi * WARP_GROUP_TILE_V;
                res_rv[vi] = __bfloat162float(vb[gv]);
                float shk = 0, shq = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        shk += S[vi][ki][j] * k_reg[ki][j];
                        shq += S[vi][ki][j] * q_reg[ki][j];
                    }
                res_hk[vi] = reduce_sum_k(shk) * exp_g;
                res_hq[vi] = reduce_sum_k(shq) * exp_g;
            }
        }
        // q_reg is now out of scope → compiler frees those 16 VGPRs

        // ===== Phase B: delta rule + state update (no q needed) =====
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            float vn = (res_rv[vi] - res_hk[vi]) * beta;
            vn = shfl_idx(vn, leader);
            float ov = res_hq[vi] + vn * dot_kq;

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + k_reg[ki][j] * vn;

            if (k_tid == 0)
                output[(int64_t)(batch_idx * seq_length + sq) * NUM_V_HEADS * HEAD_V_DIM
                       + hv_idx * HEAD_V_DIM + global_v_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(ov);
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}
#endif // removed gdn_decode_kernel_asm

// ============================================================
// TUNED kernel: Maximized ILP through restructured loops
// ============================================================
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_tuned(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NUM_V_HEADS;
    const int hv_idx = b_hv % NUM_V_HEADS;
    const int hk_idx = hv_idx / (NUM_V_HEADS / NUM_K_HEADS);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NUM_V_HEADS * shs + hv_idx * shs;

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = __bfloat162float(dt_bias[hv_idx]);

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * NUM_V_HEADS + sq * NUM_V_HEADS + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * NUM_K_HEADS * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * NUM_K_HEADS * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * NUM_V_HEADS * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        // ===== Phase A: q_reg scoped, max ILP =====
        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
                float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
                float ks = __frsqrt_rn(sq_k + 1e-6f);
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            // Prefetch ALL v values before FMA loop
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);

            // All dot products with max ILP: vi as innermost loop
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = 0.0f;
                res_hq[vi] = 0.0f;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            // Interleaved reduce: all values per round
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 1);
                res_hq[vi] += shfl_xor(res_hq[vi], 1);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 2);
                res_hq[vi] += shfl_xor(res_hq[vi], 2);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 4);
                res_hq[vi] += shfl_xor(res_hq[vi], 4);
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }
        // q_reg freed

        // ===== Phase B: interleaved state update =====
        // Batch all vn computations (interleaved shfl_idx)
        float vn[V_ITERS];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            float tmp = (res_rv[vi] - res_hk[vi]) * beta;
            vn[vi] = shfl_idx(tmp, leader);
        }

        // Compute outputs, then single branch for all stores
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * NUM_V_HEADS * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        // Interleaved state update: ki,j outer, vi inner
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// ============================================================
// IASM kernel: TUNED + inline ASM for reduce/shuffle
// ============================================================
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM, int NKH, int NVH>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_iasm(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / NVH;
    const int hv_idx = b_hv % NVH;
    const int hk_idx = hv_idx / (NVH / NKH);

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader_addr = ((lane_id / WARP_K_THREADS) * WARP_K_THREADS) << 2;

    const int xor1 = (lane_id ^ 1) << 2;
    const int xor2 = (lane_id ^ 2) << 2;
    const int xor4 = (lane_id ^ 4) << 2;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * NVH * shs + hv_idx * shs;

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = __bfloat162float(dt_bias[hv_idx]);

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * NVH + sq * NVH + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * NKH * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * NVH * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                reduce_2way_asm(sq_q, sq_k, xor1, xor2, xor4);
                float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
                float ks = __frsqrt_rn(sq_k + 1e-6f);
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = 0.0f;
                res_hq[vi] = 0.0f;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            if constexpr (V_ITERS == 4) {
                reduce_8way_asm(
                    res_hk[0], res_hk[1], res_hk[2], res_hk[3],
                    res_hq[0], res_hq[1], res_hq[2], res_hq[3],
                    xor1, xor2, xor4);
            } else {
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++) {
                    res_hk[vi] += shfl_xor(res_hk[vi], 1);
                    res_hq[vi] += shfl_xor(res_hq[vi], 1);
                }
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++) {
                    res_hk[vi] += shfl_xor(res_hk[vi], 2);
                    res_hq[vi] += shfl_xor(res_hq[vi], 2);
                }
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++) {
                    res_hk[vi] += shfl_xor(res_hk[vi], 4);
                    res_hq[vi] += shfl_xor(res_hq[vi], 4);
                }
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }

        float vn[V_ITERS];
        if constexpr (V_ITERS == 4) {
            float tmp0 = (res_rv[0] - res_hk[0]) * beta;
            float tmp1 = (res_rv[1] - res_hk[1]) * beta;
            float tmp2 = (res_rv[2] - res_hk[2]) * beta;
            float tmp3 = (res_rv[3] - res_hk[3]) * beta;
            shfl_idx_4way_asm(vn[0], vn[1], vn[2], vn[3],
                              tmp0, tmp1, tmp2, tmp3, leader_addr);
        } else {
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                float tmp = (res_rv[vi] - res_hk[vi]) * beta;
                vn[vi] = shfl_idx(tmp, leader_addr >> 2);
            }
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * NVH * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

// ============================================================
// Launchers
// ============================================================
extern "C" {

#define LAUNCH(KERNEL, NVB, L2) \
   hipLaunchKernelGGL(( KERNEL<NVB, L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale)

void launch_gdn_decode(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    hipStream_t stream
) {
    dim3 grid(batch_size * NUM_V_HEADS * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const __hip_bfloat16*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);

    if (num_v_blocks == 1) {
        if (use_qk_l2norm) LAUNCH(gdn_decode_kernel_hip, 1, true);
        else               LAUNCH(gdn_decode_kernel_hip, 1, false);
    } else {
        if (use_qk_l2norm) LAUNCH(gdn_decode_kernel_hip, 2, true);
        else               LAUNCH(gdn_decode_kernel_hip, 2, false);
    }
}

void launch_gdn_decode_tuned(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    hipStream_t stream
) {
    dim3 grid(batch_size * NUM_V_HEADS * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const __hip_bfloat16*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);

    if (num_v_blocks == 1) {
        if (use_qk_l2norm) LAUNCH(gdn_decode_kernel_tuned, 1, true);
        else               LAUNCH(gdn_decode_kernel_tuned, 1, false);
    } else {
        if (use_qk_l2norm) LAUNCH(gdn_decode_kernel_tuned, 2, true);
        else               LAUNCH(gdn_decode_kernel_tuned, 2, false);
    }
}

} // close extern "C" for template dispatch

template <int NKH, int NVH>
static void dispatch_iasm(
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v,
    const __hip_bfloat16* a, const __hip_bfloat16* b, const __hip_bfloat16* dt,
    const float* al, const int* idx, float* st, __hip_bfloat16* o,
    int batch_size, int seq_length, int num_v_blocks, bool use_qk_l2norm,
    float scale, hipStream_t stream
) {
    dim3 grid(batch_size * NVH * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    #define LAUNCH_IASM(NVB, L2) \
        hipLaunchKernelGGL((gdn_decode_kernel_iasm<NVB, L2, NKH, NVH>), \
            dim3(grid), dim3(block), 0, stream, \
            q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale)
    if (num_v_blocks == 1) {
        if (use_qk_l2norm) LAUNCH_IASM(1, true);
        else               LAUNCH_IASM(1, false);
    } else {
        if (use_qk_l2norm) LAUNCH_IASM(2, true);
        else               LAUNCH_IASM(2, false);
    }
    #undef LAUNCH_IASM
}

extern "C"
void launch_gdn_decode_iasm(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const __hip_bfloat16*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);

    if (num_k_heads == 16 && num_v_heads == 32)
        dispatch_iasm<16,32>(q,k,v,a,b,dt,al,idx,st,o,batch_size,seq_length,num_v_blocks,use_qk_l2norm,scale,stream);
    else if (num_k_heads == 2 && num_v_heads == 4)
        dispatch_iasm<2,4>(q,k,v,a,b,dt,al,idx,st,o,batch_size,seq_length,num_v_blocks,use_qk_l2norm,scale,stream);
    else if (num_k_heads == 4 && num_v_heads == 8)
        dispatch_iasm<4,8>(q,k,v,a,b,dt,al,idx,st,o,batch_size,seq_length,num_v_blocks,use_qk_l2norm,scale,stream);
    else if (num_k_heads == 8 && num_v_heads == 16)
        dispatch_iasm<8,16>(q,k,v,a,b,dt,al,idx,st,o,batch_size,seq_length,num_v_blocks,use_qk_l2norm,scale,stream);
}

#undef LAUNCH

// ============================================================
// Sglang-compatible TUNED kernel: state layout [pool, HV, K, V]
// (Triton/sglang convention: K outer, V inner)
// ============================================================
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_tuned_kv(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const float* __restrict__          dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale,
    int num_k_heads, int num_v_heads
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    const int gva_ratio = num_v_heads / num_k_heads;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / num_v_heads;
    const int hv_idx = b_hv % num_v_heads;
    const int hk_idx = hv_idx / gva_ratio;

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_K_DIM * HEAD_V_DIM;
    float* sb = state + (int64_t)pool_idx * num_v_heads * shs + hv_idx * shs;

    // State load: [K, V] layout — sb[k_idx * HEAD_V_DIM + v_idx]
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                S[vi][ki][j] = sb[(warp_k_vec_start + ki * WARP_TILE_K + j) * HEAD_V_DIM + gv];
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = dt_bias[hv_idx];

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * num_v_heads + sq * num_v_heads + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
                float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
                float ks = __frsqrt_rn(sq_k + 1e-6f);
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = 0.0f;
                res_hq[vi] = 0.0f;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 1);
                res_hq[vi] += shfl_xor(res_hq[vi], 1);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 2);
                res_hq[vi] += shfl_xor(res_hq[vi], 2);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 4);
                res_hq[vi] += shfl_xor(res_hq[vi], 4);
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }

        float vn[V_ITERS];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            float tmp = (res_rv[vi] - res_hk[vi]) * beta;
            vn[vi] = shfl_idx(tmp, leader);
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    // State store: [K, V] layout
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                sb[(warp_k_vec_start + ki * WARP_TILE_K + j) * HEAD_V_DIM + gv] = S[vi][ki][j];
    }
}

extern "C" {

#define LAUNCH_KV(KERNEL, NVB, L2) \
   hipLaunchKernelGGL(( KERNEL<NVB, L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale, nkh, nvh)

void launch_gdn_decode_tuned_kv(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const float*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);
    int nkh = num_k_heads, nvh = num_v_heads;

    if (num_v_blocks == 1) {
        if (use_qk_l2norm) LAUNCH_KV(gdn_decode_kernel_tuned_kv, 1, true);
        else               LAUNCH_KV(gdn_decode_kernel_tuned_kv, 1, false);
    } else {
        if (use_qk_l2norm) LAUNCH_KV(gdn_decode_kernel_tuned_kv, 2, true);
        else               LAUNCH_KV(gdn_decode_kernel_tuned_kv, 2, false);
    }
}

#undef LAUNCH_KV

} // extern "C"

// ============================================================
// TUNED kernel with [V, K] state layout (float4 vectorized) +
// runtime head counts + float32 dt_bias.
// For sglang integration: caller transposes state [K,V]→[V,K]
// before call and [V,K]→[K,V] after call.
// ============================================================
template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_tuned_vk(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const __hip_bfloat16* __restrict__ dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale,
    int num_k_heads, int num_v_heads
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    const int gva_ratio = num_v_heads / num_k_heads;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / num_v_heads;
    const int hv_idx = b_hv % num_v_heads;
    const int hk_idx = hv_idx / gva_ratio;

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = v_block_id * TILE_V + warp_id * WARP_V_THREADS + v_tid;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    float S[V_ITERS][K_ITERS][VEC_K];
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * num_v_heads * shs + hv_idx * shs;

    // State load: [V, K] layout with float4 vectorized access
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = *reinterpret_cast<const float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = __bfloat162float(dt_bias[hv_idx]);

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * num_v_heads + sq * num_v_heads + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
                float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
                float ks = __frsqrt_rn(sq_k + 1e-6f);
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = 0.0f;
                res_hq[vi] = 0.0f;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 1);
                res_hq[vi] += shfl_xor(res_hq[vi], 1);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 2);
                res_hq[vi] += shfl_xor(res_hq[vi], 2);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 4);
                res_hq[vi] += shfl_xor(res_hq[vi], 4);
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }

        float vn[V_ITERS];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            float tmp = (res_rv[vi] - res_hk[vi]) * beta;
            vn[vi] = shfl_idx(tmp, leader);
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    // State store: [V, K] layout with float4
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + gv * HEAD_K_DIM + warp_k_vec_start + ki * WARP_TILE_K) = t;
        }
    }
}

extern "C" {

#define LAUNCH_VK(KERNEL, NVB, L2) \
   hipLaunchKernelGGL(( KERNEL<NVB, L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale, nkh, nvh)

void launch_gdn_decode_tuned_vk(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const __hip_bfloat16*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);
    int nkh = num_k_heads, nvh = num_v_heads;

    if (num_v_blocks == 1) {
        if (use_qk_l2norm) LAUNCH_VK(gdn_decode_kernel_tuned_vk, 1, true);
        else               LAUNCH_VK(gdn_decode_kernel_tuned_vk, 1, false);
    } else {
        if (use_qk_l2norm) LAUNCH_VK(gdn_decode_kernel_tuned_vk, 2, true);
        else               LAUNCH_VK(gdn_decode_kernel_tuned_vk, 2, false);
    }
}

#undef LAUNCH_VK

} // extern "C"

// ============================================================
// Fast in-place 128×128 float32 matrix transpose.
// One block per (matrix, row). Each block handles swaps for one
// row: swap mat[row][col] with mat[col][row] for col > row.
// 512 blocks for BS=1 HV=4 → good GPU utilization.
// ============================================================
constexpr int TRANS_THREADS = 128;

__global__
void state_transpose_128x128_kernel(
    float* __restrict__ state,
    const int* __restrict__ indices,
    int batch_size,
    int num_v_heads
) {
    constexpr int N = 128;
    const int global_idx = blockIdx.x;
    const int mat_idx = global_idx / N;
    const int row = global_idx % N;
    const int batch_idx = mat_idx / num_v_heads;
    const int hv_idx = mat_idx % num_v_heads;
    if (batch_idx >= batch_size) return;

    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    float* mat = state + (int64_t)pool_idx * num_v_heads * N * N + (int64_t)hv_idx * N * N;

    for (int col = row + 1 + threadIdx.x; col < N; col += TRANS_THREADS) {
        int idx1 = row * N + col;
        int idx2 = col * N + row;
        float tmp = mat[idx1];
        mat[idx1] = mat[idx2];
        mat[idx2] = tmp;
    }
}

extern "C" {

void launch_state_transpose(
    void* state, const void* indices,
    int batch_size, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads * 128);
    dim3 block(TRANS_THREADS);
    hipLaunchKernelGGL(state_transpose_128x128_kernel,
        grid, block, 0, stream,
        reinterpret_cast<float*>(state),
        reinterpret_cast<const int*>(indices),
        batch_size, num_v_heads);
}

} // extern "C" (transpose)

// ============================================================
// KV4 kernel: works directly on [K, V] state layout (sglang).
// float4 vectorized loads along V dimension (V is innermost
// in [K,V]). No transpose needed.
// Thread mapping: WARP_K_THREADS=8 (scalar K), WARP_V_THREADS=8
// Each V-thread loads VEC_V=4 consecutive V values via float4.
// ============================================================
constexpr int KV4_VEC_V = 4;
constexpr int KV4_WARP_TILE_V = WARP_V_THREADS * KV4_VEC_V;  // 8*4=32
constexpr int KV4_WARP_TILE_K = WARP_K_THREADS;               // 8 (scalar)
constexpr int KV4_K_ITERS = HEAD_K_DIM / KV4_WARP_TILE_K;     // 128/8=16

template <int NUM_V_BLOCKS, bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 3)
__attribute__((amdgpu_waves_per_eu(3, 5)))
void gdn_decode_kernel_kv4(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const float* __restrict__          dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale,
    int num_k_heads, int num_v_heads
) {
    constexpr int TILE_V = HEAD_V_DIM / NUM_V_BLOCKS;
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * KV4_WARP_TILE_V;
    constexpr int V_ITERS = TILE_V / WARP_GROUP_TILE_V;
    const int gva_ratio = num_v_heads / num_k_heads;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv_vb = blockIdx.x;
    const int v_block_id = b_hv_vb % NUM_V_BLOCKS;
    const int b_hv = b_hv_vb / NUM_V_BLOCKS;
    const int batch_idx = b_hv / num_v_heads;
    const int hv_idx = b_hv % num_v_heads;
    const int hk_idx = hv_idx / gva_ratio;

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int v_base = v_block_id * TILE_V + warp_id * KV4_WARP_TILE_V + v_tid * KV4_VEC_V;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb = state + (int64_t)pool_idx * num_v_heads * shs + hv_idx * shs;

    float S[V_ITERS][KV4_K_ITERS][KV4_VEC_V];

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            int k_pos = k_tid + ki * KV4_WARP_TILE_K;
            float4 t = *reinterpret_cast<const float4*>(sb + k_pos * HEAD_V_DIM + gv);
            S[vi][ki][0] = t.x; S[vi][ki][1] = t.y; S[vi][ki][2] = t.z; S[vi][ki][3] = t.w;
        }
    }

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = dt_bias[hv_idx];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * num_v_heads + sq * num_v_heads + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        float q_reg[KV4_K_ITERS], k_reg[KV4_K_ITERS];
        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            int k_pos = k_tid + ki * KV4_WARP_TILE_K;
            q_reg[ki] = __bfloat162float(qb[k_pos]);
            k_reg[ki] = __bfloat162float(kb[k_pos]);
        }

        if constexpr (USE_QK_L2NORM) {
            float sq_q = 0, sq_k = 0;
            #pragma unroll
            for (int ki = 0; ki < KV4_K_ITERS; ki++) {
                sq_q += q_reg[ki] * q_reg[ki];
                sq_k += k_reg[ki] * k_reg[ki];
            }
            sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
            float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
            float ks = __frsqrt_rn(sq_k + 1e-6f);
            #pragma unroll
            for (int ki = 0; ki < KV4_K_ITERS; ki++) {
                q_reg[ki] *= qs;
                k_reg[ki] *= ks;
            }
        } else {
            #pragma unroll
            for (int ki = 0; ki < KV4_K_ITERS; ki++)
                q_reg[ki] *= scale;
        }

        float dot_kq = 0;
        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++)
            dot_kq += q_reg[ki] * k_reg[ki];
        dot_kq = reduce_sum_k(dot_kq);

        float res_rv[V_ITERS][KV4_VEC_V];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            int gv = v_base + vi * WARP_GROUP_TILE_V;
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++)
                res_rv[vi][jv] = __bfloat162float(vb[gv + jv]);
        }

        float res_hk[V_ITERS][KV4_VEC_V];
        float res_hq[V_ITERS][KV4_VEC_V];

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] = 0.0f;
                res_hq[vi][jv] = 0.0f;
            }

        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            float kv = k_reg[ki];
            float qv = q_reg[ki];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                #pragma unroll
                for (int jv = 0; jv < KV4_VEC_V; jv++) {
                    res_hk[vi][jv] += S[vi][ki][jv] * kv;
                    res_hq[vi][jv] += S[vi][ki][jv] * qv;
                }
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] += shfl_xor(res_hk[vi][jv], 1);
                res_hq[vi][jv] += shfl_xor(res_hq[vi][jv], 1);
            }
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] += shfl_xor(res_hk[vi][jv], 2);
                res_hq[vi][jv] += shfl_xor(res_hq[vi][jv], 2);
            }
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] += shfl_xor(res_hk[vi][jv], 4);
                res_hq[vi][jv] += shfl_xor(res_hq[vi][jv], 4);
            }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                res_hk[vi][jv] *= exp_g;
                res_hq[vi][jv] *= exp_g;
            }

        float vn[V_ITERS][KV4_VEC_V];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++) {
                float tmp = (res_rv[vi][jv] - res_hk[vi][jv]) * beta;
                vn[vi][jv] = shfl_idx(tmp, leader);
            }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            #pragma unroll
            for (int jv = 0; jv < KV4_VEC_V; jv++)
                res_hq[vi][jv] = res_hq[vi][jv] + vn[vi][jv] * dot_kq;

        if (k_tid == 0) {
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                int gv = v_base + vi * WARP_GROUP_TILE_V;
                const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM
                                         + hv_idx * HEAD_V_DIM + gv;
                #pragma unroll
                for (int jv = 0; jv < KV4_VEC_V; jv++)
                    output[out_base + jv] = __float2bfloat16(res_hq[vi][jv]);
            }
        }

        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            float kv = k_reg[ki];
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                #pragma unroll
                for (int jv = 0; jv < KV4_VEC_V; jv++)
                    S[vi][ki][jv] = S[vi][ki][jv] * exp_g + kv * vn[vi][jv];
        }
    }

    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < KV4_K_ITERS; ki++) {
            int k_pos = k_tid + ki * KV4_WARP_TILE_K;
            float4 t = {S[vi][ki][0], S[vi][ki][1], S[vi][ki][2], S[vi][ki][3]};
            *reinterpret_cast<float4*>(sb + k_pos * HEAD_V_DIM + gv) = t;
        }
    }
}

extern "C" {

#define LAUNCH_KV4(KERNEL, NVB, L2) \
   hipLaunchKernelGGL(( KERNEL<NVB, L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale, nkh, nvh)

void launch_gdn_decode_kv4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads * num_v_blocks);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const float*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);
    int nkh = num_k_heads, nvh = num_v_heads;

    (void)num_v_blocks;
    if (use_qk_l2norm) LAUNCH_KV4(gdn_decode_kernel_kv4, 1, true);
    else               LAUNCH_KV4(gdn_decode_kernel_kv4, 1, false);
}

#undef LAUNCH_KV4

} // extern "C"

// ============================================================
// FUSED kernel: reads [K,V] state, transposes in LDS, computes
// using VK-style float4-along-K, writes back [K,V].
// Uses 64KB LDS (full MI300X budget). 1 block per CU max.
// 256 threads load 64KB cooperatively via float4.
// ============================================================
constexpr int LDS_K = HEAD_K_DIM;     // 128
constexpr int LDS_V = HEAD_V_DIM;     // 128
constexpr int LDS_STRIDE = LDS_K;     // 128 floats per V-row (no pad, exactly 64KB)

template <bool USE_QK_L2NORM>
__global__
__launch_bounds__(BLOCK_THREADS, 1)
void gdn_decode_kernel_fused(
    const __hip_bfloat16* __restrict__ query,
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    const __hip_bfloat16* __restrict__ a_input,
    const __hip_bfloat16* __restrict__ b_input,
    const float* __restrict__          dt_bias,
    const float* __restrict__          A_log,
    const int* __restrict__            indices,
    float* __restrict__                state,
    __hip_bfloat16* __restrict__       output,
    int batch_size, int seq_length, float scale,
    int num_k_heads, int num_v_heads
) {
    constexpr int WARP_GROUP_TILE_V = NUM_WARPS * WARP_V_THREADS;  // 4*8=32
    constexpr int V_ITERS = HEAD_V_DIM / WARP_GROUP_TILE_V;       // 128/32=4
    const int gva_ratio = num_v_heads / num_k_heads;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int k_tid = lane_id % WARP_K_THREADS;
    const int v_tid = lane_id / WARP_K_THREADS;

    const int b_hv = blockIdx.x;
    const int batch_idx = b_hv / num_v_heads;
    const int hv_idx = b_hv % num_v_heads;
    const int hk_idx = hv_idx / gva_ratio;

    if (batch_idx >= batch_size) return;
    const int pool_idx = indices[batch_idx];
    if (pool_idx < 0) return;

    const int warp_k_vec_start = k_tid * VEC_K;
    const int global_v_base = warp_id * WARP_V_THREADS + v_tid;
    const int leader = (lane_id / WARP_K_THREADS) * WARP_K_THREADS;

    // Global state pointer: [pool, HV, K, V] layout
    const int shs = HEAD_V_DIM * HEAD_K_DIM;
    float* sb_global = state + (int64_t)pool_idx * num_v_heads * shs + hv_idx * shs;

    // LDS: [V, K+pad] — transposed from global [K, V]
    // 128 * 129 * 4 = 66,048 bytes ≈ 64.5KB
    __shared__ float lds[LDS_V][LDS_STRIDE];

    // Cooperative load: global [K,V] → LDS [V, K]
    // Each thread loads multiple elements
    // Global: row k, col v → lds[v][k]
    constexpr int TOTAL_ELEMS = LDS_K * LDS_V;  // 16384
    constexpr int ELEMS_PER_THREAD = (TOTAL_ELEMS + BLOCK_THREADS - 1) / BLOCK_THREADS;  // 64

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int elem = tid + i * BLOCK_THREADS;
        if (elem < TOTAL_ELEMS) {
            int gk = elem / LDS_V;  // global K index
            int gv = elem % LDS_V;  // global V index
            lds[gv][gk] = sb_global[gk * HEAD_V_DIM + gv];
        }
    }
    __syncthreads();

    // Load from LDS in [V, K] order with float4 along K (like VK kernel)
    float S[V_ITERS][K_ITERS][VEC_K];
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int k_base = warp_k_vec_start + ki * WARP_TILE_K;
            S[vi][ki][0] = lds[gv][k_base];
            S[vi][ki][1] = lds[gv][k_base + 1];
            S[vi][ki][2] = lds[gv][k_base + 2];
            S[vi][ki][3] = lds[gv][k_base + 3];
        }
    }
    // No more LDS needed for state loading; barrier before any writes
    __syncthreads();

    const float r_A_log = A_log[hv_idx];
    const float r_dt_bias = dt_bias[hv_idx];

    float k_reg[K_ITERS][VEC_K];

    for (int sq = 0; sq < seq_length; sq++) {
        const int ab = batch_idx * seq_length * num_v_heads + sq * num_v_heads + hv_idx;
        float x = __bfloat162float(a_input[ab]) + r_dt_bias;
        float sp = (x <= 20.0f) ? __logf(1.0f + __expf(x)) : x;
        float exp_g = __expf(-__expf(r_A_log) * sp);
        float beta = __frcp_rn(1.0f + __expf(-__bfloat162float(b_input[ab])));

        const auto* qb = query + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* kb = key + (int64_t)(batch_idx * seq_length + sq) * num_k_heads * HEAD_K_DIM + hk_idx * HEAD_K_DIM;
        const auto* vb = value + (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM + hv_idx * HEAD_V_DIM;

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int ko = warp_k_vec_start + ki * WARP_TILE_K;
            #pragma unroll
            for (int j = 0; j < VEC_K; j++)
                k_reg[ki][j] = __bfloat162float(kb[ko + j]);
        }

        float dot_kq = 0;
        float res_hk[V_ITERS], res_hq[V_ITERS], res_rv[V_ITERS];

        {
            float q_reg[K_ITERS][VEC_K];
            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++) {
                int ko = warp_k_vec_start + ki * WARP_TILE_K;
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    q_reg[ki][j] = __bfloat162float(qb[ko + j]);
            }

            if constexpr (USE_QK_L2NORM) {
                float sq_q = 0, sq_k = 0;
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        sq_q += q_reg[ki][j] * q_reg[ki][j];
                        sq_k += k_reg[ki][j] * k_reg[ki][j];
                    }
                sq_q = reduce_sum_k(sq_q); sq_k = reduce_sum_k(sq_k);
                float qs = scale * __frsqrt_rn(sq_q + 1e-6f);
                float ks = __frsqrt_rn(sq_k + 1e-6f);
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++) {
                        q_reg[ki][j] *= qs;
                        k_reg[ki][j] *= ks;
                    }
            } else {
                #pragma unroll
                for (int ki = 0; ki < K_ITERS; ki++)
                    #pragma unroll
                    for (int j = 0; j < VEC_K; j++)
                        q_reg[ki][j] *= scale;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++)
                    dot_kq += k_reg[ki][j] * q_reg[ki][j];
            dot_kq = reduce_sum_k(dot_kq);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                res_rv[vi] = __bfloat162float(vb[global_v_base + vi * WARP_GROUP_TILE_V]);

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] = 0.0f;
                res_hq[vi] = 0.0f;
            }

            #pragma unroll
            for (int ki = 0; ki < K_ITERS; ki++)
                #pragma unroll
                for (int j = 0; j < VEC_K; j++) {
                    float kv = k_reg[ki][j];
                    float qv = q_reg[ki][j];
                    #pragma unroll
                    for (int vi = 0; vi < V_ITERS; vi++) {
                        res_hk[vi] += S[vi][ki][j] * kv;
                        res_hq[vi] += S[vi][ki][j] * qv;
                    }
                }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 1);
                res_hq[vi] += shfl_xor(res_hq[vi], 1);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 2);
                res_hq[vi] += shfl_xor(res_hq[vi], 2);
            }
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] += shfl_xor(res_hk[vi], 4);
                res_hq[vi] += shfl_xor(res_hq[vi], 4);
            }

            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++) {
                res_hk[vi] *= exp_g;
                res_hq[vi] *= exp_g;
            }
        }

        float vn[V_ITERS];
        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++) {
            float tmp = (res_rv[vi] - res_hk[vi]) * beta;
            vn[vi] = shfl_idx(tmp, leader);
        }

        #pragma unroll
        for (int vi = 0; vi < V_ITERS; vi++)
            res_hq[vi] = res_hq[vi] + vn[vi] * dot_kq;

        if (k_tid == 0) {
            const int64_t out_base = (int64_t)(batch_idx * seq_length + sq) * num_v_heads * HEAD_V_DIM
                                     + hv_idx * HEAD_V_DIM + global_v_base;
            #pragma unroll
            for (int vi = 0; vi < V_ITERS; vi++)
                output[out_base + vi * WARP_GROUP_TILE_V] = __float2bfloat16(res_hq[vi]);
        }

        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++)
            #pragma unroll
            for (int j = 0; j < VEC_K; j++) {
                float kv = k_reg[ki][j];
                #pragma unroll
                for (int vi = 0; vi < V_ITERS; vi++)
                    S[vi][ki][j] = S[vi][ki][j] * exp_g + kv * vn[vi];
            }
    }

    // State store: registers → LDS [V, K] → global [K, V]
    #pragma unroll
    for (int vi = 0; vi < V_ITERS; vi++) {
        int gv = global_v_base + vi * WARP_GROUP_TILE_V;
        #pragma unroll
        for (int ki = 0; ki < K_ITERS; ki++) {
            int k_base = warp_k_vec_start + ki * WARP_TILE_K;
            lds[gv][k_base]     = S[vi][ki][0];
            lds[gv][k_base + 1] = S[vi][ki][1];
            lds[gv][k_base + 2] = S[vi][ki][2];
            lds[gv][k_base + 3] = S[vi][ki][3];
        }
    }
    __syncthreads();

    // Cooperative store: LDS [V, K] → global [K, V]
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int elem = tid + i * BLOCK_THREADS;
        if (elem < TOTAL_ELEMS) {
            int gk = elem / LDS_V;
            int gv = elem % LDS_V;
            sb_global[gk * HEAD_V_DIM + gv] = lds[gv][gk];
        }
    }
}

extern "C" {

#define LAUNCH_FUSED(KERNEL, L2) \
   hipLaunchKernelGGL(( KERNEL<L2>), dim3(grid), dim3(block), 0, stream,  \
        q, k, v, a, b, dt, al, idx, st, o, batch_size, seq_length, scale, nkh, nvh)

void launch_gdn_decode_fused(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
) {
    dim3 grid(batch_size * num_v_heads);
    dim3 block(BLOCK_THREADS);
    auto q = reinterpret_cast<const __hip_bfloat16*>(query);
    auto k = reinterpret_cast<const __hip_bfloat16*>(key);
    auto v = reinterpret_cast<const __hip_bfloat16*>(value);
    auto a = reinterpret_cast<const __hip_bfloat16*>(a_input);
    auto b = reinterpret_cast<const __hip_bfloat16*>(b_input);
    auto dt = reinterpret_cast<const float*>(dt_bias);
    auto al = reinterpret_cast<const float*>(A_log);
    auto idx = reinterpret_cast<const int*>(indices);
    auto st = reinterpret_cast<float*>(state);
    auto o = reinterpret_cast<__hip_bfloat16*>(output);
    int nkh = num_k_heads, nvh = num_v_heads;

    if (use_qk_l2norm) LAUNCH_FUSED(gdn_decode_kernel_fused, true);
    else               LAUNCH_FUSED(gdn_decode_kernel_fused, false);
}

#undef LAUNCH_FUSED

} // extern "C"
