#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <cstdlib>

static int gdn_sort_threshold() {
    // BS=128 with sorted path under-saturates HBM (~1.62 TB/s -> 79us).
    // The unsorted path at BS=128 hits ~2.16 TB/s -> 59us (-25% kernel time, +30% vs FlyDSL).
    // Sort still helps at BS=256 (97us vs 103us). Setting threshold=192 keeps sort only for
    // BS in [192, +inf), so BS=128 uses the faster unsorted kernel and BS=256 keeps sort.
    static int t = []() {
        const char* e = std::getenv("HIP_GDN_SORT_IDX_BS");
        return e ? std::atoi(e) : 192;
    }();
    return t;
}

// Cache sorted indices + permutation across layers in a decode step.
// All GDN layers in one step share the same indices tensor, so sorting
// once and reusing is a ~64x reduction in sort overhead.
static struct {
    const void* last_ptr = nullptr;
    int last_bs = 0;
    torch::Tensor sorted_indices;
    torch::Tensor perm_i32;
} sort_cache;

extern "C" {
void launch_gdn_decode_iasm(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    const int* batch_perm,
    hipStream_t stream
);

void launch_gdn_decode_iasm_generic(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);

void launch_gdn_decode_tuned(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    hipStream_t stream
);

void launch_gdn_decode_tuned_kv(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);

void launch_gdn_decode_tuned_vk(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);

void launch_state_transpose(
    void* state, const void* indices,
    int batch_size, int num_v_heads,
    hipStream_t stream
);

void launch_gdn_decode_kv4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb2_lb1(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb2_lb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb2_lb3(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb2_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb2_lb8(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb2_lb12(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb1(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb1_lb1(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb1_lb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb1_lb4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb1_lb2w2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb1_lb12(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_lds(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb3(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb2_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb2_opt_lb5(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb2_fixedbs(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb3_fixedbs(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_fixedbs(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb6_fixedbs(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_vm(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_fuse(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_vm_fuse(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb3_vm(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_dot2_opt(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_dot2_nvb4_opt(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_dot2_nvb4_opt_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_streaming(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs1_hyper(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs1_hyper_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs1_kpipe(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs1_extreme(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs1_extreme_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs1_ultra(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs1_ultra_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_opt(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb2_opt(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_builtin_nvh8_nvb2_v64_lb8(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_builtin_nvh8_nvb2_v80_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb4_opt(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb4_memopt(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs2_nvb4_lb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_nvb4_lb3(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_nvb4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_nvb4_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_nvb2_lb3(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_nvb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs2_extreme(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_extreme_lb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_extreme_lb4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_extreme_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_extreme_nvb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs2_extreme_nvb2_lb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_bs16_extreme(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_extreme_lb1(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_extreme_lb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_extreme_lb3(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);
void launch_gdn_decode_iasm_nvh8_bs16_extreme_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb4_opt_lb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb4_opt_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb4_opt_lb6_fusepost(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_builtin_nvh8_nvb4_v80_lb6_fusestore(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_builtin_nvh8_nvb4_v64_lb8_fusestore(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb4_opt_lb6_vmem(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb4_opt_lb4_vmem(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb4_opt_lb10v48_vmem(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_iasm_nvh8_nvb4_sched_hint(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_fused(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, int seq_length,
    bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads,
    hipStream_t stream
);

// DPP-based kernel launchers (use VALU DPP instead of ds_bpermute)
void launch_gdn_decode_dpp_nvh8_nvb4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_dpp_nvh8_nvb4_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_dpp_nvh8_nvb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

void launch_gdn_decode_dpp_nvh8_nvb1(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    int batch_size, float scale,
    hipStream_t stream
);

// EXTREME BS=1 kernel launchers
void launch_gdn_decode_extreme_bs1_nvb4_hi_occ(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_extreme_bs1_nvb4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_extreme_bs1_nvb4_lb6(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_extreme_bs1_nvb4_lb4(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);

void launch_gdn_decode_extreme_bs1_nvb4_lb2(
    const void* query, const void* key, const void* value,
    const void* a_input, const void* b_input, const void* dt_bias,
    const void* A_log, const void* indices,
    void* state, void* output,
    float scale,
    hipStream_t stream
);
}

void hip_gdn_decode_tuned_inplace(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor A_log, torch::Tensor indices,
    torch::Tensor state, torch::Tensor output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale
) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    launch_gdn_decode_tuned(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
        A_log.data_ptr(), indices.data_ptr(),
        state.data_ptr(), output.data_ptr(),
        batch_size, seq_length,
        num_v_blocks, use_qk_l2norm, scale,
        stream
    );
}

void hip_gdn_decode_tuned_kv_inplace(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor A_log, torch::Tensor indices,
    torch::Tensor state, torch::Tensor output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads
) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    launch_gdn_decode_tuned_kv(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
        A_log.data_ptr(), indices.data_ptr(),
        state.data_ptr(), output.data_ptr(),
        batch_size, seq_length,
        num_v_blocks, use_qk_l2norm, scale,
        num_k_heads, num_v_heads,
        stream
    );
}

void hip_gdn_decode_tuned_vk_inplace(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
    torch::Tensor A_log, torch::Tensor indices,
    torch::Tensor state, torch::Tensor output,
    int batch_size, int seq_length,
    int num_v_blocks, bool use_qk_l2norm, float scale,
    int num_k_heads, int num_v_heads
) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    launch_gdn_decode_tuned_vk(
        query.data_ptr(), key.data_ptr(), value.data_ptr(),
        a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
        A_log.data_ptr(), indices.data_ptr(),
        state.data_ptr(), output.data_ptr(),
        batch_size, seq_length,
        num_v_blocks, use_qk_l2norm, scale,
        num_k_heads, num_v_heads,
        stream
    );
}

void hip_state_transpose(
    torch::Tensor state, torch::Tensor indices,
    int batch_size, int num_v_heads
) {
    auto stream = at::hip::getCurrentHIPStream().stream();
    launch_state_transpose(
        state.data_ptr(), indices.data_ptr(),
        batch_size, num_v_heads, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hip_gdn_decode_tuned_inplace", &hip_gdn_decode_tuned_inplace,
          "GDN decode TUNED kernel (state layout: [pool, HV, V, K])");
    m.def("hip_gdn_decode_tuned_kv_inplace", &hip_gdn_decode_tuned_kv_inplace,
          "GDN decode TUNED kernel (state layout: [pool, HV, K, V] - sglang compatible)");
    m.def("hip_gdn_decode_tuned_vk_inplace", &hip_gdn_decode_tuned_vk_inplace,
          "GDN decode TUNED kernel (state layout: [pool, HV, V, K] with runtime heads)");
    m.def("hip_state_transpose", &hip_state_transpose,
          "In-place 128x128 state transpose [K,V] <-> [V,K]");
    m.def("hip_gdn_decode_vk_auto_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        int num_v_blocks, bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_state_transpose(state.data_ptr(), indices.data_ptr(),
                               batch_size, num_v_heads, stream);
        launch_gdn_decode_tuned_vk(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            num_v_blocks, use_qk_l2norm, scale,
            num_k_heads, num_v_heads, stream);
        launch_state_transpose(state.data_ptr(), indices.data_ptr(),
                               batch_size, num_v_heads, stream);
    }, "GDN decode VK kernel with auto state transpose (single C++ call)");
    m.def("hip_gdn_decode_fused_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_fused(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            use_qk_l2norm, scale,
            num_k_heads, num_v_heads, stream);
    }, "GDN decode FUSED kernel (LDS transpose, [K,V] state, single kernel)");
    m.def("hip_gdn_decode_asm_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        int num_v_blocks, bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        auto err_before = hipGetLastError();

        const int* batch_perm_ptr = nullptr;
        torch::Tensor sorted_indices, perm_i32;
        int sort_thr = gdn_sort_threshold();
        if (sort_thr > 0 && batch_size >= sort_thr &&
            num_k_heads == 2 && num_v_heads == 8) {
            const void* cur_ptr = indices.data_ptr();
            if (sort_cache.last_ptr == cur_ptr &&
                sort_cache.last_bs == batch_size) {
                sorted_indices = sort_cache.sorted_indices;
                perm_i32 = sort_cache.perm_i32;
            } else {
                auto perm_i64 = torch::argsort(indices, /*dim=*/int64_t(0), /*descending=*/false);
                sorted_indices = indices.index_select(0, perm_i64);
                perm_i32 = perm_i64.to(torch::kInt32);
                sort_cache.last_ptr = cur_ptr;
                sort_cache.last_bs = batch_size;
                sort_cache.sorted_indices = sorted_indices;
                sort_cache.perm_i32 = perm_i32;
            }
            batch_perm_ptr = perm_i32.data_ptr<int>();
        }

        launch_gdn_decode_iasm(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(),
            batch_perm_ptr ? sorted_indices.data_ptr() : indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            num_v_blocks, use_qk_l2norm, scale,
            num_k_heads, num_v_heads,
            batch_perm_ptr, stream);
        auto err = hipGetLastError();
        TORCH_CHECK(err == hipSuccess,
            "hip_gdn_decode_asm_inplace launch failed: ", hipGetErrorString(err),
            " (BS=", batch_size, " SQ=", seq_length, " NKH=", num_k_heads,
            " NVH=", num_v_heads, " NVB=", num_v_blocks,
            " l2norm=", use_qk_l2norm, " dt_dtype=", dt_bias.dtype(),
            " q_dtype=", query.dtype(), " state_dtype=", state.dtype(), ")");
    }, "GDN decode ASM kernel (inline asm reduces, state [V,K], template-specialized heads)");
    m.def("hip_gdn_decode_asm_generic_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        int num_v_blocks, bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_iasm_generic(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            num_v_blocks, use_qk_l2norm, scale,
            num_k_heads, num_v_heads, stream);
    }, "GDN decode ASM generic fallback kernel (pre-extreme dispatch)");
    // NKH=2 NVH=8 per-variant entry points for benchmark A/B testing
    #define DEF_NVH8_VARIANT(NAME, LAUNCH_FN) \
    m.def(NAME, []( \
        torch::Tensor query, torch::Tensor key, torch::Tensor value, \
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias, \
        torch::Tensor A_log, torch::Tensor indices, \
        torch::Tensor state, torch::Tensor output, \
        int batch_size, float scale \
    ) { \
        auto stream = at::hip::getCurrentHIPStream().stream(); \
        LAUNCH_FN( \
            query.data_ptr(), key.data_ptr(), value.data_ptr(), \
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(), \
            A_log.data_ptr(), indices.data_ptr(), \
            state.data_ptr(), output.data_ptr(), \
            batch_size, scale, stream); \
    })

    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb2", launch_gdn_decode_iasm_nvh8_nvb2);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb2_lb1", launch_gdn_decode_iasm_nvh8_nvb2_lb1);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb2_lb2", launch_gdn_decode_iasm_nvh8_nvb2_lb2);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb2_lb3", launch_gdn_decode_iasm_nvh8_nvb2_lb3);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb2_lb6", launch_gdn_decode_iasm_nvh8_nvb2_lb6);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb2_lb8", launch_gdn_decode_iasm_nvh8_nvb2_lb8);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb2_lb12", launch_gdn_decode_iasm_nvh8_nvb2_lb12);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1", launch_gdn_decode_iasm_nvh8_nvb1);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1_lb1", launch_gdn_decode_iasm_nvh8_nvb1_lb1);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1_lb2", launch_gdn_decode_iasm_nvh8_nvb1_lb2);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1_lb4", launch_gdn_decode_iasm_nvh8_nvb1_lb4);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1_lb2w2", launch_gdn_decode_iasm_nvh8_nvb1_lb2w2);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1_lb12", launch_gdn_decode_iasm_nvh8_nvb1_lb12);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb4_lds", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_lds);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb3", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb3);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb2_lb6", launch_gdn_decode_iasm_nvh8_bs16_nvb2_lb6);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb2_opt_lb5", launch_gdn_decode_iasm_nvh8_bs16_nvb2_opt_lb5);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb2_fixedbs", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb2_fixedbs);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb3_fixedbs", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb3_fixedbs);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb4_fixedbs", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_fixedbs);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb6_fixedbs", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb6_fixedbs);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb4_vm", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_vm);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb4_fuse", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_fuse);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb4_vm_fuse", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb4_vm_fuse);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_nvb1_lb3_vm", launch_gdn_decode_iasm_nvh8_bs16_nvb1_lb3_vm);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_dot2_opt", launch_gdn_decode_iasm_nvh8_dot2_opt);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_dot2_nvb4_opt", launch_gdn_decode_iasm_nvh8_dot2_nvb4_opt);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_dot2_nvb4_opt_lb6", launch_gdn_decode_iasm_nvh8_dot2_nvb4_opt_lb6);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_streaming", launch_gdn_decode_iasm_nvh8_streaming);
    // Optimized variants: latency hiding
    DEF_NVH8_VARIANT("hip_gdn_nvh8_opt", launch_gdn_decode_iasm_nvh8_opt);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb2_opt", launch_gdn_decode_iasm_nvh8_nvb2_opt);
    DEF_NVH8_VARIANT("hip_gdn_builtin_nvh8_nvb2_v64_lb8", launch_gdn_decode_builtin_nvh8_nvb2_v64_lb8);
    DEF_NVH8_VARIANT("hip_gdn_builtin_nvh8_nvb2_v80_lb6", launch_gdn_decode_builtin_nvh8_nvb2_v80_lb6);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb4_opt", launch_gdn_decode_iasm_nvh8_nvb4_opt);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb4_memopt", launch_gdn_decode_iasm_nvh8_nvb4_memopt);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb4_opt_lb2", launch_gdn_decode_iasm_nvh8_nvb4_opt_lb2);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb4_opt_lb6", launch_gdn_decode_iasm_nvh8_nvb4_opt_lb6);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb4_opt_lb6_fusepost", launch_gdn_decode_iasm_nvh8_nvb4_opt_lb6_fusepost);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_builtin_nvb4_v80_lb6_fusestore", launch_gdn_decode_builtin_nvh8_nvb4_v80_lb6_fusestore);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_builtin_nvb4_v64_lb8_fusestore", launch_gdn_decode_builtin_nvh8_nvb4_v64_lb8_fusestore);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb4_opt_lb6_vmem", launch_gdn_decode_iasm_nvh8_nvb4_opt_lb6_vmem);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb4_opt_lb4_vmem", launch_gdn_decode_iasm_nvh8_nvb4_opt_lb4_vmem);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb4_opt_lb10v48_vmem", launch_gdn_decode_iasm_nvh8_nvb4_opt_lb10v48_vmem);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb4_sched_hint", launch_gdn_decode_iasm_nvh8_nvb4_sched_hint);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_extreme", launch_gdn_decode_iasm_nvh8_bs16_extreme);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_extreme_lb1", launch_gdn_decode_iasm_nvh8_bs16_extreme_lb1);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_extreme_lb2", launch_gdn_decode_iasm_nvh8_bs16_extreme_lb2);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_extreme_lb3", launch_gdn_decode_iasm_nvh8_bs16_extreme_lb3);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_bs16_extreme_lb6", launch_gdn_decode_iasm_nvh8_bs16_extreme_lb6);
    #undef DEF_NVH8_VARIANT

    #define DEF_BS2_NVH8_VARIANT(NAME, LAUNCH_FN) \
    m.def(NAME, []( \
        torch::Tensor query, torch::Tensor key, torch::Tensor value, \
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias, \
        torch::Tensor A_log, torch::Tensor indices, \
        torch::Tensor state, torch::Tensor output, \
        int batch_size, float scale \
    ) { \
        (void)batch_size; \
        auto stream = at::hip::getCurrentHIPStream().stream(); \
        LAUNCH_FN( \
            query.data_ptr(), key.data_ptr(), value.data_ptr(), \
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(), \
            A_log.data_ptr(), indices.data_ptr(), \
            state.data_ptr(), output.data_ptr(), \
            scale, stream); \
    })

    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_nvb4_lb2", launch_gdn_decode_iasm_nvh8_bs2_nvb4_lb2);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_nvb4_lb3", launch_gdn_decode_iasm_nvh8_bs2_nvb4_lb3);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_nvb4", launch_gdn_decode_iasm_nvh8_bs2_nvb4);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_nvb4_lb6", launch_gdn_decode_iasm_nvh8_bs2_nvb4_lb6);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_nvb2_lb3", launch_gdn_decode_iasm_nvh8_bs2_nvb2_lb3);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_nvb2", launch_gdn_decode_iasm_nvh8_bs2_nvb2);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_extreme", launch_gdn_decode_iasm_nvh8_bs2_extreme);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_extreme_lb2", launch_gdn_decode_iasm_nvh8_bs2_extreme_lb2);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_extreme_lb4", launch_gdn_decode_iasm_nvh8_bs2_extreme_lb4);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_extreme_lb6", launch_gdn_decode_iasm_nvh8_bs2_extreme_lb6);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_extreme_nvb2", launch_gdn_decode_iasm_nvh8_bs2_extreme_nvb2);
    DEF_BS2_NVH8_VARIANT("hip_gdn_nvh8_bs2_extreme_nvb2_lb2", launch_gdn_decode_iasm_nvh8_bs2_extreme_nvb2_lb2);
    #undef DEF_BS2_NVH8_VARIANT

    // BS=1 ultra-optimized variants (no batch_size param)
    m.def("hip_gdn_nvh8_bs1_ultra", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_iasm_nvh8_bs1_ultra(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    });
    m.def("hip_gdn_nvh8_bs1_hyper", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_iasm_nvh8_bs1_hyper(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    });
    m.def("hip_gdn_nvh8_bs1_hyper_lb6", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_iasm_nvh8_bs1_hyper_lb6(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    });
    m.def("hip_gdn_nvh8_bs1_kpipe", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_iasm_nvh8_bs1_kpipe(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    });
    m.def("hip_gdn_nvh8_bs1_extreme", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_iasm_nvh8_bs1_extreme(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    });
    m.def("hip_gdn_nvh8_bs1_extreme_lb6", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_iasm_nvh8_bs1_extreme_lb6(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    });
    m.def("hip_gdn_nvh8_bs1_ultra_lb6", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_iasm_nvh8_bs1_ultra_lb6(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    });

    m.def("hip_gdn_decode_kv4_inplace", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, int seq_length,
        int num_v_blocks, bool use_qk_l2norm, float scale,
        int num_k_heads, int num_v_heads
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_kv4(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, seq_length,
            num_v_blocks, use_qk_l2norm, scale,
            num_k_heads, num_v_heads, stream);
    }, "GDN decode KV4 kernel (state layout: [pool, HV, K, V] with float4 along V)");

    // DPP-based kernels: use VALU DPP instead of ds_bpermute for reduces
    // Better for small batch sizes due to eliminated lgkmcnt waits
    m.def("hip_gdn_dpp_nvb4", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_dpp_nvh8_nvb4(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, scale, stream);
    }, "DPP-based GDN decode (nvb4): uses VALU DPP row_xor for reduces");
    
    m.def("hip_gdn_dpp_nvb4_lb6", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_dpp_nvh8_nvb4_lb6(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, scale, stream);
    }, "DPP-based GDN decode (nvb4, lb6): uses VALU DPP, higher occupancy");
    
    m.def("hip_gdn_dpp_nvb2", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_dpp_nvh8_nvb2(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, scale, stream);
    }, "DPP-based GDN decode (nvb2): uses VALU DPP row_xor for reduces");
    
    m.def("hip_gdn_dpp_nvb1", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        int batch_size, float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_dpp_nvh8_nvb1(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            batch_size, scale, stream);
    }, "DPP-based GDN decode (nvb1): uses VALU DPP row_xor for reduces");

    // EXTREME BS=1 kernels: maximum optimization for small batch
    m.def("hip_gdn_extreme_bs1_hi_occ", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_extreme_bs1_nvb4_hi_occ(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    }, "EXTREME BS=1 GDN decode (high occupancy): 10 waves/EU");

    m.def("hip_gdn_extreme_bs1", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_extreme_bs1_nvb4(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    }, "EXTREME BS=1 GDN decode: ASM transcendentals, 4-way accumulators");

    m.def("hip_gdn_extreme_bs1_lb6", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_extreme_bs1_nvb4_lb6(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    }, "EXTREME BS=1 GDN decode (lb6): higher occupancy variant");

    m.def("hip_gdn_extreme_bs1_lb4", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_extreme_bs1_nvb4_lb4(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    }, "EXTREME BS=1 GDN decode (lb4): balanced occupancy");

    m.def("hip_gdn_extreme_bs1_lb2", [](
        torch::Tensor query, torch::Tensor key, torch::Tensor value,
        torch::Tensor a, torch::Tensor b, torch::Tensor dt_bias,
        torch::Tensor A_log, torch::Tensor indices,
        torch::Tensor state, torch::Tensor output,
        float scale
    ) {
        auto stream = at::hip::getCurrentHIPStream().stream();
        launch_gdn_decode_extreme_bs1_nvb4_lb2(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            a.data_ptr(), b.data_ptr(), dt_bias.data_ptr(),
            A_log.data_ptr(), indices.data_ptr(),
            state.data_ptr(), output.data_ptr(),
            scale, stream);
    }, "EXTREME BS=1 GDN decode (lb2): more registers per thread");
}
