#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <cstdlib>

static int gdn_sort_threshold() {
    static int t = []() {
        const char* e = std::getenv("HIP_GDN_SORT_IDX_BS");
        return e ? std::atoi(e) : 64;
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

void launch_gdn_decode_iasm_nvh8_nvb1(
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

void launch_gdn_decode_iasm_nvh8_streaming(
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
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1", launch_gdn_decode_iasm_nvh8_nvb1);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1_lb2", launch_gdn_decode_iasm_nvh8_nvb1_lb2);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1_lb4", launch_gdn_decode_iasm_nvh8_nvb1_lb4);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_nvb1_lb2w2", launch_gdn_decode_iasm_nvh8_nvb1_lb2w2);
    DEF_NVH8_VARIANT("hip_gdn_nvh8_streaming", launch_gdn_decode_iasm_nvh8_streaming);
    #undef DEF_NVH8_VARIANT

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
}
