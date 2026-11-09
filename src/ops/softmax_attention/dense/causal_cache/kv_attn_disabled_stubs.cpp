// Rejections for the KV attention codings a deployment build omits. The dispatch trees keep their
// calls; a build without a coding's translation units links these instead, so selecting that cache
// format throws instead of reaching another coding's kernel.

#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/kv_attn_coverage.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void kv_attn_compiled_out(const char* coding) {
    throw std::runtime_error(std::string("this build was compiled without the ") + coding +
                             " KV attention coding");
}

} // namespace

#if !NINFER_KV_ATTN_FP8
void causal_attention_small_t_fp8_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                         const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                         CausalAttentionExecutionEnvelope, std::int32_t,
                                         std::int32_t, Tensor&, Tensor&, Tensor&, Tensor&,
                                         cudaStream_t, float*) {
    kv_attn_compiled_out("fp8");
}

void causal_attention_cached_small_t_fp8_launch(const Tensor&, const Tensor&, float,
                                                const PagedKVLayerView&,
                                                CausalAttentionExecutionEnvelope, Tensor&, Tensor&,
                                                Tensor&, Tensor&, cudaStream_t) {
    kv_attn_compiled_out("fp8");
}

void causal_attention_prompt_fp8_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                        Tensor&, cudaStream_t) {
    kv_attn_compiled_out("fp8");
}

void causal_attention_prompt_fp8_attention_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&, Tensor&, cudaStream_t) {
    kv_attn_compiled_out("fp8");
}
#endif

#if !NINFER_KV_ATTN_NVFP4
void causal_attention_small_t_nvfp4_launch(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, float,
    PagedKVBatchLayerView, CausalAttentionExecutionEnvelope, std::int32_t, std::int32_t, Tensor&,
    Tensor&, Tensor&, Tensor&, cudaStream_t, float*) {
    kv_attn_compiled_out("nvfp4");
}

void causal_attention_cached_small_t_nvfp4_launch(const Tensor&, const Tensor&, float,
                                                  const PagedKVLayerView&,
                                                  CausalAttentionExecutionEnvelope, Tensor&,
                                                  Tensor&, Tensor&, Tensor&, cudaStream_t) {
    kv_attn_compiled_out("nvfp4");
}

void causal_attention_prompt_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, Tensor&, cudaStream_t) {
    kv_attn_compiled_out("nvfp4");
}

void causal_attention_prompt_nvfp4_attention_launch(const Tensor&, const Tensor&, float,
                                                    const PagedKVLayerView&, Tensor&,
                                                    cudaStream_t) {
    kv_attn_compiled_out("nvfp4");
}
#endif

#if !NINFER_KV_ATTN_K8V4
void causal_attention_small_t_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                          CausalAttentionExecutionEnvelope, std::int32_t,
                                          std::int32_t, Tensor&, Tensor&, Tensor&, Tensor&,
                                          cudaStream_t, float*) {
    kv_attn_compiled_out("k8v4");
}

void causal_attention_cached_small_t_k8v4_launch(const Tensor&, const Tensor&, float,
                                                 const PagedKVLayerView&,
                                                 CausalAttentionExecutionEnvelope, Tensor&,
                                                 Tensor&, Tensor&, Tensor&, cudaStream_t) {
    kv_attn_compiled_out("k8v4");
}

void causal_attention_prompt_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                         const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                                         Tensor&, cudaStream_t) {
    kv_attn_compiled_out("k8v4");
}

void causal_attention_prompt_k8v4_attention_launch(const Tensor&, const Tensor&, float,
                                                   const PagedKVLayerView&, Tensor&, cudaStream_t) {
    kv_attn_compiled_out("k8v4");
}
#endif

} // namespace ninfer::ops::detail
