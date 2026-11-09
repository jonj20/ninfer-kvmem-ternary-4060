// Rejections for the KV append codings a deployment build omits. The append dispatcher keeps its
// calls; a build without a coding's translation units links these instead, so selecting that cache
// format throws instead of falling through to another coding's kernel.

#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/kv_attn_coverage.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void kv_append_compiled_out(const char* coding) {
    throw std::runtime_error(std::string("this build was compiled without the ") + coding +
                             " KV cache append coding");
}

} // namespace

#if !NINFER_KV_ATTN_NVFP4
void kv_cache_append_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                  cudaStream_t) {
    kv_append_compiled_out("nvfp4");
}

void kv_cache_append_nvfp4_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    kv_append_compiled_out("nvfp4");
}
#endif

#if !NINFER_KV_ATTN_K8V4
void kv_cache_append_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                 cudaStream_t) {
    kv_append_compiled_out("k8v4");
}

void kv_cache_append_k8v4_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    kv_append_compiled_out("k8v4");
}
#endif

} // namespace ninfer::ops::detail
