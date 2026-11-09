#pragma once

#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"

namespace ninfer::ops::detail {

// GQA4 uses the prompt route at every width. Do not instantiate the specialized small-T
// kernels for it: the same query-head count may also name a GQA6 or GQA8 TP share.
template <typename F>
decltype(auto) with_causal_small_t_geometry(std::int32_t query_heads, std::int32_t kv_heads,
                                           F&& f) {
    if (!causal_attention_geometry_supported(query_heads, kv_heads) ||
        (query_heads / kv_heads != CausalD256H24Kv4::GroupSize &&
         query_heads / kv_heads != CausalD256H16Kv2::GroupSize)) {
        throw std::invalid_argument("causal small-T attention: unsupported head geometry");
    }
    if (query_heads / kv_heads == CausalD256H24Kv4::GroupSize) {
#if NINFER_KV_ATTN_GEOM_H24
        return f.template operator()<CausalD256H24Kv4>();
#else
        throw std::invalid_argument("causal small-T attention: this build omits the H24 geometry");
#endif
    }
#if NINFER_KV_ATTN_GEOM_H16
    return f.template operator()<CausalD256H16Kv2>();
#else
    throw std::invalid_argument("causal small-T attention: this build omits the H16 geometry");
#endif
}

} // namespace ninfer::ops::detail
