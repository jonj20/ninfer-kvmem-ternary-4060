#pragma once

#include "ops/softmax_attention/kv_attn_coverage.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

// A causal attention geometry fixes the query heads per KV head (GroupSize) and the small-T split
// scale at compile time. How many KV heads a call attends with is a run-time count: the model's,
// or a tensor-parallel rank's run of them, at most ModelKVHeads. Every kernel takes it from its
// launch grid, which spans the call's KV heads (small-T partials, packed-GQA prompts) or query
// heads (reductions, prompts, merges). ModelQHeads names the model geometry whose device-profile
// routes a call takes.
template <int GroupSizeValue, int ModelKVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry {
    static_assert(GroupSizeValue > 0 && ModelKVHeadsValue > 0 && SmallTSplitScaleValue > 0);

    static constexpr int GroupSize           = GroupSizeValue;
    static constexpr int ModelKVHeads        = ModelKVHeadsValue;
    static constexpr int ModelQHeads         = GroupSize * ModelKVHeads;
    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = 85 * SmallTSplitScale;
};

using CausalD256H8Kv2 = CausalAttentionGeometry<4, 2, 2>;
using CausalD256H16Kv4 = CausalAttentionGeometry<4, 4, 1>;

using CausalD256H24Kv4 = CausalAttentionGeometry<6, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<8, 2, 2>;

// Whether `query_heads` over `kv_heads` is one of those geometries or a run of its KV heads.
[[nodiscard]] constexpr bool causal_attention_geometry_supported(std::int32_t query_heads,
                                                                 std::int32_t kv_heads) {
    if (kv_heads <= 0 || query_heads % kv_heads != 0) { return false; }
    const std::int32_t group = query_heads / kv_heads;
    return (NINFER_KV_ATTN_GEOM_H24 &&
            group == CausalD256H24Kv4::GroupSize && kv_heads <= CausalD256H24Kv4::ModelKVHeads) ||
           (NINFER_KV_ATTN_GEOM_H16 &&
            group == CausalD256H16Kv2::GroupSize && kv_heads <= CausalD256H16Kv2::ModelKVHeads) ||
           (NINFER_KV_ATTN_GEOM_G4 && group == 4 && kv_heads <= 4);
}

// Calls `f.template operator()<Geometry>()` with the geometry of `query_heads` over `kv_heads`.
// Throws for any other pair.
template <typename F>
decltype(auto) with_causal_geometry(std::int32_t query_heads, std::int32_t kv_heads, F&& f) {
    if (!causal_attention_geometry_supported(query_heads, kv_heads)) {
        throw std::invalid_argument("causal attention: unsupported head geometry");
    }
    if (query_heads / kv_heads == CausalD256H24Kv4::GroupSize) {
#if NINFER_KV_ATTN_GEOM_H24
        return f.template operator()<CausalD256H24Kv4>();
#else
        throw std::invalid_argument("causal attention: this build omits the H24 geometry");
#endif
    }
    if (query_heads / kv_heads == 4) {
#if NINFER_KV_ATTN_GEOM_G4
        if (kv_heads <= 2) return f.template operator()<CausalD256H8Kv2>();
        return f.template operator()<CausalD256H16Kv4>();
#else
        throw std::invalid_argument("causal attention: this build omits the group-4 geometry");
#endif
    }
#if NINFER_KV_ATTN_GEOM_H16
    return f.template operator()<CausalD256H16Kv2>();
#else
    throw std::invalid_argument("causal attention: this build omits the H16 geometry");
#endif
}

} // namespace ninfer::ops
