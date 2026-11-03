#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

inline constexpr int kKVCacheAppendFullHeadDim = 256;

template <int KVHeadsValue>
struct KVCacheAppendFullGeometry {
    static_assert(KVHeadsValue >= 1 && KVHeadsValue <= 4);
    static constexpr int KVHeads = KVHeadsValue;
};

using KVCacheAppendD256Kv4 = KVCacheAppendFullGeometry<4>;
using KVCacheAppendD256Kv2 = KVCacheAppendFullGeometry<2>;
// Tensor-parallel shares of the four-head geometry (three heads, one) and of either (one).
using KVCacheAppendD256Kv3 = KVCacheAppendFullGeometry<3>;
using KVCacheAppendD256Kv1 = KVCacheAppendFullGeometry<1>;

// Calls `f.template operator()<Geometry>()` with the geometry of `kv_heads` heads.
template <typename F>
decltype(auto) with_kv_append_geometry(std::int32_t kv_heads, F&& f) {
    switch (kv_heads) {
    case 4:
        return f.template operator()<KVCacheAppendD256Kv4>();
    case 2:
        return f.template operator()<KVCacheAppendD256Kv2>();
    case 3:
        return f.template operator()<KVCacheAppendD256Kv3>();
    case 1:
        return f.template operator()<KVCacheAppendD256Kv1>();
    default:
        throw std::invalid_argument("kv_cache_append: unsupported KV-head count");
    }
}

} // namespace ninfer::ops
