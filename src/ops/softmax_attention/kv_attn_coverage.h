#pragma once

// What this build carries of the KV attention: the KV cache codings and the head geometries.
//
// The build decides (the CMake options NINFER_KV_ATTN_CODING_* and NINFER_KV_ATTN_GEOM_*, declared
// in src/ops/CMakeLists.txt) and hands the values to the units that read them as compile
// definitions (src/ops/softmax_attention/sources.cmake). There is no default here on purpose: a
// unit that reads the values without them is a compile error below, not a silent "everything is
// compiled". A unit that believed in a coding this build does not carry would select a kernel that
// was never compiled, and the failure would surface at the first launch instead of at the build.
//
// Two questions are answered here. Which storage each coding switch covers is
// kv_attn_storage_in_build() below, which the Op dispatchers ask before they dispatch. Which
// geometry each geometry switch covers is the dispatch trees' own #if blocks, one per geometry they
// select (src/ops/softmax_attention/dense/causal_cache).
//
// The public form of the same question is in include/ninfer/kv_cache_storage.h, which the Engine
// asks at admission.

#include "ninfer/kv_cache_storage.h"

#if !defined(NINFER_KV_ATTN_CODING_RK8V4) || !defined(NINFER_KV_ATTN_CODING_RK4V4) ||      \
    !defined(NINFER_KV_ATTN_CODING_RK4V4E8) || !defined(NINFER_KV_ATTN_CODING_RK2V4E8) ||  \
    !defined(NINFER_KV_ATTN_FP8) || !defined(NINFER_KV_ATTN_NVFP4) ||                      \
    !defined(NINFER_KV_ATTN_K8V4) || !defined(NINFER_KV_ATTN_GEOM_H24) ||                    \
    !defined(NINFER_KV_ATTN_GEOM_H16) || !defined(NINFER_KV_ATTN_GEOM_G4)
#    error "the KV attention coverage values are missing; this unit must be compiled with them (src/ops/softmax_attention/sources.cmake)"
#endif

namespace ninfer::ops::detail {

// Whether this build contains the selected cache format's attention and append code. A build that
// leaves a format out must reject it here before any dispatch can fall through to another coding.
// The public form of this question is ninfer::kv_attn_storage_compiled (include/ninfer/
// kv_cache_storage.h), which asks it from outside the Ops.
[[nodiscard]] constexpr bool kv_attn_storage_in_build(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
    case KvCacheStorage::Int8Group64:
        return true;
    case KvCacheStorage::Fp8E4M3Row256:
        return NINFER_KV_ATTN_FP8 != 0;
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return NINFER_KV_ATTN_CODING_RK8V4 != 0;
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return NINFER_KV_ATTN_CODING_RK4V4 != 0;
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        return NINFER_KV_ATTN_CODING_RK4V4E8 != 0;
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
        return NINFER_KV_ATTN_CODING_RK2V4E8 != 0;
    case KvCacheStorage::Nvfp4Group16:
        return NINFER_KV_ATTN_NVFP4 != 0;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return NINFER_KV_ATTN_K8V4 != 0;
    }
    return false;
}

} // namespace ninfer::ops::detail
