// The KV cache codings this build carries, as the Engine and the Op dispatchers see them.
//
// The compile-time switches live in kv_attn_coverage.h. This unit is compiled with them (see
// src/ops/softmax_attention/sources.cmake), so the answer it gives is the build's own coverage
// and not the product default. The Engine asks it at admission, so a --kv-dtype this build
// omitted is refused by name before any weight is uploaded, instead of failing inside an Op on
// the first generation.

#include "ops/softmax_attention/kv_attn_coverage.h"

#include "ninfer/kv_cache_storage.h"

#include <string>

namespace ninfer {

bool kv_attn_storage_compiled(KvCacheStorage storage) {
    return ops::detail::kv_attn_storage_in_build(storage);
}

const char* kv_attn_storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return "rk8v4";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return "rk4v4";
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
        return "rk4v4-e8";
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
        return "rk2v4-e8";
    }
    return "unknown";
}

std::string kv_attn_compiled_storage_names() {
    std::string names;
    for (const KvCacheStorage storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64,
                                         KvCacheStorage::Fp8E4M3Row256,
                                         KvCacheStorage::RotatedInt8KeyInt4ValueGroup64,
                                         KvCacheStorage::Nvfp4Group16,
                                         KvCacheStorage::Fp8KeyNvfp4Value,
                                         KvCacheStorage::RotatedLloyd4KeyInt4Value,
                                         KvCacheStorage::RotatedInt4KeyInt4ValueE8,
                                         KvCacheStorage::RotatedE8RootKeyInt4Value}) {
        if (!kv_attn_storage_compiled(storage)) { continue; }
        if (!names.empty()) { names += ", "; }
        names += kv_attn_storage_name(storage);
    }
    return names;
}

} // namespace ninfer
