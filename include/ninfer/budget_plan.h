#pragma once
#include "ninfer/engine_options.h"
#include <cstddef>
#include <cstdint>

namespace ninfer {
// A static native reservation quote after loading weights. No runtime arenas or
// CUDA graphs are allocated; readiness and peak inference memory remain untested.
struct KvmemBudgetPlan {
    bool available = false;
    std::uint32_t budget = 0, minimum_budget = 0, maximum_budget = 0;
    std::uint32_t reserve = 0, prefill = 0, context = 0, concurrency = 0;
    KvCacheStorage kv_cache = KvCacheStorage::Int8Group64;
    std::size_t available_bytes = 0, runtime_bytes = 0, safety_bytes = 0;
    std::uint64_t planning_us = 0;
};
} // namespace ninfer
