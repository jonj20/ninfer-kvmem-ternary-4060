#pragma once

#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

// Geometry shared by the pre-load split estimate and actual Program planning.
struct MemoryStatisticsPlanning {
    std::uint32_t capacity = 0;
    std::uint32_t window_tokens = 0; // Zero disables all KVMem allocations.
    std::uint32_t prefill_chunk = 0;
    std::uint32_t verify_window = 0;
    std::uint32_t max_concurrency = 1;
    std::uint32_t query_max_tokens = 0;
    bool mean_query = false;
};

} // namespace ninfer::models::qwen3_5::detail
