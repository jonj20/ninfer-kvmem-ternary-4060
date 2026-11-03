#pragma once

#include "ninfer/engine_options.h"
#include "ninfer/budget_plan.h"
#include <stdexcept>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>

namespace ninfer::runtime {
template<class Quote>
KvmemBudgetPlan maximum_memory_budget(const EngineOptions& original, std::size_t available,
                                      std::uint32_t maximum, Quote&& quote) {
    const auto started = std::chrono::steady_clock::now();
    KvmemBudgetPlan out;
    out.reserve = original.kvmem.reserve_tokens;
    out.prefill = original.prefill_chunk;
    out.context = original.max_context;
    out.concurrency = original.max_concurrency;
    out.kv_cache = original.kv_cache;
    out.available_bytes = available;
    out.safety_bytes = std::max<std::size_t>(128ULL << 20, available / 20);
    out.minimum_budget = std::max({128U,
        (original.kvmem.sink_blocks() + original.kvmem.recent_blocks()) * 64,
        out.reserve < 256 ? 256 - out.reserve : 0U});
    out.maximum_budget = (out.context > out.reserve ?
        std::min(maximum, out.context - out.reserve) : 0U) / 64 * 64;
    if (out.maximum_budget >= out.minimum_budget && available > out.safety_bytes) {
        const auto limit = available - out.safety_bytes;
        auto candidate = original;
        auto measure = [&](std::uint32_t budget) {
            candidate.kvmem.selected_tokens = budget;
            const auto tokens = std::uint64_t(candidate.kvmem.device_tokens()) * candidate.max_concurrency;
            if (tokens > UINT32_MAX) throw std::overflow_error("KVMem capacity exceeds uint32");
            candidate.kv_capacity = KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(tokens));
            return quote(candidate);
        };
        auto low = out.minimum_budget / 64, high = out.maximum_budget / 64;
        auto needed = measure(low * 64);
        if (needed <= limit) {
            out.available = true; out.budget = low * 64; out.runtime_bytes = needed;
            while (low < high) {
                const auto middle = low + (high - low + 1) / 2;
                needed = measure(middle * 64);
                if (needed <= limit) {
                    low = middle; out.budget = middle * 64; out.runtime_bytes = needed;
                } else { high = middle - 1; }
            }
        }
    }
    out.planning_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}

struct MemoryRecoveryOption {
    const char* id;
    bool available = false;
    std::uint32_t budget = 0, reserve = 0, prefill = 0;
    bool fast_prefill = false;
    std::size_t required = 0;
    const char* reason = "minimum_does_not_fit";
};

// Quote must build the native layout only: no Program, graph capture or allocation.
// Search page-aligned B with R fixed. Re-quote every candidate because workspace,
// draft state and graph bounds need not be affine in the history window.
template<class Quote>
std::string memory_recovery_plan(const EngineOptions& original, std::size_t available,
                                 std::size_t failed_required, Quote&& quote) {
    const auto started = std::chrono::steady_clock::now();
    const std::size_t margin = std::max<std::size_t>(128ULL << 20, available / 20);
    const auto reserve = std::min(original.kvmem.reserve_tokens, 4096U);
    const auto prefill = std::min(original.prefill_chunk, reserve) / 128 * 128;
    std::array<MemoryRecoveryOption, 3> choices{{{"int8"}, {"nvfp4"}, {"rk8v4"}}};
    constexpr std::array formats{KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16,
                                KvCacheStorage::RotatedInt8KeyInt4ValueGroup64};
    for (std::size_t index = 0; index < choices.size(); ++index) {
        auto& out = choices[index];
        auto candidate = original;
        candidate.kv_cache = formats[index];
        candidate.kvmem.reserve_tokens = out.reserve = reserve;
        candidate.prefill_chunk = out.prefill = prefill;
        candidate.fast_prefill_kernel = out.fast_prefill = index != 1 && original.fast_prefill_kernel;
        // Recompute each format's static graph allowance. A measurement from the
        // failing configuration is not transferable to another format/window.
        candidate.cuda_graph_allowance_bytes = 0;
        if (prefill == 0) { continue; }
        try {
            const auto plan = maximum_memory_budget(candidate, available, original.kvmem.selected_tokens, quote);
            if (!plan.available) { continue; }
            out.available = true; out.reason = ""; out.budget = plan.budget; out.required = plan.runtime_bytes;
        } catch (const std::exception&) {
            // Recommendation failure must not replace the original startup error.
            out.available = false; out.budget = 0; out.required = 0;
            out.reason = "planning_unavailable";
        }
    }
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();
    std::ostringstream json;
    json << "{\"version\":1,\"required_bytes\":" << failed_required
         << ",\"available_bytes\":" << available << ",\"safety_bytes\":" << margin
         << ",\"planning_us\":" << micros << ",\"original\":{\"Budget\":" << original.kvmem.selected_tokens
         << ",\"Reserve\":" << original.kvmem.reserve_tokens << ",\"Prefill\":" << original.prefill_chunk
         << ",\"Context\":" << original.max_context << ",\"Concurrency\":" << original.max_concurrency
         << ",\"kv_storage\":" << static_cast<unsigned>(original.kv_cache) << "},\"options\":[";
    for (std::size_t i = 0; i < choices.size(); ++i) {
        const auto& c = choices[i];
        if (i) { json << ','; }
        json << "{\"id\":\"" << c.id << "\",\"available\":" << (c.available ? "true" : "false")
             << ",\"reason\":\"" << c.reason << "\",\"budget\":" << c.budget
             << ",\"reserve\":" << c.reserve << ",\"prefill\":" << c.prefill
             << ",\"fast_prefill\":" << (c.fast_prefill ? "true" : "false")
             << ",\"runtime_bytes\":" << c.required << '}';
    }
    json << "]}";
    return json.str();
}
} // namespace ninfer::runtime
