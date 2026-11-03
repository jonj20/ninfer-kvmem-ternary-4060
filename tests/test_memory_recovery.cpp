#include "runtime/engine/memory_recovery.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

void require(bool value) { if (!value) { throw std::runtime_error("memory recovery invariant failed"); } }
int main() {
    try {
        ninfer::EngineOptions options;
        options.max_context = 262144;
        options.max_concurrency = 2;
        options.kvmem.selected_tokens = 65536;
        options.kvmem.sink_tokens = 12288;
        options.kvmem.recent_tokens = 1024;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.fast_prefill_kernel = true;
        options.cuda_graph_allowance_bytes = 123;
        constexpr std::size_t MiB = 1ULL << 20;
        std::size_t calls = 0;
        auto cost = [](const ninfer::EngineOptions& c) {
            // Independent monotone layout with a discrete workspace jump; an
            // affine subtraction of the old KV payload gives the wrong boundary.
            const std::size_t stride = c.kv_cache == ninfer::KvCacheStorage::Int8Group64 ? 32768 :
                c.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16 ? 18432 : 24576;
            return 400 * MiB + c.kvmem.device_tokens() * c.max_concurrency * stride
                + (c.kvmem.selected_tokens >= 32768 ? 96 * MiB : 0);
        };
        for (const auto reserve : {1024U, 4096U, 8192U}) {
            options.kvmem.reserve_tokens = reserve;
            options.prefill_chunk = reserve;
            auto report = nlohmann::json::parse(ninfer::runtime::memory_recovery_plan(
                options, 2400 * MiB, 5000 * MiB, [&](const auto& c) {
                    ++calls;
                    require(c.cuda_graph_allowance_bytes == 0);
                    require(c.kv_capacity.explicit_tokens == c.kvmem.device_tokens() * c.max_concurrency);
                    require(c.kvmem.reserve_tokens == std::min(reserve, 4096U));
                    require(c.prefill_chunk == std::min(reserve, 4096U));
                    return cost(c);
                }));
            for (const auto& result : report["options"]) {
                require(result["available"].get<bool>());
                auto chosen = options;
                chosen.kvmem.selected_tokens = result["budget"];
                chosen.kvmem.reserve_tokens = result["reserve"];
                chosen.kv_cache = result["id"] == "int8" ? ninfer::KvCacheStorage::Int8Group64 :
                    result["id"] == "nvfp4" ? ninfer::KvCacheStorage::Nvfp4Group16 : ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64;
                require(chosen.kvmem.selected_tokens % 64 == 0);
                require(chosen.kvmem.selected_tokens >= 13312);
                require(cost(chosen) + report["safety_bytes"].get<std::size_t>() <= 2400 * MiB);
                require(result["runtime_bytes"].get<std::size_t>() == cost(chosen));
                chosen.kvmem.selected_tokens += 64;
                require(chosen.kvmem.selected_tokens > options.kvmem.selected_tokens ||
                    cost(chosen) + report["safety_bytes"].get<std::size_t>() > 2400 * MiB);
                require(result["fast_prefill"].get<bool>() == (result["id"] != "nvfp4"));
            }
        }
        require(calls <= 108); // bounded page search, never a model-loading loop
        auto none = nlohmann::json::parse(ninfer::runtime::memory_recovery_plan(
            options, 64 * MiB, 5000 * MiB, cost));
        require(!none["options"][0]["available"].get<bool>());
        auto failed = nlohmann::json::parse(ninfer::runtime::memory_recovery_plan(
            options, 2400 * MiB, 5000 * MiB, [](const auto&) -> std::size_t { throw std::runtime_error("unsupported"); }));
        require(failed["options"][0]["reason"] == "planning_unavailable");
        require(!failed["options"][1]["available"].get<bool>());
        // Proactive planning keeps R/KV/prefill fixed and can grow beyond old B.
        options.kvmem.selected_tokens = 13312;
        for (auto format : {ninfer::KvCacheStorage::Int8Group64, ninfer::KvCacheStorage::Nvfp4Group16,
                            ninfer::KvCacheStorage::RotatedInt8KeyInt4ValueGroup64}) {
            options.kv_cache = format;
            for (auto reserve : {1024U, 4096U, 8192U}) {
                options.kvmem.reserve_tokens = reserve;
                options.prefill_chunk = 256;
                const auto plan = ninfer::runtime::maximum_memory_budget(options, 2400 * MiB, options.max_context,
                    [&](const auto& c) {
                        require(c.kvmem.reserve_tokens == reserve && c.kv_cache == format);
                        require(c.prefill_chunk == options.prefill_chunk && c.fast_prefill_kernel == options.fast_prefill_kernel);
                        require(c.cuda_graph_allowance_bytes == options.cuda_graph_allowance_bytes);
                        return cost(c);
                    });
                require(plan.available && plan.budget > options.kvmem.selected_tokens);
                auto chosen = options; chosen.kvmem.selected_tokens = plan.budget;
                require(cost(chosen) + plan.safety_bytes <= plan.available_bytes);
                chosen.kvmem.selected_tokens += 64;
                require(cost(chosen) + plan.safety_bytes > plan.available_bytes);
            }
        }
        options.max_context = 32768;
        const auto capped = ninfer::runtime::maximum_memory_budget(options, 100000 * MiB, options.max_context, cost);
        require(capped.budget == options.max_context - options.kvmem.reserve_tokens);
        const auto too_small = ninfer::runtime::maximum_memory_budget(options, 64 * MiB, options.max_context, cost);
        require(!too_small.available && too_small.budget == 0);
        std::cout << "PASS: recovery and proactive budget bounds, next-page boundary, fixed reserve/format/prefill, context ceiling, failure handling\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
