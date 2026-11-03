#pragma once

#include "core/layout.h"
#include "kvmem/page_softmax_cuda.hpp"
#include "models/qwen3_5/program/planning/memory_statistics_options.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

struct MemoryStatisticsLayout {
    std::uint32_t first_layer = 0;
    std::uint32_t key_offset = 0;
    std::uint32_t query_offset = 0;
    std::uint32_t layers = 0;
    TensorRegion key_sums, candidate_keys, candidate_origin, prefill_origin, query_values;
    TensorRegion attention_queries, attention_tokens, attention_rows, attention_positions;
    TensorRegion attention_scratch, attention_norms, attention_output, ranges;
};

// The estimates retain the existing split solver: head-sharded bytes divide,
// control/output buffers repeat on attention ranks, and Query scoring stays on
// rank 0. The actual Program and this estimate use the same allocation recipe.
enum class MemoryStatisticsOwner : std::size_t { Divided, Every, First };

inline std::int32_t memory_statistics_dimension(std::uint64_t value) {
    if (!value || value > std::uint64_t(std::numeric_limits<std::int32_t>::max()))
        throw std::overflow_error("KVMem statistics dimension exceeds int32");
    return static_cast<std::int32_t>(value);
}

template<class Allocate>
void plan_memory_statistics_shard(MemoryStatisticsLayout& out,
    const MemoryStatisticsPlanning& plan, std::uint64_t key_width, std::uint64_t query_width,
    std::uint32_t query_heads, Allocate&& allocate) {
    if (!plan.window_tokens || !out.layers) return;
    const auto d = memory_statistics_dimension;
    const auto prefill = std::min(plan.prefill_chunk, plan.capacity);
    const auto candidates = (std::uint64_t(plan.verify_window) + 1) * plan.max_concurrency;
    const auto capture = std::max<std::uint64_t>(prefill, candidates);
    const auto pages = (std::uint64_t(plan.window_tokens) + 63) / 64;
    const auto divided = MemoryStatisticsOwner::Divided, every = MemoryStatisticsOwner::Every;
    out.key_sums = allocate(divided, DType::FP32,
        {d(key_width), d((std::uint64_t(prefill)+126)/64), d(out.layers)}, "memory key sums");
    out.candidate_keys = allocate(divided, DType::FP32,
        {d(key_width), d(candidates), d(out.layers)}, "memory candidate keys");
    out.candidate_origin = allocate(every, DType::I32, {1}, "memory candidate origin");
    out.query_values = allocate(divided, plan.mean_query ? DType::FP32 : DType::BF16,
        {d(query_width), d(plan.mean_query ? 1U : std::min(prefill, plan.query_max_tokens)), d(out.layers)},
        plan.mean_query ? "memory query sums" : "memory query rows");
    out.attention_queries = allocate(divided, DType::BF16,
        {d(query_width), d(capture), 3}, "memory attention Q capture");
    out.attention_rows = allocate(every, DType::I32, {64}, "memory attention sample rows");
    out.attention_positions = allocate(every, DType::I32, {64}, "memory attention causal ends");
    out.attention_scratch = allocate(divided, DType::FP32,
        {d(pages), d(query_heads), 64}, "memory attention page LSE");
    out.attention_norms = allocate(divided, DType::FP32,
        {2, d(query_heads), 64}, "memory attention row norms");
    out.attention_output = allocate(every, DType::FP32, {d(pages), 64}, "memory attention page mass");
    out.ranges = allocate(every, DType::I32, {4}, "memory statistics ranges");
}

template<class Allocate>
void plan_memory_statistics_rank0(MemoryStatisticsLayout& out, std::optional<TensorRegion>& workspace,
    const MemoryStatisticsPlanning& plan, std::uint32_t query_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, Allocate&& allocate) {
    if (!plan.window_tokens) return;
    const auto d = memory_statistics_dimension;
    const auto prefill = std::min(plan.prefill_chunk, plan.capacity);
    const auto candidates = (std::uint64_t(plan.verify_window) + 1) * plan.max_concurrency;
    const auto first = MemoryStatisticsOwner::First;
    out.prefill_origin = allocate(first, DType::I32, {1}, "memory logical prefill origin");
    out.attention_tokens = allocate(first, DType::I32,
        {d(std::max<std::uint64_t>(prefill, candidates))}, "memory consumed token capture");
    if (!plan.mean_query) {
        const auto bytes = kvmem::PageSoftmaxCuda::workspace_bytes(
            static_cast<std::uint32_t>((std::uint64_t(plan.capacity)+63)/64),
            query_heads, kv_heads, head_dim);
        workspace = allocate(first, DType::U8, {d(bytes)}, "memory page-softmax workspace");
    }
}

struct MemoryStatisticsBudget {
    std::size_t divided = 0, every = 0, first = 0;
};

inline MemoryStatisticsBudget memory_statistics_budget(const MemoryStatisticsPlanning& plan,
    std::uint32_t layers, std::uint64_t key_width, std::uint64_t query_width,
    std::uint32_t query_heads, std::uint32_t kv_heads, std::uint32_t head_dim) {
    std::array<LayoutBuilder,3> builders;
    const auto allocate = [&](MemoryStatisticsOwner owner, DType dtype,
        std::initializer_list<std::int32_t> shape, const char* label) {
        return builders[std::size_t(owner)].add_tensor(dtype,shape,256,label);
    };
    MemoryStatisticsLayout layout;
    layout.layers = layers;
    std::optional<TensorRegion> workspace;
    plan_memory_statistics_shard(layout,plan,key_width,query_width,query_heads,allocate);
    plan_memory_statistics_rank0(layout,workspace,plan,query_heads,kv_heads,head_dim,allocate);
    return {builders[0].finish(256),builders[1].finish(256),builders[2].finish(256)};
}
} // namespace ninfer::models::qwen3_5::detail
