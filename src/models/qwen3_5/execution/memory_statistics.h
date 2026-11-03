#pragma once
#include "core/tensor.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
// Stable Program-owned buffers, borrowed by eager execution and graph capture.
// Each shard holds a stage's layers or a TP rank's heads of all layers.
// Host stores use global layer and head order, independent of device partition.
// Key sums: FP32 [key_width,buckets,layers]. Query values are either compact
// BF16 [query_width,query_capacity,layers] or seeded FP32 [query_width,1,layers]
// for Mean-Q; attention feedback uses its separate post-RoPE capture.
// Ranges: [K begin,end,Q begin,end].
// Only normalized, represented BF16 values BEFORE RoPE enter these reductions.
struct MemoryStatisticsShard {
    Tensor key_sums;
    Tensor query_values;
    Tensor attention_queries; // BF16 post-RoPE target Q, three global spread3 slots
    Tensor attention_tokens;
    Tensor attention_rows, attention_positions, attention_scratch, attention_norms, attention_output;
    Tensor ranges;
    // Prefill aggregates complete logical blocks; speculative verification keeps
    // one column per candidate until the Frontend commits an exact prefix.
    std::uint32_t block_tokens = 64;
    Tensor candidate_origin;
    Tensor prefill_origin;
    std::uint32_t first_layer = 0;
    // Feature offsets in a whole layer, derived from the rank's KV/query head range.
    std::uint32_t key_offset = 0;
    std::uint32_t query_offset = 0;
};

// One contiguous shard's columns scattered into the complete host head order.
// Sources have width_bytes pitch; destinations have destination_pitch. Keeping
// this geometry independent of CUDA lets the host tests check actual values in
// the same layout used by the asynchronous device copies.
struct MemoryStatisticsCopy {
    std::size_t rank;
    const void* source;
    std::size_t destination_offset;
    std::size_t width_bytes;
    std::size_t destination_pitch;
    std::size_t rows;
};

struct MemoryStatistics {
    std::vector<MemoryStatisticsShard> ranks;
    std::uint32_t key_width = 0;
    std::uint32_t query_width = 0;
    std::uint32_t layers = 0;
    std::uint32_t columns = 0;
    std::uint32_t query_columns = 0;
    bool mean_query = false;

    void validate_layout() const {
        if (!layers || !key_width || !query_width || !columns || !query_columns ||
            query_width % key_width)
            throw std::logic_error("KVMem statistics have an invalid global geometry");
        const auto ratio = query_width / key_width;
        std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> coverage(layers);
        for (const auto& shard : ranks) {
            if (!shard.key_sums.data) continue; // rank with no attention heads/layers
            const auto& key = shard.key_sums;
            const auto& query = shard.query_values;
            if (key.dtype != DType::FP32 || !key.is_contiguous() || key.ne[3] != 1 ||
                key.ne[0] <= 0 || key.ne[1] != columns || key.ne[2] <= 0 ||
                std::uint64_t(shard.first_layer) + key.ne[2] > layers ||
                std::uint64_t(shard.key_offset) + key.ne[0] > key_width ||
                !query.data || query.dtype != (mean_query ? DType::FP32 : DType::BF16) ||
                !query.is_contiguous() || query.ne[3] != 1 ||
                query.ne[2] != key.ne[2] || query.ne[1] != query_columns ||
                query.ne[0] != std::uint64_t(key.ne[0]) * ratio ||
                shard.query_offset != std::uint64_t(shard.key_offset) * ratio)
                throw std::logic_error("KVMem statistics shard disagrees with its global geometry");
            for (std::int32_t i = 0; i < key.ne[2]; ++i)
                coverage[shard.first_layer + i].emplace_back(shard.key_offset,
                    shard.key_offset + static_cast<std::uint32_t>(key.ne[0]));
        }
        for (auto& layer : coverage) {
            std::sort(layer.begin(), layer.end());
            std::uint32_t next = 0;
            for (const auto& range : layer) {
                if (range.first != next)
                    throw std::logic_error("KVMem statistics heads overlap or leave a gap");
                next = range.second;
            }
            if (next != key_width)
                throw std::logic_error("KVMem statistics do not cover every model head");
        }
    }

    [[nodiscard]] std::size_t query_element_bytes() const noexcept {
        return mean_query ? sizeof(float) : sizeof(std::uint16_t);
    }

    [[nodiscard]] std::size_t key_bytes() const noexcept {
        return std::size_t(key_width) * columns * layers * sizeof(float);
    }
    [[nodiscard]] std::size_t query_elements() const noexcept {
        return std::size_t(query_width) * query_columns * layers;
    }
    [[nodiscard]] std::size_t query_bytes() const noexcept {
        return query_elements() * query_element_bytes();
    }

    // validate_layout() runs when the Program binds its buffers. query_count is
    // the accepted capture prefix; unused candidate/query columns are not read.
    template<class Visit>
    void for_each_host_copy(bool include_query, std::uint32_t query_count, Visit&& visit) const {
        const auto count = mean_query ? 1U : query_count;
        if (include_query && (!count || count > query_columns))
            throw std::logic_error("KVMem Query capture exceeds its planned capacity");
        const auto key_pitch = std::size_t(key_width) * sizeof(float);
        const auto query_pitch = std::size_t(query_width) * query_element_bytes();
        for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
            const auto& shard = ranks[rank];
            if (!shard.key_sums.data) continue;
            if (std::uint64_t(shard.first_layer) + shard.key_sums.ne[2] > layers ||
                std::uint64_t(shard.key_offset) + shard.key_sums.ne[0] > key_width ||
                std::uint64_t(shard.query_offset) + shard.query_values.ne[0] > query_width)
                throw std::logic_error("KVMem statistics shard is outside the global head layout");
            for (std::int32_t layer = 0; layer < shard.key_sums.ne[2]; ++layer) {
                const auto key = shard.key_sums.slice(2, layer, 1);
                visit(MemoryStatisticsCopy{
                    rank, key.data,
                    (shard.first_layer + layer) * key_pitch * columns +
                        std::size_t(shard.key_offset) * sizeof(float),
                    std::size_t(key.ne[0]) * sizeof(float), key_pitch, columns});
                if (include_query) {
                    const auto query = shard.query_values.slice(2, layer, 1);
                    visit(MemoryStatisticsCopy{
                        rank, query.data,
                        key_bytes() + (shard.first_layer + layer) * query_pitch * query_columns +
                            std::size_t(shard.query_offset) * query_element_bytes(),
                        std::size_t(query.ne[0]) * query_element_bytes(), query_pitch, count});
                }
            }
        }
    }
};
} // namespace ninfer::models::qwen3_5::execution
