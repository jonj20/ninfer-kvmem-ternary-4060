#pragma once

#include "kvmem/graph_recall.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

// Native page mass is a mean over one rank's local heads. Keep one slot per
// compact page while adding sampled layers/ranks, in the same addition order
// as the native calls. Repeating GraphEdges per rank retains no extra information
// and makes the host graph budget grow with the tensor-parallel rank count.
class AttentionPageMassAccumulator {
public:
    AttentionPageMassAccumulator(std::span<kvmem::AcceptedAttentionRow> rows,
                                std::span<const std::uint32_t> logical_pages,
                                std::uint32_t sampled_layers, std::uint32_t global_heads)
        : rows_(rows), pages_(logical_pages.size()), global_heads_(global_heads),
          divisor_(double(sampled_layers) * global_heads) {
        if (!sampled_layers || !global_heads)
            throw std::logic_error("KVMem attention mass has no sampled layers or heads");
        for (auto& row : rows_) {
            if (!row.mass.empty())
                throw std::logic_error("KVMem attention mass was initialized twice");
            row.mass.reserve(pages_);
            for (const auto page : logical_pages)
                row.mass.push_back({std::uint64_t(page) * 64, 0.0});
        }
    }

    void add(std::span<const float> native_mass, std::size_t native_pages,
             std::uint32_t local_heads) {
        if (!native_pages || native_pages < pages_ || !local_heads || local_heads > global_heads_ ||
            native_mass.size() / native_pages != rows_.size() ||
            native_mass.size() % native_pages)
            throw std::logic_error("KVMem attention mass has invalid shard geometry");
        for (std::size_t i = 0; i < rows_.size(); ++i) {
            for (std::size_t p = 0; p < native_pages; ++p) {
                const auto value = native_mass[i * native_pages + p] * double(local_heads) / divisor_;
                if (!std::isfinite(value) || value < 0)
                    throw std::runtime_error("KVMem sampled invalid attention mass");
                if (p >= pages_) {
                    if (value > 1e-7)
                        throw std::runtime_error("KVMem sampled an unmapped compact page");
                    continue;
                }
                rows_[i].mass[p].mass += value;
            }
        }
    }

    void finish() {
        for (auto& row : rows_)
            std::erase_if(row.mass, [](const kvmem::GraphEdge& edge) { return edge.mass == 0; });
    }

private:
    std::span<kvmem::AcceptedAttentionRow> rows_;
    std::size_t pages_;
    std::uint32_t global_heads_;
    double divisor_;
};

inline std::size_t attention_feedback_workspace_bytes(
    std::span<const kvmem::AcceptedAttentionRow> rows) {
    std::size_t bytes = 0;
    for (const auto& row : rows) {
        // finish() removes zero entries but retains their allocated storage.
        // Count that capacity once and the two normalized/feedback working
        // copies by their positive entries; do not hide sparse-row allocation.
        bytes += row.mass.capacity() * sizeof(kvmem::GraphEdge) +
                 row.mass.size() * 2 * sizeof(std::pair<std::uint64_t, double>);
    }
    return bytes;
}

} // namespace ninfer::models::qwen3_5::detail
