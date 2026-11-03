#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// The Frontend has already mapped both boundaries exactly through tokenization
// and media expansion. Add a deeper input source only when conservative rewrite
// recovery moved earlier; ordinary same-boundary requests keep their old policy.
inline std::uint32_t structural_input_reuse_frontier(std::uint32_t prompt_tokens,
                                                     std::uint32_t input_end,
                                                     std::uint32_t rewrite_frontier) {
    return rewrite_frontier && rewrite_frontier < input_end && input_end < prompt_tokens
        ? input_end : 0U;
}

// Publish only after capture/restore succeeded. The two roles replace the old
// single Base role across requests, so splitting their names must not retain
// an obsolete extra image. During a real replay its exact Query source remains
// protected even if a separate ordinary tail is captured by the second pass.
template<class State>
void publish_checkpoint_execution_source(std::shared_ptr<State>& query,
                                         std::shared_ptr<State>& tail,
                                         std::shared_ptr<State> source,
                                         bool query_source, bool query_active) {
    if (query_source) {
        query = std::move(source);
        tail.reset();
    } else {
        tail = std::move(source);
        if (!query_active) query.reset();
    }
}

struct CheckpointReplayCompletion {
    bool prefill_committed = false;
    bool replay_active = false;
    bool tail_complete = false;
    std::uint64_t tail_frontier = 0;
    std::uint64_t query_end = 0;
};

// Retire a consumed execution role at accepted Begin, not at final reply. A
// subsequent decode cancellation then keeps the same single-Base footprint as
// T=0 previously did. An unfinished replay or incomplete recovery tail cannot
// release it; T>0 retention remains the separate successful-finish policy.
template<class State>
bool retire_query_checkpoint_after_prefill(std::shared_ptr<State>& query,
                                           const std::shared_ptr<State>& tail,
                                           std::uint64_t limit,
                                           const CheckpointReplayCompletion& completion) {
    if (limit || !query || !tail || !completion.prefill_committed || completion.replay_active ||
        !completion.tail_complete || completion.tail_frontier < completion.query_end) return false;
    query.reset();
    return true;
}

// This is an extra-token budget, not a time, I/O or StateImage-copy budget.
inline bool checkpoint_recompute_allowed(std::uint64_t earlier, std::uint64_t frontier,
                                         std::uint64_t limit, std::uint64_t minimum) {
    return limit && earlier && earlier < frontier && earlier >= minimum &&
           frontier - earlier <= limit;
}

// A same-boundary native image can be shared even with thinning disabled. The
// caller still has to prove complete native identity, state and KV coverage.
inline bool input_reuse_source_allowed(std::uint64_t source, std::uint64_t frontier,
                                       std::uint64_t limit, std::uint64_t query_end) {
    return source && source >= query_end &&
        (source == frontier || checkpoint_recompute_allowed(source, frontier, limit, query_end));
}

struct CheckpointRecomputePoint {
    std::uint64_t frontier = 0;
    // A post-Query input point must retain an alternative that also contains the
    // whole Query. A consumed pre-Query execution point has no such requirement.
    std::uint64_t minimum_surrogate = 0;
    bool complete = false;
    bool protected_point = false;
};

inline constexpr auto no_checkpoint_surrogate = std::numeric_limits<std::size_t>::max();

// The caller proves native identity/KV completeness and removes payload aliases.
// A result entry names a retained earlier point, or no_checkpoint_surrogate when
// this point stays. Removed neighbours never become representatives: 100/110/120
// with a budget of 16 retains 100 and 120, rather than silently accumulating 20.
inline std::vector<std::size_t> checkpoint_recompute_plan(
        std::span<const CheckpointRecomputePoint> points, std::uint64_t limit) {
    std::vector<std::size_t> result(points.size(), no_checkpoint_surrogate);
    if (!limit) return result;
    std::vector<std::size_t> order(points.size()), retained;
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        return points[a].frontier < points[b].frontier;
    });
    for (auto index : order) {
        const auto& point = points[index];
        if (point.complete && !point.protected_point) {
            for (auto it = retained.rbegin(); it != retained.rend(); ++it) {
                const auto& earlier = points[*it];
                if (earlier.complete && checkpoint_recompute_allowed(
                        earlier.frontier, point.frontier, limit, point.minimum_surrogate)) {
                    result[index] = *it;
                    break;
                }
            }
        }
        if (result[index] == no_checkpoint_surrogate) retained.push_back(index);
    }
    return result;
}

} // namespace ninfer::models::qwen3_5::detail
