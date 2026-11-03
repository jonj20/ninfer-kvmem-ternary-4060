#include "models/qwen3_5/program/storage/checkpoint_recompute_policy.h"

#include <array>
#include <iostream>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)

int main() {
    using namespace ninfer::models::qwen3_5::detail;
    constexpr auto keep = no_checkpoint_surrogate;
    // These are exact Frontend token boundaries, not a prompt-minus-constant
    // estimate: the long tool input retains its 34-token safe rewrite source.
    CHECK(structural_input_reuse_frontier(1452, 1450, 34) == 1450);
    CHECK(structural_input_reuse_frontier(8002, 8000, 34) == 8000);
    CHECK(structural_input_reuse_frontier(499, 492, 492) == 0);
    CHECK(structural_input_reuse_frontier(499, 492, 0) == 0);
    CHECK(structural_input_reuse_frontier(499, 0, 34) == 0);
    CHECK(structural_input_reuse_frontier(499, 499, 34) == 0);
    CHECK(structural_input_reuse_frontier(499, 501, 34) == 0);
    CHECK(structural_input_reuse_frontier(499, 492, 493) == 0);
    CHECK(input_reuse_source_allowed(1450, 1450, 0, 1440));
    CHECK(input_reuse_source_allowed(1444, 1450, 16, 1440));
    CHECK(!input_reuse_source_allowed(1444, 1450, 0, 1440));
    CHECK(!input_reuse_source_allowed(1444, 1450, 5, 1440));
    CHECK(!input_reuse_source_allowed(1444, 1450, 16, 1445));
    CHECK(!input_reuse_source_allowed(1451, 1450, 16, 1440));
    CHECK(!input_reuse_source_allowed(0, 10, 16, 0));
    {
        // The structured input and safe rewrite sources survive independently
        // of execution-role publication. Query rewind invalidates the deeper
        // first-pass state; only a fresh second-pass state may replace it.
        auto fallback = std::make_shared<int>(34);
        auto query = std::make_shared<int>(1200);
        auto input = std::make_shared<int>(1450);
        const auto probe_query = query;
        std::weak_ptr<int> first_pass = input;
        std::shared_ptr<int> tail;
        CheckpointReplayCompletion completion{false, true, false, 0, 1440};
        CHECK(!retire_query_checkpoint_after_prefill(query, tail, 0, completion));
        CHECK(query == probe_query && *fallback == 34);
        input.reset(); // Native restore truncates all roles after query_begin.
        CHECK(first_pass.expired() && query == probe_query && *fallback == 34);
        input = std::make_shared<int>(1450);
        completion = {true, false, true, 1450, 1440};
        // An InputReuse image is not an ordinary PromptTail publication: the
        // existing T=0 execution retirement contract is not silently widened.
        CHECK(!retire_query_checkpoint_after_prefill(query, tail, 0, completion));
        CHECK(query == probe_query && *input == 1450 && *fallback == 34);
        auto shared_input = input;
        std::weak_ptr<int> retained_input = input;
        input.reset();
        CHECK(!retained_input.expired()); // Other role/history leases own bytes.
        CHECK(*shared_input == 1450 && *fallback == 34);
    }
    {
        // Short -> long: the successfully captured Query replaces the old tail
        // without retaining a second unreferenced native image at T=0.
        std::shared_ptr<int> query;
        auto tail = std::make_shared<int>(498);
        std::weak_ptr<int> old_tail = tail;
        auto captured = std::make_shared<int>(4096);
        CHECK(!old_tail.expired()); // New capture does not consume the old one.
        publish_checkpoint_execution_source(query, tail, std::move(captured), true, false);
        CHECK(query && *query == 4096 && !tail && old_tail.expired());

        // A source still serving InputFallback is not destroyed by role retirement.
        auto fallback = query;
        auto next_tail = std::make_shared<int>(8000);
        publish_checkpoint_execution_source(query, tail, std::move(next_tail), false, false);
        CHECK(!query && *tail == 8000 && *fallback == 4096);

        // A failed capture never reaches publication; its previous recovery
        // source remains live. Restoring that source can change its role safely.
        auto old = tail;
        try { throw std::runtime_error("capture failed before publication"); }
        catch (const std::runtime_error&) {}
        CHECK(tail == old && !query);
        publish_checkpoint_execution_source(query, tail, tail, true, false);
        CHECK(query == old && !tail);

        // The replay's second pass must not drop the mandatory Query source.
        publish_checkpoint_execution_source(query, tail, std::make_shared<int>(8010), false, true);
        CHECK(query == old && *tail == 8010);
        // After successful completion the same publication rule retires its
        // execution role, while any independent recovery alias remains valid.
        publish_checkpoint_execution_source(query, tail, tail, false, false);
        CHECK(!query && *tail == 8010 && *old == 8000);
    }
    CHECK(!checkpoint_recompute_allowed(492, 498, 0, 490));
    CHECK(!checkpoint_recompute_allowed(492, 498, 5, 490));
    CHECK(checkpoint_recompute_allowed(492, 498, 6, 490));
    CHECK(!checkpoint_recompute_allowed(498, 498, 16, 490));
    CHECK(!checkpoint_recompute_allowed(499, 498, 16, 490));
    CHECK(!checkpoint_recompute_allowed(0, 6, 16, 0));
    CHECK(!checkpoint_recompute_allowed(489, 498, 16, 490));
    {
        // Near-window MTP replay can produce both Query and PromptTail images.
        // Follow the lifecycle through probe/replay, accepted Begin and a later
        // decode cancellation. Only accepted Begin may retire the old role.
        auto query = std::make_shared<int>(1600);
        auto tail = std::make_shared<int>(1789);
        std::weak_ptr<int> old_query = query;
        CheckpointReplayCompletion state{false, true, true, 1789, 1783};
        CHECK(!retire_query_checkpoint_after_prefill(query, tail, 0, state));
        state.replay_active = false; // complete_prefill_step alone is not commit.
        CHECK(!retire_query_checkpoint_after_prefill(query, tail, 0, state));
        state.prefill_committed = true;
        state.tail_complete = false;
        CHECK(!retire_query_checkpoint_after_prefill(query, tail, 0, state));
        state.tail_complete = true;
        state.query_end = 1790; // A tail inside the Query is not an alternative.
        CHECK(!retire_query_checkpoint_after_prefill(query, tail, 0, state));
        state.query_end = 1783;
        CHECK(!retire_query_checkpoint_after_prefill(query, tail, 16, state));
        CHECK(retire_query_checkpoint_after_prefill(query, tail, 0, state));
        CHECK(old_query.expired() && tail && *tail == 1789);
        // Cancellation after this transition has only the valid tail to retain.
        const auto cancelled_history = tail;
        CHECK(!query && cancelled_history == tail);
    }
    {
        // A repeat cannot restore the endpoint beyond the 499-token prompt.
        // Keep a complete post-Query input representative, independently of it.
        const std::array points{
            CheckpointRecomputePoint{492, 490, true, false},
            CheckpointRecomputePoint{498, 490, true, false},
            CheckpointRecomputePoint{626, 0, true, true}};
        CHECK(checkpoint_recompute_plan(points, 16) == std::vector<std::size_t>({keep, 0, keep}));
        CHECK(checkpoint_recompute_plan(points, 0) == std::vector<std::size_t>({keep, keep, keep}));
    }
    {
        // Missing/failed native capture cannot stand in for the requested tail.
        const std::array points{CheckpointRecomputePoint{492, 490, false, false},
                                CheckpointRecomputePoint{498, 490, true, false}};
        CHECK(checkpoint_recompute_plan(points, 16) == std::vector<std::size_t>({keep, keep}));
    }
    {
        // Even a short Query must not become unevaluated merely because the
        // only remaining input checkpoint moved to its preceding prefix.
        const std::array points{CheckpointRecomputePoint{489, 0, true, false},
                                CheckpointRecomputePoint{498, 490, true, false},
                                CheckpointRecomputePoint{506, 0, true, true}};
        CHECK(checkpoint_recompute_plan(points, 16) == std::vector<std::size_t>({keep, keep, keep}));
    }
    {
        // An executing Query snapshot is protected. Once consumed it may go;
        // the long Query's completed input checkpoint remains independently.
        std::array points{CheckpointRecomputePoint{4090, 0, true, false},
                          CheckpointRecomputePoint{4096, 0, true, true},
                          CheckpointRecomputePoint{8000, 7990, true, false}};
        CHECK(checkpoint_recompute_plan(points, 16) == std::vector<std::size_t>({keep, keep, keep}));
        points[1].protected_point = false;
        CHECK(checkpoint_recompute_plan(points, 16) == std::vector<std::size_t>({keep, 0, keep}));
    }
    {
        // Evaluate against the final representatives, including unsorted input.
        const std::array points{CheckpointRecomputePoint{120, 0, true, false},
                                CheckpointRecomputePoint{100, 0, true, false},
                                CheckpointRecomputePoint{110, 0, true, false},
                                CheckpointRecomputePoint{126, 0, true, false}};
        CHECK(checkpoint_recompute_plan(points, 16) == std::vector<std::size_t>({keep, keep, 1, 0}));
    }
    {
        // Equal position alone never aliases different native states.
        const std::array points{CheckpointRecomputePoint{100, 0, true, false},
                                CheckpointRecomputePoint{100, 0, true, false}};
        CHECK(checkpoint_recompute_plan(points, 16) == std::vector<std::size_t>({keep, keep}));
    }
    std::cout << "PASS checkpoint recompute boundary, lifecycle and coverage contracts\n";
}
