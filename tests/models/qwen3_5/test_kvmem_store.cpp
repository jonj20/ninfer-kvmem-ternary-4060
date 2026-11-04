#include "models/qwen3_5/program/storage/kvmem_store.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace store = ninfer::models::qwen3_5::detail;

// Production geometry: 128-token blocks (two 64-token pages), four blocks of budget.
constexpr std::uint32_t kBlock = 128;
constexpr std::uint32_t kPage  = 64;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void expect_eq(std::uint32_t actual, std::uint32_t expected, std::string_view message) {
    if (actual == expected) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << " expected " << expected << ", got " << actual << '\n';
}

template <typename Fn>
bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

std::vector<std::uint32_t> ids_of(const store::KVMemPlan& plan) { return plan.selected; }

// budget = four blocks, sink = one block by default.
store::KVMemConfig small_config() {
    store::KVMemConfig cfg;
    cfg.block_tokens  = kBlock;
    cfg.budget_tokens = 4 * kBlock;
    cfg.sink_tokens   = 0;
    cfg.recent_blocks = 0;
    return cfg;
}

store::KVMemBlockStore make_store(std::uint32_t tokens, const store::KVMemConfig& cfg) {
    store::KVMemBlockStore block_store(cfg);
    block_store.register_append(tokens);
    return block_store;
}

void test_append_and_lookup() {
    store::KVMemBlockStore block_store(small_config());
    expect_eq(block_store.register_append(2 * kBlock), 2, "two blocks became full");
    expect_eq(block_store.total_tokens(), 2 * kBlock, "total tracks appended tokens");
    expect_eq(block_store.block_count(), 2, "two full blocks");

    expect_eq(block_store.register_append(3), 0, "a partial trailing block does not become full");
    expect_eq(block_store.block_count(), 3, "three tokens opened no extra block");
    expect_eq(block_store.block(2).n_tokens, 3, "the trailing block holds the remainder");
    expect_eq(block_store.block(2).orig_pos_start, 2 * kBlock,
              "the trailing block starts at the old end");

    expect_eq(block_store.block_id_containing(0), 0, "position 0 is in block 0");
    expect_eq(block_store.block_id_containing(kBlock - 1), 0, "last position of block 0");
    expect_eq(block_store.block_id_containing(kBlock), 1, "first position of block 1");
    expect_eq(block_store.block_id_containing(2 * kBlock), 2, "first position of block 2");
    expect_eq(block_store.block_id_containing(2 * kBlock + 3), 3, "past the end");

    expect_eq(block_store.register_append(kBlock - 3), 1, "filling the trailing block marks it full");
    expect_eq(block_store.total_tokens(), 3 * kBlock, "total is the sum of every block");
    expect_eq(block_store.block_count(), 3, "filling the trailing block opens no new one");
}

void test_block_starts_stay_page_aligned() {
    // Invariant I2 rests on this: truncation shortens the trailing block but never moves a start.
    store::KVMemBlockStore block_store(small_config());
    block_store.register_append(5 * kBlock);
    block_store.truncate_to(300);
    for (const store::KVMemBlock& entry : block_store.blocks()) {
        expect(entry.orig_pos_start % kBlock == 0, "block start stays a whole block after truncate");
        expect(entry.orig_pos_start % kPage == 0, "block start stays a whole page after truncate");
    }
    block_store.register_append(kBlock);
    for (const store::KVMemBlock& entry : block_store.blocks()) {
        expect(entry.orig_pos_start % kBlock == 0, "block start stays a whole block after refill");
    }
}

void test_truncate() {
    store::KVMemBlockStore block_store(small_config());
    block_store.register_append(5 * kBlock); // five blocks

    const std::vector<store::KVMemBlock> dropped = block_store.truncate_to(300);
    expect_eq(static_cast<std::uint32_t>(dropped.size()), 2, "two whole blocks were popped");
    expect_eq(dropped[0].block_id, 4, "the newest block pops first");
    expect_eq(dropped[1].block_id, 3, "then the next newest");
    expect_eq(block_store.total_tokens(), 300, "the store holds exactly token_pos tokens");
    expect_eq(block_store.block_count(), 3, "the partially covered block survives");
    expect_eq(block_store.block(2).n_tokens, 300 - 2 * kBlock, "the trailing block shrinks to the cut");
    expect_eq(block_store.block(2).orig_pos_start, 2 * kBlock, "a shrunk block keeps its start");

    expect(block_store.truncate_to(6000).empty(), "truncating past the end is a no-op");
    expect_eq(block_store.total_tokens(), 300, "a no-op truncate changes nothing");

    expect_eq(static_cast<std::uint32_t>(block_store.truncate_to(0).size()), 3,
              "truncating to zero pops every survivor");
    expect_eq(block_store.total_tokens(), 0, "truncating to zero empties the store");
    expect_eq(block_store.block_count(), 0, "truncating to zero pops every block");
    expect_eq(block_store.register_append(kBlock), 1, "the store restarts cleanly after a full truncate");
}

void test_window_is_identity_when_the_context_fits() {
    store::KVMemBlockStore block_store = make_store(4 * kBlock, small_config());
    const store::KVMemPlan plan         = block_store.select_recency();
    expect(!plan.stage_in.empty(), "selecting an empty window stages everything in");
    expect(plan.stage_out.empty(), "an empty window has nothing to stage out");
    expect(ids_of(plan) == std::vector<std::uint32_t>({0, 1, 2, 3}),
           "full-budget selection is every block in order");

    const store::KVMemWindow window = block_store.window_for(plan, 0, 4 * kBlock - 1);
    expect(window.is_identity(), "a context that fits the budget is the identity view");
    expect(!window.has_hole(), "a context that fits the budget has no hole");
    expect_eq(window.block_tokens, kBlock, "the window records its block geometry");
    expect_eq(window.total_tokens, 4 * kBlock, "the window records the selected context");
    expect_eq(window.window_tokens, 4 * kBlock, "the window covers the whole context");
    expect_eq(window.retained_pages, 8, "the identity covers every page of the context");
    expect(window.row_base_by_block.empty(), "the identity leaves the block map empty");
    for (std::uint32_t abs_pos = 0; abs_pos < 4 * kBlock; ++abs_pos) {
        if (window.rel(abs_pos) != abs_pos) {
            expect(false, "identity window must map every position to itself");
            break;
        }
    }
}

void test_window_renumbers_sink_and_tail_into_one_prefix() {
    store::KVMemBlockStore block_store = make_store(5 * kBlock, small_config());
    const store::KVMemPlan plan         = block_store.select_recency();
    expect(ids_of(plan) == std::vector<std::uint32_t>({0, 2, 3, 4}),
           "sink head plus newest tail, skipping the middle the budget cannot hold");

    const std::uint32_t total = 5 * kBlock;
    const store::KVMemWindow window = block_store.window_for(plan, total - 1, total - 1);

    expect(window.has_hole(), "a budget that skips a block has a hole");
    expect_eq(window.total_tokens, total, "the window records the full context");
    expect_eq(window.window_tokens, 4 * kBlock, "window tokens equal the selected token count");
    expect_eq(window.retained_pages, 8, "four retained blocks cover eight pages");
    expect_eq(window.row_base_by_block.size(), 5, "one map entry per context block");
    expect_eq(window.row_base_by_block[0], 0, "sink block starts at row 0");
    expect(window.row_base_by_block[1] == std::numeric_limits<std::uint32_t>::max(),
           "the skipped middle block has no row");
    expect_eq(window.row_base_by_block[2], kBlock, "the tail starts right after the sink");
    expect_eq(window.row_base_by_block[3], 2 * kBlock, "the next tail block follows");
    expect_eq(window.row_base_by_block[4], 3 * kBlock, "the newest block is last");

    // The renumbered rows must be a dense prefix: sink rows 0..127, tail rows 128..511.
    expect_eq(window.rel(0), 0, "sink starts at window row 0");
    expect_eq(window.rel(kBlock - 1), kBlock - 1, "sink ends at its last row");
    expect_eq(window.rel(2 * kBlock), kBlock, "the tail starts right after the sink");
    expect_eq(window.rel(total - 1), 4 * kBlock - 1, "the newest position is the last window row");

    // rel() <= rel(p)  <=>  k <= p  over every retained pair: this is what makes the unchanged
    // causal mask select exactly the retained positions at or before the query.
    std::vector<std::uint32_t> retained;
    for (std::uint32_t abs_pos = 0; abs_pos < total; ++abs_pos) {
        if (window.retains(abs_pos)) { retained.push_back(abs_pos); }
    }
    expect_eq(static_cast<std::uint32_t>(retained.size()), 4 * kBlock,
              "retained positions number exactly the window");
    bool order_holds = true;
    for (std::size_t i = 0; i < retained.size() && order_holds; ++i) {
        for (std::size_t j = 0; j < retained.size(); ++j) {
            const bool forward  = window.rel(retained[i]) <= window.rel(retained[j]);
            const bool expected = retained[i] <= retained[j];
            if (forward != expected) { order_holds = false; break; }
        }
    }
    expect(order_holds, "rel() order over retained positions equals true-position order");

    expect(!window.retains(kBlock), "hole positions are not retained");
    expect(window.retains(total - 1), "the frontier is retained");
}

void test_window_rejects_a_call_inside_the_hole() {
    store::KVMemBlockStore block_store = make_store(5 * kBlock, small_config());
    const store::KVMemPlan plan         = block_store.select_recency();
    const store::KVMemWindow window = block_store.window_for(plan, 5 * kBlock - 1, 5 * kBlock - 1);
    expect(window.has_hole(), "this fixture has a hole");

    // A call whose range straddles the hole would need rel() at a hole position.
    expect(throws([&] { (void)block_store.window_for(plan, kBlock, kBlock + 10); }),
           "a call inside the hole throws (I1)");
    expect(throws([&] { (void)block_store.window_for(plan, kBlock, 2 * kBlock); }),
           "a call spanning into the hole throws (I1)");
    // A call entirely in the sink or entirely in the tail is legal.
    expect(!throws([&] { (void)block_store.window_for(plan, 0, kBlock - 1); }),
           "a call entirely in the sink is legal");
    expect(!throws([&] { (void)block_store.window_for(plan, 2 * kBlock, 5 * kBlock - 1); }),
           "a call entirely in the tail is legal");
}

void test_frontier_block_survives_a_budget_of_one_block() {
    store::KVMemConfig cfg = small_config();
    cfg.budget_tokens      = kBlock;
    store::KVMemBlockStore block_store = make_store(5 * kBlock, cfg);

    const store::KVMemPlan plan = block_store.select_recency();
    expect(ids_of(plan) == std::vector<std::uint32_t>({4}),
           "a one-block budget must spend its slot on the frontier, not the sink");

    const store::KVMemWindow window = block_store.window_for(plan, 5 * kBlock - 1, 5 * kBlock - 1);
    expect_eq(window.retained_pages, 2, "only the frontier block's two pages remain");
    expect(window.block_retained(4) && !window.block_retained(0),
           "the frontier block is retained and the dropped sink is not");
    expect_eq(window.rel(5 * kBlock - 1), kBlock - 1, "the frontier maps to the last window row");
    expect(throws([&] { (void)block_store.window_for(plan, 0, 10); }),
           "the dropped prefix is no longer visible");
}

void test_selection_diff_is_incremental() {
    store::KVMemBlockStore block_store = make_store(5 * kBlock, small_config());
    const store::KVMemPlan first        = block_store.select_recency();
    block_store.apply(first);

    block_store.register_append(kBlock); // sixth block
    const store::KVMemPlan second = block_store.select_recency();
    expect(ids_of(second) == std::vector<std::uint32_t>({0, 3, 4, 5}),
           "the window keeps the sink and slides its tail");
    expect_eq(static_cast<std::uint32_t>(second.stage_out.size()), 1,
              "only the block that fell out of the tail is staged out");
    expect_eq(static_cast<std::uint32_t>(second.stage_in.size()), 1,
              "only the newly entered tail block is staged in");
    expect_eq(second.stage_out[0], 2, "the oldest tail block is the one leaving");
    expect_eq(second.stage_in[0], 5, "the newest block is the one entering");

    const store::KVMemWindow window = block_store.window_for(second, 6 * kBlock - 1, 6 * kBlock - 1);
    expect_eq(window.retained_pages, 8, "a sliding window still retains four blocks of pages");
    expect_eq(window.row_base_by_block[3], kBlock, "the oldest tail block starts after the sink");
    expect_eq(window.rel(6 * kBlock - 1), 4 * kBlock - 1,
              "the window still ends at its last row after a slide");
}

void test_topk_degenerates_to_recency_without_scores() {
    store::KVMemBlockStore block_store = make_store(5 * kBlock, small_config());
    expect(ids_of(block_store.select_topk()) == ids_of(block_store.select_recency()),
           "an unscored top-k is exactly the recency selection");

    std::vector<double> scores(block_store.block_count(), 0.0);
    scores[1] = 10.0;
    block_store.set_scores(scores);
    const store::KVMemPlan plan = block_store.select_topk();
    expect(ids_of(plan) == std::vector<std::uint32_t>({0, 1, 3, 4}),
           "a scored middle block displaces the lowest scored tail block");
    expect_eq(static_cast<std::uint32_t>(plan.selected.size()), 4, "top-k still honors the budget");
}

void test_window_supports_blocks_wider_than_one_page_pair() {
    store::KVMemConfig cfg;
    cfg.block_tokens  = 256;
    cfg.budget_tokens = 512;
    cfg.sink_tokens   = 0;
    cfg.recent_blocks = 0;
    store::KVMemBlockStore block_store(cfg);
    block_store.register_append(3 * 256);
    const store::KVMemPlan plan = block_store.select_recency();
    expect(ids_of(plan) == std::vector<std::uint32_t>({0, 2}),
           "a two-block budget keeps the sink and the newest wide block");

    const store::KVMemWindow window = block_store.window_for(plan, 3 * 256 - 1, 3 * 256 - 1);
    expect_eq(window.block_tokens, 256, "the window keeps the configured block size");
    expect_eq(window.retained_pages, 8, "two 256-token blocks cover eight pages");
    expect_eq(window.row_base_by_block[2], 256, "the tail starts after the wide sink");
    expect_eq(window.rel(2 * 256), 256, "the tail maps to its dense row");
    expect(window.retains(512) && !window.retains(300),
           "retains() uses the configured block size, not a hard-coded 128");
}

void test_config_validation() {
    store::KVMemConfig bad_tokens = small_config();
    bad_tokens.block_tokens       = 0;
    expect(throws([&] { store::KVMemBlockStore store_(bad_tokens); }),
           "block_tokens must be positive");

    store::KVMemConfig bad_page = small_config();
    bad_page.block_tokens       = 100;
    expect(throws([&] { store::KVMemBlockStore store_(bad_page); }),
           "block_tokens must be a whole number of KV pages (I2)");

    store::KVMemConfig bad_budget = small_config();
    bad_budget.budget_tokens      = 2;
    expect(throws([&] { store::KVMemBlockStore store_(bad_budget); }),
           "budget must cover at least one block");

    store::KVMemBlockStore block_store(small_config());
    expect(throws([&] { (void)block_store.block(99); }), "an out of range block id throws");
    expect(throws([&] { (void)block_store.window_for(block_store.select_recency(), 0, 0); }),
           "a call past the context throws");
}

// Mean-K scoring geometry: two key heads of four dimensions each, so a hand-computable oracle
// covers the per-head scaling and the sum over heads.
constexpr std::uint32_t kHeads = 2;
constexpr std::uint32_t kDim   = 4;
constexpr std::uint32_t kWidth = kHeads * kDim;

store::KVMemScoreStore make_score_store() { return store::KVMemScoreStore(kBlock, kWidth, kDim); }

// One block token: eight F32 values, the shape a capture round reduces before the host stores it.
using Token = std::vector<float>;

// Independent FP64 oracle over the token lists themselves (never over the store's own mean): the
// block mean is the FP64 mean of its tokens, similarity is the per-head scaled dot product summed
// over heads, and the score is the softmax of those similarities.
std::vector<double> oracle_scores(const std::vector<std::vector<Token>>& tokens_per_block,
                                  const Token& query) {
    const std::size_t blocks = tokens_per_block.size();
    std::vector<double> similarity(blocks, 0.0);
    double largest           = -std::numeric_limits<double>::infinity();
    for (std::size_t id = 0; id < blocks; ++id) {
        double total = 0.0;
        for (std::uint32_t head = 0; head < kHeads; ++head) {
            double head_dot = 0.0;
            for (std::uint32_t index = 0; index < kDim; ++index) {
                const std::size_t column = static_cast<std::size_t>(head) * kDim + index;
                double mean              = 0.0;
                for (const Token& token : tokens_per_block[id]) {
                    mean += static_cast<double>(token[column]);
                }
                mean /= static_cast<double>(tokens_per_block[id].size());
                head_dot += static_cast<double>(query[column]) * mean;
            }
            total += head_dot;
        }
        similarity[id] = total / std::sqrt(static_cast<double>(kDim));
        largest        = std::max(largest, similarity[id]);
    }
    double denominator = 0.0;
    for (std::size_t id = 0; id < blocks; ++id) { denominator += std::exp(similarity[id] - largest); }
    for (std::size_t id = 0; id < blocks; ++id) {
        similarity[id] = std::exp(similarity[id] - largest) / denominator;
    }
    return similarity;
}

// Feed one block through `rounds` capture rounds: every round reduces `kBlock / rounds` copies of
// the same token into a partial sum, exactly as the device reduction and the host accumulation
// would see it.
void feed_block(store::KVMemScoreStore& scores, std::uint32_t block_id, const Token& token,
                std::uint32_t rounds, std::vector<Token>& tokens_out) {
    std::vector<float> partial(kWidth, 0.0F);
    for (std::uint32_t round = 0; round < rounds; ++round) {
        std::fill(partial.begin(), partial.end(), 0.0F);
        const std::uint32_t per_round = kBlock / rounds;
        for (std::uint32_t repeat = 0; repeat < per_round; ++repeat) {
            for (std::uint32_t index = 0; index < kWidth; ++index) { partial[index] += token[index]; }
            tokens_out.push_back(token);
        }
        scores.accumulate(block_id, partial.data(), per_round);
    }
    scores.finalize(block_id);
}

void expect_close(double actual, double expected, std::string_view message) {
    const double tolerance = 1e-6;
    if (std::abs(actual - expected) <= tolerance) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << " expected " << expected << ", got " << actual << '\n';
}

void test_score_matches_the_mean_k_oracle() {
    constexpr std::uint32_t blocks = 4;
    store::KVMemScoreStore scores = make_score_store();
    // Block vectors chosen so the ranking is not the position order: the query points at the first
    // and last coordinate, so the newest block scores highest and the first block lowest.
    std::vector<std::vector<Token>> tokens(blocks);
    std::vector<Token> vectors(blocks, Token(kWidth, 0.0F));
    for (std::uint32_t id = 0; id < blocks; ++id) {
        vectors[id][0]          = 1.0F + static_cast<float>(id);
        vectors[id][kWidth - 1] = static_cast<float>(id) * 0.5F;
        feed_block(scores, id, vectors[id], 4, tokens[id]);
    }

    Token query(kWidth, 0.0F);
    query[0]          = 1.0F;
    query[kWidth - 1] = 1.0F;
    const std::vector<double> actual = scores.score(query.data(), kWidth);
    const std::vector<double> oracle = oracle_scores(tokens, query);
    for (std::uint32_t id = 0; id < blocks; ++id) {
        expect_close(actual[id], oracle[id], "mean-K score matches the independent oracle");
    }
    // Softmax output: probabilities that sum to one over the scored blocks.
    double total = 0.0;
    for (const double value : actual) { total += value; }
    expect_close(total, 1.0, "mean-K scores form a distribution");
    for (std::uint32_t id = 0; id + 1 < blocks; ++id) {
        expect(actual[id] < actual[id + 1], "scores are monotone in this fixture");
    }
    expect(actual.back() > actual.front(),
           "the block most aligned with the query scores above the least aligned one");
}

void test_scores_survive_partial_capture_and_truncation() {
    store::KVMemScoreStore scores = make_score_store();
    expect(!scores.has_mean(0), "an uncaptured block has no mean");
    expect(throws([&] { scores.accumulate(0, nullptr, 8); }), "a null partial sum is rejected");
    expect(throws([&] { (void)scores.mean(0); }), "an unfinalized block has no mean to read");
    expect(throws([&] { (void)scores.mean(7); }), "an out of range block id throws");

    // A block captured over several rounds has the same mean as one single round.
    const Token constant(kWidth, 2.0F);
    std::vector<Token> tokens;
    feed_block(scores, 0, constant, 8, tokens);
    for (const float value : scores.mean(0)) {
        expect_close(value, 2.0, "accumulated mean equals the block mean");
    }

    // A partially captured block stays unfinalized and therefore unscorable.
    const std::vector<float> partial(kWidth, 1.0F);
    scores.accumulate(1, partial.data(), kBlock / 2);
    expect(throws([&] { scores.finalize(1); }), "an incomplete block cannot be finalized");
    expect(!scores.has_mean(1), "an incomplete block has no mean");
    expect_eq(scores.captured_tokens(1), kBlock / 2, "the partial token count is retained");
    expect(throws([&] { scores.accumulate(1, partial.data(), kBlock); }),
           "a partial sum that would overfill the block is rejected");

    expect(throws([&] { scores.accumulate(0, partial.data(), 1); }),
           "an already finalized block cannot accumulate");

    scores.truncate(1);
    expect_eq(scores.block_count(), 1, "truncate drops the blocks past the frontier");
    expect(scores.has_mean(0), "truncation keeps the surviving blocks' means");
    expect(throws([&] { scores.truncate(2); }), "truncate cannot extend the score store");
}

void test_topk_selection_uses_the_scores() {
    // Score store and block store agree on geometry: five blocks, budget of three, sink of one.
    constexpr std::uint32_t blocks = 5;
    store::KVMemConfig cfg;
    cfg.block_tokens  = kBlock;
    cfg.budget_tokens = 3 * kBlock;
    cfg.sink_tokens   = 0;
    cfg.recent_blocks = 0;
    store::KVMemBlockStore selection(cfg);
    selection.register_append(blocks * kBlock);

    store::KVMemScoreStore scores = make_score_store();
    for (std::uint32_t id = 0; id < blocks; ++id) {
        std::vector<float> summed(kWidth, 0.0F);
        summed[0] = id == 2 ? 10.0F : (id == 1 ? 5.0F : 0.0F);
        scores.accumulate(id, summed.data(), kBlock);
        scores.finalize(id);
    }
    Token query(kWidth, 0.0F);
    query[0] = 1.0F;
    selection.set_scores(scores.score(query.data(), kWidth));
    const store::KVMemPlan plan = selection.select_topk();
    // Budget of three blocks: sink block 0, the newest tail block 4, and the best-scoring middle
    // block 2 - a true interior block with holes on both sides of it.
    expect(ids_of(plan) == std::vector<std::uint32_t>({0, 2, 4}),
           "retrieval keeps the sink, an interior best-scoring block and the frontier block");
}

// The point of retrieval is an interior block that neither recency nor the call range would keep.
// The window must publish it at its dense row so attention can read it alongside the sink and tail.
void test_window_retains_interior_topk_blocks() {
    constexpr std::uint32_t blocks = 6;
    store::KVMemConfig cfg;
    cfg.block_tokens  = kBlock;
    cfg.budget_tokens = 4 * kBlock;
    cfg.sink_tokens   = 0;
    cfg.recent_blocks = 0;
    store::KVMemBlockStore selection(cfg);
    selection.register_append(blocks * kBlock);

    store::KVMemScoreStore scores = make_score_store();
    for (std::uint32_t id = 0; id < blocks; ++id) {
        std::vector<float> summed(kWidth, 0.0F);
        summed[0] = (id == 1 || id == 3) ? 10.0F : 0.0F;
        scores.accumulate(id, summed.data(), kBlock);
        scores.finalize(id);
    }
    Token query(kWidth, 0.0F);
    query[0] = 1.0F;
    selection.set_scores(scores.score(query.data(), kWidth));
    const store::KVMemPlan plan = selection.select_topk();
    expect(ids_of(plan) == std::vector<std::uint32_t>({0, 1, 3, 5}),
           "top-k keeps the sink, two interior blocks and the frontier block");

    const std::uint32_t total = blocks * kBlock;
    const store::KVMemWindow window = selection.window_for(plan, total - 1, total - 1);
    expect(window.has_hole(), "the skipped interior blocks are holes");
    expect_eq(window.retained_pages, 8, "four retained blocks cover eight pages");
    expect_eq(window.rel(1 * kBlock + 5), kBlock + 5, "the first interior block keeps its dense row");
    expect_eq(window.rel(3 * kBlock + 7), 2 * kBlock + 7, "the second interior block follows");
    expect(window.retains(1 * kBlock) && window.retains(3 * kBlock) &&
               !window.retains(2 * kBlock) && !window.retains(4 * kBlock),
           "retains() agrees with the selected interior blocks");
    expect_eq(window.rel(total - 1), 4 * kBlock - 1, "the frontier closes the dense prefix");
}

// The captured range is partitioned into one share per block it touches, exactly as the reduction
// wrote one column per block. These are the shares' own invariants: ascending contiguous block
// ids, every count positive, and the counts summing back to the range length - which together make
// column b of a captured segment attributable to shares[b].
void test_block_shares_partition_the_captured_range() {
    using Share = store::KVMemBlockShare;
    const auto same_shares = [](const std::vector<Share>& actual,
                               std::initializer_list<Share> expected_list) {
        const std::vector<Share> expected(expected_list);
        if (actual.size() != expected.size()) { return false; }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            if (actual[i].block_id != expected[i].block_id ||
                actual[i].tokens != expected[i].tokens) {
                return false;
            }
        }
        return true;
    };
    const auto shares_of = [](std::int64_t first, std::int64_t tokens, std::uint32_t block) {
        return store::kvmem_block_shares(first, tokens, block);
    };
    // A range strictly inside one block: one share, the whole range.
    expect(same_shares(shares_of(5, 10, kBlock), {{.block_id = 0, .tokens = 10}}),
           "a range inside one block is one share");
    // Starting 3 tokens before a block boundary and running 3 blocks covers four: a 3-token head, two
// whole blocks, and a 125-token tail that stops 3 short of the fourth boundary.
    expect(same_shares(shares_of(kBlock - 3, 3 * kBlock, kBlock),
                       {{.block_id = 0, .tokens = 3},
                        {.block_id = 1, .tokens = kBlock},
                        {.block_id = 2, .tokens = kBlock},
                        {.block_id = 3, .tokens = 3 * kBlock - 2 * kBlock - 3}}),
           "a range spanning blocks splits its head and tail");
    // A single token still yields exactly one share - the decode shape, one token per lane.
    expect(same_shares(shares_of(7 * kBlock + 5, 1, kBlock), {{.block_id = 7, .tokens = 1}}),
           "one token is one share");
    // Exactly block aligned on both ends: whole blocks only, no empty shares at either end.
    expect(same_shares(shares_of(kBlock, 2 * kBlock, kBlock),
                       {{.block_id = 1, .tokens = kBlock}, {.block_id = 2, .tokens = kBlock}}),
           "a block-aligned range yields whole blocks");
    // Every share is positive and the counts return the range length, across a wide sweep that
    // includes unaligned starts, sizes and block sizes.
    for (std::int64_t first : {0, 1, 63, 127, 128, 129, 1000, 4096}) {
        for (std::int64_t tokens : {1, 2, 7, 128, 129, 255, 1024, 2049}) {
            for (std::uint32_t block : {1U, 64U, 128U, 384U}) {
                const std::vector<store::KVMemBlockShare> shares = shares_of(first, tokens, block);
                std::int64_t total = 0;
                for (std::size_t i = 0; i < shares.size(); ++i) {
                    expect(shares[i].tokens > 0, "every captured block share has tokens");
                    total += shares[i].tokens;
                    if (i > 0) {
                        expect(shares[i].block_id == shares[i - 1U].block_id + 1U,
                               "captured block shares are ascending and contiguous");
                    }
                }
                expect(total == tokens, "captured block shares cover the range exactly");
            }
        }
    }
    // Malformed ranges are rejected rather than silently misattributed.
    expect(throws([&] { (void)shares_of(-1, 10, kBlock); }),
           "a negative captured position throws");
    expect(throws([&] { (void)shares_of(0, 0, kBlock); }),
           "an empty captured range throws");
    expect(throws([&] { (void)shares_of(0, 10, 0); }),
           "a zero block size throws");
    expect(throws([&] { (void)shares_of(static_cast<std::int64_t>(INT32_MAX), 4, kBlock); }),
           "a range past the position domain throws");
}

} // namespace

int main() {
    test_append_and_lookup();
    test_block_starts_stay_page_aligned();
    test_truncate();
    test_window_is_identity_when_the_context_fits();
    test_window_renumbers_sink_and_tail_into_one_prefix();
    test_window_rejects_a_call_inside_the_hole();
    test_frontier_block_survives_a_budget_of_one_block();
    test_selection_diff_is_incremental();
    test_topk_degenerates_to_recency_without_scores();
    test_window_supports_blocks_wider_than_one_page_pair();
    test_config_validation();
    test_score_matches_the_mean_k_oracle();
    test_scores_survive_partial_capture_and_truncation();
    test_topk_selection_uses_the_scores();
    test_window_retains_interior_topk_blocks();
    test_block_shares_partition_the_captured_range();

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
