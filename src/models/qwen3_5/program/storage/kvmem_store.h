#pragma once

// Block repository and bounded-working-set selection for KVMem-style retrieval.
//
// Pure host logic: no CUDA, no Program dependency, no physical page handles. The executor
// consumes KVMemPlan to decide which logical blocks back the execution-table window this round
// and which blocks must cross the Device/Host tier boundary.
//
// Granularity: selection and scoring work in `block_tokens`-token logical blocks (128 by
// default), while physical storage and transfer stay in the page store's 64-token logical pages.
// One block is exactly two pages, so a block is a natural extent membership.
//
// Window order is always ascending TRUE token position. The executor publishes those blocks into
// a dense window-row prefix through KVMemWindow: the causal mask reads
// `row <= cache_positions[t]`, and renumbering keeps row order identical to true-position order,
// so mask order and visibility order agree while K itself stays baked at its true position.
// See docs/maintainer/kvmem-port.md 2.12.

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct KVMemConfig {
    // Selection/transfer block granularity in tokens. Physical pages stay at 64 tokens.
    std::uint32_t block_tokens = 128;
    // Tokens the retrieved working set may occupy on the Device pool.
    std::uint32_t budget_tokens = 32768;
    // Always-kept prefix. Zero selects one block, matching the reference implementation, so the
    // sink is never accidentally empty.
    std::uint32_t sink_tokens = 0;
    // Always-kept suffix blocks under the Recency policy. Zero keeps no suffix block beyond what
    // the budget already buys from the tail.
    std::uint32_t recent_blocks = 0;
};

struct KVMemBlock {
    std::uint32_t block_id         = 0;
    std::uint32_t orig_pos_start   = 0;
    std::uint32_t n_tokens         = 0;
    bool          in_window        = false;
    bool          host_resident    = false;
    double        score            = 0.0;

    [[nodiscard]] std::uint32_t orig_pos_end() const noexcept {
        return orig_pos_start + n_tokens;
    }
};

// Diff of one selection round. `selected` is ascending true position and defines the window row
// layout: the window row of selected[i] is the sum of n_tokens over selected[0..i).
struct KVMemPlan {
    std::vector<std::uint32_t> stage_in;
    std::vector<std::uint32_t> stage_out;
    std::vector<std::uint32_t> selected;
    std::uint32_t window_tokens = 0;
};

// Renumbered-prefix view of one selection (docs/maintainer/kvmem-port.md 2.12).
//
// `row_base_by_block[id]` is the dense window row of the first token of block `id`, or UINT32_MAX
// when the selection dropped that block. The map is empty exactly when the selection retains every
// block (the identity view), so rel() and retains() stay branch-light on the full-context path.
// A selection may retain arbitrary interior blocks: every retained page is published contiguously
// in ascending true-position order, so the unchanged causal mask `row <= cache_positions[t]`
// selects exactly the retained positions at or before the query.
//
// rel() is defined only for retained positions. Hole positions are never selected and must never
// be queried (invariants I1 and I3); the store validates I2, that block boundaries are whole
// 64-token pages, which is what makes a reselect a pure execution-table update with no device byte
// movement.
struct KVMemWindow {
    // Selection block granularity. Zero only for a default-constructed identity window.
    std::uint32_t block_tokens = 0;
    // Absolute tokens the selection covers (the block store's total at selection time).
    std::uint32_t total_tokens = 0;
    // Tokens the retained blocks hold.
    std::uint32_t window_tokens = 0;
    // 64-token pages the retained blocks cover; the execution table's contiguous row count.
    std::uint32_t retained_pages = 0;
    std::vector<std::uint32_t> row_base_by_block;

    [[nodiscard]] bool is_identity() const noexcept { return row_base_by_block.empty(); }

    [[nodiscard]] bool block_retained(std::uint32_t block_id) const noexcept {
        if (is_identity()) { return true; }
        return block_id < row_base_by_block.size() &&
               row_base_by_block[block_id] != std::numeric_limits<std::uint32_t>::max();
    }

    [[nodiscard]] bool has_hole() const noexcept {
        if (is_identity()) { return false; }
        for (const std::uint32_t base : row_base_by_block) {
            if (base == std::numeric_limits<std::uint32_t>::max()) { return true; }
        }
        return false;
    }

    // Window row of a retained absolute position. Throws std::out_of_range for a hole or a
    // position outside the selected context, because a silent fallback would read the wrong KV.
    [[nodiscard]] std::uint32_t rel(std::uint32_t abs_pos) const {
        if (is_identity()) { return abs_pos; }
        if (block_tokens == 0) {
            throw std::out_of_range("kvmem window has no block geometry");
        }
        const std::uint32_t block_id = abs_pos / block_tokens;
        if (block_id >= row_base_by_block.size() ||
            row_base_by_block[block_id] == std::numeric_limits<std::uint32_t>::max()) {
            throw std::out_of_range("kvmem window does not retain this position");
        }
        return row_base_by_block[block_id] + (abs_pos - block_id * block_tokens);
    }

    // Whether an absolute position belongs to the retained union. The identity view retains the
    // whole address space (including pages mapped ahead of the context), matching the dense
    // publication every caller assumes before a budget opens a hole.
    [[nodiscard]] bool retains(std::uint32_t abs_pos) const noexcept {
        if (is_identity()) { return true; }
        if (block_tokens == 0 || abs_pos >= total_tokens) { return false; }
        return block_retained(abs_pos / block_tokens);
    }

    // Whether a reselect can keep the published execution table: same block geometry, same
    // retained page count and the same block->row map. Token growth inside an already retained page
    // does not move a row.
    [[nodiscard]] bool same_layout(const KVMemWindow& other) const noexcept {
        return block_tokens == other.block_tokens &&
               retained_pages == other.retained_pages &&
               row_base_by_block == other.row_base_by_block;
    }
};

class KVMemBlockStore {
public:
    explicit KVMemBlockStore(KVMemConfig cfg);

    [[nodiscard]] const KVMemConfig& config() const noexcept { return cfg_; }
    [[nodiscard]] std::uint32_t block_count() const noexcept {
        return static_cast<std::uint32_t>(blocks_.size());
    }
    [[nodiscard]] std::uint32_t total_tokens() const noexcept { return total_tokens_; }
    [[nodiscard]] const std::vector<KVMemBlock>& blocks() const noexcept { return blocks_; }
    [[nodiscard]] const KVMemBlock& block(std::uint32_t block_id) const;
    [[nodiscard]] std::uint32_t budget_blocks() const noexcept;

    // Blocks that own original token position `pos`, or block_count() when no block does.
    [[nodiscard]] std::uint32_t block_id_containing(std::uint32_t pos) const noexcept;

    // Register newly appended context tokens. Partial trailing blocks extend in place; a full
    // trailing block starts a new one. Returns how many blocks became newly full (0 or 1, since
    // only the trailing block can change state).
    std::uint32_t register_append(std::uint32_t n_tokens);

    // Inverse of register_append: drop trailing blocks so the store holds exactly token_pos
    // tokens. A partially covered trailing block shrinks in place; fully past blocks are popped.
    // Returns the popped blocks so the caller can release their tier slots. token_pos beyond
    // total_tokens is a no-op.
    std::vector<KVMemBlock> truncate_to(std::uint32_t token_pos);

    // Recency selection: keep the sink head, then fill the remaining budget from the newest tail.
    // When the whole context fits the budget every block is selected, which is exact full
    // attention. Produces a diff against the current window membership; it does not apply it.
    [[nodiscard]] KVMemPlan select_recency() const;

    // Apply a plan produced by this store: flips in_window membership and records window order.
    void apply(const KVMemPlan& plan);

    // Derive the renumbered-prefix window for a selection, given the true-position range of the
    // next attention call. Pure: it reads `plan.selected`, the block table and the range only.
    // Throws std::logic_error when a validated invariant would be violated, because every one of
    // them fails silently at output rather than crashing:
    //   I1 every block the next call's position range touches is retained
    //   I2 block boundaries are multiples of the 64-token page (no byte movement on reselect)
    //   the window fits the configured budget
    [[nodiscard]] KVMemWindow window_for(const KVMemPlan& plan, std::uint32_t call_min_pos,
                                         std::uint32_t call_max_pos) const;

    // Window row of a block under the current membership, or false when it is not in the window.
    [[nodiscard]] bool window_row(std::uint32_t block_id, std::uint32_t& row_out) const noexcept;

    // Selector/scoring hook for a later retrieval round (phase 4b). Overwrites scores and returns
    // a plan over the same budget; identical to select_recency until a scorer exists.
    void set_scores(std::vector<double> scores);
    // `recent_floor` forces the newest `recent_floor` blocks into the window regardless of score, so
    // a caller whose next call spans a contiguous tail (MTP verify/draft) can guarantee every block
    // it touches is retained. The remaining interior budget is still ranked by the pending scores.
    // The floor is capped by the budget after the sink; a call range that cannot fit then fails
    // window_for's I1 instead of silently reading holes.
    [[nodiscard]] KVMemPlan select_topk(std::uint32_t recent_floor = 0) const;

private:
    void validate_config(const KVMemConfig& cfg) const;
    [[nodiscard]] KVMemPlan finish_plan(std::vector<std::uint32_t> selected) const;
    [[nodiscard]] std::uint32_t sink_blocks() const noexcept;

    KVMemConfig cfg_;
    std::vector<KVMemBlock> blocks_;
    std::uint32_t total_tokens_ = 0;
    std::uint32_t window_tokens_ = 0;
};

// Mean-K retrieval scores (kvmem-port.md 1.4, 2.6 stage 4b). One F32 pre-RoPE key mean per
// selection block, accumulated on the host from device-reduced partial sums, plus the similarity
// ranking a query vector produces over those means. Pure host logic - no CUDA, no Program
// dependency, no physical page handles - so the whole scoring path is unit-testable (2.8).
//
// A block spans capture rounds: every round that touches it contributes a partial sum for the
// tokens of that block it covered, and `finalize` divides by the token count once the block is
// full. `score` ranks blocks by the scaled dot product between the query and their means,
// softmaxed over the blocks that have one; softmax is monotone in the dot product, so the
// selection ranking is the raw inner product and the stored score is its probability.
class KVMemScoreStore {
public:
    // `vector_width` is the captured key width (key heads x head dimension), `head_dim` the
    // per-head dimension the similarity scales by. Both are fixed by the model geometry.
    KVMemScoreStore(std::uint32_t block_tokens, std::uint32_t vector_width, std::uint32_t head_dim);

    [[nodiscard]] std::uint32_t block_tokens() const noexcept { return block_tokens_; }
    [[nodiscard]] std::uint32_t vector_width() const noexcept { return vector_width_; }
    [[nodiscard]] std::uint32_t block_count() const noexcept {
        return static_cast<std::uint32_t>(blocks_.size());
    }

    // Add one device partial sum for `block_id`: the sum over `count` of that block's tokens.
    // Throws when the vector width, a zero count, or the accumulated token count does not fit
    // the block, because a partial sum that silently overflows becomes a wrong mean.
    void accumulate(std::uint32_t block_id, const float* partial_sum, std::uint32_t count);

    // Promote a full block to its mean. A block whose accumulated token count is not the block
    // size (still growing, or truncated) stays unfinalized and cannot be scored.
    void finalize(std::uint32_t block_id);

    // Forget every block at or beyond `count` (context truncation or prefix fork).
    void truncate(std::uint32_t count);

    [[nodiscard]] bool has_mean(std::uint32_t block_id) const;
    [[nodiscard]] bool has_tokens(std::uint32_t block_id) const;
    [[nodiscard]] std::uint32_t captured_tokens(std::uint32_t block_id) const;
    [[nodiscard]] const std::vector<float>& mean(std::uint32_t block_id) const;
    // Mean of the captured keys of a block that may still be filling. The retrieval query lives in
    // the frontier block, which holds the prompt's tail until the next tokens arrive, so the first
    // decode cannot wait for it to finalize.
    [[nodiscard]] std::vector<float> mean_for_query(std::uint32_t block_id) const;

    // Softmaxed similarity of every block mean against `query`. Blocks without a mean score 0,
    // which places them after every scored block while still letting them be selected when the
    // budget has room left.
    [[nodiscard]] std::vector<double> score(const float* query, std::uint32_t width) const;

private:
    struct Block {
        std::vector<float> sum;   // device partial sums, cleared once finalized
        std::vector<float> mean;  // sum / block_tokens, empty until finalized
        std::uint32_t tokens = 0;
    };

    [[nodiscard]] const Block& require_block(std::uint32_t block_id) const;
    [[nodiscard]] Block& require_block(std::uint32_t block_id);

    std::uint32_t block_tokens_ = 0;
    std::uint32_t vector_width_ = 0;
    std::uint32_t head_dim_ = 0;
    std::vector<Block> blocks_;
};

// One block a captured range contributes to, and how much of that range falls inside it.
//
// block_sum_by_position reduces a whole range of tokens into one column per block that range
// touches, so the host side has to recover each column's block identity and token count from the
// same arithmetic the Op used. Deriving it here - pure, from the range alone - keeps the two
// descriptions of the same range from drifting apart.
struct KVMemBlockShare {
    std::uint32_t block_id = 0;
    std::uint32_t tokens   = 0;  // tokens of this block inside the range; always positive
};

// Blocks that the token range [first_position, first_position + tokens) touches, ascending.
// Throws std::invalid_argument for a negative position or an empty range, because such a range
// misattributes partial sums instead of failing.
[[nodiscard]] std::vector<KVMemBlockShare>
kvmem_block_shares(std::int64_t first_position, std::int64_t tokens, std::uint32_t block_tokens);

// Blocks that the token range [first_position, first_position + tokens) touches in this store's
// actual (possibly truncated and re-appended) block layout. Unlike kvmem_block_shares above, this
// uses the store's real block boundaries, so capture stays correct after a non-block-aligned
// truncation or prefix fork.
[[nodiscard]] std::vector<KVMemBlockShare>
kvmem_store_block_shares(const KVMemBlockStore& store, std::uint32_t first_position,
                         std::uint32_t tokens);

} // namespace ninfer::models::qwen3_5::detail
