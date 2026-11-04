#include "models/qwen3_5/program/storage/kvmem_store.h"

#include "core/paged_kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::detail {

namespace {

constexpr std::uint32_t page_tokens() noexcept {
    return static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / page_tokens();
}

} // namespace

void KVMemBlockStore::validate_config(const KVMemConfig& cfg) const {
    if (cfg.block_tokens == 0) {
        throw std::invalid_argument("kvmem block_tokens must be positive");
    }
    if (cfg.block_tokens % page_tokens() != 0) {
        // Invariant I2 rests on block boundaries landing on page boundaries, so a block that is
        // not a whole number of pages cannot be renumbered without moving bytes.
        throw std::invalid_argument("kvmem block_tokens must be a whole number of KV pages");
    }
    if (cfg.budget_tokens < cfg.block_tokens) {
        throw std::invalid_argument("kvmem budget_tokens must cover at least one block");
    }
}

KVMemBlockStore::KVMemBlockStore(KVMemConfig cfg) : cfg_(cfg) { validate_config(cfg_); }

std::uint32_t KVMemBlockStore::budget_blocks() const noexcept {
    return cfg_.budget_tokens / cfg_.block_tokens;
}

std::uint32_t KVMemBlockStore::sink_blocks() const noexcept {
    // A zero sink still keeps one block so a working set is never empty, matching the reference
    // implementation's default.
    return std::max<std::uint32_t>(
        1, (cfg_.sink_tokens + cfg_.block_tokens - 1U) / cfg_.block_tokens);
}

const KVMemBlock& KVMemBlockStore::block(std::uint32_t block_id) const {
    if (block_id >= block_count()) {
        throw std::out_of_range("kvmem block id " + std::to_string(block_id) + " is out of range");
    }
    return blocks_[block_id];
}

std::uint32_t KVMemBlockStore::block_id_containing(std::uint32_t pos) const noexcept {
    if (pos >= total_tokens_) { return block_count(); }
    // Blocks are stored ascending, non-overlapping and gap-free, so the first block whose end
    // passes pos contains it.
    for (std::uint32_t id = 0; id < block_count(); ++id) {
        if (pos < blocks_[id].orig_pos_end()) { return id; }
    }
    return block_count();
}

std::uint32_t KVMemBlockStore::register_append(std::uint32_t n_tokens) {
    std::uint32_t newly_full = 0;
    while (n_tokens > 0) {
        if (blocks_.empty() || blocks_.back().n_tokens == cfg_.block_tokens) {
            KVMemBlock next;
            next.block_id       = static_cast<std::uint32_t>(blocks_.size());
            next.orig_pos_start = total_tokens_;
            blocks_.push_back(next);
        }
        KVMemBlock& trailing     = blocks_.back();
        const std::uint32_t room = cfg_.block_tokens - trailing.n_tokens;
        const std::uint32_t take = std::min(n_tokens, room);
        trailing.n_tokens += take;
        total_tokens_ += take;
        n_tokens -= take;
        if (take != 0 && trailing.n_tokens == cfg_.block_tokens) { ++newly_full; }
    }
    return newly_full;
}

std::vector<KVMemBlock> KVMemBlockStore::truncate_to(std::uint32_t token_pos) {
    std::vector<KVMemBlock> dropped;
    if (token_pos >= total_tokens_) { return dropped; }

    while (!blocks_.empty()) {
        KVMemBlock& trailing = blocks_.back();
        if (trailing.orig_pos_start >= token_pos) {
            dropped.push_back(trailing);
            blocks_.pop_back();
            continue;
        }
        if (trailing.orig_pos_end() > token_pos) {
            trailing.n_tokens = token_pos - trailing.orig_pos_start;
        }
        break;
    }

    total_tokens_  = token_pos;
    for (KVMemBlock& entry : blocks_) { entry.in_window = false; }
    window_tokens_ = 0;
    return dropped;
}

KVMemPlan KVMemBlockStore::finish_plan(std::vector<std::uint32_t> selected) const {
    const std::uint32_t count = block_count();
    std::vector<bool> chosen(count, false);
    KVMemPlan plan;
    plan.selected = std::move(selected);
    for (const std::uint32_t id : plan.selected) {
        if (id >= count) {
            throw std::logic_error("kvmem selection names a block that does not exist");
        }
        if (chosen[id]) { throw std::logic_error("kvmem selection repeats a block"); }
        chosen[id]     = true;
        plan.window_tokens += blocks_[id].n_tokens;
    }
    if (std::is_sorted(plan.selected.begin(), plan.selected.end()) == false) {
        throw std::logic_error("kvmem selection is not ascending");
    }

    plan.stage_in.reserve(plan.selected.size());
    plan.stage_out.reserve(plan.selected.size());
    for (std::uint32_t id = 0; id < count; ++id) {
        if (chosen[id] && !blocks_[id].in_window) { plan.stage_in.push_back(id); }
        if (!chosen[id] && blocks_[id].in_window) { plan.stage_out.push_back(id); }
    }
    return plan;
}

KVMemPlan KVMemBlockStore::select_recency() const {
    const std::uint32_t count = block_count();
    const std::uint32_t keep  = std::min(count, budget_blocks());
    std::uint32_t head        = std::min(sink_blocks(), keep);
    // The trailing block owns the frontier, so a budget that only reaches as far as the sink can
    // never decode. Spend the last slot on the tail instead of making the window undecodable.
    if (count > keep && head == keep) { head = keep - 1; }
    const std::uint32_t tail = keep - head;

    // Fill the whole budget from the newest tail. When the context fits the budget every block
    // is selected, which is exact full attention over the context.
    std::vector<std::uint32_t> selected;
    selected.reserve(keep);
    for (std::uint32_t id = 0; id < head; ++id) { selected.push_back(id); }
    for (std::uint32_t id = count - tail; id < count; ++id) { selected.push_back(id); }
    if (selected.size() != keep) {
        throw std::logic_error("kvmem recency selection did not fill its budget");
    }
    return finish_plan(std::move(selected));
}

KVMemPlan KVMemBlockStore::select_topk() const {
    const std::uint32_t count = block_count();
    const std::uint32_t keep  = std::min(count, budget_blocks());
    std::uint32_t head        = std::min(sink_blocks(), keep);
    // Same frontier guarantee as select_recency: the newest block must survive selection.
    if (count > keep && head == keep) { head = keep - 1; }
    const std::uint32_t keep_head = keep - head;
    // recent_blocks only bites here: under Recency the budget already fills from the tail, so a
    // recent floor smaller than the remaining budget is indistinguishable from no floor at all.
    // One trailing block is always reserved because it owns the frontier.
    const std::uint32_t recent =
        std::min(keep_head, std::max(cfg_.recent_blocks, count > keep ? 1U : 0U));
    const std::uint32_t mid_budget = keep_head - recent;
    const std::uint32_t mid_begin  = head;
    const std::uint32_t mid_end    = count - recent;

    std::vector<std::uint32_t> middle;
    middle.reserve(mid_end - mid_begin);
    for (std::uint32_t id = mid_begin; id < mid_end; ++id) { middle.push_back(id); }
    // Scores are all zero until phase 4b installs a scorer, so ties fall back to the newest
    // block id and this degenerates into exactly the Recency selection.
    std::sort(middle.begin(), middle.end(), [this](std::uint32_t lhs, std::uint32_t rhs) {
        if (blocks_[lhs].score != blocks_[rhs].score) {
            return blocks_[lhs].score > blocks_[rhs].score;
        }
        return lhs > rhs;
    });
    middle.resize(std::min<std::size_t>(mid_budget, middle.size()));

    std::vector<std::uint32_t> selected;
    selected.reserve(keep);
    for (std::uint32_t id = 0; id < head; ++id) { selected.push_back(id); }
    for (std::uint32_t id : middle) { selected.push_back(id); }
    for (std::uint32_t id = mid_end; id < count; ++id) { selected.push_back(id); }
    std::sort(selected.begin(), selected.end());
    if (selected.size() != keep) {
        throw std::logic_error("kvmem top-k selection did not fill its budget");
    }
    return finish_plan(std::move(selected));
}

void KVMemBlockStore::apply(const KVMemPlan& plan) {
    const std::uint32_t count = block_count();
    for (const std::uint32_t id : plan.selected) {
        if (id >= count) { throw std::logic_error("kvmem plan names a block that does not exist"); }
    }
    for (KVMemBlock& entry : blocks_) { entry.in_window = false; }
    window_tokens_ = 0;
    for (const std::uint32_t id : plan.selected) {
        blocks_[id].in_window = true;
        window_tokens_ += blocks_[id].n_tokens;
    }
}

KVMemWindow KVMemBlockStore::window_for(const KVMemPlan& plan, std::uint32_t call_min_pos,
                                        std::uint32_t call_max_pos) const {
    const std::uint32_t count = block_count();
    if (call_min_pos > call_max_pos || call_max_pos >= total_tokens_) {
        throw std::logic_error("kvmem window asked about a position past the context");
    }

    KVMemWindow window;
    window.block_tokens = cfg_.block_tokens;
    window.total_tokens = total_tokens_;
    window.window_tokens = plan.window_tokens;

    // The identity view is the whole selection: leave the map empty so rel()/retains() take the
    // dense path and the executor publishes every mapped page.
    bool identity = plan.selected.size() == count;
    for (std::size_t index = 0; identity && index < plan.selected.size(); ++index) {
        identity = plan.selected[index] == index;
    }
    if (identity) {
        if (window.window_tokens != total_tokens_) {
            throw std::logic_error("kvmem identity plan does not cover the whole context");
        }
        window.retained_pages = pages_for_tokens(total_tokens_);
    } else {
        window.row_base_by_block.assign(count, std::numeric_limits<std::uint32_t>::max());
        std::uint32_t row      = 0;
        std::uint32_t pages    = 0;
        std::uint32_t previous = 0;
        for (std::size_t index = 0; index < plan.selected.size(); ++index) {
            const std::uint32_t id = plan.selected[index];
            if (id >= count) {
                throw std::logic_error("kvmem selection names a block that does not exist");
            }
            if (index != 0 && id <= previous) {
                throw std::logic_error("kvmem selection is not strictly ascending");
            }
            previous                       = id;
            window.row_base_by_block[id]   = row;
            row += blocks_[id].n_tokens;
            pages += pages_for_tokens(blocks_[id].n_tokens);
        }
        if (row != window.window_tokens) {
            throw std::logic_error("kvmem plan window_tokens disagrees with its selected blocks");
        }
        window.retained_pages = pages;
    }

    if (window.window_tokens > cfg_.budget_tokens) {
        throw std::logic_error("kvmem window exceeds the configured budget");
    }

    // Every block the next call touches must be retained. Checking only the endpoints would let a
    // call span a hole, where one constant row shift cannot describe the whole range.
    const std::uint32_t first = block_id_containing(call_min_pos);
    const std::uint32_t last  = block_id_containing(call_max_pos);
    if (first >= count || last >= count) {
        throw std::logic_error("kvmem window asked about a position past the context");
    }
    for (std::uint32_t id = first; id <= last; ++id) {
        if (!window.block_retained(id)) {
            throw std::logic_error("kvmem window does not retain the next call's position range");
        }
    }
    return window;
}

bool KVMemBlockStore::window_row(std::uint32_t block_id, std::uint32_t& row_out) const noexcept {
    if (block_id >= block_count() || !blocks_[block_id].in_window) { return false; }
    std::uint32_t row = 0;
    for (std::uint32_t id = 0; id < block_id; ++id) {
        if (blocks_[id].in_window) { row += blocks_[id].n_tokens; }
    }
    row_out = row;
    return true;
}

void KVMemBlockStore::set_scores(std::vector<double> scores) {
    for (std::uint32_t id = 0; id < block_count(); ++id) {
        blocks_[id].score = id < scores.size() ? scores[id] : 0.0;
    }
}

KVMemScoreStore::KVMemScoreStore(std::uint32_t block_tokens, std::uint32_t vector_width,
                                 std::uint32_t head_dim)
    : block_tokens_(block_tokens), vector_width_(vector_width), head_dim_(head_dim) {
    if (block_tokens == 0 || vector_width == 0 || head_dim == 0 || vector_width % head_dim != 0) {
        throw std::invalid_argument("kvmem mean-K geometry is invalid");
    }
}

const KVMemScoreStore::Block& KVMemScoreStore::require_block(std::uint32_t block_id) const {
    if (block_id >= block_count()) {
        throw std::logic_error("kvmem score store block does not exist");
    }
    return blocks_[block_id];
}

KVMemScoreStore::Block& KVMemScoreStore::require_block(std::uint32_t block_id) {
    if (block_id >= block_count()) {
        throw std::logic_error("kvmem score store block does not exist");
    }
    return blocks_[block_id];
}

void KVMemScoreStore::accumulate(std::uint32_t block_id, const float* partial_sum,
                                 std::uint32_t count) {
    if (partial_sum == nullptr || count == 0 || count > block_tokens_) {
        throw std::invalid_argument("kvmem mean-K partial sum is not applicable to this block");
    }
    // Writes extend the store: capture appends blocks as the context grows, exactly as the block
    // store's register_append does. Only reads require an existing block.
    if (block_id >= block_count()) { blocks_.resize(static_cast<std::size_t>(block_id) + 1U); }
    Block& block = require_block(block_id);
    if (!block.mean.empty() || count > block_tokens_ - block.tokens) {
        throw std::invalid_argument("kvmem mean-K partial sum does not fit this block");
    }
    if (block.sum.empty()) { block.sum.assign(vector_width_, 0.0F); }
    for (std::uint32_t index = 0; index < vector_width_; ++index) {
        block.sum[index] += partial_sum[index];
    }
    block.tokens += count;
}

void KVMemScoreStore::finalize(std::uint32_t block_id) {
    Block& block = require_block(block_id);
    if (block.tokens != block_tokens_ || block.sum.empty()) {
        throw std::logic_error("kvmem mean-K finalize needs a complete block");
    }
    const float scale = 1.0F / static_cast<float>(block_tokens_);
    block.mean.resize(vector_width_);
    for (std::uint32_t index = 0; index < vector_width_; ++index) {
        block.mean[index] = block.sum[index] * scale;
    }
    block.sum.clear();
    block.sum.shrink_to_fit();
}

void KVMemScoreStore::truncate(std::uint32_t count) {
    if (count > block_count()) {
        throw std::invalid_argument("kvmem mean-K truncate cannot extend the score store");
    }
    blocks_.resize(count);
}

bool KVMemScoreStore::has_mean(std::uint32_t block_id) const {
    return block_id < block_count() && !blocks_[block_id].mean.empty();
}

std::uint32_t KVMemScoreStore::captured_tokens(std::uint32_t block_id) const {
    return require_block(block_id).tokens;
}

const std::vector<float>& KVMemScoreStore::mean(std::uint32_t block_id) const {
    const Block& block = require_block(block_id);
    if (block.mean.empty()) { throw std::logic_error("kvmem mean-K block is not finalized"); }
    return block.mean;
}

std::vector<double> KVMemScoreStore::score(const float* query, std::uint32_t width) const {
    if (query == nullptr || width != vector_width_) {
        throw std::invalid_argument("kvmem mean-K query width does not match the store");
    }
    // Scaled dot product per key head, summed over heads: the 1/sqrt(head_dim) factor is the same
    // temperature attention uses, so a dot product here is comparable to a pre-softmax score.
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));
    std::vector<double> similarity(block_count(), 0.0);
    double largest = 0.0;
    bool scored    = false;
    for (std::uint32_t id = 0; id < block_count(); ++id) {
        const Block& block = blocks_[id];
        if (block.mean.empty()) { continue; }
        double total = 0.0;
        for (std::uint32_t head = 0; head * head_dim_ < vector_width_; ++head) {
            const std::uint32_t base = head * head_dim_;
            double head_dot            = 0.0;
            for (std::uint32_t index = 0; index < head_dim_; ++index) {
                head_dot += static_cast<double>(query[base + index]) *
                            static_cast<double>(block.mean[base + index]);
            }
            total += head_dot;
        }
        similarity[id] = total * scale;
        if (!scored || similarity[id] > largest) { largest = similarity[id]; }
        scored = true;
    }
    if (!scored) { return similarity; }
    // Softmax over the scored blocks; blocks without a mean keep 0, which ranks them behind every
    // scored block (a positive probability is always greater) while staying selectable.
    double denominator = 0.0;
    std::vector<double> weights(block_count(), 0.0);
    for (std::uint32_t id = 0; id < block_count(); ++id) {
        if (blocks_[id].mean.empty()) { continue; }
        weights[id] = std::exp(similarity[id] - largest);
        denominator += weights[id];
    }
    for (std::uint32_t id = 0; id < block_count(); ++id) {
        if (weights[id] != 0.0) { similarity[id] = weights[id] / denominator; }
    }
    return similarity;
}

std::vector<KVMemBlockShare> kvmem_block_shares(std::int64_t first_position, std::int64_t tokens,
                                               std::uint32_t block_tokens) {
    if (first_position < 0 || tokens <= 0 || block_tokens == 0) {
        throw std::invalid_argument("kvmem captured range is not applicable");
    }
    // The last position must stay inside the Op's INT32_MAX domain, which is what makes the
    // block_count the Op derived and the shares derived here the same partition.
    if (first_position + tokens - 1 > static_cast<std::int64_t>(INT32_MAX)) {
        throw std::invalid_argument("kvmem captured range exceeds the position domain");
    }
    const std::int64_t span  = static_cast<std::int64_t>(block_tokens);
    const std::int64_t first = first_position / span;
    const std::int64_t last  = (first_position + tokens - 1) / span;
    std::vector<KVMemBlockShare> shares;
    shares.reserve(static_cast<std::size_t>(last - first + 1));
    for (std::int64_t block = first; block <= last; ++block) {
        const std::int64_t begin = std::max(first_position, block * span);
        const std::int64_t end   = std::min(first_position + tokens, (block + 1) * span);
        shares.push_back(KVMemBlockShare{.block_id = static_cast<std::uint32_t>(block),
                                         .tokens   = static_cast<std::uint32_t>(end - begin)});
    }
    return shares;
}

std::vector<KVMemBlockShare> kvmem_store_block_shares(const KVMemBlockStore& store,
                                                     std::uint32_t first_position,
                                                     std::uint32_t tokens) {
    if (tokens == 0) {
        throw std::invalid_argument("kvmem captured range is not applicable");
    }
    const std::uint64_t range_end = static_cast<std::uint64_t>(first_position) + tokens;
    if (range_end > store.total_tokens()) {
        throw std::invalid_argument("kvmem captured range exceeds the store");
    }
    std::vector<KVMemBlockShare> shares;
    shares.reserve(store.block_count());
    for (std::uint32_t id = 0; id < store.block_count(); ++id) {
        const KVMemBlock& block = store.block(id);
        const std::uint32_t begin = std::max(first_position, block.orig_pos_start);
        const std::uint32_t end   = std::min(static_cast<std::uint32_t>(range_end),
                                            block.orig_pos_end());
        if (end > begin) {
            shares.push_back(KVMemBlockShare{.block_id = id, .tokens = end - begin});
        }
        if (block.orig_pos_start >= range_end) { break; }
    }
    if (shares.empty()) {
        throw std::invalid_argument("kvmem captured range touches no block");
    }
    return shares;
}

} // namespace ninfer::models::qwen3_5::detail
