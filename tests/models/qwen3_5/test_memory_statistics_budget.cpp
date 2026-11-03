#include "models/qwen3_5/program/planning/memory_statistics_layout.h"
#include "models/qwen3_5/program/storage/attention_feedback_mass.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::models::qwen3_5::detail;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F>
void expect_rejected(F&& call) {
    bool rejected = false;
    try { call(); } catch (const std::exception&) { rejected = true; }
    expect(rejected, "invalid attention mass was accepted");
}

void test_page_mass() {
    // Physical compact order differs from logical order; each local mean is
    // independently represented by three identical heads and one other head.
    const std::vector<std::uint32_t> pages{11, 2, 40, 99};
    const float means[3][2][3]{
        {{.5f, .25f, .25f}, {0, .5f, .5f}},
        {{.25f, .25f, .5f}, {.75f, .25f, 0}},
        {{.125f, .625f, .25f}, {.25f, .25f, .5f}}};
    std::vector<kvmem::AcceptedAttentionRow> rows(2);
    rows[0].source_position = 6400;
    rows[1].source_position = 6416;
    AttentionPageMassAccumulator accumulator(rows, pages, 3, 4);
    for (std::size_t layer = 0; layer < 3; ++layer) {
        for (std::size_t rank = 0; rank < 2; ++rank) {
            std::vector<float> native(12, 0); // 2 rows, 6 compact slots each.
            for (std::size_t p = 0; p < 3; ++p) {
                native[p] = means[layer][rank][p];
                native[6 + p] = means[layer][rank][2 - p];
            }
            native[4] = 1e-8f; // The existing unmapped-page roundoff tolerance.
            accumulator.add(native, 6, rank ? 1 : 3);
        }
    }
    accumulator.finish();
    expect(rows[0].source_position == 6400 && rows[1].source_position == 6416,
           "aggregation changed accepted source positions");
    for (std::size_t row = 0; row < rows.size(); ++row) {
        expect(rows[row].mass.size() == 3, "zero pages were retained or rank edges duplicated");
        for (std::size_t p = 0; p < 3; ++p) {
            double oracle = 0;
            // Average actual individual heads and layers, without a rank weight.
            for (std::size_t layer = 0; layer < 3; ++layer)
                for (std::size_t head = 0; head < 4; ++head)
                    oracle += means[layer][head == 3 ? 1 : 0][row ? 2 - p : p] / 12.0;
            expect(rows[row].mass[p].target == std::uint64_t(pages[p]) * 64 &&
                   std::abs(rows[row].mass[p].mass - oracle) < 1e-14,
                   "page mapping or unequal-head/layer weighting differs from head oracle");
        }
    }
    const auto expected_bytes =
        (rows[0].mass.capacity() + rows[1].mass.capacity()) * sizeof(kvmem::GraphEdge) +
        12 * sizeof(std::pair<std::uint64_t, double>);
    expect(attention_feedback_workspace_bytes(rows) == expected_bytes,
           "sparse observation allocation was omitted from the graph budget");

    // Rejection is local: production publishes graph/feedback only after all
    // these contributions have succeeded. A zero-head rank is skipped by the
    // caller; pretending it supplied a native attention mean is invalid.
    const auto invalid = [&](float value, std::size_t slot, std::uint32_t heads = 4) {
        std::vector<kvmem::AcceptedAttentionRow> sample(1);
        AttentionPageMassAccumulator sum(sample, pages, 1, 4);
        std::vector<float> native(6, 0); native[slot] = value;
        sum.add(native, 6, heads);
    };
    expect_rejected([&] { invalid(-.1f, 0); });
    expect_rejected([&] { invalid(std::numeric_limits<float>::infinity(), 0); });
    expect_rejected([&] { invalid(std::numeric_limits<float>::quiet_NaN(), 0); });
    expect_rejected([&] { invalid(.1f, 4); });
    expect_rejected([&] { invalid(0, 0, 0); });
    expect_rejected([&] { invalid(0, 0, 5); });
    expect_rejected([&] {
        std::vector<kvmem::AcceptedAttentionRow> sample(1);
        AttentionPageMassAccumulator sum(sample, pages, 1, 4);
        sum.add(std::array<float, 3>{}, 3, 4);
    });
    std::vector<kvmem::AcceptedAttentionRow> zero(1);
    AttentionPageMassAccumulator zero_sum(zero, pages, 1, 4);
    zero_sum.add(std::array<float, 4>{}, 4, 4);
    zero_sum.finish();
    expect(zero[0].mass.empty() && attention_feedback_workspace_bytes(zero) ==
               zero[0].mass.capacity() * sizeof(kvmem::GraphEdge) &&
               zero[0].mass.capacity() >= pages.size(),
           "all-zero pages hid their allocated capacity");
}

struct WindowResult {
    std::vector<double> pressure;
    std::size_t last_edges = 0, last_extra = 0;
};

WindowResult run_long_window(std::span<const std::uint32_t> heads, bool oracle) {
    constexpr std::uint32_t window = 131072 + 2048, chunk = 2048;
    constexpr std::size_t limit = 32u << 20;
    auto parent = std::make_shared<kvmem::GraphStatisticsBudget>(256u << 20);
    auto budget = std::make_shared<kvmem::GraphStatisticsBudget>(limit, parent);
    kvmem::GraphSnapshot graph;
    graph.set_budget(budget);
    kvmem::AttentionFeedback feedback;
    const auto total_heads = std::accumulate(heads.begin(), heads.end(), 0u);
    WindowResult result;
    for (std::uint32_t begin = 0; begin < window; begin += chunk) {
        const auto end = begin + chunk;
        const auto samples = kvmem::prefill_sample_rows(begin, chunk,
            feedback.snapshot().prefill_end, begin, window);
        expect(samples.size() == 64 && samples.front() == 31 && samples.back() == 2047,
               "long-window fixture has different native sampling boundaries");
        std::vector<std::uint32_t> pages(end / 64);
        std::iota(pages.begin(), pages.end(), 0u);
        std::vector<kvmem::AcceptedAttentionRow> rows(samples.size());
        for (std::size_t i = 0; i < samples.size(); ++i) rows[i].source_position = begin + samples[i];
        if (oracle) {
            // Independent Q=0 reference: probability is visible token count in
            // a page divided by the row's whole causal token count.
            for (auto& row : rows) {
                const auto visible = row.source_position + 1;
                for (std::uint64_t page = 0; page * 64 < visible; ++page)
                    row.mass.push_back({page * 64, double(std::min<std::uint64_t>(64, visible - page * 64)) / visible});
            }
        } else {
            AttentionPageMassAccumulator sum(rows, pages, 3, total_heads);
            std::vector<float> native(rows.size() * pages.size(), 0);
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const auto visible = rows[i].source_position + 1;
                for (std::size_t p = 0; p * 64 < visible; ++p)
                    native[i * pages.size() + p] = float(double(std::min<std::uint64_t>(64, visible - p * 64)) / visible);
            }
            for (std::size_t layer = 0; layer < 3; ++layer)
                for (const auto local_heads : heads)
                    if (local_heads) sum.add(native, pages.size(), local_heads);
            sum.finish();
            for (const auto& row : rows) {
                const auto visible = row.source_position + 1;
                expect(row.mass.size() == (visible + 63) / 64,
                       "rank count enlarged a complete positive attention row");
                double total = 0;
                for (const auto& edge : row.mass) {
                    const auto expected = double(std::min<std::uint64_t>(64, visible - edge.target)) / visible;
                    expect(std::abs(edge.mass - expected) <= expected * 1e-7,
                           "native FP32 page mass disagrees with the independent token oracle");
                    total += edge.mass;
                }
                expect(std::abs(total - 1) < 1e-7, "long-window sampled row lost causal mass");
            }
        }
        auto next = kvmem::prepare_graph_accept(graph, graph.accepted_through, end, rows, limit);
        result.last_extra = feedback.snapshot().bytes() + attention_feedback_workspace_bytes(rows);
        next.reserve_extra(result.last_extra);
        auto next_feedback = feedback;
        std::vector<std::uint64_t> exposed;
        for (auto page : pages) exposed.push_back(std::uint64_t(page) * 64);
        next_feedback.observe_prefill(begin, end, chunk, rows, exposed);
        next.include_feedback(next_feedback.snapshot().bytes());
        graph = std::move(next);
        feedback = std::move(next_feedback);
        result.last_edges = 0;
        for (const auto& row : rows) result.last_edges += row.mass.size();
        expect(graph.accepted_through == end && budget->used() <= limit,
               "long-window graph publication exceeded the accepted frontier or budget");
    }
    std::vector<std::uint64_t> pages;
    for (std::uint64_t p = 0; p < window; p += 64) pages.push_back(p);
    result.pressure = feedback.pressure_scores(pages);
    return result;
}

void test_long_window() {
    const auto reference = run_long_window(std::array{4u}, true);
    const auto single = run_long_window(std::array{4u}, false);
    const auto unequal = run_long_window(std::array{3u, 1u}, false);
    const auto zero_rank = run_long_window(std::array{3u, 0u, 1u}, false);
    for (const auto* result : {&single, &unequal, &zero_rank}) {
        expect(result->last_edges == 132128 && result->last_extra < (8u << 20),
               "long-window representation grows with tensor rank count");
        expect(result->pressure.size() == reference.pressure.size(), "wrong Pressure window");
        for (std::size_t p = 0; p < reference.pressure.size(); ++p)
            expect(std::abs(result->pressure[p] - reference.pressure[p]) < 1e-7,
                   "Pressure disagrees with independent uniform-token reference");
    }
    expect(single.last_extra == unequal.last_extra && unequal.last_extra == zero_rank.last_extra,
           "equal logical observations have rank-dependent host memory cost");
    std::cout << "OK: single/3:1/zero-head-rank windows reach 133120 tokens under 32 MiB; "
              << "last edges=" << unequal.last_edges << " extra=" << unequal.last_extra << '\n';
}

int main() {
    test_page_mass();
    test_long_window();
    const MemoryStatisticsPlanning plan{
        .capacity=8192, .window_tokens=2048, .prefill_chunk=512,
        .verify_window=7, .max_concurrency=2, .query_max_tokens=128};
    const auto budget=memory_statistics_budget(plan,2,1024,4096,16,4,256);
    // Hand-counted bytes for this fixture: 9 K buckets, 16 candidates,
    // 128 Query rows, 512 capture rows at 3 layers, 32 sampled pages.
    expect(budget.divided==15024128,"divided statistics cost omits a native buffer");
    expect(budget.every==9216,"per-rank controls/output cost is wrong");
    const auto scoring=kvmem::PageSoftmaxCuda::workspace_bytes(128,16,4,256);
    expect(budget.first==2304+scoring,"rank 0 Query workspace or token capture was omitted");

    LayoutBuilder actual;
    MemoryStatisticsLayout layout;
    layout.layers=2;
    std::optional<TensorRegion> workspace;
    const auto allocate=[&](MemoryStatisticsOwner,DType dtype,
        std::initializer_list<std::int32_t> shape,const char* label) {
        return actual.add_tensor(dtype,shape,256,label);
    };
    plan_memory_statistics_shard(layout,plan,1024,4096,16,allocate);
    plan_memory_statistics_rank0(layout,workspace,plan,16,4,256,allocate);
    expect(actual.finish(256)==budget.divided+budget.every+budget.first,
           "estimate differs from the actual aligned allocation recipe");
    expect(layout.key_sums.shape[1]==9 && layout.candidate_keys.shape[1]==16 &&
           layout.query_values.shape[1]==128 && layout.attention_queries.shape[1]==512,
           "fixture geometry differs from its hand-counted oracle");

    const auto big=memory_statistics_budget(plan,2,768,3072,12,3,256);
    const auto small=memory_statistics_budget(plan,2,256,1024,4,1,256);
    expect(big.divided==budget.divided*3/4 && small.divided==budget.divided/4 &&
           big.every==budget.every && small.every==budget.every,
           "unequal heads must divide statistics and retain per-rank output/control");

    auto mean=plan; mean.mean_query=true;
    const auto mean_budget=memory_statistics_budget(mean,2,1024,4096,16,4,256);
    expect(mean_budget.divided==12959744 && mean_budget.first==2304,
           "MeanQ must use FP32 sums and omit PageSoftmax workspace");
    auto concurrent=plan; concurrent.max_concurrency=8;
    const auto concurrent_budget=memory_statistics_budget(concurrent,2,1024,4096,16,4,256);
    expect(concurrent_budget.divided==budget.divided+393216 &&
           concurrent_budget.first==budget.first,
           "candidate columns must follow concurrency without multiplying Program-wide Query buffers");
    auto longer=plan; longer.capacity=131072;
    const auto long_budget=memory_statistics_budget(longer,2,1024,4096,16,4,256);
    expect(long_budget.first>budget.first && long_budget.divided==budget.divided &&
           long_budget.every==budget.every,
           "logical history must size rank 0 scoring independently of resident window");
    const auto disabled=memory_statistics_budget({},2,1024,4096,16,4,256);
    expect(disabled.divided==0 && disabled.every==0 && disabled.first==0,
           "KVMem disabled must not change native tensor split estimates");
    std::cout << "OK: KVMem tensor-split costs share the exact startup allocation recipe\n";
}
