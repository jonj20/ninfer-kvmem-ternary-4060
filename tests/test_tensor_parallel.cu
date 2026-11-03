// Qualification of `TensorParallelExchange`, the all-reduce of a tensor-parallel group.
//
// The oracle is the reduction itself evaluated on the host in the same order: each rank's FP32
// partials summed in rank order, added to the BF16 residual in FP32, rounded once. Only adds
// appear, so the host evaluates the same IEEE operations and the comparison is exact. Every rank's
// replica must also equal every other rank's. Ranks share device 0, as in the stage-link test:
// `force_staged` runs the real mapped-host path, and without it ranks on one card take the peer
// path. With two or more devices the ranks are also dealt over them, so the staged path crosses
// cards. A captured sequence of exchanges with changed inputs between replays checks that the
// protocol holds inside one graph spanning every rank's stream. Two ranks on two cards without
// peer access exchange decode-sized partials as low-latency words; a sequence alternating them
// with staged exchanges and with one contributor checks that the two protocols share the slots.
// A partial of at least the scatter threshold reduces as a reduce-scatter and an all-gather, from a
// larger one where the ranks read peer memory or share a card; the default thresholds send a prompt
// chunk that way, and a threshold of four bytes sends every
// exchange that is not low-latency words, at sizes whose ranges end in a partial block or leave
// some ranks without a range; two ranks storing an FP32 sum still reduce whole. The captured pass
// then reuses a slot within one graph.
//
// The host writes the inputs with blocking copies from pageable memory, which may return before
// the data reaches the device; the rank streams do not wait for them (they are non-blocking), so
// every device is synchronized before the exchange runs.

#include "core/device.h"
#include "core/tensor_parallel.h"
#include "cuda_availability.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

using ninfer::test::cuda_unavailable;

int expect(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "expectation failed: " << label << '\n';
    return 1;
}

std::uint16_t bits(__nv_bfloat16 value) {
    std::uint16_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

struct Case {
    std::size_t ranks;
    std::int64_t values;
    bool force_staged;
    int devices; // ranks are dealt over devices [0, devices)
    std::size_t scatter_bytes;
    std::size_t near_scatter_bytes;
};

// One exchange on every rank's stream: ranks [0, contributors) publish their partials of `values`,
// the others nothing, and every rank reduces into `out[r]` and completes it.
template <class Out>
void exchange_all(ninfer::TensorParallelExchange& exchange, ninfer::DeviceContext& ctx,
                  std::size_t slot, const std::vector<Out*>& out, std::int64_t values,
                  std::size_t contributors,
                  ninfer::TensorParallelSum sum = ninfer::TensorParallelSum::AddBf16) {
    const auto bytes = static_cast<std::size_t>(values) * sizeof(float);
    for (std::size_t r = 0; r < out.size(); ++r) {
        exchange.publish(r, slot, r < contributors ? bytes : 0, ctx.rank(r).stream, sum);
    }
    for (std::size_t r = 0; r < out.size(); ++r) {
        exchange.reduce(r, slot, out[r], values, contributors, ctx.rank(r).stream, sum);
    }
    for (std::size_t r = 0; r < out.size(); ++r) {
        exchange.complete(r, slot, out[r], values, ctx.rank(r).stream, sum);
    }
}

// The replica every rank must hold: the residual plus the rank-order sum of the partials.
std::vector<std::uint16_t> oracle(const std::vector<__nv_bfloat16>& residual,
                                  const std::vector<std::vector<float>>& partials) {
    std::vector<std::uint16_t> out(residual.size());
    for (std::size_t i = 0; i < residual.size(); ++i) {
        float sum = partials[0][i];
        for (std::size_t r = 1; r < partials.size(); ++r) { sum += partials[r][i]; }
        out[i] = bits(__float2bfloat16(__bfloat162float(residual[i]) + sum));
    }
    return out;
}

int compare_replicas(const std::vector<__nv_bfloat16*>& replica, std::int64_t values,
                     const std::vector<std::uint16_t>& want, const char* label) {
    int failures = 0;
    for (std::size_t r = 0; r < replica.size(); ++r) {
        std::vector<std::uint16_t> got(want.size());
        CUDA_CHECK(cudaMemcpy(got.data(), replica[r], values * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < want.size(); ++i) { wrong += got[i] != want[i]; }
        if (wrong != 0) {
            std::cerr << "rank " << r << " of " << replica.size() << ", " << values << " values: "
                      << wrong << " differ\n";
        }
        failures += expect(wrong == 0, label);
    }
    return failures;
}

int run_case(const Case& c, std::mt19937& rng) {
    std::vector<int> ids(c.ranks);
    for (std::size_t r = 0; r < c.ranks; ++r) { ids[r] = static_cast<int>(r) % c.devices; }
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    const auto settle = [&] {
        for (int device = 0; device < c.devices; ++device) {
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaSetDevice(0));
    };
    const std::size_t bytes = static_cast<std::size_t>(c.values) * sizeof(float);
    ninfer::TensorParallelExchange exchange(ctx, {.slot_bytes         = bytes,
                                                  .force_staged       = c.force_staged,
                                                  .scatter_bytes      = c.scatter_bytes,
                                                  .near_scatter_bytes = c.near_scatter_bytes});
    // Across cards the transport follows their peer access; on one card it is the override's.
    int failures = expect((c.devices > 1 && !c.force_staged) ||
                              exchange.transport() == (c.force_staged
                                                           ? ninfer::TensorParallelTransport::StagedHost
                                                           : ninfer::TensorParallelTransport::Peer),
                          "transport follows the topology and the override");
    bool distinct = true;
    for (std::size_t a = 0; a < c.ranks; ++a) {
        for (std::size_t b = a + 1; b < c.ranks; ++b) { distinct = distinct && ids[a] != ids[b]; }
    }
    const bool host =
        exchange.transport() == ninfer::TensorParallelTransport::StagedHost && distinct;
    const bool words = host && bytes <= (32U << 10U);
    failures += expect(exchange.low_latency(bytes) == words,
                       "low-latency words exactly for small staged exchanges between cards");
    const std::size_t threshold = host ? c.scatter_bytes : c.near_scatter_bytes;
    failures += expect(exchange.scatter(bytes) == (!words && threshold != 0 && bytes >= threshold),
                       "a reduce-scatter exactly from its topology's threshold, never for words");
    failures += expect(exchange.scatter(bytes, ninfer::TensorParallelSum::StoreFp32) ==
                           (exchange.scatter(bytes) && c.ranks > 2),
                       "two ranks storing an FP32 sum reduce whole");

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16*> replica(c.ranks);
    for (std::size_t r = 0; r < c.ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&replica[r]),
                              c.values * sizeof(__nv_bfloat16)));
    }
    CUDA_CHECK(cudaSetDevice(0));
    const auto residual_of = [&](float scale) {
        std::vector<__nv_bfloat16> residual(static_cast<std::size_t>(c.values));
        for (auto& v : residual) { v = __float2bfloat16(scale * normal(rng)); }
        return residual;
    };
    const auto partials_of = [&](float scale) {
        std::vector<std::vector<float>> partials(c.ranks,
                                                 std::vector<float>(static_cast<std::size_t>(c.values)));
        for (auto& p : partials) {
            for (auto& v : p) { v = scale * normal(rng); }
        }
        return partials;
    };
    const auto load = [&](const std::vector<__nv_bfloat16>& residual,
                          const std::vector<std::vector<float>>& partials, std::size_t slot) {
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaMemcpy(replica[r], residual.data(), c.values * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(exchange.partial(r, slot), partials[r].data(), bytes,
                                  cudaMemcpyHostToDevice));
        }
        settle();
    };

    // Three exchanges in a row, so both slots are used and the first is reused.
    for (int round = 0; round < 3; ++round) {
        const std::size_t slot = static_cast<std::size_t>(round) % 2;
        const auto residual    = residual_of(4.0f);
        const auto partials    = partials_of(1.0f + static_cast<float>(round));
        load(residual, partials, slot);
        exchange_all(exchange, ctx, slot, replica, c.values, c.ranks);
        ctx.synchronize();
        failures += compare_replicas(replica, c.values, oracle(residual, partials),
                                     "every replica equals the rank-order sum");
    }

    // A projection only rank 0 runs: the others publish nothing and every replica adds rank 0's.
    {
        const auto residual = residual_of(4.0f);
        const auto partials = partials_of(1.0f);
        load(residual, partials, 1);
        exchange_all(exchange, ctx, 1, replica, c.values, 1);
        ctx.synchronize();
        const std::vector<std::vector<float>> first(partials.begin(), partials.begin() + 1);
        failures += compare_replicas(replica, c.values, oracle(residual, first),
                                     "one contributor reaches every replica");
    }

    // A block output: the sum stored in place of the output, rounded once to BF16 or kept in FP32.
    {
        const auto residual = residual_of(4.0f);
        const auto partials = partials_of(1.0f);
        load(residual, partials, 0);
        exchange_all(exchange, ctx, 0, replica, c.values, c.ranks,
                     ninfer::TensorParallelSum::StoreBf16);
        ctx.synchronize();
        const std::vector<__nv_bfloat16> zero(residual.size(), __float2bfloat16(0.0f));
        failures += compare_replicas(replica, c.values, oracle(zero, partials),
                                     "a stored BF16 sum is the rank-order sum rounded once");

        std::vector<float*> sums(c.ranks);
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaSetDevice(ids[r]));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sums[r]), bytes));
        }
        CUDA_CHECK(cudaSetDevice(0));
        load(residual, partials, 1);
        exchange_all(exchange, ctx, 1, sums, c.values, c.ranks,
                     ninfer::TensorParallelSum::StoreFp32);
        ctx.synchronize();
        std::vector<float> want(static_cast<std::size_t>(c.values));
        for (std::size_t i = 0; i < want.size(); ++i) {
            float sum = partials[0][i];
            for (std::size_t r = 1; r < c.ranks; ++r) { sum += partials[r][i]; }
            want[i] = sum;
        }
        for (std::size_t r = 0; r < c.ranks; ++r) {
            std::vector<float> got(want.size());
            CUDA_CHECK(cudaMemcpy(got.data(), sums[r], bytes, cudaMemcpyDeviceToHost));
            failures += expect(std::memcmp(got.data(), want.data(), bytes) == 0,
                               "a stored FP32 sum is the rank-order sum");
            CUDA_CHECK(cudaSetDevice(ids[r]));
            CUDA_CHECK(cudaFree(sums[r]));
        }
        CUDA_CHECK(cudaSetDevice(0));
    }

    // A captured pass: three exchanges back to back on every rank's stream, joined into rank 0's;
    // the third reuses the first one's slot and partials.
    {
        cudaStream_t origin = ctx.rank(0).stream;
        std::vector<cudaEvent_t> fork(c.ranks), join(c.ranks);
        // An event is recorded on a stream of its own device.
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaSetDevice(ids[r]));
            CUDA_CHECK(cudaEventCreateWithFlags(&fork[r], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&join[r], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaSetDevice(0));
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
        CUDA_CHECK(cudaEventRecord(fork[0], origin));
        for (std::size_t r = 1; r < c.ranks; ++r) {
            CUDA_CHECK(cudaStreamWaitEvent(ctx.rank(r).stream, fork[0], 0));
        }
        for (std::size_t slot : {0U, 1U, 0U}) {
            exchange_all(exchange, ctx, slot, replica, c.values, c.ranks);
        }
        for (std::size_t r = 1; r < c.ranks; ++r) {
            CUDA_CHECK(cudaEventRecord(join[r], ctx.rank(r).stream));
            CUDA_CHECK(cudaStreamWaitEvent(origin, join[r], 0));
        }
        CUDA_CHECK(cudaStreamEndCapture(origin, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            const auto residual = residual_of(1.0f);
            const auto first    = partials_of(1.0f);
            const auto second   = partials_of(1.0f);
            load(residual, first, 0);
            for (std::size_t r = 0; r < c.ranks; ++r) {
                CUDA_CHECK(cudaMemcpy(exchange.partial(r, 1), second[r].data(), bytes,
                                      cudaMemcpyHostToDevice));
            }
            settle();
            CUDA_CHECK(cudaGraphLaunch(executable, origin));
            CUDA_CHECK(cudaStreamSynchronize(origin));
            std::vector<__nv_bfloat16> after(residual);
            for (const auto* partials : {&first, &second, &first}) {
                const auto next = oracle(after, *partials);
                std::memcpy(after.data(), next.data(), next.size() * sizeof(std::uint16_t));
            }
            std::vector<std::uint16_t> want(after.size());
            std::memcpy(want.data(), after.data(), want.size() * sizeof(std::uint16_t));
            failures +=
                compare_replicas(replica, c.values, want, "a captured pass replays every exchange");
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        for (std::size_t r = 0; r < c.ranks; ++r) {
            CUDA_CHECK(cudaEventDestroy(fork[r]));
            CUDA_CHECK(cudaEventDestroy(join[r]));
        }
    }

    try {
        exchange.publish(0, 0, bytes + sizeof(float), ctx.rank(0).stream);
        failures += expect(false, "an oversized partial is refused");
    } catch (const std::invalid_argument&) {}
    for (std::size_t r = 0; r < c.ranks; ++r) {
        CUDA_CHECK(cudaSetDevice(ids[r]));
        CUDA_CHECK(cudaFree(replica[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    return failures;
}

// Two ranks on two cards: low-latency words, whole staged reduces and reduce-scatters interleaved
// over both slots, with all ranks or only rank 0 contributing.
int run_mixed(std::mt19937& rng) {
    const int ids[] = {0, 1};
    ninfer::DeviceContext ctx(std::span<const int>(ids, 2));
    constexpr std::int64_t kWide = 5120 * 4;
    constexpr std::int64_t kWhole = 5120 * 2;
    ninfer::TensorParallelExchange exchange(ctx, {.slot_bytes    = kWide * sizeof(float),
                                                  .force_staged  = true,
                                                  .scatter_bytes = kWide * sizeof(float)});
    int failures = expect(exchange.low_latency(5120 * sizeof(float)) &&
                              !exchange.low_latency(kWhole * sizeof(float)) &&
                              !exchange.scatter(kWhole * sizeof(float)) &&
                              exchange.scatter(kWide * sizeof(float)),
                          "decode columns as words, a short window whole, a wide one scattered");
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<__nv_bfloat16*> replica(2);
    for (int r = 0; r < 2; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&replica[r]), kWide * sizeof(__nv_bfloat16)));
    }
    const std::int64_t sizes[]       = {5120, kWide, 1031, kWhole, kWide, 5120, kWhole, kWide};
    const std::size_t contributors[] = {2, 2, 1, 1, 1, 2, 2, 2};
    for (std::size_t step = 0; step < std::size(sizes); ++step) {
        const std::int64_t values = sizes[step];
        const std::size_t slot    = step % 2;
        std::vector<__nv_bfloat16> residual(static_cast<std::size_t>(values));
        for (auto& v : residual) { v = __float2bfloat16(4.0f * normal(rng)); }
        std::vector<std::vector<float>> partials(2, std::vector<float>(residual.size()));
        for (auto& p : partials) {
            for (auto& v : p) { v = normal(rng); }
        }
        for (int r = 0; r < 2; ++r) {
            CUDA_CHECK(cudaSetDevice(r));
            CUDA_CHECK(cudaMemcpy(replica[r], residual.data(), values * sizeof(__nv_bfloat16),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(exchange.partial(r, slot), partials[r].data(),
                                  values * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        exchange_all(exchange, ctx, slot, replica, values, contributors[step]);
        ctx.synchronize();
        partials.resize(contributors[step]);
        failures += compare_replicas(replica, values, oracle(residual, partials),
                                     "interleaved exchanges match the rank-order sum");
    }
    for (int r = 0; r < 2; ++r) {
        CUDA_CHECK(cudaSetDevice(r));
        CUDA_CHECK(cudaFree(replica[r]));
    }
    CUDA_CHECK(cudaSetDevice(0));
    return failures;
}

int run() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);
    CUDA_CHECK(cudaSetDevice(0));
    std::mt19937 rng(20261025);
    int failures = 0;
    // One decode column of the 27B residual, a verify window, an odd size and a prompt chunk at the
    // default threshold; then every exchange above words as a reduce-scatter, down to ranges of one
    // partial block and ranks without one.
    const ninfer::TensorParallelExchangeOptions defaults{};
    for (const int devices : {1, std::min(count, 2)}) {
        for (const bool staged : {true, false}) {
            for (const std::size_t ranks : {2U, 3U, 4U, 8U}) {
                for (const std::int64_t values : {5120LL, 5120LL * 4, 1031LL, 5120LL * 512}) {
                    failures += run_case({ranks, values, staged, devices, defaults.scatter_bytes,
                                          defaults.near_scatter_bytes},
                                         rng);
                }
                for (const std::int64_t values : {5120LL, 5120LL * 4, 1031LL, 100LL}) {
                    failures += run_case(
                        {ranks, values, staged, devices, sizeof(float), sizeof(float)}, rng);
                }
            }
        }
        if (count < 2) { break; }
    }
    if (count >= 2) { failures += run_mixed(rng); }
    if (failures != 0) { return 1; }
    std::cout << "tensor-parallel exchange holds\n";
    return 0;
}

} // namespace

int run_tensor_parallel_test() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        return 1;
    }
}
