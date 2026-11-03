// Times TensorParallelExchange, the all-reduce of tensor parallelism, at the partial sizes of
// decode steps, verifications and prompt chunks. For every rank count the device list allows (2, 4,
// 8 and the whole list), through peer reads where every pair of cards has peer access and through
// host memory, it compares a whole reduce with the reduce-scatter and all-gather and names the
// smallest partial from which the second wins at every larger size. A reference moves the bytes the
// reduce-scatter reads with the copy engines alone, without a reduction, to show how close the
// kernels come to what the links carry.
//
// Each measurement captures `--exchanges` back-to-back exchanges of every rank's partial on every
// rank's stream into one graph, forked from rank 0's stream and joined back as a pass is, replays
// it `--replays` times and divides the time on rank 0's stream by the exchanges replayed. Partials
// and replicas hold zeros: the time does not depend on the values, which tests/test_tensor_parallel
// qualifies. The reference runs eagerly: after a barrier, each rank copies from every other rank
// 1/N of its FP32 partial and its range of the output (6/N bytes per value for BF16 outputs, 8/N
// for FP32), one copy after another on its own stream (a peer copy, or a host-to-device copy from a
// pinned host buffer of the other rank); it leaves out the publish's device-to-host copies.
//
// `--sum` picks what a reduce does with the sum: adds it to a BF16 replica (`add`, the Qwen3.5-core
// residual), or stores it as BF16 (`bf16`) or FP32 (`fp32`, Qwen3.8-Flash-Next's routed experts).
//
// Run: ninfer-tp-exchange-bench --devices 0,1,2,3 [--hidden 5120] [--sum add|bf16|fp32]
//          [--columns 1,2,4,8,13,16,32,64,128,256,512,1024,2048] [--exchanges 16] [--replays 10]
#include "core/device.h"
#include "core/tensor_parallel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t err, const char* what) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s: %s: %s\n", what, cudaGetErrorName(err), cudaGetErrorString(err));
    std::exit(2);
}

std::vector<long long> parse_list(const std::string& text) {
    std::vector<long long> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) { out.push_back(std::stoll(item)); }
    return out;
}

struct Options {
    std::vector<int> devices;
    long long hidden              = 5120;
    ninfer::TensorParallelSum sum = ninfer::TensorParallelSum::AddBf16;
    std::vector<long long> columns{1, 2, 4, 8, 13, 16, 32, 64, 128, 256, 512, 1024, 2048};
    int exchanges = 16;
    int replays   = 10;
};

// Bytes of one output value.
std::size_t output_bytes(ninfer::TensorParallelSum sum) {
    return sum == ninfer::TensorParallelSum::StoreFp32 ? sizeof(float) : 2;
}

// Each rank's replica of the widest partial and the reference's landing buffer, with the events
// that fork and join the ranks' streams.
struct RankBuffers {
    std::vector<void*> replica;
    std::vector<void*> landing;
    std::vector<cudaEvent_t> fork;
    std::vector<cudaEvent_t> join;
};

RankBuffers allocate(const ninfer::DeviceContext& ctx, std::size_t replica_bytes,
                     std::size_t landing_bytes) {
    RankBuffers out;
    const std::size_t n = ctx.size();
    out.replica.resize(n);
    out.landing.resize(n);
    out.fork.resize(n);
    out.join.resize(n);
    for (std::size_t r = 0; r < n; ++r) {
        ninfer::RankBinding bind(ctx, r);
        ck(cudaMalloc(&out.replica[r], replica_bytes), "replica");
        ck(cudaMemset(out.replica[r], 0, replica_bytes), "replica zero");
        ck(cudaMalloc(&out.landing[r], landing_bytes), "landing");
        ck(cudaEventCreateWithFlags(&out.fork[r], cudaEventDisableTiming), "event");
        ck(cudaEventCreateWithFlags(&out.join[r], cudaEventDisableTiming), "event");
    }
    return out;
}

void release(const ninfer::DeviceContext& ctx, RankBuffers& buffers) {
    for (std::size_t r = 0; r < ctx.size(); ++r) {
        ninfer::RankBinding bind(ctx, r);
        (void)cudaFree(buffers.replica[r]);
        (void)cudaFree(buffers.landing[r]);
        (void)cudaEventDestroy(buffers.fork[r]);
        (void)cudaEventDestroy(buffers.join[r]);
    }
}

// Rank 0's stream forks every other rank's stream, or joins them back.
void fork(const ninfer::DeviceContext& ctx, RankBuffers& buffers) {
    const cudaStream_t origin = ctx.rank(0).stream;
    ck(cudaEventRecord(buffers.fork[0], origin), "fork record");
    for (std::size_t r = 1; r < ctx.size(); ++r) {
        ck(cudaStreamWaitEvent(ctx.rank(r).stream, buffers.fork[0], 0), "fork wait");
    }
}

void join(const ninfer::DeviceContext& ctx, RankBuffers& buffers) {
    const cudaStream_t origin = ctx.rank(0).stream;
    for (std::size_t r = 1; r < ctx.size(); ++r) {
        ck(cudaEventRecord(buffers.join[r], ctx.rank(r).stream), "join record");
        ck(cudaStreamWaitEvent(origin, buffers.join[r], 0), "join wait");
    }
}

// Milliseconds of `launch` repeated `repeats` times on rank 0's stream, after one unmeasured run.
template <class Launch>
double time_on_origin(const ninfer::DeviceContext& ctx, int repeats, const Launch& launch) {
    ninfer::RankBinding bind(ctx, 0);
    const cudaStream_t origin = ctx.rank(0).stream;
    cudaEvent_t start         = nullptr;
    cudaEvent_t stop          = nullptr;
    ck(cudaEventCreate(&start), "timing event");
    ck(cudaEventCreate(&stop), "timing event");
    launch();
    ck(cudaStreamSynchronize(origin), "warm-up");
    ck(cudaEventRecord(start, origin), "timing record");
    for (int i = 0; i < repeats; ++i) { launch(); }
    ck(cudaEventRecord(stop, origin), "timing record");
    ck(cudaEventSynchronize(stop), "timing synchronize");
    float ms = 0.0F;
    ck(cudaEventElapsedTime(&ms, start, stop), "timing elapsed");
    (void)cudaEventDestroy(start);
    (void)cudaEventDestroy(stop);
    return ms;
}

// Microseconds per exchange of `values` per partial, every rank contributing, in a captured graph.
double time_exchanges(ninfer::TensorParallelExchange& exchange, const ninfer::DeviceContext& ctx,
                      RankBuffers& buffers, std::int64_t values, const Options& options) {
    const std::size_t n       = ctx.size();
    const cudaStream_t origin = ctx.rank(0).stream;
    const auto bytes          = static_cast<std::size_t>(values) * sizeof(float);
    ninfer::RankBinding bind(ctx, 0);
    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    ck(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal), "capture");
    fork(ctx, buffers);
    for (int e = 0; e < options.exchanges; ++e) {
        const auto slot = static_cast<std::size_t>(e) % ninfer::TensorParallelExchange::kSlots;
        for (std::size_t r = 0; r < n; ++r) {
            exchange.publish(r, slot, bytes, ctx.rank(r).stream, options.sum);
        }
        for (std::size_t r = 0; r < n; ++r) {
            exchange.reduce(r, slot, buffers.replica[r], values, n, ctx.rank(r).stream,
                            options.sum);
        }
        for (std::size_t r = 0; r < n; ++r) {
            exchange.complete(r, slot, buffers.replica[r], values, ctx.rank(r).stream, options.sum);
        }
    }
    join(ctx, buffers);
    ck(cudaStreamEndCapture(origin, &graph), "end capture");
    ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
    const double ms = time_on_origin(
        ctx, options.replays, [&] { ck(cudaGraphLaunch(executable, origin), "graph launch"); });
    (void)cudaGraphExecDestroy(executable);
    (void)cudaGraphDestroy(graph);
    return ms * 1000.0 / (static_cast<double>(options.replays) * options.exchanges);
}

// Microseconds for every rank's copy engine to bring in what a reduce-scatter and all-gather of
// `values` per partial reads from the other ranks (1/N of a partial and of an output per value).
double time_copies(ninfer::TensorParallelExchange& exchange, const ninfer::DeviceContext& ctx,
                   RankBuffers& buffers, std::int64_t values, const Options& options) {
    const std::size_t n = ctx.size();
    const auto share =
        static_cast<std::size_t>(values) * (4 + output_bytes(options.sum)) / n / 16 * 16;
    const bool peer = exchange.transport() == ninfer::TensorParallelTransport::Peer;
    // Host slots hold what a staged publish copied; their addresses are not exposed, so the
    // reference reads the same number of bytes from pinned host buffers of its own.
    std::vector<void*> host(n, nullptr);
    if (!peer) {
        for (std::size_t r = 0; r < n; ++r) {
            ninfer::RankBinding bind(ctx, r);
            ck(cudaHostAlloc(&host[r], share, cudaHostAllocMapped | cudaHostAllocPortable), "host");
        }
    }
    const double ms = time_on_origin(ctx, options.replays, [&] {
        fork(ctx, buffers);
        for (std::size_t r = 0; r < n; ++r) {
            ninfer::RankBinding bind(ctx, r);
            auto* landing = static_cast<unsigned char*>(buffers.landing[r]);
            for (std::size_t q = 0; q < n; ++q) {
                if (q == r) { continue; }
                if (peer) {
                    ck(cudaMemcpyPeerAsync(landing + q * share, ctx.rank(r).device,
                                           exchange.partial(q, 0), ctx.rank(q).device, share,
                                           ctx.rank(r).stream),
                       "peer copy");
                } else {
                    ck(cudaMemcpyAsync(landing + q * share, host[q], share, cudaMemcpyHostToDevice,
                                       ctx.rank(r).stream),
                       "host copy");
                }
            }
        }
        join(ctx, buffers);
    });
    for (void* buffer : host) {
        if (buffer != nullptr) { (void)cudaFreeHost(buffer); }
    }
    return ms * 1000.0 / options.replays;
}

void run_group(const std::vector<int>& ids, const Options& options) {
    ninfer::DeviceContext ctx(std::span<const int>(ids.data(), ids.size()));
    const std::size_t n    = ctx.size();
    const long long widest = *std::max_element(options.columns.begin(), options.columns.end());
    const std::size_t slot_bytes = static_cast<std::size_t>(widest * options.hidden) * 4;
    const std::size_t out        = output_bytes(options.sum);
    RankBuffers buffers          = allocate(ctx, slot_bytes / 4 * out, slot_bytes / 4 * (4 + out));
    std::string list;
    for (const int id : ids) { list += (list.empty() ? "" : ",") + std::to_string(id); }
    const char* sum = options.sum == ninfer::TensorParallelSum::AddBf16     ? "add"
                      : options.sum == ninfer::TensorParallelSum::StoreBf16 ? "bf16"
                                                                            : "fp32";
    for (const bool staged : {false, true}) {
        if (staged && !ctx.peer_access()) { break; } // the first pass already went through host
        std::vector<double> whole;
        std::vector<double> scattered;
        std::vector<double> copies;
        ninfer::TensorParallelTransport transport{};
        bool scatters = true;
        for (const std::size_t scatter_bytes : {sizeof(float), std::size_t{0}}) {
            ninfer::TensorParallelExchange exchange(ctx, {.slot_bytes         = slot_bytes,
                                                          .force_staged       = staged,
                                                          .scatter_bytes      = scatter_bytes,
                                                          .near_scatter_bytes = scatter_bytes});
            transport = exchange.transport();
            // Two ranks storing an FP32 sum reduce whole at every size.
            if (scatter_bytes != 0 && !exchange.scatter(slot_bytes, options.sum)) {
                scatters = false;
                break;
            }
            for (std::size_t r = 0; r < n; ++r) {
                ninfer::RankBinding bind(ctx, r);
                for (std::size_t slot = 0; slot < ninfer::TensorParallelExchange::kSlots; ++slot) {
                    ck(cudaMemset(exchange.partial(r, slot), 0, slot_bytes), "partial zero");
                }
            }
            for (const long long columns : options.columns) {
                const auto values = static_cast<std::int64_t>(columns * options.hidden);
                (scatter_bytes == 0 ? whole : scattered)
                    .push_back(time_exchanges(exchange, ctx, buffers, values, options));
                if (scatter_bytes != 0) {
                    copies.push_back(time_copies(exchange, ctx, buffers, values, options));
                }
            }
        }
        const char* name = transport == ninfer::TensorParallelTransport::Peer ? "peer" : "host";
        std::printf("\ndevices %s, %zu ranks, %s transport, hidden %lld, sum %s\n", list.c_str(), n,
                    name, options.hidden, sum);
        if (!scatters) {
            std::printf("every exchange reduces whole\n");
            continue;
        }
        std::printf("%8s %10s %10s %10s %8s %10s %12s %12s\n", "columns", "partial", "whole_us",
                    "scatter_us", "ratio", "copy_us", "whole_GB/s", "scatter_GB/s");
        std::size_t threshold = options.columns.size();
        for (std::size_t i = options.columns.size(); i-- > 0;) {
            if (scattered[i] >= whole[i]) { break; }
            threshold = i;
        }
        for (std::size_t i = 0; i < options.columns.size(); ++i) {
            const double values = static_cast<double>(options.columns[i] * options.hidden);
            // Bytes one rank reads from the others: N-1 whole FP32 partials, or 1/N of each plus
            // the others' ranges of the output.
            const double whole_read = 4.0 * values * static_cast<double>(n - 1);
            const double scatter_read =
                static_cast<double>(4 + out) * values * static_cast<double>(n - 1) / n;
            std::printf("%8lld %9.0fK %10.1f %10.1f %8.3f %10.1f %12.2f %12.2f\n",
                        options.columns[i], values * 4.0 / 1024.0, whole[i], scattered[i],
                        scattered[i] / whole[i], copies[i], whole_read / whole[i] / 1e3,
                        scatter_read / scattered[i] / 1e3);
        }
        if (threshold < options.columns.size()) {
            std::printf("reduce-scatter wins from %lld columns (%.0f KiB per partial) upward\n",
                        options.columns[threshold],
                        static_cast<double>(options.columns[threshold] * options.hidden) * 4.0 /
                            1024.0);
        } else {
            std::printf("reduce-scatter does not win at the widest size\n");
        }
    }
    release(ctx, buffers);
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value  = i + 1 < argc;
        if (arg == "--devices" && has_value) {
            for (const long long id : parse_list(argv[++i])) {
                options.devices.push_back(static_cast<int>(id));
            }
        } else if (arg == "--hidden" && has_value) {
            options.hidden = std::stoll(argv[++i]);
        } else if (arg == "--columns" && has_value) {
            options.columns = parse_list(argv[++i]);
        } else if (arg == "--exchanges" && has_value) {
            options.exchanges = std::max(2, std::stoi(argv[++i]) / 2 * 2);
        } else if (arg == "--replays" && has_value) {
            options.replays = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--sum" && has_value && std::string(argv[i + 1]) == "add") {
            options.sum = ninfer::TensorParallelSum::AddBf16;
            ++i;
        } else if (arg == "--sum" && has_value && std::string(argv[i + 1]) == "bf16") {
            options.sum = ninfer::TensorParallelSum::StoreBf16;
            ++i;
        } else if (arg == "--sum" && has_value && std::string(argv[i + 1]) == "fp32") {
            options.sum = ninfer::TensorParallelSum::StoreFp32;
            ++i;
        } else {
            std::fprintf(stderr,
                         "usage: %s --devices 0,1[,...] [--hidden N] [--sum add|bf16|fp32] "
                         "[--columns 1,2,...] [--exchanges N] [--replays N]\n",
                         argv[0]);
            return 2;
        }
    }
    if (options.devices.size() < 2 || options.devices.size() > ninfer::kMaxRanks ||
        options.columns.empty() || options.hidden <= 0) {
        std::fprintf(stderr, "need 2 to %zu devices, columns and a positive hidden size\n",
                     ninfer::kMaxRanks);
        return 2;
    }
    try {
        std::vector<std::size_t> counts;
        for (const std::size_t count : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
            if (count <= options.devices.size()) { counts.push_back(count); }
        }
        if (counts.back() != options.devices.size()) { counts.push_back(options.devices.size()); }
        for (const std::size_t count : counts) {
            run_group(std::vector<int>(options.devices.begin(), options.devices.begin() + count),
                      options);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
