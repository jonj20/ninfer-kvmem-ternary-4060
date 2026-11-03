#include "core/tensor_parallel.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) { return; }
    throw std::runtime_error(std::string("TensorParallelExchange ") + what + ": " +
                             cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
}

struct PartialSet {
    const float* part[kMaxRanks];
    int count;
};

// One value's sum into the output, as TensorParallelSum says.
template <TensorParallelSum Sum>
__device__ __forceinline__ void store_sum(void* out, std::int64_t i, float sum) {
    if constexpr (Sum == TensorParallelSum::AddBf16) {
        auto* residual = static_cast<__nv_bfloat16*>(out);
        residual[i]    = __float2bfloat16(__bfloat162float(residual[i]) + sum);
    } else if constexpr (Sum == TensorParallelSum::StoreBf16) {
        static_cast<__nv_bfloat16*>(out)[i] = __float2bfloat16(sum);
    } else {
        static_cast<float*>(out)[i] = sum;
    }
}

struct alignas(8) Bf16x4 {
    __nv_bfloat16 v[4];
};

// Four consecutive values' sums, from a multiple of four, each rounded as store_sum rounds it.
template <TensorParallelSum Sum>
__device__ __forceinline__ void store_sum4(void* out, std::int64_t i, float4 sum) {
    if constexpr (Sum == TensorParallelSum::AddBf16) {
        auto* residual = reinterpret_cast<Bf16x4*>(static_cast<__nv_bfloat16*>(out) + i);
        Bf16x4 word    = *residual;
        word.v[0]      = __float2bfloat16(__bfloat162float(word.v[0]) + sum.x);
        word.v[1]      = __float2bfloat16(__bfloat162float(word.v[1]) + sum.y);
        word.v[2]      = __float2bfloat16(__bfloat162float(word.v[2]) + sum.z);
        word.v[3]      = __float2bfloat16(__bfloat162float(word.v[3]) + sum.w);
        *residual      = word;
    } else if constexpr (Sum == TensorParallelSum::StoreBf16) {
        const Bf16x4 word{{__float2bfloat16(sum.x), __float2bfloat16(sum.y),
                           __float2bfloat16(sum.z), __float2bfloat16(sum.w)}};
        *reinterpret_cast<Bf16x4*>(static_cast<__nv_bfloat16*>(out) + i) = word;
    } else {
        *reinterpret_cast<float4*>(static_cast<float*>(out) + i) = sum;
    }
}

// Values [begin, end); the adds run in rank order. `vector` (begin a multiple of four and every
// pointer aligned for it) reads each partial four values at a time, which crosses the bus in
// fewer, larger requests; the arithmetic per value is the same.
template <TensorParallelSum Sum>
__global__ void tensor_parallel_reduce_kernel(void* __restrict__ out, PartialSet parts,
                                              std::int64_t begin, std::int64_t end, bool vector) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::int64_t tail         = begin;
    if (vector) {
        const std::int64_t quads = (end - begin) / 4;
        for (std::int64_t q = thread; q < quads; q += stride) {
            const std::int64_t i = begin + 4 * q;
            float4 sum           = *reinterpret_cast<const float4*>(parts.part[0] + i);
            for (int r = 1; r < parts.count; ++r) {
                const float4 part = *reinterpret_cast<const float4*>(parts.part[r] + i);
                sum.x += part.x;
                sum.y += part.y;
                sum.z += part.z;
                sum.w += part.w;
            }
            store_sum4<Sum>(out, i, sum);
        }
        tail = begin + 4 * quads;
    }
    for (std::int64_t i = tail + thread; i < end; i += stride) {
        float sum = parts.part[0][i];
        for (int r = 1; r < parts.count; ++r) { sum += parts.part[r][i]; }
        store_sum<Sum>(out, i, sum);
    }
}

struct GatherSet {
    const unsigned char* source[kMaxRanks];
    std::int64_t offset[kMaxRanks];
    std::int64_t bytes[kMaxRanks];
};

// Row y of the grid copies `bytes[y]` from `source[y]` to `out + offset[y]`, 16 bytes at a time
// where `vector` says every pointer and offset allows it.
__global__ void tensor_parallel_gather_kernel(unsigned char* __restrict__ out, GatherSet set,
                                              bool vector) {
    const unsigned char* source = set.source[blockIdx.y];
    unsigned char* destination  = out + set.offset[blockIdx.y];
    const std::int64_t bytes    = set.bytes[blockIdx.y];
    const std::int64_t stride   = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t thread   = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::int64_t tail           = 0;
    if (vector) {
        const std::int64_t words = bytes / 16;
        for (std::int64_t w = thread; w < words; w += stride) {
            reinterpret_cast<uint4*>(destination)[w] = reinterpret_cast<const uint4*>(source)[w];
        }
        tail = words * 16;
    }
    for (std::int64_t b = tail + thread; b < bytes; b += stride) { destination[b] = source[b]; }
}

bool aligned(const void* pointer, std::size_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

std::size_t output_bytes(TensorParallelSum sum) {
    return sum == TensorParallelSum::StoreFp32 ? sizeof(float) : sizeof(__nv_bfloat16);
}

struct LowLatencyArgs {
    void* out;
    const float* mine;
    const unsigned long long* rows[kMaxRanks];
    std::uint32_t* epoch;
    std::int64_t values;
    std::int64_t capacity;
    int ranks;
    int rank;
    int contributors;
};

__device__ __forceinline__ unsigned long long low_latency_clock() {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// The value of a word once it carries `epoch`.
__device__ __forceinline__ std::uint32_t await_word(const unsigned long long* word,
                                                    std::uint32_t epoch,
                                                    unsigned long long begun) {
    unsigned long long seen = *reinterpret_cast<const volatile unsigned long long*>(word);
    for (std::uint32_t spins = 0; static_cast<std::uint32_t>(seen >> 32U) != epoch;) {
        if ((++spins & 1023U) == 0 &&
            low_latency_clock() - begun > TensorParallelExchange::kLowLatencyTimeoutNs) {
            printf("ninfer: a tensor-parallel exchange waited %llu s for epoch %u\n",
                   TensorParallelExchange::kLowLatencyTimeoutNs / 1'000'000'000ULL, epoch);
            __trap();
        }
        seen = *reinterpret_cast<const volatile unsigned long long*>(word);
    }
    return static_cast<std::uint32_t>(seen);
}

template <TensorParallelSum Sum>
__global__ void tensor_parallel_low_latency_kernel(LowLatencyArgs a) {
    const std::uint32_t epoch = *reinterpret_cast<const volatile std::uint32_t*>(a.epoch) + 1U;
    const std::int64_t base   = static_cast<std::int64_t>(epoch & 1U) * a.capacity;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t first  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool contributes    = a.rank < a.contributors;
    auto* own = const_cast<volatile unsigned long long*>(a.rows[a.rank]) + base;
    const unsigned long long tag = static_cast<unsigned long long>(epoch) << 32U;
    if (contributes) {
        for (std::int64_t i = first; i < a.values; i += stride) {
            own[i] = tag | __float_as_uint(a.mine[i]);
        }
    } else if (first == 0) {
        own[0] = tag;
    }
    const unsigned long long begun = low_latency_clock();
    for (std::int64_t i = first; i < a.values; i += stride) {
        float sum = 0.0F;
        for (int r = 0; r < a.contributors; ++r) {
            const float part = r == a.rank
                                   ? a.mine[i]
                                   : __uint_as_float(await_word(a.rows[r] + base + i, epoch, begun));
            sum = r == 0 ? part : sum + part;
        }
        store_sum<Sum>(a.out, i, sum);
    }
    if (first == 0) {
        for (int r = a.contributors; r < a.ranks; ++r) {
            if (r != a.rank) { (void)await_word(a.rows[r] + base, epoch, begun); }
        }
    }
    // Every block read the count before it finished, so the last one to finish advances it.
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(a.epoch + 1, 1U) + 1U == gridDim.x) {
            a.epoch[1] = 0;
            a.epoch[0] = epoch;
        }
    }
}

} // namespace

namespace detail {

void tensor_parallel_reduce(void* out, const float* const* parts, std::size_t count,
                            std::int64_t begin, std::int64_t end, cudaStream_t stream,
                            TensorParallelSum sum) {
    if (count == 0 || count > kMaxRanks || begin < 0 || end < begin) {
        throw std::invalid_argument("tensor_parallel_reduce: invalid rank or value range");
    }
    if (end == begin) { return; }
    PartialSet set{};
    bool vector = begin % 4 == 0 && aligned(out, 4 * output_bytes(sum));
    for (std::size_t r = 0; r < count; ++r) {
        set.part[r] = parts[r];
        vector      = vector && aligned(parts[r], sizeof(float4));
    }
    set.count                 = static_cast<int>(count);
    constexpr int kBlock      = 256;
    const std::int64_t items  = vector ? (end - begin + 3) / 4 : end - begin;
    const std::int64_t blocks = std::min<std::int64_t>((items + kBlock - 1) / kBlock, 4096);
    const auto grid           = static_cast<unsigned>(blocks);
    switch (sum) {
    case TensorParallelSum::AddBf16:
        tensor_parallel_reduce_kernel<TensorParallelSum::AddBf16>
            <<<grid, kBlock, 0, stream>>>(out, set, begin, end, vector);
        break;
    case TensorParallelSum::StoreBf16:
        tensor_parallel_reduce_kernel<TensorParallelSum::StoreBf16>
            <<<grid, kBlock, 0, stream>>>(out, set, begin, end, vector);
        break;
    case TensorParallelSum::StoreFp32:
        tensor_parallel_reduce_kernel<TensorParallelSum::StoreFp32>
            <<<grid, kBlock, 0, stream>>>(out, set, begin, end, vector);
        break;
    }
    check(cudaGetLastError(), "reduce launch");
}

TensorParallelRange tensor_parallel_scatter_range(std::size_t rank, std::size_t ranks,
                                                  std::int64_t values) noexcept {
    constexpr std::int64_t kBlock = 64;
    const auto n                  = static_cast<std::int64_t>(ranks);
    const auto r                  = static_cast<std::int64_t>(rank);
    const std::int64_t blocks     = (values + kBlock - 1) / kBlock;
    const std::int64_t per        = blocks / n;
    const std::int64_t extra      = blocks % n;
    const std::int64_t first      = r * per + std::min(r, extra);
    const std::int64_t last       = first + per + (r < extra ? 1 : 0);
    return {std::min(first * kBlock, values), std::min(last * kBlock, values)};
}

void tensor_parallel_gather(void* out, const void* const* sources, const std::int64_t* offsets,
                            const std::int64_t* bytes, std::size_t count, cudaStream_t stream) {
    if (count > kMaxRanks) {
        throw std::invalid_argument("tensor_parallel_gather: too many ranges");
    }
    GatherSet set{};
    std::size_t rows     = 0;
    std::int64_t largest = 0;
    bool vector          = aligned(out, 16);
    for (std::size_t i = 0; i < count; ++i) {
        if (bytes[i] < 0 || offsets[i] < 0) {
            throw std::invalid_argument("tensor_parallel_gather: invalid range");
        }
        if (bytes[i] == 0) { continue; }
        set.source[rows] = static_cast<const unsigned char*>(sources[i]);
        set.offset[rows] = offsets[i];
        set.bytes[rows]  = bytes[i];
        vector           = vector && aligned(sources[i], 16) && offsets[i] % 16 == 0;
        largest          = std::max(largest, bytes[i]);
        ++rows;
    }
    if (rows == 0) { return; }
    constexpr int kBlock      = 256;
    const std::int64_t items  = vector ? (largest + 15) / 16 : largest;
    const std::int64_t blocks = std::min<std::int64_t>((items + kBlock - 1) / kBlock, 1024);
    tensor_parallel_gather_kernel<<<
        dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(rows)), kBlock, 0, stream>>>(
        static_cast<unsigned char*>(out), set, vector);
    check(cudaGetLastError(), "gather launch");
}

void tensor_parallel_low_latency(void* out, const float* mine,
                                 const unsigned long long* const* rows, std::size_t ranks,
                                 std::size_t rank, std::size_t contributors, std::int64_t values,
                                 std::size_t capacity, std::uint32_t* epoch, cudaStream_t stream,
                                 TensorParallelSum sum) {
    if (ranks < 2 || ranks > kMaxRanks || rank >= ranks || contributors == 0 ||
        contributors > ranks || values <= 0 || static_cast<std::size_t>(values) > capacity) {
        throw std::invalid_argument("tensor_parallel_low_latency: invalid ranks or value count");
    }
    LowLatencyArgs args{};
    args.out      = out;
    args.mine     = mine;
    for (std::size_t r = 0; r < ranks; ++r) { args.rows[r] = rows[r]; }
    args.epoch        = epoch;
    args.values       = values;
    args.capacity     = static_cast<std::int64_t>(capacity);
    args.ranks        = static_cast<int>(ranks);
    args.rank         = static_cast<int>(rank);
    args.contributors = static_cast<int>(contributors);
    // About two values per thread: a poll is a bus round trip, so the waits overlap across blocks.
    constexpr int kBlock = 256;
    const std::int64_t blocks = std::clamp<std::int64_t>(values / (2 * kBlock), 1, 64);
    const auto grid = static_cast<unsigned>(blocks);
    switch (sum) {
    case TensorParallelSum::AddBf16:
        tensor_parallel_low_latency_kernel<TensorParallelSum::AddBf16>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreBf16:
        tensor_parallel_low_latency_kernel<TensorParallelSum::StoreBf16>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    case TensorParallelSum::StoreFp32:
        tensor_parallel_low_latency_kernel<TensorParallelSum::StoreFp32>
            <<<grid, kBlock, 0, stream>>>(args);
        break;
    }
    check(cudaGetLastError(), "low-latency exchange launch");
}

} // namespace detail

std::vector<std::uint64_t> rank_free_bytes(const DeviceContext& context) {
    std::vector<std::uint64_t> out(context.size(), 0);
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < context.size(); ++other) {
            if (context.same_physical_device(rank, other)) { ++sharing; }
        }
        DeviceBinding bind(context.rank(rank).device);
        std::size_t free = 0, total = 0;
        check(cudaMemGetInfo(&free, &total), "free device memory");
        out[rank] = free / sharing;
    }
    return out;
}

std::vector<double> rank_copy_bandwidth(const DeviceContext& context) {
    // Large enough to stream past the L2 of every supported card, small enough to fit beside a
    // CUDA context on any of them.
    constexpr std::size_t kBytes  = 128ULL << 20;
    constexpr int kTimedCopies    = 3;
    std::vector<double> out(context.size(), 0.0);
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        std::size_t first = rank;
        for (std::size_t other = 0; other < rank; ++other) {
            if (context.same_physical_device(rank, other)) {
                first = other;
                break;
            }
        }
        if (first != rank) {
            out[rank] = out[first];
            continue;
        }
        DeviceBinding bind(context.rank(rank).device);
        void* source      = nullptr;
        void* destination = nullptr;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop  = nullptr;
        cudaStream_t stream = nullptr;
        const auto release = [&] {
            if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
            if (start != nullptr) { (void)cudaEventDestroy(start); }
            if (stop != nullptr) { (void)cudaEventDestroy(stop); }
            if (source != nullptr) { (void)cudaFree(source); }
            if (destination != nullptr) { (void)cudaFree(destination); }
        };
        try {
            check(cudaMalloc(&source, kBytes), "bandwidth cudaMalloc");
            check(cudaMalloc(&destination, kBytes), "bandwidth cudaMalloc");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "bandwidth stream");
            check(cudaEventCreate(&start), "bandwidth event");
            check(cudaEventCreate(&stop), "bandwidth event");
            check(cudaMemsetAsync(source, 0, kBytes, stream), "bandwidth memset");
            check(cudaMemcpyAsync(destination, source, kBytes, cudaMemcpyDeviceToDevice, stream),
                  "bandwidth warm-up");
            check(cudaEventRecord(start, stream), "bandwidth record");
            for (int copy = 0; copy < kTimedCopies; ++copy) {
                check(cudaMemcpyAsync(destination, source, kBytes, cudaMemcpyDeviceToDevice, stream),
                      "bandwidth copy");
            }
            check(cudaEventRecord(stop, stream), "bandwidth record");
            check(cudaEventSynchronize(stop), "bandwidth synchronize");
            float milliseconds = 0.0F;
            check(cudaEventElapsedTime(&milliseconds, start, stop), "bandwidth elapsed");
            // A copy reads and writes every byte.
            const double moved = 2.0 * static_cast<double>(kBytes) * kTimedCopies;
            out[rank] = moved / (std::max(static_cast<double>(milliseconds), 1e-3) * 1e-3);
        } catch (...) {
            release();
            throw;
        }
        release();
    }
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < context.size(); ++other) {
            if (context.same_physical_device(rank, other)) { ++sharing; }
        }
        out[rank] /= static_cast<double>(sharing);
    }
    return out;
}

TensorParallelExchange::TensorParallelExchange(const DeviceContext& context,
                                               TensorParallelExchangeOptions options)
    : slot_bytes_(options.slot_bytes) {
    if (context.size() < 2) {
        throw std::invalid_argument("a tensor-parallel exchange needs at least two ranks");
    }
    if (slot_bytes_ == 0 || slot_bytes_ % sizeof(float) != 0) {
        throw std::invalid_argument("a tensor-parallel slot holds a positive number of floats");
    }
    bool peer = !options.force_staged;
    for (std::size_t a = 0; a < context.size() && peer; ++a) {
        for (std::size_t b = 0; b < context.size(); ++b) {
            if (a != b && !context.same_physical_device(a, b) && !context.peer_access(b, a)) {
                peer = false;
                break;
            }
        }
    }
    transport_ = peer ? TensorParallelTransport::Peer : TensorParallelTransport::StagedHost;

    // Low-latency words and the smaller reduce-scatter threshold hold through host memory with
    // every rank on its own card.
    bool own_cards = true;
    for (std::size_t a = 0; a < context.size(); ++a) {
        for (std::size_t b = a + 1; b < context.size(); ++b) {
            if (context.same_physical_device(a, b)) { own_cards = false; }
        }
    }
    const bool host = transport_ == TensorParallelTransport::StagedHost && own_cards;
    if (host && context.size() <= kMaxRanks) {
        low_latency_bytes_ = std::min(options.low_latency_bytes, slot_bytes_) / sizeof(float) *
                             sizeof(float);
    }
    scatter_bytes_ = host ? options.scatter_bytes : options.near_scatter_bytes;
    ranks_.resize(context.size());
    try {
        for (std::size_t r = 0; r < ranks_.size(); ++r) {
            Rank& rank  = ranks_[r];
            rank.device = context.rank(r).device;
            DeviceBinding bind(rank.device);
            for (std::size_t slot = 0; slot < kSlots; ++slot) {
                check(cudaMalloc(reinterpret_cast<void**>(&rank.device_slot[slot]), slot_bytes_),
                      "cudaMalloc");
                if (transport_ == TensorParallelTransport::StagedHost) {
                    // Mapped, so a reduce on any rank reads it in place; portable, so every
                    // rank's context treats it as pinned.
                    check(cudaHostAlloc(reinterpret_cast<void**>(&rank.host_slot[slot]),
                                        slot_bytes_, cudaHostAllocMapped | cudaHostAllocPortable),
                          "cudaHostAlloc");
                }
                // Events cannot be created while a stream captures, so all exist up front. A
                // rank records its own events.
                check(cudaEventCreateWithFlags(&rank.published[slot], cudaEventDisableTiming),
                      "cudaEventCreate");
                check(cudaEventCreateWithFlags(&rank.reduced[slot], cudaEventDisableTiming),
                      "cudaEventCreate");
            }
            if (low_latency_bytes_ != 0) {
                const std::size_t words = kSlots * (low_latency_bytes_ / sizeof(float));
                check(cudaHostAlloc(reinterpret_cast<void**>(&rank.words),
                                    words * sizeof(unsigned long long),
                                    cudaHostAllocMapped | cudaHostAllocPortable),
                      "cudaHostAlloc");
                std::fill_n(rank.words, words, 0ULL);
                check(cudaMalloc(reinterpret_cast<void**>(&rank.epoch), 2 * sizeof(std::uint32_t)),
                      "cudaMalloc");
                check(cudaMemset(rank.epoch, 0, 2 * sizeof(std::uint32_t)), "cudaMemset");
                // Loaded now, so no first launch waits on module loading while a peer spins.
                cudaFuncAttributes attributes{};
                for (const auto kernel :
                     {tensor_parallel_low_latency_kernel<TensorParallelSum::AddBf16>,
                      tensor_parallel_low_latency_kernel<TensorParallelSum::StoreBf16>,
                      tensor_parallel_low_latency_kernel<TensorParallelSum::StoreFp32>}) {
                    check(cudaFuncGetAttributes(&attributes, kernel), "low-latency kernel");
                }
            }
        }
    } catch (...) {
        release();
        throw;
    }
}

TensorParallelExchange::~TensorParallelExchange() { release(); }

void TensorParallelExchange::release() noexcept {
    for (Rank& rank : ranks_) {
        DeviceBinding bind(rank.device);
        for (std::size_t slot = 0; slot < kSlots; ++slot) {
            if (rank.published[slot] != nullptr) { (void)cudaEventDestroy(rank.published[slot]); }
            if (rank.reduced[slot] != nullptr) { (void)cudaEventDestroy(rank.reduced[slot]); }
            if (rank.device_slot[slot] != nullptr) { (void)cudaFree(rank.device_slot[slot]); }
            if (rank.host_slot[slot] != nullptr) { (void)cudaFreeHost(rank.host_slot[slot]); }
            rank.published[slot]   = nullptr;
            rank.reduced[slot]     = nullptr;
            rank.device_slot[slot] = nullptr;
            rank.host_slot[slot]   = nullptr;
        }
        if (rank.words != nullptr) { (void)cudaFreeHost(rank.words); }
        if (rank.epoch != nullptr) { (void)cudaFree(rank.epoch); }
    }
    ranks_.clear();
}

void TensorParallelExchange::check_slot(std::size_t rank, std::size_t slot,
                                        std::size_t bytes) const {
    if (rank >= ranks_.size() || slot >= kSlots) {
        throw std::out_of_range("tensor-parallel rank or slot is out of range");
    }
    if (bytes > slot_bytes_) {
        throw std::invalid_argument("tensor-parallel partial exceeds its slot");
    }
}

float* TensorParallelExchange::partial(std::size_t rank, std::size_t slot) const {
    check_slot(rank, slot, 0);
    return ranks_[rank].device_slot[slot];
}

void TensorParallelExchange::publish(std::size_t rank, std::size_t slot, std::size_t bytes,
                                     cudaStream_t stream, TensorParallelSum sum) {
    check_slot(rank, slot, bytes);
    Rank& self = ranks_[rank];
    DeviceBinding bind(self.device);
    // Low-latency words leave the slot where it is: the reduce's kernel stores them itself. A
    // reduce-scatter's own range of the partial stays on the device, since no other rank reads it.
    if (transport_ == TensorParallelTransport::StagedHost && bytes != 0 && !low_latency(bytes)) {
        std::size_t own_begin = bytes;
        std::size_t own_end   = bytes;
        if (scatter(bytes, sum)) {
            const auto range = detail::tensor_parallel_scatter_range(
                rank, ranks_.size(), static_cast<std::int64_t>(bytes / sizeof(float)));
            own_begin = static_cast<std::size_t>(range.begin) * sizeof(float);
            own_end   = static_cast<std::size_t>(range.end) * sizeof(float);
        }
        auto* host         = reinterpret_cast<unsigned char*>(self.host_slot[slot]);
        const auto* device = reinterpret_cast<const unsigned char*>(self.device_slot[slot]);
        if (own_begin != 0) {
            check(cudaMemcpyAsync(host, device, own_begin, cudaMemcpyDeviceToHost, stream),
                  "publish copy");
        }
        if (own_end != bytes) {
            check(cudaMemcpyAsync(host + own_end, device + own_end, bytes - own_end,
                                  cudaMemcpyDeviceToHost, stream),
                  "publish copy");
        }
    }
    check(cudaEventRecord(self.published[slot], stream), "publish fence");
}

void TensorParallelExchange::publish_whole(std::size_t rank, std::size_t slot, std::size_t bytes,
                                           cudaStream_t stream) {
    check_slot(rank, slot, bytes);
    Rank& self = ranks_[rank];
    DeviceBinding bind(self.device);
    if (transport_ == TensorParallelTransport::StagedHost && bytes != 0) {
        check(cudaMemcpyAsync(self.host_slot[slot], self.device_slot[slot], bytes,
                              cudaMemcpyDeviceToHost, stream),
              "publish copy");
    }
    check(cudaEventRecord(self.published[slot], stream), "publish fence");
}

void TensorParallelExchange::gather(std::size_t rank, std::size_t slot,
                                    std::span<const float*> parts, cudaStream_t stream) {
    check_slot(rank, slot, 0);
    if (parts.size() != ranks_.size()) {
        throw std::invalid_argument("tensor-parallel gather needs one pointer per rank");
    }
    DeviceBinding bind(ranks_[rank].device);
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        if (r == rank) {
            parts[r] = ranks_[r].device_slot[slot];
            continue;
        }
        check(cudaStreamWaitEvent(stream, ranks_[r].published[slot], 0), "gather wait");
        parts[r] = transport_ == TensorParallelTransport::Peer ? ranks_[r].device_slot[slot]
                                                              : ranks_[r].host_slot[slot];
    }
}

void TensorParallelExchange::reduce(std::size_t rank, std::size_t slot, void* out,
                                    std::int64_t count, std::size_t contributors,
                                    cudaStream_t stream, TensorParallelSum sum) {
    check_slot(rank, slot, static_cast<std::size_t>(count) * sizeof(float));
    if (contributors == 0 || contributors > ranks_.size()) {
        throw std::invalid_argument("tensor-parallel reduce needs one to all ranks as contributors");
    }
    DeviceBinding bind(ranks_[rank].device);
    if (low_latency(static_cast<std::size_t>(count) * sizeof(float))) {
        std::array<const unsigned long long*, kMaxRanks> rows{};
        for (std::size_t r = 0; r < ranks_.size(); ++r) { rows[r] = ranks_[r].words; }
        detail::tensor_parallel_low_latency(out, ranks_[rank].device_slot[slot], rows.data(),
                                            ranks_.size(), rank, contributors, count,
                                            low_latency_bytes_ / sizeof(float),
                                            ranks_[rank].epoch, stream, sum);
        return;
    }
    std::array<const float*, kMaxRanks> parts{};
    for (std::size_t r = 0; r < ranks_.size(); ++r) {
        if (r == rank) {
            parts[r] = ranks_[r].device_slot[slot];
            continue;
        }
        check(cudaStreamWaitEvent(stream, ranks_[r].published[slot], 0), "reduce wait");
        parts[r] = transport_ == TensorParallelTransport::Peer ? ranks_[r].device_slot[slot]
                                                              : ranks_[r].host_slot[slot];
    }
    if (!scatter(static_cast<std::size_t>(count) * sizeof(float), sum)) {
        detail::tensor_parallel_reduce(out, parts.data(), contributors, 0, count, stream, sum);
        return;
    }
    const auto range = detail::tensor_parallel_scatter_range(rank, ranks_.size(), count);
    detail::tensor_parallel_reduce(out, parts.data(), contributors, range.begin, range.end, stream,
                                   sum);
    Rank& self        = ranks_[rank];
    self.output[slot] = out;
    if (transport_ == TensorParallelTransport::StagedHost && range.end > range.begin) {
        const std::size_t size = output_bytes(sum);
        check(cudaMemcpyAsync(reinterpret_cast<unsigned char*>(self.host_slot[slot]) +
                                  static_cast<std::size_t>(range.begin) * sizeof(float),
                              static_cast<const unsigned char*>(out) +
                                  static_cast<std::size_t>(range.begin) * size,
                              static_cast<std::size_t>(range.end - range.begin) * size,
                              cudaMemcpyDeviceToHost, stream),
              "reduce copy");
    }
    check(cudaEventRecord(self.reduced[slot], stream), "reduce fence");
}

void TensorParallelExchange::complete(std::size_t rank, std::size_t slot, void* out,
                                      std::int64_t count, cudaStream_t stream,
                                      TensorParallelSum sum) {
    check_slot(rank, slot, static_cast<std::size_t>(count) * sizeof(float));
    if (!scatter(static_cast<std::size_t>(count) * sizeof(float), sum)) { return; }
    DeviceBinding bind(ranks_[rank].device);
    const std::size_t size = output_bytes(sum);
    std::array<const void*, kMaxRanks> sources{};
    std::array<std::int64_t, kMaxRanks> offsets{};
    std::array<std::int64_t, kMaxRanks> bytes{};
    std::size_t copies = 0;
    for (std::size_t owner = 0; owner < ranks_.size(); ++owner) {
        const auto range = detail::tensor_parallel_scatter_range(owner, ranks_.size(), count);
        if (owner == rank || range.end == range.begin) { continue; }
        const Rank& from = ranks_[owner];
        if (from.output[slot] == nullptr) {
            throw std::logic_error("tensor-parallel complete before every rank's reduce");
        }
        check(cudaStreamWaitEvent(stream, from.reduced[slot], 0), "complete wait");
        sources[copies] = transport_ == TensorParallelTransport::Peer
                              ? static_cast<const unsigned char*>(from.output[slot]) +
                                    static_cast<std::size_t>(range.begin) * size
                              : reinterpret_cast<const unsigned char*>(from.host_slot[slot]) +
                                    static_cast<std::size_t>(range.begin) * sizeof(float);
        offsets[copies] = range.begin * static_cast<std::int64_t>(size);
        bytes[copies]   = (range.end - range.begin) * static_cast<std::int64_t>(size);
        ++copies;
    }
    detail::tensor_parallel_gather(out, sources.data(), offsets.data(), bytes.data(), copies,
                                   stream);
}

} // namespace ninfer
