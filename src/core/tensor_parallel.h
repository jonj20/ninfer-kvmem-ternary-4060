#pragma once

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// The all-reduce of a tensor-parallel group. Every rank of a DeviceContext holds a replica of the
// BF16 residual and an FP32 partial of one row-split projection; each rank adds the sum of all
// partials, taken in rank order, to its replica and rounds once:
//
//   x_r <- bf16(float(x_r) + ((p_0 + p_1) + ... + p_{N-1}))
//
// Every rank evaluates the same expression on the same values, so the replicas stay identical, and
// the one rounding is the one a single device's residual-adding projection makes. A projection only
// rank 0 runs is an exchange with one contributor: every replica adds rank 0's output.
//
// A rank writes its partial into its slot (`partial`), publishes it (`publish`) and, once every
// rank has published, reduces (`reduce`). Every rank publishes in every exchange, one outside the
// contributors with zero bytes. Two slots alternate between exchanges. A rank writes a
// slot again two exchanges later, after its reduce of the exchange in between, which waits for
// every other rank's publish of that exchange, which each rank issues after its own reduce of the
// earlier one: so no rank overwrites a slot another is still reading. Consecutive passes are
// separate in the same way as long as a pass ends in a synchronize, as every round does.
//
// Records and waits pair within one exchange, so the protocol is legal eagerly and inside one
// capture that spans every rank's stream.
//
// Transports, chosen once from the topology:
//   Peer        every pair of ranks has peer access or shares a device: a reduce reads the other
//               ranks' device slots directly.
//   StagedHost  a publish copies the slot into a pinned, mapped host slot, and a reduce reads the
//               other ranks' host slots across the bus. Works between any two cards.
//
// Low-latency words. Under StagedHost with every rank on its own card, an exchange of at most
// `low_latency_bytes` per partial (a decode or verification step) skips the copy and
// the cross-device fences: the reduce is one kernel that stores each FP32 value of the rank's
// partial with the exchange's 32-bit epoch as one 8-byte word into the rank's own mapped host row,
// then polls the other ranks' rows until their words carry the same epoch, and adds in rank order
// as above. A rank outside the contributors stores one word. Each rank counts its own exchanges in
// device memory, so every rank derives the same epoch and a captured graph replays without new
// arguments; the row's slot is the epoch's parity. A rank writes a slot again two exchanges later,
// after its kernel of the exchange in between, which waited for every other rank's word of that
// exchange, which each rank stores only after its own earlier kernel ended: no rank overwrites
// words another is still reading. A wait that outlasts kLowLatencyTimeoutNs traps the context.
// Ranks that share a card keep the staged protocol, since two spinning kernels on one device need
// not run together.
//
// Reduce-scatter and all-gather. An exchange of at least `scatter_bytes` per partial
// (`near_scatter_bytes` where the ranks read peer memory or share a card; a prompt chunk or a long
// verification) gives each rank one contiguous range of the values, whole 64-value blocks as even
// as they divide. A rank's reduce sums only its own range of the contributors' partials, in rank
// order as above, into its output and records that the range is done; its `complete` then copies
// every other rank's finished range into its own output. Each value is still evaluated once, by the
// same expression on the same operands, so every output is bit-identical to a whole reduce, while a
// rank reads 1/N of each other rank's FP32 partial and (N-1)/N of an output instead of N-1 whole
// partials. Two ranks storing an FP32 sum would read half a partial and half an output, as many
// bytes as a whole partial, so they reduce whole. Under StagedHost a publish leaves the rank's own
// range of its partial on the device, since no other rank reads it, and the owner copies its
// finished range into that range of its host slot, where the others read it. A rank writes its
// range of an output again in its reduce of a later exchange, which waits for every other rank's
// publish of that exchange, which each rank issues after its own `complete` of this one: so no rank
// overwrites a range another is still copying. The caller issues `complete` for every rank after
// `reduce` for every rank, since a wait issued before its event's record would see an earlier
// record.
enum class TensorParallelTransport { Peer, StagedHost };

// What a reduce does with the sum of the partials: adds it to a BF16 residual and rounds once (the
// residual stream of a Qwen3.5-core model), or stores it in place of the output, rounded to BF16 or
// kept in FP32 (a block output that the caller then adds itself, as Qwen3.8-Flash-Next's
// hyper-connections do).
enum class TensorParallelSum : std::uint8_t { AddBf16, StoreBf16, StoreFp32 };

struct TensorParallelExchangeOptions {
    // Largest partial one slot holds.
    std::size_t slot_bytes = 0;
    // Use the staged protocol even where peer access exists (tests).
    bool force_staged = false;
    // Largest partial exchanged as low-latency words; 0 turns them off.
    std::size_t low_latency_bytes = 32U << 10U;
    // Smallest partial reduced as a reduce-scatter and an all-gather; 0 reduces every partial
    // whole. `scatter_bytes` holds where each rank reads the others through host memory from its
    // own card: for two and four ranks the two tie at 80 KiB and the reduce-scatter is faster from
    // 160 KiB. `near_scatter_bytes` holds where the ranks read peer memory or share a card, where a
    // whole reduce reads faster: through peer reads on one card it stays ahead up to 260 KiB and
    // ties at 320 KiB (tools/tp_exchange_bench.cu).
    std::size_t scatter_bytes      = 128U << 10U;
    std::size_t near_scatter_bytes = 320U << 10U;
};

class TensorParallelExchange {
public:
    static constexpr std::size_t kSlots = 2;
    static constexpr unsigned long long kLowLatencyTimeoutNs = 300'000'000'000ULL;

    TensorParallelExchange(const DeviceContext& context, TensorParallelExchangeOptions options);
    ~TensorParallelExchange();

    TensorParallelExchange(const TensorParallelExchange&)            = delete;
    TensorParallelExchange& operator=(const TensorParallelExchange&) = delete;
    TensorParallelExchange(TensorParallelExchange&&)                 = delete;
    TensorParallelExchange& operator=(TensorParallelExchange&&)      = delete;

    [[nodiscard]] std::size_t ranks() const noexcept { return ranks_.size(); }
    [[nodiscard]] TensorParallelTransport transport() const noexcept { return transport_; }
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }
    // Whether an exchange of `bytes` per partial travels as low-latency words.
    [[nodiscard]] bool low_latency(std::size_t bytes) const noexcept {
        return bytes != 0 && bytes <= low_latency_bytes_;
    }

    // Whether an exchange of `bytes` per partial whose reduce does `sum` runs as a reduce-scatter
    // and an all-gather.
    [[nodiscard]] bool scatter(std::size_t bytes,
                               TensorParallelSum sum = TensorParallelSum::AddBf16) const noexcept {
        return scatter_bytes_ != 0 && bytes >= scatter_bytes_ && !low_latency(bytes) &&
               (sum != TensorParallelSum::StoreFp32 || ranks_.size() > 2);
    }

    // Where `rank` writes its FP32 partial for an exchange in `slot`: device memory of that rank.
    [[nodiscard]] float* partial(std::size_t rank, std::size_t slot) const;
    // Makes `rank`'s partial of `bytes` in `slot` readable by the other ranks; `sum` is the
    // exchange's reduce.
    void publish(std::size_t rank, std::size_t slot, std::size_t bytes, cudaStream_t stream,
                 TensorParallelSum sum = TensorParallelSum::AddBf16);
    // Adds the partials ranks [0, contributors) published in `slot` to `rank`'s BF16 residual of
    // `count` values, or stores their sum there (`sum`); for a reduce-scatter, only in the rank's
    // own range.
    void reduce(std::size_t rank, std::size_t slot, void* out, std::int64_t count,
                std::size_t contributors, cudaStream_t stream,
                TensorParallelSum sum = TensorParallelSum::AddBf16);
    // Copies the other ranks' finished ranges of a reduce-scatter into `rank`'s output; nothing for
    // a smaller exchange. Every rank's `complete` follows every rank's `reduce` of the exchange,
    // with the same `out`, `count` and `sum`.
    void complete(std::size_t rank, std::size_t slot, void* out, std::int64_t count,
                  cudaStream_t stream, TensorParallelSum sum = TensorParallelSum::AddBf16);
    // A gather in place of the sum (a position-split attention's partials, merged by
    // ops::causal_attention_merge_partials): `publish_whole` makes `rank`'s partial of `bytes` in
    // `slot` readable by the others, never as low-latency words; `gather` points `parts[r]` at
    // every rank's partial in `slot` as `rank`'s kernels on `stream` read it (its own device slot,
    // another's device slot under Peer or its mapped host slot under StagedHost), after `stream`
    // waits for the other ranks' publishes. It takes its place in the slot alternation as an
    // exchange does: a rank's next exchange in the slot follows its own reads of it.
    void publish_whole(std::size_t rank, std::size_t slot, std::size_t bytes, cudaStream_t stream);
    void gather(std::size_t rank, std::size_t slot, std::span<const float*> parts,
                cudaStream_t stream);

private:
    struct Rank {
        int device = 0;
        std::array<float*, kSlots> device_slot{};
        std::array<float*, kSlots> host_slot{};
        std::array<cudaEvent_t, kSlots> published{};
        // A reduce-scatter: the output the rank's reduce wrote its range into, and its record.
        std::array<void*, kSlots> output{};
        std::array<cudaEvent_t, kSlots> reduced{};
        // Low-latency words: the rank's mapped host row, [kSlots][low_latency_bytes / 4], and its
        // {exchanges counted, blocks finished} in device memory.
        unsigned long long* words = nullptr;
        std::uint32_t* epoch      = nullptr;
    };

    void release() noexcept;
    void check_slot(std::size_t rank, std::size_t slot, std::size_t bytes) const;

    std::vector<Rank> ranks_;
    TensorParallelTransport transport_ = TensorParallelTransport::StagedHost;
    std::size_t slot_bytes_            = 0;
    std::size_t low_latency_bytes_     = 0; // 0 where the transport and topology exclude them
    std::size_t scatter_bytes_         = 0;
};

// Each rank's speed for dividing work by: the device-to-device copy bandwidth of its card in bytes
// per second, shared evenly by the ranks one card holds (`--devices 0,0`). A decode step reads every
// weight once, so a rank's share of the weights takes time in proportion to this. Measured with a
// short copy on each card.
[[nodiscard]] std::vector<double> rank_copy_bandwidth(const DeviceContext& context);

// Each rank's free device memory now, shared evenly by the ranks one card holds.
[[nodiscard]] std::vector<std::uint64_t> rank_free_bytes(const DeviceContext& context);

namespace detail {

// One rank's reduction of values [begin, end): out[i] <- bf16(float(out[i]) + ((part[0][i] +
// part[1][i]) + ...)), or the sum stored as BF16 or FP32 (`sum`).
void tensor_parallel_reduce(void* out, const float* const* parts, std::size_t count,
                            std::int64_t begin, std::int64_t end, cudaStream_t stream,
                            TensorParallelSum sum = TensorParallelSum::AddBf16);

// Values [begin, end).
struct TensorParallelRange {
    std::int64_t begin = 0;
    std::int64_t end   = 0;
};

// Rank `rank`'s range of a reduce-scatter of `values` over `ranks` ranks.
[[nodiscard]] TensorParallelRange tensor_parallel_scatter_range(std::size_t rank, std::size_t ranks,
                                                                std::int64_t values) noexcept;

// Byte copies into `out`: `bytes[i]` from `sources[i]` to `out + offsets[i]`, for i < `count`.
void tensor_parallel_gather(void* out, const void* const* sources, const std::int64_t* offsets,
                            const std::int64_t* bytes, std::size_t count, cudaStream_t stream);

// One rank's low-latency exchange (see TensorParallelExchange): stores `mine` (a contributor's
// partial) as words of the next epoch into `row`, waits for the words of every other rank's row,
// and adds the contributors' values in rank order, to the residual or into `out` as `sum` says.
// `rows[r]` is rank r's row of `capacity` words per slot; `epoch` is the rank's counter pair.
void tensor_parallel_low_latency(void* out, const float* mine,
                                 const unsigned long long* const* rows, std::size_t ranks,
                                 std::size_t rank, std::size_t contributors, std::int64_t values,
                                 std::size_t capacity, std::uint32_t* epoch, cudaStream_t stream,
                                 TensorParallelSum sum = TensorParallelSum::AddBf16);

} // namespace detail

} // namespace ninfer
