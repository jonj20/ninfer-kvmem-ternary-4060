#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SparseMoeWeights {
    Weight router_shared_gate;
    Weight routed_gate_up;
    Weight routed_down;
    Weight shared_gate_up;
    Weight shared_down;
};

enum class SparseMoeEpilogue : std::uint8_t {
    AddResidual,
};

/**
 * Optional per-call execution hints. Every field is a pure cache hint with no numeric effect:
 * the same call with a default-constructed SparseMoeHints produces bit-identical output.
 *
 * next_weight_prefetch names a weight span the next decode-step consumer will stream; the decode
 * D4 epilogue issues fire-and-forget L2 prefetches over its first bytes. The span is caller-owned
 * and read once, inside the call: the Op keeps no state between calls, and no hidden channel
 * carries it.
 */
struct SparseMoeHints {
    const void* next_weight_prefetch       = nullptr;
    std::size_t next_weight_prefetch_bytes = 0;
};

/**
 * Returns the transient capacity required by SparseMoe for every T in the inclusive
 * [min_tokens,max_tokens] interval. The routed QTypes are the fixed implementation profile.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(QType routed_gate_up,
                                                              QType routed_down,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Closed sparse-MoE Op for the exact future 35B-A3B geometry.
 *
 * For contiguous BF16 x [2048,T] and destination [2048,T] with T>0, 256 routed experts, top-8
 * selection, and one always-on shared expert, this Op owns router projection and selection,
 * selected routed and shared SwiGLU projections, down projections, their merge, and the
 * AddResidual epilogue independently for every token column. At an exact top-8 boundary tie the
 * lower expert id wins. destination is the only observable mutation: its incoming value is the
 * residual and its outgoing value is the BF16 sparse-MoE result plus that residual.
 *
 * The complete mathematical oracle starts from represented BF16 inputs, exact stored-weight
 * decode, and evaluates the logical formula naively in FP32/FP64. Scores, route weights, expert
 * activations, workspace representation, reduction association, and scale placement are private
 * execution choices rather than semantic rounding boundaries.
 *
 * The five weights have the exact registered shapes: BF16 router/shared gate [257,2048], routed
 * gate/up [256*1024,2048], routed down [256*2048,512], shared gate/up [1024,2048], and shared down
 * [2048,512]. Admitted codec profiles are Q4+Q5, Q4+Q6, and Q8+Q8 for the two routed banks; both
 * shared banks are Q8. Expert e directly selects its stored row spans; no selected-weight gather
 * or repack occurs.
 *
 * Every positive T is supported.
 *
 * x, destination, all weight planes, and live workspace must be pairwise non-overlapping.
 * Execution is enqueued on stream without host synchronization. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * The same Op with caller-supplied execution hints. Semantics, workspace requirement and output
 * are exactly those of the overload above; hints only steer cache warming.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, const SparseMoeHints& hints, WorkspaceArena& workspace,
                cudaStream_t stream);

/**
 * One device's part of SparseMoe under expert parallelism: routed experts
 * [expert_begin, expert_begin + expert_count) of the 256, and the shared expert when
 * `shared_expert`, each with `intermediate` of its 512 intermediate columns: all of them, or one
 * half (256) where the devices divide every expert's intermediate columns instead.
 */
struct SparseMoeShare {
    std::int32_t expert_begin = 0;
    std::int32_t expert_count = 256;
    bool shared_expert        = true;
    std::int32_t intermediate = 512;
};

/**
 * SparseMoe over one device's share of the experts, before the residual.
 *
 * The device routes every token exactly as sparse_moe does, from its own router replica, and
 * evaluates only the selected paths of the experts it holds, and the shared path when it holds the
 * shared expert. It writes their weighted FP32 sum to `partial` [2048,T] (contiguous FP32,
 * 16-byte aligned) and leaves the residual alone. Over shares that partition the routed experts
 * and their intermediate columns, with each column of the shared expert in exactly one, the sum of
 * the partials is sparse_moe's result less its residual: the caller adds that sum to the residual
 * and rounds once to BF16. A divided expert's SwiGLU columns are exactly the whole expert's; only
 * the association of its down product's FP32 sum changes.
 *
 * router_shared_gate is the whole [257,2048] router. With W = intermediate, routed_gate_up
 * [expert_count*2W,2048] and routed_down [expert_count*2048,W] hold the share's experts in order,
 * expert e at rows of e - expert_begin (its W gate rows, then its W up rows, and its W columns of
 * down), and shared_gate_up [2W,2048] and shared_down [2048,W] the shared expert's columns. The
 * shared weights are read only when shared_expert. Admitted profiles are the groupwise ones of
 * sparse_moe; NVFP4 banks have no share.
 *
 * Every device's routes have to agree, so the share chooses its route by T alone, from the codec's
 * own prefill frontier rather than a device profile's: two different cards score the tokens with
 * the same kernel. Its workspace requirement is sparse_moe_share_workspace_capacity_bytes.
 */
[[nodiscard]] std::size_t sparse_moe_share_workspace_capacity_bytes(QType routed_gate_up,
                                                                    QType routed_down,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);
void sparse_moe_share(const Tensor& x, const SparseMoeWeights& weights, const SparseMoeShare& share,
                      Tensor& partial, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
