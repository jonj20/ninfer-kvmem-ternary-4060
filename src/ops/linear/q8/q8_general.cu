#include "ops/linear/q8/q8_simt_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {
namespace {

// The runtime-shape routes of a product no shape table names, such as a tensor-parallel rank's
// share of a projection: unified templates whose rows and K come from the operands. A BF16 result
// takes the instances the shape tables use; an FP32 result (a rank's partial sum) the same
// schedules writing FP32.

template <class Schedule>
void mma_fp32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, w),
        LinearFp32Output{static_cast<float*>(out.data), w.n}, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void simt_fp32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_simt<Schedule>(q8_linear_operands(x, w),
        LinearFp32Output{static_cast<float*>(out.data), w.n}, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced_k_mma_fp32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(q8_linear_operands(x, w),
        LinearFp32Output{static_cast<float*>(out.data), w.n}, LinearIdentityEpilogue{}, stream);
}

} // namespace

Q8Launch select_q8_general_launch(std::int32_t tokens, DType out) {
    using namespace q8_instances;
    if (out == DType::FP32) {
        if (tokens <= 4) return simt_fp32<SimtR4T4W2G16S2>;
        if (tokens <= 16) return sliced_k_mma_fp32<SlicedR16T16W4S2>;
        if (tokens <= 32) return sliced_k_mma_fp32<SlicedR16T32W4S2>;
        if (tokens <= 64) return mma_fp32<MmaR32T64>;
        return mma_fp32<MmaR64T128>;
    }
    if (tokens <= 4) return launch_q8_a16_simt_r4_t4_w2_g16_s2;
    if (tokens <= 16) return launch_q8_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q8_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
