#include "ops/linear/q6/q6_simt_launch.cuh"
#include "ops/linear/q6/q6_mma_launch.cuh"
#include "ops/linear/q6/q6_sliced_k_launch.cuh"
#include "ops/linear/q6/q6_instances.cuh"
#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {
namespace {

// The runtime-shape routes of a product no shape table names, such as a tensor-parallel rank's
// share of a projection: unified templates whose rows and K come from the operands. A BF16 result
// takes the instances the shape tables use; an FP32 result (a rank's partial sum) the same
// schedules writing FP32.

template <class Schedule>
void gemv_fp32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q6_a16_gemv<Schedule>(q6_linear_operands(x, w),
        LinearFp32Output{static_cast<float*>(out.data), w.n}, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void mma_fp32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q6_a16_mma<Schedule>(q6_linear_operands(x, w),
        LinearFp32Output{static_cast<float*>(out.data), w.n}, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced_k_mma_fp32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q6_a16_sliced_k_mma<Schedule>(q6_linear_operands(x, w),
        LinearFp32Output{static_cast<float*>(out.data), w.n}, LinearIdentityEpilogue{}, stream);
}

} // namespace

Q6Launch select_q6_general_launch(std::int32_t tokens, DType out) {
    using namespace q6_instances;
    if (out == DType::FP32) {
        if (tokens <= 1) return gemv_fp32<GemvR4W2G16>;
        if (tokens <= 8) return sliced_k_mma_fp32<SlicedR16T8W4S2>;
        if (tokens <= 16) return sliced_k_mma_fp32<SlicedR32T16W4S2>;
        if (tokens <= 64) return sliced_k_mma_fp32<SlicedR32T32W4S2>;
        return mma_fp32<MmaR64T128>;
    }
    if (tokens <= 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 64) return launch_q6_a16_sliced_r32_t32_w4_s2;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
