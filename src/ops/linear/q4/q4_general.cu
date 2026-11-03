#include "ops/linear/q4/q4_instance_launch.cuh"
#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
namespace {

// The runtime-shape routes of a product no shape table names, such as a tensor-parallel rank's
// share of a projection: the shape tables' schedules with the rows and K read from the operands,
// writing BF16 or FP32 (a rank's partial sum).
template <class Output>
Output result(Tensor& out, std::int32_t rows);

template <>
LinearBf16Output result(Tensor& out, std::int32_t rows) {
    return {static_cast<__nv_bfloat16*>(out.data), rows};
}

template <>
LinearFp32Output result(Tensor& out, std::int32_t rows) {
    return {static_cast<float*>(out.data), rows};
}

template <class Schedule, class Output>
void gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_gemv<Schedule>(q4_linear_operands(x, w), result<Output>(out, w.n),
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_sliced_k_mma<Schedule>(q4_linear_operands(x, w), result<Output>(out, w.n),
                                         LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Output>
void mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q4_a16_mma<Schedule>(q4_linear_operands(x, w), result<Output>(out, w.n),
                                LinearIdentityEpilogue{}, stream);
}

template <class Output>
Q4Launch select(std::int32_t tokens) {
    using namespace q4_instances;
    if (tokens <= 1) return gemv<GemvR1W8, Output>;
    if (tokens <= 4) return sliced<SlicedR16T8Capacity4, Output>;
    if (tokens <= 8) return sliced<SlicedR16T8W4S2, Output>;
    if (tokens <= 16) return sliced<SlicedR32T16W4S2, Output>;
    if (tokens <= 64) return sliced<SlicedR32T32W4S2, Output>;
    return mma<MmaR64T128K64Wr64Wt32S2A2B1, Output>;
}

} // namespace

Q4Launch select_q4_general_launch(std::int32_t tokens, DType out) {
    return out == DType::FP32 ? select<LinearFp32Output>(tokens) : select<LinearBf16Output>(tokens);
}

} // namespace ninfer::ops::detail
