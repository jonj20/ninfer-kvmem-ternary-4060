#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_launch_unified.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using namespace unified;

// The [5120,17408] table's A16 schedules without its compile-time K: a tensor-parallel rank's
// share of a projection, whose rows and K come from the operands.
using Gemv     = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Sliced8  = Fp8SlicedInstance<8, 8, 2>;
using Sliced16 = Fp8SlicedInstance<16, 8, 2>;
using Sliced32 = Fp8SlicedInstance<32, 8, 1>;
using Sliced64 = Fp8SlicedInstance<64, 2, 2>;
using Mma128   = Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>;
using MmaWide  = Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;

static_assert(kFp8ShareUnits.rows % Gemv::kBlockRows == 0 &&
              kFp8ShareUnits.rows % Sliced8::kBlockRows == 0 &&
              kFp8ShareUnits.rows % Mma128::kBlockRows == 0 &&
              kFp8ShareUnits.rows % MmaWide::kBlockRows == 0);
static_assert(kFp8ShareUnits.columns % (32 * Gemv::kValuesPerLane) == 0 &&
              kFp8ShareUnits.columns % Sliced8::kBlockK == 0 &&
              kFp8ShareUnits.columns % Sliced64::kBlockK == 0 &&
              kFp8ShareUnits.columns % Mma128::kBlockK == 0 &&
              kFp8ShareUnits.columns % MmaWide::kBlockK == 0);

template <class Output>
void launch(const Tensor& x, const Weight& w, Output output, cudaStream_t stream) {
    const auto p     = fp8_a16_operands(x, w);
    const int tokens = x.ne[1];
    const LinearIdentityEpilogue epilogue{};
    if (tokens == 1) return launch_fp8_a16_gemv<Gemv>(p, output, epilogue, stream);
    if (tokens <= 8) return launch_fp8_a16_sliced_k_mma<Sliced8>(p, output, epilogue, stream);
    if (tokens <= 16) return launch_fp8_a16_sliced_k_mma<Sliced16>(p, output, epilogue, stream);
    if (tokens <= 32) return launch_fp8_a16_sliced_k_mma<Sliced32>(p, output, epilogue, stream);
    if (tokens <= 64) return launch_fp8_a16_sliced_k_mma<Sliced64>(p, output, epilogue, stream);
    if (tokens <= 128) return launch_fp8_a16_mma<Mma128>(p, output, epilogue, stream);
    launch_fp8_a16_mma<MmaWide>(p, output, epilogue, stream);
}

} // namespace

void fp8_general_a16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.n % kFp8ShareUnits.rows != 0 || w.k % kFp8ShareUnits.columns != 0) {
        throw std::invalid_argument("fp8 linear: a runtime shape needs N a multiple of 64 and K "
                                    "of 512");
    }
    if (out.dtype == DType::FP32) {
        launch(x, w, LinearFp32Output{static_cast<float*>(out.data), w.n}, stream);
    } else {
        launch(x, w, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, stream);
    }
}

} // namespace ninfer::ops::detail
