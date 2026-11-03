#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/nvfp4/nvfp4_launch_unified.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using namespace unified;

// The [5120,17408] table's A16 schedules without its compile-time K: a tensor-parallel rank's
// share of a projection, whose rows and K come from the operands. Its GEMV and SIMT routes need a
// static K, so T = 1..8 take the first sliced-K schedule.
using Sliced8  = Nvfp4SlicedInstance<8, 4, 2>;
using Sliced16 = Nvfp4SlicedInstance<16, 4, 2>;
using Sliced32 = Nvfp4SlicedInstance<32, 4, 1>;
using Sliced64 = Nvfp4SlicedInstance<64, 2, 2>;
using Mma128   = Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>;
using MmaWide  = Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;

static_assert(kNvfp4ShareUnits.rows % Sliced8::kBlockRows == 0 &&
              kNvfp4ShareUnits.rows % Mma128::kBlockRows == 0 &&
              kNvfp4ShareUnits.rows % MmaWide::kBlockRows == 0);
static_assert(kNvfp4ShareUnits.columns % Sliced8::kBlockK == 0 &&
              kNvfp4ShareUnits.columns % Sliced64::kBlockK == 0 &&
              kNvfp4ShareUnits.columns % Mma128::kBlockK == 0 &&
              kNvfp4ShareUnits.columns % MmaWide::kBlockK == 0);

template <class Output>
void launch(const Tensor& x, const Weight& w, Output output, cudaStream_t stream) {
    const auto p     = nvfp4_a16_operands(x, w);
    const int tokens = x.ne[1];
    const LinearIdentityEpilogue epilogue{};
    if (tokens <= 8) return launch_nvfp4_a16_sliced_k_mma<Sliced8>(p, output, epilogue, stream);
    if (tokens <= 16) return launch_nvfp4_a16_sliced_k_mma<Sliced16>(p, output, epilogue, stream);
    if (tokens <= 32) return launch_nvfp4_a16_sliced_k_mma<Sliced32>(p, output, epilogue, stream);
    if (tokens <= 64) return launch_nvfp4_a16_sliced_k_mma<Sliced64>(p, output, epilogue, stream);
    if (tokens <= 128) return launch_nvfp4_a16_mma<Mma128>(p, output, epilogue, stream);
    launch_nvfp4_a16_mma<MmaWide>(p, output, epilogue, stream);
}

} // namespace

void nvfp4_general_a16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.n % kNvfp4ShareUnits.rows != 0 || w.k % kNvfp4ShareUnits.columns != 0) {
        throw std::invalid_argument("nvfp4 linear: a runtime shape needs N a multiple of 128 and "
                                    "K of 256");
    }
    if (out.dtype == DType::FP32) {
        launch(x, w, LinearFp32Output{static_cast<float*>(out.data), w.n}, stream);
    } else {
        launch(x, w, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, stream);
    }
}

} // namespace ninfer::ops::detail
