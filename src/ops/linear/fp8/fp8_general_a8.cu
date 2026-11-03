#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/common/blackwell_routes.h"
#include "ops/linear/common/wave_tiles.h"
#include "ops/linear/fp8/fp8_launch_unified.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using namespace unified;

// The registered tables' A8 TMA schedules without their compile-time K: a tensor-parallel rank's
// share of a projection, whose rows and K come from the operands. A share's rows are whole 64-row
// units; rows in whole 128-row tiles also take the 128-row tiles.
using Tma32x64   = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
using Tma64x64   = Fp8A8TmaMmaSchedule<64, 64, 128, 2, 2, 3, 2>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
using Tma128x64  = Fp8A8TmaMmaSchedule<128, 64, 128, 2, 2, 3, 1>;
using Tma128x128 = Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>;

static_assert(kFp8ShareUnits.rows % Tma32x64::kBlockRows == 0 &&
              kFp8ShareUnits.rows % Tma128x64::kBlockRows == 0);
static_assert(kFp8ShareUnits.columns % Tma32x64::kBlockK == 0);

// A share takes the tile whose CTAs fill the fewest whole waves of the card's multiprocessors, each
// wave weighted by what one costs that tile (common/wave_tiles.h). The weights are the RTX 5090's
// per-wave times in tenths of Tma128x128's, fitted over the shares of Qwen3.8-27B's projections at
// two and four ranks (bench/ops/linear_bench.cu, 17 to 2048 tokens): the tile they pick runs within
// one 2 us timer tick of the fastest of the five at every width measured.
enum class Tile : std::uint8_t { T32x64, T64x64, T64x128, T128x64, T128x128 };
constexpr std::array kTiles{
    WaveTile{Tma32x64::kBlockTokens, Tma32x64::kBlockRows, 5},
    WaveTile{Tma64x64::kBlockTokens, Tma64x64::kBlockRows, 6},
    WaveTile{Tma64x128::kBlockTokens, Tma64x128::kBlockRows, 7},
    WaveTile{Tma128x64::kBlockTokens, Tma128x64::kBlockRows, 8},
    WaveTile{Tma128x128::kBlockTokens, Tma128x128::kBlockRows, 10},
};

template <class Schedule, class Output>
void launch_schedule(const Weight& w, Output output, Fp8A8Workspace scratch, std::int32_t tokens,
                     cudaStream_t stream) {
    launch_fp8_a8_tma_mma<Schedule>(fp8_a8_operands(w, scratch, tokens), output,
                                    LinearIdentityEpilogue{}, stream);
}

template <class Output>
void launch(const Weight& w, Output output, Fp8A8Workspace scratch, std::int32_t tokens,
            cudaStream_t stream) {
    switch (static_cast<Tile>(cheapest_wave_tile(kTiles, w.n, tokens))) {
    case Tile::T32x64:
        return launch_schedule<Tma32x64>(w, output, scratch, tokens, stream);
    case Tile::T64x64:
        return launch_schedule<Tma64x64>(w, output, scratch, tokens, stream);
    case Tile::T64x128:
        return launch_schedule<Tma64x128>(w, output, scratch, tokens, stream);
    case Tile::T128x64:
        return launch_schedule<Tma128x64>(w, output, scratch, tokens, stream);
    case Tile::T128x128:
        return launch_schedule<Tma128x128>(w, output, scratch, tokens, stream);
    }
}

} // namespace

void fp8_general_a8(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                    cudaStream_t stream) {
    require_fp8_a8_card();
    if (w.n % kFp8ShareUnits.rows != 0 || w.k % kFp8ShareUnits.columns != 0) {
        throw std::invalid_argument("fp8 A8 linear: a runtime shape needs N a multiple of 64 and "
                                    "K of 512");
    }
    const std::int32_t tokens = x.ne[1];
    auto scope                = workspace.scope();
    const auto scratch        = allocate_fp8_a8_workspace(workspace, tokens, w.k);
    launch_fp8_a8_quantize(x, w, scratch, stream);
    if (out.dtype == DType::FP32) {
        launch(w, LinearFp32Output{static_cast<float*>(out.data), w.n}, scratch, tokens, stream);
    } else {
        launch(w, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, scratch, tokens,
               stream);
    }
}

} // namespace ninfer::ops::detail
