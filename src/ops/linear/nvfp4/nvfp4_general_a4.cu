#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/common/blackwell_routes.h"
#include "ops/linear/common/wave_tiles.h"
#include "ops/linear/nvfp4/nvfp4_launch_unified.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using namespace unified;

// The registered tables' A4 schedules without their compile-time K: a tensor-parallel rank's share
// of a projection, whose rows and K come from the operands. Every schedule takes whole 128-row
// tiles and 256-column K tiles, the share units, and the MMA schedules two K tiles at most. The
// TMA route (nvfp4_a4_tma.cu) runs 128-row tiles of 128 or 256 tokens, the token tile set by the
// activation scales' layout.
using T32R64  = Nvfp4A4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using T32R128 = Nvfp4A4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using T64R128 = Nvfp4A4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;

static_assert(kNvfp4ShareUnits.rows % T32R64::kBlockRows == 0 &&
              kNvfp4ShareUnits.rows % T32R128::kBlockRows == 0);
static_assert(kNvfp4ShareUnits.columns % T32R128::kBlockK == 0 &&
              kNvfp4GeneralA4MinK / T32R128::kBlockK >= T32R128::kStages);

// A share takes the tile whose CTAs fill the fewest whole waves of the card's multiprocessors, each
// wave weighted by what one costs that tile (common/wave_tiles.h). The weights are the RTX 5090's
// per-wave times in tenths of T64R128's, fitted over the shares of Qwen3.8-27B's projections at two
// and four ranks (bench/ops/linear_bench.cu, 8 to 2048 tokens): the tile they pick runs within
// 0.2% of the fastest of the five at every width measured.
enum class Tile : std::uint8_t { T32R64, T32R128, T64R128, Tma128, Tma256 };
constexpr std::array kTiles{
    WaveTile{T32R64::kBlockTokens, T32R64::kBlockRows, 6},
    WaveTile{T32R128::kBlockTokens, T32R128::kBlockRows, 8},
    WaveTile{T64R128::kBlockTokens, T64R128::kBlockRows, 10},
    WaveTile{128, 128, 10},
    WaveTile{256, 128, 15},
};

template <class Schedule, class Output>
void launch_schedule(const Weight& w, Output output, Nvfp4A4Workspace scratch, std::int32_t tokens,
                     cudaStream_t stream) {
    launch_nvfp4_a4_mma<Schedule>(nvfp4_a4_operands(w, scratch, tokens, Nvfp4ScaleLayout::RowMajor),
                                  output, LinearIdentityEpilogue{}, stream, {},
                                  pdl::Dependency::Programmatic);
}

template <class Output>
void launch_mma(Tile tile, const Weight& w, Output output, Nvfp4A4Workspace scratch,
                std::int32_t tokens, cudaStream_t stream) {
    if (tile == Tile::T32R64) return launch_schedule<T32R64>(w, output, scratch, tokens, stream);
    if (tile == Tile::T32R128) return launch_schedule<T32R128>(w, output, scratch, tokens, stream);
    launch_schedule<T64R128>(w, output, scratch, tokens, stream);
}

} // namespace

void nvfp4_general_a4(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    require_nvfp4_a4_card();
    if (w.n % kNvfp4ShareUnits.rows != 0 || w.k % kNvfp4ShareUnits.columns != 0 ||
        w.k < kNvfp4GeneralA4MinK) {
        throw std::invalid_argument("nvfp4 A4 linear: a runtime shape needs N a multiple of 128 "
                                    "and K of 256, from 512");
    }
    const std::int32_t tokens = x.ne[1];
    auto scope                = workspace.scope();
    const auto scratch        = allocate_nvfp4_a4_workspace(workspace, tokens, w.k);
    const auto tile           = static_cast<Tile>(cheapest_wave_tile(kTiles, w.n, tokens));
    // Every tile quantizes by division, as one device's TMA routes do, so a share's codes do not
    // depend on the tile it runs on. One device's MMA routes multiply by a reciprocal instead,
    // which can move a value at an exact tie by one code. The MMA tiles read row-major scales; the
    // TMA tiles read whole tiles of 128 or 256 tokens.
    if (tile == Tile::Tma128 || tile == Tile::Tma256) {
        const Nvfp4ScaleLayout layout =
            tile == Tile::Tma128 ? Nvfp4ScaleLayout::Tiled128 : Nvfp4ScaleLayout::Tiled256;
        launch_nvfp4_a4_quantize(x, w, scratch, layout, stream);
        launch_nvfp4_a4_tma_general(nvfp4_a4_operands(w, scratch, tokens, layout), out, stream);
        return;
    }
    launch_nvfp4_a4_quantize(x, w, scratch, Nvfp4ScaleLayout::RowMajor, stream);
    if (out.dtype == DType::FP32) {
        launch_mma(tile, w, LinearFp32Output{static_cast<float*>(out.data), w.n}, scratch, tokens,
                   stream);
    } else {
        launch_mma(tile, w, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n}, scratch,
                   tokens, stream);
    }
}

} // namespace ninfer::ops::detail
