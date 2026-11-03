#pragma once

#include "core/weight_view.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ninfer::artifact {

// A piece of a stored matrix that a device object holds on its own: the rows of every range in
// `rows`, in order, and, of each, the columns of every range in `columns`, side by side. A vector
// is one row. Several row ranges take, for instance, one part of each expert of a bank. A tensor-parallel
// rank holds such pieces of the projections it computes (models/qwen3_5/load/text.cpp).
//
// Every stored layout but the device-only row-split panel cuts this way: a piece of a row is a run
// of each plane's bytes for that row (codes, high bits, group scales), and a per-row record such as
// an FP8 row scale goes along whole, so the piece is a gather of byte runs. Column bounds fall on
// whole stored runs: one column of a contiguous matrix or an FP8 row, a ggml block, 128 columns of
// a row-split matrix (its rows are padded to 128), 64 columns of NVFP4. NVFP4 row bounds fall on
// its 128-row scale tiles and on its divisors' rows.
struct ObjectSelection {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> rows;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> columns;
    // INT32 vectors only: subtracted from every selected value at upload, so a piece of an index
    // vector can index a piece of what it indexes.
    std::int32_t rebase = 0;

    bool operator==(const ObjectSelection&) const = default;
};

// One run of a selection: `bytes` from `source` (an offset in the stored object) to `destination`
// (an offset in the selected object).
struct SelectedRun {
    std::uint64_t source      = 0;
    std::uint64_t bytes       = 0;
    std::uint64_t destination = 0;
};

// The columns and rows a piece's bounds fall on (an end may also be the object's last column or
// row). Throws ArtifactError for a layout no byte gather cuts.
[[nodiscard]] std::uint64_t selection_column_unit(const WeightGeometry& stored);
[[nodiscard]] std::uint64_t selection_row_unit(const WeightGeometry& stored);

// The geometry the selected piece takes. Throws ArtifactError for a selection outside the object,
// an empty or unordered range, a layout no byte gather can cut, or a bound inside a stored run.
[[nodiscard]] WeightGeometry selected_geometry(const WeightGeometry& stored,
                                               const ObjectSelection& selection);

// The byte runs that copy the piece out of the stored object, in destination order. Whole rows that
// follow each other are one run.
[[nodiscard]] std::vector<SelectedRun> selected_runs(const WeightGeometry& stored,
                                                     const ObjectSelection& selection);

} // namespace ninfer::artifact
