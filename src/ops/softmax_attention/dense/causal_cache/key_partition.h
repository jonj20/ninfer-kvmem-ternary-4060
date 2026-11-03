#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#    define NINFER_PARTITION_HD __host__ __device__
#else
#    define NINFER_PARTITION_HD
#endif

namespace ninfer::ops {

// How a split-KV attention launch divides each window's keys between its splits, and which keys
// the device holds. Both default to the tuned behaviour: splits sized by the kernel's tiers, every
// page on this device.
//
//   segment_keys > 0  invariant attention: split s covers keys [s * segment_keys,
//                     (s + 1) * segment_keys) whatever the window, width, batch or launch
//                     capacity. With one key tile and one reduction order per kernel family, a
//                     column's partials and output are then the same bytes in every launch that
//                     holds it. A multiple of the page size.
//   page_stride > 1   a rank of a position-split cache (--tensor-attention positions): its planes
//                     hold the physical pages p with p % page_stride == page_residue, at page
//                     p / page_stride, and its launches attend over those pages' keys only and
//                     append only the rows that land in them. Splits then cover whole pages.
struct CausalKeyPartition {
    std::int32_t segment_keys = 0;
    std::int32_t page_stride  = 1;
    std::int32_t page_residue = 0;

    [[nodiscard]] NINFER_PARTITION_HD constexpr bool segmented() const noexcept {
        return segment_keys > 0;
    }
    [[nodiscard]] NINFER_PARTITION_HD constexpr bool strided() const noexcept {
        return page_stride > 1;
    }
};

} // namespace ninfer::ops

#undef NINFER_PARTITION_HD
