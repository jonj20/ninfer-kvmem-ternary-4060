#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

// One device's part of SparseMoe under expert parallelism, as the kernels read it: its routed banks
// hold experts [begin, begin + count), so a selected expert e is that bank's expert e - begin and
// the other experts' paths contribute nothing; the shared expert runs where `shared`; and where
// `partial` is set the device writes the FP32 sum [2048, T] of its paths there instead of adding
// the result to the residual. `intermediate` is the width of the intermediate columns every
// expert it holds keeps, routed and shared: 512, or 256 when the devices divide each expert's
// intermediate columns (each expert's gate rows, then its up rows, then down's columns, of that
// width). The default is the whole Op.
struct SparseMoeShareArgs {
    std::int32_t begin        = 0;
    std::int32_t count        = 256;
    bool shared               = true;
    float* partial            = nullptr;
    std::int32_t intermediate = 512;
};

} // namespace ninfer::ops::detail
