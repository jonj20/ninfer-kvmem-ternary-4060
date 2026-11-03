#include "ops/linear/ptq1/ptq1_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

bool ptq1_a8_shape_supported(std::int32_t n, std::int32_t k) {
    return n > 0 && n % 128 == 0 && (k == 5120 || k == 6144 || k == 17408);
}

void validate_ptq1_shape(std::int32_t n, std::int32_t k, std::int32_t tokens, LinearPolicy policy) {
    // Runtime-sized tensor shares retain complete packed groups and the
    // small-T kernel's row/slab units. Do not limit them to whole-model shapes.
    const bool shape = n > 0 && n % 64 == 0 && k > 0 && k % 512 == 0;
    if (!shape || tokens <= 0) {
        throw std::invalid_argument("PTQ1 linear: unsupported shape or token count");
    }
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA8Int:
    case LinearPolicy::AllowA8IntDecode:
    case LinearPolicy::AllowPrefillCublas:
        return;
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("PTQ1 linear: unsupported policy");
}

void ptq1_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    // Workspace planning and execution share this contract without consulting T2 routes.
    validate_ptq1_shape(w.n, w.k, x.ne[1], policy);
    if (w.qtype != QType::PTQ1_G128_FP16 || w.layout != QuantLayout::RowSplit ||
        w.qdata == nullptr || w.scales == nullptr || w.qhigh == nullptr ||
        w.high_plane_bytes == 0 || w.group != 128 || w.group_size != 128 ||
        w.scale_dtype != DType::FP16 || w.padded_shape[1] != w.k) {
        throw std::invalid_argument("PTQ1 linear: unsupported shape, policy or planes");
    }
    if (x.ne[1] <= 16) {
        launch_t2_small_t_v2_ptq1(x, w, out, stream);
    } else {
        launch_t2_ptq1_wide_t(x, w, out, out.ne[0], stream);
    }
}

} // namespace ninfer::ops::detail
