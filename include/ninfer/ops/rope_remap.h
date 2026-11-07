#pragma once
// Route A (kvmem-port.md §2.4): after prefill has cached every prompt key with absolute RoPE, the
// retained keys are re-rotated in place from their absolute positions to their compact window rows,
// so that query and key share the compressed retrieval frame. The kernel is launched from the host
// with the list of (physical page, delta position in tokens) pairs; delta is constant within a page
// because pages are repacked contiguously without reordering tokens inside a page.
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

void rope_remap_text_kv(void* k_data, int head_dim, int kv_heads,
                        const int* physical_page_list, const float* delta_position_tokens,
                        int count, int rotary_dim, float theta, int physical_pages,
                        cudaStream_t stream);

} // namespace ninfer::ops
