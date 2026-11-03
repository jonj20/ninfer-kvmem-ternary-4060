#pragma once

// The DeepSeek-V4.1 vision tower's own kernels. Activations are patch-major: an FP32 residual
// h[p][i], BF16 projection inputs x[p][k] and FP32 projection outputs, the patches of one image
// row by row of its grid_w x grid_h grid.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::deepseek_v41::vision_kernels {

// h[p] = x[p] + bias (replace), or h[p] += x[p] (+ bias when given); then the next projection's
// input xn[p] = bf16(h[p] * rsqrt(mean h[p]^2 + eps) * norm). hidden % 128 == 0, at most 4096.
void residual(float* h, const float* x, const float* bias, bool replace, int patches, int hidden,
              const float* norm, float eps, __nv_bfloat16* xn, cudaStream_t stream);

// The attention's inputs from its projection qkv [P][3 hidden] and the projection's bias: q, k and
// v BF16 [P][3 hidden] in the same places, q and k of each head (head_dim D values) turned by the
// 2D RoPE: pair (i, i + D/2) by the angle row * inv_freq[i] for i < D/4 and column *
// inv_freq[i - D/4] after, (row, column) the patch's place in the grid (inv_freq: [D/4]).
void rotate_qkv(const float* qkv, const float* bias, int patches, int grid_w, int hidden,
                int head_dim, const float* inv_freq, __nv_bfloat16* out, cudaStream_t stream);

// p[r][j] = bf16(softmax_j(s[r][j])) over `rows` rows of `keys` FP32 scores.
void softmax_rows(const float* s, int rows, int keys, __nv_bfloat16* p, cudaStream_t stream);

// mid[p][i] = bf16(silu(gu[p][i]) * gu[p][width + i]) (gate, then up).
void swiglu(const float* gu, int patches, int width, __nv_bfloat16* mid, cudaStream_t stream);

// The aligner's input from the tower's output x [grid_h * grid_w][hidden]: one row of
// hidden * r * r values per cell of r x r patches (ceil(grid_h / r) x ceil(grid_w / r) cells, row by
// row), value c * r * r + ky * r + kx of cell (cy, cx) being x at patch (cy r + ky, cx r + kx),
// channel c, or zero past the grid (F.unfold over the zero-padded grid).
void unfold(const __nv_bfloat16* x, int grid_w, int grid_h, int hidden, int r,
            __nv_bfloat16* cells, cudaStream_t stream);

// mid[r][i] = bf16(gelu(x[r][i] + bias[i])) with the erf GELU.
void gelu_bias(const float* x, const float* bias, int rows, int width, __nv_bfloat16* mid,
               cudaStream_t stream);

// An image's span, BF16 [rows (columns + 1) + 2][width]: the start row, then per row of the
// aligner's grid its `columns` rows x + bias and the newline row, then the end row.
void span(const float* x, const float* bias, const float* start, const float* newline,
          const float* end, int rows, int columns, int width, __nv_bfloat16* out,
          cudaStream_t stream);

} // namespace ninfer::models::deepseek_v41::vision_kernels
