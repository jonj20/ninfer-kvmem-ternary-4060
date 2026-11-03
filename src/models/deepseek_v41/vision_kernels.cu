#include "models/deepseek_v41/vision_kernels.h"

#include <cuda_bf16.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::models::deepseek_v41::vision_kernels {
namespace {

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("deepseek_v41 vision ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

int blocks_for(std::int64_t n, int threads) { return int((n + threads - 1) / threads); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o)); }
    return v;
}

// Over the block (blockDim.x a multiple of 32, at most 1024); every thread gets the result.
template <bool Max>
__device__ float block_reduce(float v) {
    __shared__ float partial[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = Max ? warp_max(v) : warp_sum(v);
    __syncthreads();
    if (lane == 0) { partial[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        float s = lane < int(blockDim.x >> 5) ? partial[lane] : (Max ? -INFINITY : 0.0f);
        s       = Max ? warp_max(s) : warp_sum(s);
        if (lane == 0) { total = s; }
    }
    __syncthreads();
    return total;
}

// Block p, one float4 of the row per thread.
__global__ void residual_kernel(float* __restrict__ h, const float* __restrict__ x,
                                const float* __restrict__ bias, bool replace, int hidden,
                                const float* __restrict__ norm, float eps,
                                __nv_bfloat16* __restrict__ xn) {
    const std::int64_t row = std::int64_t(blockIdx.x) * hidden;
    const int i            = threadIdx.x * 4;
    float4 v = *reinterpret_cast<const float4*>(x + row + i);
    if (bias != nullptr) {
        const float4 b = *reinterpret_cast<const float4*>(bias + i);
        v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w;
    }
    if (!replace) {
        const float4 r = *reinterpret_cast<const float4*>(h + row + i);
        v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w;
    }
    *reinterpret_cast<float4*>(h + row + i) = v;
    const float sum   = block_reduce<false>(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    const float scale = rsqrtf(sum / float(hidden) + eps);
    const float4 w    = *reinterpret_cast<const float4*>(norm + i);
    xn[row + i]       = __float2bfloat16_rn(v.x * scale * w.x);
    xn[row + i + 1]   = __float2bfloat16_rn(v.y * scale * w.y);
    xn[row + i + 2]   = __float2bfloat16_rn(v.z * scale * w.z);
    xn[row + i + 3]   = __float2bfloat16_rn(v.w * scale * w.w);
}

// Block p; thread j over the row's 3 hidden / 2 pairs: section (q, k, v), head and pair i < D/2.
__global__ void rotate_qkv_kernel(const float* __restrict__ qkv, const float* __restrict__ bias,
                                  int grid_w, int hidden, int head_dim,
                                  const float* __restrict__ inv_freq,
                                  __nv_bfloat16* __restrict__ out) {
    const int p = blockIdx.x, half = head_dim / 2, quarter = head_dim / 4;
    const float row = float(p / grid_w), column = float(p % grid_w);
    const std::int64_t base = std::int64_t(p) * 3 * hidden;
    for (int j = threadIdx.x; j < 3 * hidden / 2; j += blockDim.x) {
        const int section = j / (hidden / 2);
        const int within  = j % (hidden / 2);
        const int head = within / half, i = within % half;
        const int a = section * hidden + head * head_dim + i, b = a + half;
        const float x1 = qkv[base + a] + bias[a], x2 = qkv[base + b] + bias[b];
        if (section == 2) {
            out[base + a] = __float2bfloat16_rn(x1);
            out[base + b] = __float2bfloat16_rn(x2);
            continue;
        }
        const float angle = (i < quarter ? row : column) * inv_freq[i % quarter];
        float s, c;
        sincosf(angle, &s, &c);
        out[base + a] = __float2bfloat16_rn(x1 * c - x2 * s);
        out[base + b] = __float2bfloat16_rn(x2 * c + x1 * s);
    }
}

// Block r.
__global__ void softmax_rows_kernel(const float* __restrict__ s, int keys,
                                    __nv_bfloat16* __restrict__ p) {
    const float* row = s + std::int64_t(blockIdx.x) * keys;
    __nv_bfloat16* out = p + std::int64_t(blockIdx.x) * keys;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < keys; j += blockDim.x) { m = fmaxf(m, row[j]); }
    m = block_reduce<true>(m);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < keys; j += blockDim.x) { sum += expf(row[j] - m); }
    const float inv = 1.0f / block_reduce<false>(sum);
    for (int j = threadIdx.x; j < keys; j += blockDim.x) {
        out[j] = __float2bfloat16_rn(expf(row[j] - m) * inv);
    }
}

__global__ void swiglu_kernel(const float* __restrict__ gu, std::int64_t n, int width,
                              __nv_bfloat16* __restrict__ mid) {
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) { return; }
    const std::int64_t p = k / width;
    const int i          = int(k % width);
    const float g        = gu[p * 2 * width + i];
    const float u        = gu[p * 2 * width + width + i];
    mid[k]               = __float2bfloat16_rn(g / (1.0f + expf(-g)) * u);
}

__global__ void unfold_kernel(const __nv_bfloat16* __restrict__ x, int grid_w, int grid_h,
                              int hidden, int r, int cells_w, std::int64_t n,
                              __nv_bfloat16* __restrict__ cells) {
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) { return; }
    const int features = hidden * r * r;
    const std::int64_t cell = k / features;
    const int f = int(k % features);
    const int c = f / (r * r), ky = f % (r * r) / r, kx = f % r;
    const int y = int(cell / cells_w) * r + ky, xx = int(cell % cells_w) * r + kx;
    cells[k] = y < grid_h && xx < grid_w ? x[(std::int64_t(y) * grid_w + xx) * hidden + c]
                                         : __float2bfloat16_rn(0.0f);
}

__global__ void gelu_bias_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                 std::int64_t n, int width, __nv_bfloat16* __restrict__ mid) {
    const std::int64_t k = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k >= n) { return; }
    const float v = x[k] + bias[k % width];
    // 1 + erf(v / sqrt(2)) as erfc(-v / sqrt(2)), which keeps the negative tail's small values
    // instead of cancelling them.
    mid[k] = __float2bfloat16_rn(0.5f * v * erfcf(v * -0.70710678118654752f));
}

// Block t: span row t.
__global__ void span_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                            const float* __restrict__ start, const float* __restrict__ newline,
                            const float* __restrict__ end, int rows, int columns, int width,
                            __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x, last = rows * (columns + 1) + 1;
    const float* src = nullptr;
    const float* add = nullptr;
    if (t == 0) {
        src = start;
    } else if (t == last) {
        src = end;
    } else {
        const int y = (t - 1) / (columns + 1), c = (t - 1) % (columns + 1);
        if (c == columns) {
            src = newline;
        } else {
            src = x + (std::int64_t(y) * columns + c) * width;
            add = bias;
        }
    }
    __nv_bfloat16* dst = out + std::int64_t(t) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        dst[i] = __float2bfloat16_rn(add != nullptr ? src[i] + add[i] : src[i]);
    }
}

} // namespace

void residual(float* h, const float* x, const float* bias, bool replace, int patches, int hidden,
              const float* norm, float eps, __nv_bfloat16* xn, cudaStream_t stream) {
    if (hidden % 128 != 0 || hidden > 4096) {
        throw std::invalid_argument("deepseek_v41 vision residual: hidden % 128 != 0 or > 4096");
    }
    if (patches <= 0) { return; }
    residual_kernel<<<patches, hidden / 4, 0, stream>>>(h, x, bias, replace, hidden, norm, eps, xn);
    check_launch("residual");
}

void rotate_qkv(const float* qkv, const float* bias, int patches, int grid_w, int hidden,
                int head_dim, const float* inv_freq, __nv_bfloat16* out, cudaStream_t stream) {
    if (head_dim % 4 != 0 || hidden % head_dim != 0) {
        throw std::invalid_argument("deepseek_v41 vision rotate_qkv: heads of whole rotated pairs");
    }
    if (patches <= 0) { return; }
    rotate_qkv_kernel<<<patches, 512, 0, stream>>>(qkv, bias, grid_w, hidden, head_dim, inv_freq,
                                                   out);
    check_launch("rotate_qkv");
}

void softmax_rows(const float* s, int rows, int keys, __nv_bfloat16* p, cudaStream_t stream) {
    if (rows <= 0 || keys <= 0) { return; }
    softmax_rows_kernel<<<rows, 256, 0, stream>>>(s, keys, p);
    check_launch("softmax_rows");
}

void swiglu(const float* gu, int patches, int width, __nv_bfloat16* mid, cudaStream_t stream) {
    const std::int64_t n = std::int64_t(patches) * width;
    if (n <= 0) { return; }
    swiglu_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(gu, n, width, mid);
    check_launch("swiglu");
}

void unfold(const __nv_bfloat16* x, int grid_w, int grid_h, int hidden, int r,
            __nv_bfloat16* cells, cudaStream_t stream) {
    const int cells_w = (grid_w + r - 1) / r, cells_h = (grid_h + r - 1) / r;
    const std::int64_t n = std::int64_t(cells_w) * cells_h * hidden * r * r;
    if (n <= 0) { return; }
    unfold_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, grid_w, grid_h, hidden, r, cells_w, n,
                                                          cells);
    check_launch("unfold");
}

void gelu_bias(const float* x, const float* bias, int rows, int width, __nv_bfloat16* mid,
               cudaStream_t stream) {
    const std::int64_t n = std::int64_t(rows) * width;
    if (n <= 0) { return; }
    gelu_bias_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(x, bias, n, width, mid);
    check_launch("gelu_bias");
}

void span(const float* x, const float* bias, const float* start, const float* newline,
          const float* end, int rows, int columns, int width, __nv_bfloat16* out,
          cudaStream_t stream) {
    if (rows <= 0 || columns <= 0) { return; }
    span_kernel<<<rows * (columns + 1) + 2, 256, 0, stream>>>(x, bias, start, newline, end, rows,
                                                              columns, width, out);
    check_launch("span");
}

} // namespace ninfer::models::deepseek_v41::vision_kernels
