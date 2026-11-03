// DeepSeek-V4.1's vision kernels against FP64 oracles, at the tower's widths (16 heads of 64,
// hidden 1024, SwiGLU 2816, aligner cells of 3 x 3 patches into the text width 5120):
//  - residual: the patch projection's output with its bias into the residual, a block's delta
//    added with and without a bias, each with the next projection's RMS-normed input;
//  - rotate_qkv: the 2D RoPE of Q and K by the patch's row and column (wide and tall grids, angles
//    past 70 radians), V with its bias only, allowing the error FP32 angles carry;
//  - softmax_rows over key counts within and past one block's threads;
//  - swiglu, gelu_bias (the erf GELU, its negative tail too);
//  - unfold of grids that divide into cells and grids padded with zero rows and columns;
//  - span: the learned start, newline and end rows around the aligner's rows plus their bias.
// A BF16 output is compared with the oracle's rounding of its exact value; the two may differ by
// a step where the FP32 arithmetic lands on the other side of a rounding boundary. Values that
// only move (unfold, the span's learned rows) must match bit for bit.

#include "models/deepseek_v41/vision_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace vk = ninfer::models::deepseek_v41::vision_kernels;
using bf16   = __nv_bfloat16;

constexpr int kHidden = 1024, kHeadDim = 64, kWidth = 2816, kRatio = 3, kText = 5120;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

int failures = 0;

void report(bool ok, const std::string& what) {
    std::cout << (ok ? "OK   " : "FAIL ") << what << '\n';
    failures += ok ? 0 : 1;
}

std::string num(double x) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.3g", x);
    return text;
}

template <class T>
struct Device {
    T* p          = nullptr;
    std::size_t n = 0;
    explicit Device(std::size_t count) : n(count) {
        check(cudaMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T)), "malloc");
    }
    explicit Device(const std::vector<T>& host) : Device(host.size()) {
        check(cudaMemcpy(p, host.data(), n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    [[nodiscard]] std::vector<T> host() const {
        check(cudaDeviceSynchronize(), "kernel");
        std::vector<T> out(n);
        check(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return out;
    }
};

std::vector<float> normal(std::size_t n, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> dist(0.0f, sigma);
    std::vector<float> v(n);
    for (auto& x : v) { x = dist(rng); }
    return v;
}

std::vector<bf16> to_bf16(const std::vector<float>& v) {
    std::vector<bf16> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = __float2bfloat16_rn(v[i]); }
    return out;
}

double value(bf16 x) { return __bfloat162float(x); }

std::uint16_t bits(bf16 x) {
    std::uint16_t b = 0;
    std::memcpy(&b, &x, 2);
    return b;
}

// BF16 outputs against the oracle's rounding of the exact values: how many differ, and the largest
// difference in steps of BF16 at the oracle's value. `slack` is an absolute error the FP32
// arithmetic before the rounding may carry: an output within half a step plus it of the exact value
// counts as equal.
struct Tally {
    std::size_t values = 0, off = 0;
    double worst = 0.0;

    void add(bf16 got, double exact, double slack = 0.0) {
        ++values;
        const double want = value(__float2bfloat16_rn(float(exact)));
        if (value(got) == want) { return; }
        const double step =
            std::ldexp(1.0, std::max(std::ilogb(std::max(std::fabs(want), 1e-37)), -126) - 7);
        if (slack > 0.0 && std::fabs(value(got) - exact) <= 0.5 * step + slack) { return; }
        ++off;
        const double steps = std::isfinite(value(got)) ? std::fabs(value(got) - want) / step : INFINITY;
        if (!(steps <= worst)) { worst = steps; }
    }

    [[nodiscard]] bool within(double fraction, double steps) const {
        return double(off) <= fraction * double(values) && worst <= steps;
    }

    [[nodiscard]] std::string str() const {
        return std::to_string(off) + "/" + std::to_string(values) + " off by " + num(worst) +
               " steps";
    }
};

void residual(int patches, bool replace, bool with_bias, std::mt19937& rng) {
    const float eps = 1e-6f;
    const auto h0   = normal(std::size_t(patches) * kHidden, 2.0f, rng);
    const auto x    = normal(h0.size(), 1.0f, rng);
    const auto bias = normal(kHidden, 0.5f, rng);
    auto norm       = normal(kHidden, 0.3f, rng);
    for (auto& v : norm) { v += 1.0f; }
    Device<float> dh(h0), dx(x), db(bias), dn(norm);
    Device<bf16> xn(h0.size());
    vk::residual(dh.p, dx.p, with_bias ? db.p : nullptr, replace, patches, kHidden, dn.p, eps, xn.p,
                 nullptr);
    const auto got_h = dh.host();
    const auto got_x = xn.host();
    double worst_h = 0.0;
    Tally tally;
    for (int p = 0; p < patches; ++p) {
        const std::size_t row = std::size_t(p) * kHidden;
        std::vector<double> want(kHidden);
        double sum = 0.0;
        for (int i = 0; i < kHidden; ++i) {
            const double b = with_bias ? bias[std::size_t(i)] : 0.0, h = replace ? 0.0 : h0[row + i];
            const double v = double(x[row + i]) + b + h;
            want[std::size_t(i)] = v;
            sum += v * v;
            // Two FP32 additions, each within half a unit of its sum's last place.
            const double bound = std::ldexp(std::fabs(x[row + i]) + std::fabs(b) + std::fabs(h), -23);
            worst_h = std::max(worst_h, std::fabs(got_h[row + i] - v) / std::max(bound, 1e-30));
        }
        const double scale = 1.0 / std::sqrt(sum / kHidden + eps);
        for (int i = 0; i < kHidden; ++i) {
            tally.add(got_x[row + i], want[std::size_t(i)] * scale * norm[std::size_t(i)]);
        }
    }
    report(worst_h <= 1.0 && tally.within(0.01, 1.0),
           std::string("residual ") + (replace ? "replace" : "add") + (with_bias ? " with bias" : "") +
               ": residual error " + num(worst_h) + " of its FP32 bound, normed input " +
               tally.str());
}

void rotate_qkv(int grid_w, int grid_h, std::mt19937& rng) {
    const int patches = grid_w * grid_h, quarter = kHeadDim / 4;
    const auto qkv  = normal(std::size_t(patches) * 3 * kHidden, 1.0f, rng);
    const auto bias = normal(3 * kHidden, 0.3f, rng);
    std::vector<float> freq(quarter);
    for (int i = 0; i < quarter; ++i) {
        freq[std::size_t(i)] = float(std::pow(10000.0, -2.0 * i / double(2 * quarter)));
    }
    Device<float> dq(qkv), db(bias), df(freq);
    Device<bf16> out(qkv.size());
    vk::rotate_qkv(dq.p, db.p, patches, grid_w, kHidden, kHeadDim, df.p, out.p, nullptr);
    const auto got = out.host();
    Tally tally;
    for (int p = 0; p < patches; ++p) {
        const double row = p / grid_w, column = p % grid_w;
        const std::size_t base = std::size_t(p) * 3 * kHidden;
        for (int section = 0; section < 3; ++section) {
            for (int head = 0; head < kHidden / kHeadDim; ++head) {
                for (int i = 0; i < kHeadDim / 2; ++i) {
                    const int a = section * kHidden + head * kHeadDim + i, b = a + kHeadDim / 2;
                    const double x1 = double(qkv[base + a]) + bias[std::size_t(a)];
                    const double x2 = double(qkv[base + b]) + bias[std::size_t(b)];
                    if (section == 2) {
                        tally.add(got[base + a], x1);
                        tally.add(got[base + b], x2);
                        continue;
                    }
                    const double angle = (i < quarter ? row : column) * freq[std::size_t(i % quarter)];
                    // The FP32 angle (a unit in its last place is up to 2^-17 radians here), sine,
                    // cosine and biased inputs: an error of a few of those units times the inputs.
                    const double slack =
                        std::ldexp((std::fabs(x1) + std::fabs(x2)) * (4.0 + angle), -22);
                    tally.add(got[base + a], x1 * std::cos(angle) - x2 * std::sin(angle), slack);
                    tally.add(got[base + b], x2 * std::cos(angle) + x1 * std::sin(angle), slack);
                }
            }
        }
    }
    report(tally.within(0.01, 1.0), "rotate_qkv " + std::to_string(grid_w) + " x " +
                                        std::to_string(grid_h) + ": " + tally.str());
}

void softmax_rows(int rows, int keys, std::mt19937& rng) {
    const auto s = normal(std::size_t(rows) * keys, 4.0f, rng);
    Device<float> ds(s);
    Device<bf16> p(s.size());
    vk::softmax_rows(ds.p, rows, keys, p.p, nullptr);
    const auto got = p.host();
    Tally tally;
    for (int r = 0; r < rows; ++r) {
        const float* row = s.data() + std::size_t(r) * keys;
        const double m   = *std::max_element(row, row + keys);
        double sum       = 0.0;
        for (int j = 0; j < keys; ++j) { sum += std::exp(row[j] - m); }
        for (int j = 0; j < keys; ++j) {
            tally.add(got[std::size_t(r) * keys + j], std::exp(row[j] - m) / sum);
        }
    }
    report(tally.within(0.01, 1.0), "softmax_rows " + std::to_string(rows) + " x " +
                                        std::to_string(keys) + ": " + tally.str());
}

void swiglu(int patches, std::mt19937& rng) {
    const auto gu = normal(std::size_t(patches) * 2 * kWidth, 2.0f, rng);
    Device<float> dg(gu);
    Device<bf16> mid(std::size_t(patches) * kWidth);
    vk::swiglu(dg.p, patches, kWidth, mid.p, nullptr);
    const auto got = mid.host();
    Tally tally;
    for (int p = 0; p < patches; ++p) {
        for (int i = 0; i < kWidth; ++i) {
            const double g = gu[std::size_t(p) * 2 * kWidth + i];
            const double u = gu[std::size_t(p) * 2 * kWidth + kWidth + i];
            tally.add(got[std::size_t(p) * kWidth + i], g / (1.0 + std::exp(-g)) * u);
        }
    }
    report(tally.within(0.01, 1.0), "swiglu " + std::to_string(patches) + " patches: " + tally.str());
}

void gelu_bias(int rows, std::mt19937& rng) {
    const auto x    = normal(std::size_t(rows) * kText, 2.0f, rng);
    const auto bias = normal(kText, 0.5f, rng);
    Device<float> dx(x), db(bias);
    Device<bf16> mid(x.size());
    vk::gelu_bias(dx.p, db.p, rows, kText, mid.p, nullptr);
    const auto got = mid.host();
    Tally tally;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double v = double(x[k]) + bias[k % kText];
        tally.add(got[k], 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0))));
    }
    report(tally.within(0.01, 1.0), "gelu_bias " + std::to_string(rows) + " rows: " + tally.str());
}

void unfold(int grid_w, int grid_h, int hidden, std::mt19937& rng) {
    const auto x = to_bf16(normal(std::size_t(grid_w) * grid_h * hidden, 1.0f, rng));
    const int cells_w = (grid_w + kRatio - 1) / kRatio, cells_h = (grid_h + kRatio - 1) / kRatio;
    const int features = hidden * kRatio * kRatio;
    Device<bf16> dx(x), cells(std::size_t(cells_w) * cells_h * features);
    vk::unfold(dx.p, grid_w, grid_h, hidden, kRatio, cells.p, nullptr);
    const auto got = cells.host();
    std::size_t wrong = 0;
    for (int cy = 0; cy < cells_h; ++cy) {
        for (int cx = 0; cx < cells_w; ++cx) {
            for (int c = 0; c < hidden; ++c) {
                for (int ky = 0; ky < kRatio; ++ky) {
                    for (int kx = 0; kx < kRatio; ++kx) {
                        const int y = cy * kRatio + ky, xx = cx * kRatio + kx;
                        const std::uint16_t want =
                            y < grid_h && xx < grid_w
                                ? bits(x[(std::size_t(y) * grid_w + xx) * hidden + c])
                                : 0;
                        const std::size_t at = (std::size_t(cy) * cells_w + cx) * features +
                                               std::size_t(c) * kRatio * kRatio + ky * kRatio + kx;
                        wrong += bits(got[at]) != want ? 1 : 0;
                    }
                }
            }
        }
    }
    report(wrong == 0, "unfold " + std::to_string(grid_w) + " x " + std::to_string(grid_h) + ": " +
                           std::to_string(wrong) + " values differ");
}

void span(int rows, int columns, std::mt19937& rng) {
    const auto x       = normal(std::size_t(rows) * columns * kText, 1.0f, rng);
    const auto bias    = normal(kText, 0.5f, rng);
    const auto start   = normal(kText, 1.0f, rng);
    const auto newline = normal(kText, 1.0f, rng);
    const auto end     = normal(kText, 1.0f, rng);
    const int tokens   = rows * (columns + 1) + 2;
    Device<float> dx(x), db(bias), ds(start), dn(newline), de(end);
    Device<bf16> out(std::size_t(tokens) * kText);
    vk::span(dx.p, db.p, ds.p, dn.p, de.p, rows, columns, kText, out.p, nullptr);
    const auto got = out.host();
    std::size_t learned = 0;
    Tally tally;
    const auto same = [&](int t, const std::vector<float>& row) {
        for (int i = 0; i < kText; ++i) {
            learned += bits(got[std::size_t(t) * kText + i]) != bits(__float2bfloat16_rn(row[std::size_t(i)]));
        }
    };
    same(0, start);
    same(tokens - 1, end);
    for (int y = 0; y < rows; ++y) {
        for (int c = 0; c < columns; ++c) {
            const int t = 1 + y * (columns + 1) + c;
            for (int i = 0; i < kText; ++i) {
                tally.add(got[std::size_t(t) * kText + i],
                          double(x[(std::size_t(y) * columns + c) * kText + i]) + bias[std::size_t(i)]);
            }
        }
        same(1 + y * (columns + 1) + columns, newline);
    }
    report(learned == 0 && tally.within(0.01, 1.0),
           "span " + std::to_string(rows) + " x " + std::to_string(columns) + ": learned rows " +
               std::to_string(learned) + " values differ, aligner rows " + tally.str());
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::mt19937 rng(4141);
        residual(3, true, true, rng);
        residual(5, false, true, rng);
        residual(2, false, false, rng);
        rotate_qkv(74, 3, rng);
        rotate_qkv(3, 71, rng);
        softmax_rows(4, 37, rng);
        softmax_rows(3, 4096, rng);
        swiglu(7, rng);
        gelu_bias(4, rng);
        unfold(9, 6, 64, rng);
        unfold(8, 5, 64, rng);
        unfold(1, 4, kHidden, rng);
        span(2, 3, rng);
        span(1, 1, rng);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " DeepSeek-V4.1 vision kernels\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "DeepSeek-V4.1 vision kernels: " << error.what() << '\n';
        return 1;
    }
}
