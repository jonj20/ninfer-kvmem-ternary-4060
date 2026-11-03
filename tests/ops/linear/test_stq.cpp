// STQ products (ops/linear/stq) against an FP64 oracle over independently decoded weights: the
// trellis read bit by bit from its u32 pairs, each slot's tail-biting window, the codebook's
// binary16 arithmetic in software and the MMA fragment order of a tile, then
// y = svh * H(W^T H(suh * x)) in double. Random words and signs at the shapes of the DeepSeek-V4.1
// drafter's matrices: one item of up to eight rows (FP32 and BF16 outputs, products split over
// input tiles), the grouped output projection's groups as items, routed experts as one-row items
// addressed through an id list. The codebooks' known vectors come from sushi's own tests.
#include "ops/linear/stq/stq_linear.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer::test;
using ninfer::ops::stq::Codebook;

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000U) << 16;
    const int exponent       = (h >> 10) & 0x1F;
    const std::uint32_t man  = h & 0x3FFU;
    if (exponent == 0) {
        const float value = std::ldexp(float(man), -24);
        return sign ? -value : value;
    }
    if (exponent == 31) {
        return std::bit_cast<float>(sign | 0x7F800000U | (man << 13));
    }
    return std::bit_cast<float>(sign | (std::uint32_t(exponent + 112) << 23) | (man << 13));
}

// Round to the nearest binary16, ties to even (finite inputs in the binary16 range).
std::uint16_t float_to_half(float f) {
    const double value = std::fabs(double(f));
    const std::uint16_t sign = std::signbit(f) ? 0x8000U : 0U;
    if (value == 0.0) { return sign; }
    int exponent = std::ilogb(value);
    if (exponent < -14) { exponent = -14; }
    const double quantum = std::ldexp(1.0, exponent - 10);
    double units         = std::nearbyint(value / quantum); // default rounding: ties to even
    if (units >= 2048.0) {
        units /= 2;
        ++exponent;
    }
    const auto u = std::uint32_t(units);
    if (u < 1024U) { return std::uint16_t(sign | u); } // subnormal
    return std::uint16_t(sign | (std::uint32_t(exponent + 15) << 10) | (u - 1024U));
}

std::uint16_t half_sum_scaled(std::uint32_t halves) {
    const std::uint16_t s = float_to_half(half_to_float(std::uint16_t(halves & 0xFFFFU)) +
                                          half_to_float(std::uint16_t(halves >> 16)));
    return float_to_half(half_to_float(s) * 4096.0F);
}

std::uint32_t rotl(std::uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

// The 256 weights of one tile, [input row][output column].
std::vector<float> decode_tile(const std::uint16_t* words, Codebook codebook) {
    const int n     = ninfer::ops::stq::tile_words(codebook);
    const int total = 16 * n;
    std::vector<int> bits(static_cast<std::size_t>(total));
    for (int at = 0; at < total; ++at) {
        const std::uint32_t pair = std::uint32_t(words[2 * (at / 32)]) |
                                   (std::uint32_t(words[2 * (at / 32) + 1]) << 16);
        bits[std::size_t(at)] = int((pair >> (31 - at % 32)) & 1U);
    }
    std::vector<std::uint32_t> windows(256);
    for (int slot = 0; slot < 256; ++slot) {
        const int end      = ((slot + 1) * n) >> 4;
        std::uint32_t word = 0;
        for (int b = 0; b < 16; ++b) {
            word = (word << 1) | std::uint32_t(bits[std::size_t(((end - 16 + b) % total + total) % total)]);
        }
        windows[std::size_t(slot)] = word;
    }
    std::vector<std::uint16_t> slots(256);
    if (codebook == Codebook::Spec1W15N64) {
        for (int slot = 0; slot < 256; ++slot) {
            slots[std::size_t(slot)] =
                half_sum_scaled(((windows[std::size_t(slot)] & 0x7FFFU) * 0x3FED4CEFU) & 0x8FF08FF0U);
        }
    } else {
        for (int m = 0; m < 128; ++m) {
            const std::uint32_t p     = windows[std::size_t(2 * m + 1)] * 0x3B22BE15U;
            slots[std::size_t(2 * m)]     = half_sum_scaled(p & 0x8ED78C13U);
            slots[std::size_t(2 * m + 1)] = half_sum_scaled(rotl(p, 3) & 0x8ECE8C03U);
        }
    }
    std::vector<float> tile(256);
    for (int lane = 0; lane < 32; ++lane) {
        const int r0 = (lane % 4) * 2, c0 = lane / 4;
        const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
        for (int j = 0; j < 8; ++j) {
            const int row = rows[j % 4], column = c0 + 8 * (j / 4);
            tile[std::size_t(row * 16 + column)] = half_to_float(slots[std::size_t(8 * lane + j)]);
        }
    }
    return tile;
}

// The orthonormal Walsh-Hadamard transform of each 128 consecutive values, in place.
void transform(std::vector<double>& v) {
    for (std::size_t base = 0; base < v.size(); base += 128) {
        for (int h = 1; h < 128; h <<= 1) {
            for (int i = 0; i < 128; i += 2 * h) {
                for (int j = i; j < i + h; ++j) {
                    const double a = v[base + j], b = v[base + j + h];
                    v[base + j]     = a + b;
                    v[base + j + h] = a - b;
                }
            }
        }
        for (int i = 0; i < 128; ++i) { v[base + i] /= std::sqrt(128.0); }
    }
}

struct MatrixSet {
    std::vector<std::uint16_t> trellis, suh, svh;
    int k = 0, n = 0, count = 0;
    Codebook codebook{};
    std::int64_t trellis_stride() const {
        return std::int64_t(k / 16) * (n / 16) * ninfer::ops::stq::tile_words(codebook);
    }
};

MatrixSet random_set(std::mt19937& rng, int k, int n, int count, Codebook codebook) {
    MatrixSet s;
    s.k = k;
    s.n = n;
    s.count    = count;
    s.codebook = codebook;
    s.trellis.resize(std::size_t(s.trellis_stride()) * count);
    for (auto& word : s.trellis) { word = std::uint16_t(rng()); }
    std::uniform_real_distribution<float> sign(0.25F, 1.0F);
    for (auto* v : {&s.suh, &s.svh}) {
        v->resize(std::size_t(v == &s.suh ? k : n) * count);
        for (auto& value : *v) { value = float_to_half((rng() & 1 ? -1.0F : 1.0F) * sign(rng)); }
    }
    return s;
}

// The FP64 product of matrix m with one row x (k floats).
std::vector<double> oracle(const MatrixSet& s, int m, const std::vector<float>& x) {
    const int k = s.k, n = s.n, words = ninfer::ops::stq::tile_words(s.codebook);
    std::vector<double> u(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        u[std::size_t(i)] = double(x[std::size_t(i)]) * half_to_float(s.suh[std::size_t(m) * k + i]);
    }
    transform(u);
    std::vector<double> z(std::size_t(n), 0.0);
    const std::uint16_t* trellis = s.trellis.data() + std::int64_t(m) * s.trellis_stride();
    for (int ti = 0; ti < k / 16; ++ti) {
        for (int to = 0; to < n / 16; ++to) {
            const auto tile = decode_tile(trellis + (std::int64_t(ti) * (n / 16) + to) * words, s.codebook);
            for (int r = 0; r < 16; ++r) {
                const double a = u[std::size_t(ti * 16 + r)];
                for (int c = 0; c < 16; ++c) { z[std::size_t(to * 16 + c)] += a * tile[std::size_t(r * 16 + c)]; }
            }
        }
    }
    transform(z);
    for (int o = 0; o < n; ++o) { z[std::size_t(o)] *= half_to_float(s.svh[std::size_t(m) * n + o]); }
    return z;
}

std::uint16_t to_bf16(float f) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
    return std::uint16_t((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}
float from_bf16(std::uint16_t h) { return std::bit_cast<float>(std::uint32_t(h) << 16); }

template <typename T>
T* upload(const std::vector<T>& host) {
    T* device = nullptr;
    if (cudaMalloc(&device, host.size() * sizeof(T)) != cudaSuccess) { throw std::runtime_error("cudaMalloc"); }
    cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device;
}

// items x rows products through the set, item i reading matrix ids[i] (i without ids) and input row
// group i / divisor; compares every output with the oracle.
int run_case(const std::string& label, std::mt19937& rng, const MatrixSet& s, int items, int rows,
             std::vector<std::int32_t> ids, int divisor, bool bf16_out) {
    const int k = s.k, n = s.n, groups = (items + divisor - 1) / divisor;
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> x(std::size_t(groups) * rows * k);
    for (auto& value : x) { value = to_bf16(normal(rng)); }
    auto* d_trellis = upload(s.trellis);
    auto* d_suh     = upload(s.suh);
    auto* d_svh     = upload(s.svh);
    auto* d_x       = upload(x);
    std::int32_t* d_ids = ids.empty() ? nullptr : upload(ids);
    const std::size_t outputs = std::size_t(items) * rows * n;
    void* d_out = nullptr;
    cudaMalloc(&d_out, outputs * (bf16_out ? 2 : 4));
    ninfer::ops::stq::Product p;
    p.m.trellis        = d_trellis;
    p.m.suh            = reinterpret_cast<const __half*>(d_suh);
    p.m.svh            = reinterpret_cast<const __half*>(d_svh);
    p.m.trellis_stride = s.trellis_stride();
    p.m.suh_stride     = k;
    p.m.svh_stride     = n;
    p.m.k              = k;
    p.m.n              = n;
    p.m.codebook       = s.codebook;
    p.ids              = d_ids;
    p.items            = items;
    p.rows             = rows;
    p.x                = reinterpret_cast<const __nv_bfloat16*>(d_x);
    p.x_item_stride    = std::int64_t(rows) * k;
    p.x_row_stride     = k;
    p.x_divisor        = divisor;
    if (bf16_out) {
        p.out_bf16 = static_cast<__nv_bfloat16*>(d_out);
    } else {
        p.out = static_cast<float*>(d_out);
    }
    p.out_item_stride = std::int64_t(rows) * n;
    p.out_row_stride  = n;
    int device = 0, sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    void* workspace = nullptr;
    cudaMalloc(&workspace, ninfer::ops::stq::workspace_bytes(p, sms));
    ninfer::ops::stq::product(p, workspace, sms, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error(label + ": kernel failed"); }
    std::vector<float> got(outputs);
    if (bf16_out) {
        std::vector<std::uint16_t> words(outputs);
        cudaMemcpy(words.data(), d_out, outputs * 2, cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < outputs; ++i) { got[i] = from_bf16(words[i]); }
    } else {
        cudaMemcpy(got.data(), d_out, outputs * 4, cudaMemcpyDeviceToHost);
    }
    for (void* ptr : {static_cast<void*>(d_trellis), static_cast<void*>(d_suh), static_cast<void*>(d_svh),
                      static_cast<void*>(d_x), static_cast<void*>(d_ids), d_out, workspace}) {
        cudaFree(ptr);
    }
    double worst = 0.0, sum_squares = 0.0;
    std::size_t count = 0;
    std::vector<std::vector<double>> want(std::size_t(items) * rows);
    for (int i = 0; i < items; ++i) {
        const int m = ids.empty() ? i : ids[std::size_t(i)];
        for (int r = 0; r < rows; ++r) {
            std::vector<float> row(static_cast<std::size_t>(k));
            for (int j = 0; j < k; ++j) {
                row[std::size_t(j)] = from_bf16(x[(std::size_t(i / divisor) * rows + r) * k + j]);
            }
            want[std::size_t(i) * rows + r] = oracle(s, m, row);
            for (const double v : want[std::size_t(i) * rows + r]) {
                sum_squares += v * v;
                ++count;
            }
        }
    }
    const double rms = std::sqrt(sum_squares / double(count));
    for (std::size_t q = 0; q < want.size(); ++q) {
        for (int o = 0; o < n; ++o) {
            const double ref   = want[q][std::size_t(o)];
            const double error = std::fabs(double(got[q * n + o]) - ref);
            // FP32 sums of k products; a BF16 output adds its own rounding of the value.
            const double bound = 2e-5 * rms + (bf16_out ? std::ldexp(std::fabs(ref), -8) : 0.0);
            worst = std::max(worst, error / bound);
        }
    }
    std::cout << label << ": rms " << rms << ", worst error " << worst << " of its bound\n";
    return worst <= 1.0 ? 0 : 1;
}

} // namespace

int main() {
    try {
        int failures = 0;
        // sushi's decodePair vector (PAIR1, 32 words).
        const std::uint32_t pair = 0x1234U * 0x3B22BE15U;
        if (half_sum_scaled(pair & 0x8ED78C13U) != 0x3822 ||
            half_sum_scaled(rotl(pair, 3) & 0x8ECE8C03U) != 0xB086) {
            std::cerr << "the host decoder misses sushi's PAIR1 vector\n";
            ++failures;
        }
        if (cuda_unavailable()) { return failures ? 1 : 77; }
        std::mt19937 rng(20261027);
        const auto spec1 = Codebook::Spec1W15N64, pair1 = Codebook::Pair1W16N32;
        // The drafter's trunk: kv (split over input tiles), the taps' projection at a seed chunk
        // of eight rows and at one, query_b into BF16, output_b, the shared expert's down.
        failures += run_case("kv 5120x512 rows 5", rng, random_set(rng, 5120, 512, 1, spec1), 1, 5, {}, 1, false);
        {
            const auto fc = random_set(rng, 15360, 5120, 1, spec1);
            failures += run_case("fc 15360x5120 rows 8", rng, fc, 1, 8, {}, 1, false);
            failures += run_case("fc 15360x5120 rows 1", rng, fc, 1, 1, {}, 1, false);
        }
        failures += run_case("query_b 1280x32768 rows 5 bf16", rng, random_set(rng, 1280, 32768, 1, spec1), 1,
                             5, {}, 1, true);
        failures += run_case("output_b 8192x5120 rows 3", rng, random_set(rng, 8192, 5120, 1, spec1), 1, 3, {},
                             1, false);
        failures += run_case("shared_down 2304x5120 rows 7", rng, random_set(rng, 2304, 5120, 1, spec1), 1, 7,
                             {}, 1, false);
        // The grouped output projection: eight groups, one item each.
        failures += run_case("output_a 8 x 4096x1024 rows 5", rng, random_set(rng, 4096, 1024, 8, spec1), 8, 5,
                             {}, 1, false);
        // Routed experts: pairs of two rows' top-3 through an id list (a repeated expert), gate/up
        // reading their row, down reading each pair's own row.
        {
            const auto gate = random_set(rng, 5120, 2304, 5, pair1);
            failures += run_case("expert gate 5120x2304 pairs 6", rng, gate, 6, 1, {4, 0, 2, 2, 1, 4}, 3, false);
            const auto down = random_set(rng, 2304, 5120, 5, pair1);
            failures += run_case("expert down 2304x5120 pairs 6", rng, down, 6, 1, {3, 1, 0, 0, 4, 2}, 1, false);
        }
        if (failures) { std::cerr << failures << " STQ case(s) failed\n"; }
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
