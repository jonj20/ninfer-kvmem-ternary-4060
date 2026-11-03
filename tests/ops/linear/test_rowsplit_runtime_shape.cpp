// The packed formats' runtime-shape routes: a tensor-parallel rank's share of a projection, whose
// rows (and, for an output projection, K) depend on the split, in BF16 and as an FP32 partial sum.
// The shapes are Qwen3.8-27B's at a 3:1 split: GDN key heads 12/4 (q and k 1536/512 rows, v and z
// 4608/1536, the output projection's K the same), FFN rows 13056/4352. T2, FP8 and NVFP4 shares are
// whole K slabs of 512 or 256 columns (ops::linear_share_units), and a rotated input's whole
// 1024-column blocks: their FFN rows split 13312/4096. On sm_70 the formats a Volta rank takes in
// the panel layout (T2, Q4, Q5, Q6) also run their FP32 partials and a small matrix in it.

#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {
using namespace ninfer::test::linear;
using ninfer::DType;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation graph(std::int32_t t) {
    return {t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true};
}

// Every width class of the routes: the GEMV or SIMT widths, each sliced-K tile and the MMA, with a
// remainder on each side of a tile.
constexpr std::array kWidths{
    a16(1),  a16(2),  a16(3),  a16(4),   a16(5),   a16(8),   a16(9),   a16(16),  a16(17),
    a16(32), a16(33), a16(64), a16(65),  a16(128), a16(129), a16(257), graph(1), graph(9),
    graph(33),
};
constexpr std::array kSmall{a16(1), a16(4), a16(9), a16(33), a16(130), graph(2)};

int conformance(const char* label, WeightGenerator generator, std::uint32_t seed) {
    int failures = 0;
    // The FFN's gate and up rows and the down projection over them, both FP32 partials.
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {13056, 5120, seed, Comparison::SampledRows, false, kWidths,
                           ActivationSigns::Centered, DType::FP32});
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {5120, 4352, seed + 1, Comparison::SampledRows, false, kWidths,
                           ActivationSigns::Biased, DType::FP32});
    // The GDN output projection over the second rank's 1536 value rows.
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {5120, 1536, seed + 2, Comparison::SampledRows, true, kWidths,
                           ActivationSigns::Centered, DType::FP32});
    // A share's GDN input parts in BF16.
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {4608, 5120, seed + 3, Comparison::SampledRows, false, kWidths});
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {512, 5120, seed + 4, Comparison::SampledRows, false, kWidths});
    // Every row of a small matrix, both outputs.
    for (const DType out : {DType::BF16, DType::FP32}) {
        failures += run_shape(label, ActivationCompute::A16, generator,
                              {384, 640, seed + 5, Comparison::Full, true, kSmall,
                               ActivationSigns::Centered, out});
    }
    return failures;
}

int slab_conformance(const char* label, WeightGenerator generator, std::uint32_t seed) {
    int failures = 0;
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {13312, 5120, seed, Comparison::SampledRows, false, kWidths,
                           ActivationSigns::Centered, DType::FP32});
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {5120, 4096, seed + 1, Comparison::SampledRows, false, kWidths,
                           ActivationSigns::Biased, DType::FP32});
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {5120, 1536, seed + 2, Comparison::SampledRows, true, kWidths,
                           ActivationSigns::Centered, DType::FP32});
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {4608, 5120, seed + 3, Comparison::SampledRows, false, kWidths});
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {512, 5120, seed + 4, Comparison::SampledRows, false, kWidths});
    for (const DType out : {DType::BF16, DType::FP32}) {
        failures += run_shape(label, ActivationCompute::A16, generator,
                              {384, 1024, seed + 5, Comparison::Full, true, kSmall,
                               ActivationSigns::Centered, out});
    }
    return failures;
}

// A Volta rank's down projection share as an FP32 partial and every row of a small matrix, both
// outputs, with the weight in the panel layout; `k` is the share's intermediate rows.
int panel_conformance(const char* label, WeightGenerator generator, std::uint32_t seed,
                      std::int32_t k) {
    int failures = 0;
    failures += run_shape(label, ActivationCompute::A16, generator,
                          {5120, k, seed, Comparison::SampledRows, true, kWidths,
                           ActivationSigns::Biased, DType::FP32, true});
    for (const DType out : {DType::BF16, DType::FP32}) {
        failures += run_shape(label, ActivationCompute::A16, generator,
                              {384, 1024, seed + 1, Comparison::Full, true, kSmall,
                               ActivationSigns::Centered, out, true});
    }
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        const struct {
            const char* label;
            WeightGenerator generator;
            std::uint32_t seed;
        } formats[] = {
            {"Q4_A16 runtime shape", make_q4_g64_fp16_weight, 401U},
            {"Q5_A16 runtime shape", make_q5_g64_fp16_weight, 503U},
            {"Q6_A16 runtime shape", make_q6_g64_fp16_weight, 607U},
            {"Q8_A16 runtime shape", make_q8_g32_fp16_weight, 809U},
        };
        for (const auto& format : formats) {
            const int format_failures = conformance(format.label, format.generator, format.seed);
            std::cout << (format_failures == 0 ? "OK " : "FAIL ") << format.label << '\n';
            failures += format_failures;
        }
        const struct {
            const char* label;
            WeightGenerator generator;
            std::uint32_t seed;
            std::int32_t k;
        } panel_formats[] = {
            {"Q4_A16 panel runtime shape", make_q4_g64_fp16_weight, 421U, 4352},
            {"Q5_A16 panel runtime shape", make_q5_g64_fp16_weight, 523U, 4352},
            {"Q6_A16 panel runtime shape", make_q6_g64_fp16_weight, 627U, 4352},
            {"T2_A16 panel runtime shape", make_t2_g128_fp16_weight, 931U, 4096},
        };
        for (const auto& format : panel_formats) {
            const int format_failures =
                panel_conformance(format.label, format.generator, format.seed, format.k);
            std::cout << (format_failures == 0 ? "OK " : "FAIL ") << format.label << '\n';
            failures += format_failures;
        }
        const struct {
            const char* label;
            WeightGenerator generator;
            std::uint32_t seed;
        } slab_formats[] = {
            {"T2_A16 runtime shape", make_t2_g128_fp16_weight, 911U},
            {"FP8_A16 runtime shape", make_fp8_weight, 1013U},
            {"NVFP4_A16 runtime shape", make_nvfp4_weight, 1117U},
        };
        for (const auto& format : slab_formats) {
            const int format_failures = slab_conformance(format.label, format.generator, format.seed);
            std::cout << (format_failures == 0 ? "OK " : "FAIL ") << format.label << '\n';
            failures += format_failures;
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "row-split runtime-shape linear: " << error.what() << '\n';
        return 1;
    }
}
