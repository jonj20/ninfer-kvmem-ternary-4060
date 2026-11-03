#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;
namespace quantized_weight = ninfer::test::quantized_weight;

quantized_weight::PackedWeight make_ptq1_weight(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    return quantized_weight::make_ptq1_patterned_weight(n, k, seed);
}

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

// The ternary profile's problems across the decode, small-T and
// prefill widths.
int ptq1_conformance() {
    int failures = 0;

    constexpr std::array kDecode{convenience(1), a16(1),   a16(2),   a16(3),   a16(4),
                                 a16(5),         a16(8),   a16(9),   a16(12),  a16(16),
                                 a16(17),        a16(18),  a16(32),  a16(33),  a16(64),
                                 a16(65),        a16(128), a16(129), a16(512), a16(1024), Invocation{3, CallForm::Policy, ops::LinearPolicy::A16Only, true},
                                 Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only, true}};

    failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_weight,
                          {1024, 5120, 201U, Comparison::Full, true, kDecode});
    failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_weight,
                          {7168, 5120, 203U, Comparison::SampledRows, false, kDecode});
    failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_weight,
                          {34816, 5120, 205U, Comparison::SampledRows, false, kDecode});
    failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_weight,
                          {5120, 6144, 207U, Comparison::Full, false, kDecode});
    failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_weight,
                          {5120, 17408, 209U, Comparison::Full, false, kDecode});
    failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_weight,
                          {131072, 5120, 211U, Comparison::SampledRows, false, kDecode});
    failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_weight,
                          {248320, 5120, 213U, Comparison::SampledRows, false, kDecode});

    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        if (argc == 2 && std::string(argv[1]) == "--heads") {
            constexpr std::array calls{a16(1), a16(4), a16(8), a16(9), a16(16),
                a16(17), a16(65), a16(129),
                Invocation{9, CallForm::Policy, ops::LinearPolicy::A16Only, true},
                Invocation{65, CallForm::Policy, ops::LinearPolicy::A16Only, true}};
            int failures = 0;
            for (const auto output : {DType::BF16, DType::FP32}) {
                for (const auto k : {512, 1536, 4608, 12800}) {
                    ShapeCase shape{192, k, 20261030U, Comparison::Full, true, calls};
                    shape.out = output;
                    failures += run_shape("PTQ1 heads", ActivationCompute::A16, make_ptq1_weight, shape);
                }
            }
            // 17408 columns cannot be split at the naive 3:1 boundary 13056:
            // the small-T activation slab needs complete 512-column units.
            for (const auto shape : {std::array{192, 13056}, std::array{96, 512}}) {
                bool refused = false;
                try {
                    (void)ops::linear_workspace_capacity_bytes(QType::PTQ1_G128_FP16,
                        shape[0], shape[1], ops::LinearPolicy::A16Only, 1, 65);
                } catch (const std::invalid_argument&) { refused = true; }
                if (!refused) throw std::runtime_error("PTQ1 invalid tensor-share cut was accepted");
            }
            return failures ? 1 : 0;
        }
        const int failures = ptq1_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " PTQ1_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "PTQ1_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
