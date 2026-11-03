#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include <array>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;
    if (!cuda_available()) return 77;
    if (argc == 2 && std::string(argv[1]) == "--heads") {
        constexpr std::array<std::int32_t, 7> widths{1, 4, 9, 16, 17, 65, 129};
        int failures = 0;
        for (const auto k : {512, 1536, 5120}) {
            failures += run_profile("PTQ1 heads LinearSwiGLU",
                {QType::PTQ1_G128_FP16, 384, k, 192, 20261030U, ActivationCompute::A16},
                widths, std::array<std::int32_t, 2>{9, 65});
        }
        return failures ? 1 : 0;
    }
    constexpr std::array<std::int32_t, 15> widths{1, 3, 8, 9, 15, 16, 17, 32, 33, 63, 64, 65, 128, 512, 1024};
    const int failures = run_profile("PTQ1 LinearSwiGLU",
        {QType::PTQ1_G128_FP16, 34816, 5120, 17408, 1751U, ActivationCompute::A16},
        widths, std::array<std::int32_t, 2>{3, 65});
    std::cout << (failures ? "FAIL" : "OK") << " PTQ1 LinearSwiGLU\n";
    return failures ? 1 : 0;
}
