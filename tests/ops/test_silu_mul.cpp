#include "core/device.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion silu_mul_bf16_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.05e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> silu_mul_oracle(const std::vector<float>& gate, const std::vector<float>& up) {
    std::vector<double> expected(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i];
        expected[i]    = (g / (1.0 + std::exp(-g))) * static_cast<double>(up[i]);
    }
    return expected;
}

int run_contiguous_case(const char* label, std::int32_t rows, std::int32_t columns,
                        std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -12.0f, 12.0f);
    fill_uniform(up, seed + 1, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected  = silu_mul_oracle(gate, up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor up_tensor(device_up.data(), DType::BF16, {rows, columns});
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    silu_mul_bf16_criterion());
    failures += verify_exact("silu_mul gate unchanged",
                             from_device<std::uint16_t>(device_gate.data(), count), gate_bits);
    failures += verify_exact("silu_mul up unchanged",
                             from_device<std::uint16_t>(device_up.data(), count), up_bits);
    failures += device_gate.verify_guards("silu_mul gate");
    failures += device_up.verify_guards("silu_mul up");
    failures += device_out.verify_guards("silu_mul out");
    return failures;
}

// Unrounded FP32 gate and up, as a runtime-shape gate/up pair hands them over (linear_swiglu).
int run_fp32_case(std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -12.0f, 12.0f);
    fill_uniform(up, seed + 1, -8.0f, 8.0f);
    const auto expected = silu_mul_oracle(gate, up);
    GuardedDeviceBuffer device_gate(count * sizeof(float));
    GuardedDeviceBuffer device_up(count * sizeof(float));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate.data(), device_gate.bytes());
    device_up.copy_from_host(up.data(), device_up.bytes());
    Tensor gate_tensor(device_gate.data(), DType::FP32, {rows, columns});
    Tensor up_tensor(device_up.data(), DType::FP32, {rows, columns});
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("silu_mul FP32 inputs", from_device_bf16(device_out.data(), count),
                                    expected, silu_mul_bf16_criterion());
    failures += device_gate.verify_guards("silu_mul FP32 gate");
    failures += device_up.verify_guards("silu_mul FP32 up");
    failures += device_out.verify_guards("silu_mul FP32 out");
    return failures;
}

int run_strided_gate_up_case() {
    constexpr std::int32_t rows    = 17408;
    constexpr std::int32_t columns = 17;
    const std::size_t count        = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, 301u, -12.0f, 12.0f);
    fill_uniform(up, 302u, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(up);

    std::vector<float> gate_up(2 * count);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t source = static_cast<std::size_t>(column) * rows;
        const std::size_t target = static_cast<std::size_t>(column) * 2 * rows;
        std::copy_n(gate.data() + source, rows, gate_up.data() + target);
        std::copy_n(up.data() + source, rows, gate_up.data() + target + rows);
    }

    const auto expected     = silu_mul_oracle(gate, up);
    const auto gate_up_bits = encode_bf16(gate_up);
    GuardedDeviceBuffer device_gate_up(gate_up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate_up.copy_from_host(gate_up_bits.data(), device_gate_up.bytes());

    Tensor gate_up_tensor(device_gate_up.data(), DType::BF16, {2 * rows, columns});
    Tensor gate_tensor = gate_up_tensor.slice(0, 0, rows);
    Tensor up_tensor   = gate_up_tensor.slice(0, rows, rows);
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures =
        verify_pointwise("silu_mul strided gate/up", from_device_bf16(device_out.data(), count),
                         expected, silu_mul_bf16_criterion());
    failures += verify_exact("silu_mul strided inputs unchanged",
                             from_device<std::uint16_t>(device_gate_up.data(), gate_up_bits.size()),
                             gate_up_bits);
    failures += device_gate_up.verify_guards("silu_mul strided inputs");
    failures += device_out.verify_guards("silu_mul strided out");
    return failures;
}

// The combined projection stores [gate; up] per token. Both row slices retain
// the full parent's column pitch; no intermediate BF16 rounding is permitted.
int run_fp32_strided_gate_up_case(std::int32_t rows, std::int32_t columns) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count), parent(2 * count);
    GuardedDeviceBuffer device_parent(parent.size() * sizeof(float));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    Tensor projected(device_parent.data(), DType::FP32, {2 * rows, columns});
    const Tensor gate_tensor = projected.slice(0, 0, rows);
    const Tensor up_tensor = projected.slice(0, rows, rows);
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    cudaStream_t stream;
    cudaGraph_t captured;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
    CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
    int failures = 0;
    for (int replay = 0; replay < 2; ++replay) {
        fill_uniform(gate, 401u + replay, -12.0f, 12.0f);
        fill_uniform(up, 411u + replay, -8.0f, 8.0f);
        for (std::int32_t column = 0; column < columns; ++column) {
            const std::size_t source = static_cast<std::size_t>(column) * rows;
            std::copy_n(gate.data() + source, rows, parent.data() + 2 * source);
            std::copy_n(up.data() + source, rows, parent.data() + 2 * source + rows);
        }
        device_parent.copy_from_host(parent.data(), device_parent.bytes());
        const auto expected = silu_mul_oracle(gate, up);
        // Check eager and captured execution independently against represented FP32 inputs.
        ops::silu_mul(gate_tensor, up_tensor, out_tensor, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        failures += verify_pointwise("silu_mul FP32 row views eager",
                                      from_device_bf16(device_out.data(), count), expected,
                                      silu_mul_bf16_criterion());
        CUDA_CHECK(cudaMemsetAsync(device_out.data(), 0xff, device_out.bytes(), stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        failures += verify_pointwise("silu_mul FP32 row views graph",
                                      from_device_bf16(device_out.data(), count), expected,
                                      silu_mul_bf16_criterion());
        failures += verify_exact("silu_mul FP32 row views unchanged",
                                  from_device<float>(device_parent.data(), parent.size()), parent);
        failures += device_parent.verify_guards("silu_mul FP32 row views");
        failures += device_out.verify_guards("silu_mul FP32 row views out");
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(captured));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}

// Independent pitches in all four dimensions, including a non-unit row stride.
int run_fp32_general_stride_case() {
    constexpr int rows = 7, columns = 3, planes = 2, batches = 2;
    std::vector<float> gate_parent(5 * 17 * 4 * 3), up_parent(19 * 6 * 5 * 4);
    fill_uniform(gate_parent, 421u, -12.0f, 12.0f);
    fill_uniform(up_parent, 422u, -8.0f, 8.0f);
    GuardedDeviceBuffer device_gate(gate_parent.size() * sizeof(float));
    GuardedDeviceBuffer device_up(up_parent.size() * sizeof(float));
    GuardedDeviceBuffer device_out(rows * columns * planes * batches * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_parent.data(), device_gate.bytes());
    device_up.copy_from_host(up_parent.data(), device_up.bytes());
    const Tensor gate_tensor = Tensor(device_gate.data(), DType::FP32, {5, 17, 4, 3})
                                   .permute({1, 0, 2, 3})
                                   .slice(0, 1, rows).slice(1, 1, columns)
                                   .slice(2, 1, planes).slice(3, 1, batches);
    const Tensor up_tensor = Tensor(device_up.data(), DType::FP32, {19, 6, 5, 4})
                                 .slice(0, 3, rows).slice(1, 2, columns)
                                 .slice(2, 2, planes).slice(3, 1, batches);
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns, planes, batches});
    std::vector<float> gate, up;
    for (int b = 0; b < batches; ++b)
        for (int p = 0; p < planes; ++p)
            for (int c = 0; c < columns; ++c)
                for (int r = 0; r < rows; ++r) {
                    gate.push_back(gate_parent[(c + 1) + 5 * ((r + 1) + 17 * ((p + 1) + 4 * (b + 1)))]);
                    up.push_back(up_parent[(r + 3) + 19 * ((c + 2) + 6 * ((p + 2) + 5 * (b + 1)))]);
                }
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise("silu_mul FP32 general strides",
                                    from_device_bf16(device_out.data(), gate.size()),
                                    silu_mul_oracle(gate, up), silu_mul_bf16_criterion());
    failures += verify_exact("silu_mul FP32 strided gate unchanged",
                              from_device<float>(device_gate.data(), gate_parent.size()), gate_parent);
    failures += verify_exact("silu_mul FP32 strided up unchanged",
                              from_device<float>(device_up.data(), up_parent.size()), up_parent);
    failures += device_gate.verify_guards("silu_mul FP32 strided gate");
    failures += device_up.verify_guards("silu_mul FP32 strided up");
    failures += device_out.verify_guards("silu_mul FP32 strided out");
    return failures;
}

int run_edge_case() {
    std::vector<float> gate{-30.0f, -8.0f, -1.0f, 0.0f, 1.0f, 8.0f, 30.0f};
    std::vector<float> up{2.0f, -3.0f, 0.5f, -4.0f, 5.0f, -6.0f, 7.0f};
    round_to_bf16(gate);
    round_to_bf16(up);

    const auto expected  = silu_mul_oracle(gate, up);
    const auto gate_bits = encode_bf16(gate);
    const auto up_bits   = encode_bf16(up);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(expected.size() * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {static_cast<int>(gate.size())});
    Tensor up_tensor(device_up.data(), DType::BF16, {static_cast<int>(up.size())});
    Tensor out_tensor(device_out.data(), DType::BF16, {static_cast<int>(expected.size())});
    ops::silu_mul(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("silu_mul edge values",
                                    from_device_bf16(device_out.data(), expected.size()), expected,
                                    silu_mul_bf16_criterion());
    failures += device_gate.verify_guards("silu_mul edge gate");
    failures += device_up.verify_guards("silu_mul edge up");
    failures += device_out.verify_guards("silu_mul edge out");
    return failures;
}

} // namespace

int main() try {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_contiguous_case("silu_mul [17408,1]", 17408, 1, 101u);
    failures += run_contiguous_case("silu_mul [17408,48]", 17408, 48, 102u);
    failures += run_fp32_case(4352, 3, 103u);
    failures += run_strided_gate_up_case();
    failures += run_fp32_strided_gate_up_case(192, 9);
    failures += run_fp32_strided_gate_up_case(896, 65);
    failures += run_fp32_general_stride_case();
    failures += run_edge_case();
    std::cout << (failures ? "FAIL" : "OK") << " silu_mul\n";
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL silu_mul: " << error.what() << std::endl;
    return 1;
}
