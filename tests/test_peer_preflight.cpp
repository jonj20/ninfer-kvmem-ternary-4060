// The host checks that keep a card pair off peer transfers before a probe touches its mappings
// (core/peer_preflight.h). Linux also reads a fabricated sysfs tree; Windows supplies policy
// facts directly because its filesystem cannot represent the PCI address directory names.

#include "core/peer_preflight.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool condition, const std::string& label) {
    if (condition) { return; }
    std::cerr << "expectation failed: " << label << '\n';
    ++failures;
}

constexpr std::uint64_t kGiB = 1ULL << 30U;

// A device directory with an optional IOMMU group type and a resource file whose BAR1 spans
// `bar1` bytes (none when zero).
#if !defined(_WIN32)
void make_device(const fs::path& root, const std::string& address, const char* iommu_type,
                 std::uint64_t bar1) {
    const fs::path device = root / "bus/pci/devices" / address;
    fs::create_directories(device);
    if (iommu_type != nullptr) {
        fs::create_directories(device / "iommu_group");
        std::ofstream(device / "iommu_group/type") << iommu_type << '\n';
    }
    std::ofstream resource(device / "resource");
    resource << "0x00000000fb000000 0x00000000fbffffff 0x0000000000040200\n";
    if (bar1 != 0) {
        const std::uint64_t start = 0x0000038000000000ULL;
        resource << "0x" << std::hex << start << " 0x" << (start + bar1 - 1)
                 << " 0x000000000014220c\n";
    } else {
        resource << "0x0000000000000000 0x0000000000000000 0x0000000000000000\n";
    }
    resource << "0x0000000000000000 0x0000000000000000 0x0000000000000000\n";
}
#endif

ninfer::PeerDeviceFacts facts(const fs::path& root, const char* name, const char* bus_id,
                              std::uint64_t memory) {
    return ninfer::read_peer_device_facts(root.string(), name, bus_id, memory);
}

void cases(const fs::path& root) {
    fs::create_directories(root);
#if defined(_WIN32)
    // These exercise the same policy as the Linux fixtures, not Linux sysfs parsing.
    const ninfer::PeerDeviceFacts rebar{
        "NVIDIA GeForce RTX 3090", "0000:01:00.0", 24 * kGiB, "identity", 32 * kGiB};
    const ninfer::PeerDeviceFacts small{
        "NVIDIA GeForce RTX 3090", "00000000:02:00.0", 24 * kGiB, "identity", 256ULL << 20U};
    const ninfer::PeerDeviceFacts dma{
        "Tesla V100-PCIE-32GB", "0000:3B:00.0", 32 * kGiB, "DMA-FQ", 32 * kGiB};
    const ninfer::PeerDeviceFacts tesla{
        "Tesla T4", "0000:5e:00.0", 16 * kGiB, std::nullopt, 256ULL << 20U};
#else
    make_device(root, "0000:01:00.0", "identity", 32 * kGiB);
    make_device(root, "0000:02:00.0", "identity", 256ULL << 20U);
    make_device(root, "0000:3b:00.0", "DMA-FQ", 32 * kGiB);
    make_device(root, "0000:5e:00.0", nullptr, 256ULL << 20U);

    const auto rebar  = facts(root, "NVIDIA GeForce RTX 3090", "0000:01:00.0", 24 * kGiB);
    const auto small  = facts(root, "NVIDIA GeForce RTX 3090", "00000000:02:00.0", 24 * kGiB);
    const auto dma    = facts(root, "Tesla V100-PCIE-32GB", "0000:3B:00.0", 32 * kGiB);
    const auto tesla  = facts(root, "Tesla T4", "0000:5e:00.0", 16 * kGiB);
    check(rebar.iommu_type == "identity", "the group type is read and trimmed");
    check(rebar.bar1_bytes == 32 * kGiB, "BAR1 is the second resource line");
    check(small.bar1_bytes == (256ULL << 20U), "an eight-digit domain finds the same directory");
    check(dma.iommu_type == "DMA-FQ", "an upper-case address finds the lower-case directory");
    check(!tesla.iommu_type.has_value(), "no group directory: no IOMMU");
#endif

    // Exercise the real reader on both platforms even where no sysfs tree can be fabricated.
    const auto absent = facts(root, "NVIDIA GeForce RTX 4090", "0000:99:00.0", 24 * kGiB);
    check(!absent.iommu_type && !absent.bar1_bytes, "a missing device reports no facts");
    check(absent.name == "NVIDIA GeForce RTX 4090" && absent.pci_bus_id == "0000:99:00.0" &&
              absent.memory_bytes == 24 * kGiB,
          "missing sysfs facts preserve the supplied device identity");
    const auto invalid = facts(root, "NVIDIA GeForce RTX 4090", "not-a-pci-address", 24 * kGiB);
    check(!invalid.iommu_type && !invalid.bar1_bytes, "an invalid PCI address reports no facts");
    const fs::path file_root = root / "not-a-directory";
    std::ofstream(file_root) << "not a sysfs directory\n";
    check(fs::is_regular_file(file_root), "the invalid sysfs root fixture is a regular file");
    const auto not_sysfs = facts(file_root, "NVIDIA GeForce RTX 4090", "0000:01:00.0", 24 * kGiB);
    check(!not_sysfs.iommu_type && !not_sysfs.bar1_bytes,
          "a sysfs root that is a regular file reports no facts");

    check(ninfer::peer_preflight_refusal(rebar, rebar).empty(), "full BAR1, identity: probed");
    check(ninfer::peer_preflight_refusal(absent, rebar).empty(), "unknown facts: probed");
    check(ninfer::peer_preflight_refusal(tesla, tesla).empty(),
          "a small BAR1 outside GeForce: probed (mailbox mappings work there)");
    const std::string translated = ninfer::peer_preflight_refusal(rebar, dma);
    check(translated.find("DMA-FQ") != std::string::npos &&
              translated.find("iommu=pt") != std::string::npos,
          "a translating IOMMU on either card: refused, naming the fix");
    const std::string mailbox = ninfer::peer_preflight_refusal(small, rebar);
    check(mailbox.find("256 MiB") != std::string::npos &&
              mailbox.find("24,576 MiB") != std::string::npos,
          "a GeForce BAR1 below its memory: refused, with both sizes");
}

} // namespace

int run_peer_preflight_test() {
    const fs::path root =
        fs::temp_directory_path() / ("ninfer-peer-preflight-" + std::to_string(std::rand()));
    try {
        fs::remove_all(root);
        cases(root);
    } catch (const std::exception& error) {
        std::cerr << "unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);
    if (failures != 0) { return 1; }
#if defined(_WIN32)
    std::cout << "peer preflight policy and missing-facts tests passed "
                 "(Windows; Linux sysfs parsing not exercised)\n";
#else
    std::cout << "peer preflight policy and sysfs parsing tests passed\n";
#endif
    return 0;
}
