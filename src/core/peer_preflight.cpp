#include "core/peer_preflight.h"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace ninfer {
namespace {

// sysfs spells the address in lower case with a four-digit domain; CUDA may print upper case or an
// eight-digit domain.
std::optional<std::string> sysfs_address(std::string_view bus_id) {
    unsigned domain = 0;
    unsigned bus    = 0;
    unsigned slot   = 0;
    unsigned fn     = 0;
    const std::string text(bus_id);
    if (std::sscanf(text.c_str(), "%x:%x:%x.%x", &domain, &bus, &slot, &fn) != 4) {
        return std::nullopt;
    }
    char out[32];
    std::snprintf(out, sizeof out, "%04x:%02x:%02x.%x", domain & 0xFFFFU, bus, slot, fn);
    return std::string(out);
}

std::string trimmed(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.pop_back();
    }
    return text;
}

std::string mib(std::uint64_t bytes) {
    std::string digits = std::to_string(bytes >> 20U);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) {
        digits.insert(static_cast<std::size_t>(at), ",");
    }
    return digits + " MiB";
}

std::string label(const PeerDeviceFacts& facts) {
    return facts.name + " (" + facts.pci_bus_id + ")";
}

bool is_geforce(const PeerDeviceFacts& facts) {
    return facts.name.find("GeForce") != std::string::npos;
}

} // namespace

PeerDeviceFacts read_peer_device_facts(std::string_view sysfs_root, std::string_view name,
                                       std::string_view pci_bus_id, std::uint64_t memory_bytes) {
    PeerDeviceFacts facts;
    facts.name         = std::string(name);
    facts.pci_bus_id   = std::string(pci_bus_id);
    facts.memory_bytes = memory_bytes;
    const std::optional<std::string> address = sysfs_address(pci_bus_id);
    if (!address) { return facts; }
    const std::string device = std::string(sysfs_root) + "/bus/pci/devices/" + *address;

    if (std::ifstream type(device + "/iommu_group/type"); type) {
        std::string text;
        std::getline(type, text);
        facts.iommu_type = trimmed(text);
    }
    // One line per BAR: start, end and flags in hex; an unused BAR reads all zeros. A 64-bit BAR
    // takes two BAR registers, so BAR1 is the second line and the third stays zero.
    if (std::ifstream resource(device + "/resource"); resource) {
        std::string line;
        for (int index = 0; index < 2 && std::getline(resource, line); ++index) {
            if (index != 1) { continue; }
            std::istringstream fields(line);
            unsigned long long start = 0;
            unsigned long long end   = 0;
            fields >> std::hex >> start >> end;
            if (fields && end > start) { facts.bar1_bytes = end - start + 1; }
        }
    }
    return facts;
}

std::string peer_preflight_refusal(const PeerDeviceFacts& from, const PeerDeviceFacts& to) {
    for (const PeerDeviceFacts* facts : {&from, &to}) {
        if (facts->iommu_type && *facts->iommu_type != "identity") {
            return label(*facts) + " is in an IOMMU group of type '" + *facts->iommu_type +
                   "'; peer DMA needs identity domains (boot with iommu=pt)";
        }
    }
    for (const PeerDeviceFacts* facts : {&from, &to}) {
        if (is_geforce(*facts) && facts->bar1_bytes && *facts->bar1_bytes < facts->memory_bytes) {
            return label(*facts) + " exposes a " + mib(*facts->bar1_bytes) + " BAR1 for its " +
                   mib(facts->memory_bytes) +
                   " of memory; GeForce peer access needs a resizable BAR that covers the memory "
                   "(a ReBAR VBIOS)";
        }
    }
    return {};
}

} // namespace ninfer
