#pragma once

// Disk-resident experts: they stay in the artifact's files and a device pool of slots holds the
// ones passes route to. Before a layer's experts run, prepare() makes every expert the layer's
// tokens routed to resident: the slots of the least recently used experts (never one the same call
// needs) take the missing ones, read from the files through the OS page cache into a page-locked
// staging ring (several reads in flight at once) and copied on the layer's stream, and the layer's
// expert tables are pointed at them. Empty slots are taken first, then released ones, then by
// CLOCK. Each device's layers share one pool whose slots fit their
// widest expert, except a layer marked own_pool whose experts are wider: it has slots of its own
// when the device's budget holds them beside the widest call of the others.
// The host memory this mode needs is the staging ring; the page cache keeps what the system can
// spare.

#include "core/arena.h"
#include "core/device.h"
#include "models/qwen4_exp/model.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp {

struct ExpertStreamLayer {
    std::size_t rank    = 0;
    cudaStream_t stream = nullptr;
    // Gate, up and down: every expert's location, and the device table the kernels read.
    std::array<std::span<const ExpertLocation>, 3> experts;
    std::array<void*, 3> tables{};
    // The most experts one call routes to, 0 for every expert of the layer.
    std::size_t widest_call = 0;
    // The MTP block, whose experts may be stored wider than the text layers': with slots of their
    // own, the text layers' slots are not sized for them.
    bool own_pool = false;
};

struct ExpertStreamStats {
    std::uint64_t routes     = 0; // distinct (call, expert) needs
    std::uint64_t hits       = 0; // of them, already resident
    std::uint64_t read_bytes = 0;
    std::uint32_t slots      = 0;
    double read_seconds      = 0.0;
};

class ExpertStream {
public:
    ExpertStream(DeviceContext& device, const std::vector<std::filesystem::path>& files,
                 std::vector<ExpertStreamLayer> layers,
                 std::span<const std::uint64_t> bytes_by_rank);
    ~ExpertStream();
    ExpertStream(const ExpertStream&)            = delete;
    ExpertStream& operator=(const ExpertStream&) = delete;

    // Makes every expert in `ids` resident for `layer` and its tables point at them, on the
    // layer's stream. Returns once the copies are queued; the staging ring is reused only after
    // the copies that read it finish. With `release_others` (a prompt span's last chunk), the
    // layer's residents `ids` lacks become the first slots later calls take once every slot holds
    // an expert, oldest release first, unless a call routes to them before: the earlier layers keep
    // their last chunk's experts while the later layers of the span read theirs.
    void prepare(std::size_t layer, std::span<const std::int32_t> ids, bool release_others);
    [[nodiscard]] ExpertStreamStats stats() const noexcept;
    // The device memory of `layer`'s slots: a table entry inside it is a resident expert.
    [[nodiscard]] std::pair<std::uintptr_t, std::uintptr_t> storage(std::size_t layer) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp
