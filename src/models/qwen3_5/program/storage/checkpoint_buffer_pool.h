#pragma once
#include "core/arena.h"
#include <memory>
#include <mutex>
#include <array>

namespace ninfer::models::qwen3_5::detail {

// Only released StateImages may enter the cache. Live checkpoints keep exclusive
// buffer leases; the pool can outlive Program while a lease is still referenced.
class KvmemCheckpointBufferPool : public std::enable_shared_from_this<KvmemCheckpointBufferPool> {
public:
    std::shared_ptr<PinnedHostBuffer> acquire(std::size_t bytes) {
        auto pool = shared_from_this();
        std::unique_ptr<PinnedHostBuffer> buffer;
        {
            std::lock_guard lock(mutex_);
            for (auto& idle : idle_) if (idle && idle->size() == bytes) {
                buffer = std::move(idle); break;
            }
        }
        if (!buffer) buffer = std::make_unique<PinnedHostBuffer>(bytes);
        return {buffer.release(), [pool = std::move(pool)](PinnedHostBuffer* value) noexcept {
            std::unique_ptr<PinnedHostBuffer> released(value);
            try {
                std::lock_guard lock(pool->mutex_);
                for (auto& idle : pool->idle_) if (!idle) {
                    idle = std::move(released); break;
                }
            } catch (...) {} // Releasing a checkpoint must remain noexcept.
        }};
    }
    std::size_t idle_bytes() const {
        std::lock_guard lock(mutex_);
        std::size_t bytes = 0;
        for (const auto& idle : idle_) if (idle) bytes += idle->size();
        return bytes;
    }
private:
    mutable std::mutex mutex_;
    // A restored anchor can release both its input and endpoint snapshots.
    // Keep that pair; live snapshots can never enter these slots.
    std::array<std::unique_ptr<PinnedHostBuffer>, 2> idle_;
};
} // namespace ninfer::models::qwen3_5::detail
