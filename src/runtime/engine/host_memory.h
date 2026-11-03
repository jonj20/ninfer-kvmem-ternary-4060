#pragma once

#include "core/host_memory.h"
#include "ninfer/engine_options.h"

#include <cstdint>
#include <optional>

namespace ninfer::runtime {

// Sizes host_cache_budget_bytes for --host-cache-mib auto from `available_bytes`, the host memory
// still free once the model is loaded: all of it but host_cache_reserve_bytes plus
// `extra_reserve_bytes` (the media caches, which grow after startup), never above
// host_cache_max_bytes, nor host_cache_percent of `total_bytes`. A machine with this little memory
// gets no Host tier rather than an error. Returns the options with the budget set and
// host_cache_auto cleared; throws when a percent is asked for without `total_bytes`.
[[nodiscard]] ContextCacheOptions
resolve_auto_host_cache(const ContextCacheOptions& requested, std::uint64_t available_bytes,
                        std::optional<std::uint64_t> total_bytes,
                        std::uint64_t extra_reserve_bytes);

// resolve_auto_host_cache on this machine's memory now; the options unchanged without
// host_cache_auto. Throws when the platform reports no available memory.
[[nodiscard]] ContextCacheOptions
resolve_auto_host_cache_now(const ContextCacheOptions& requested,
                            std::uint64_t extra_reserve_bytes);

} // namespace ninfer::runtime
