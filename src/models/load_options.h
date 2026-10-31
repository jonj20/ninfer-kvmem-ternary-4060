#pragma once

#include "ninfer/types.h"

#include <string_view>

namespace ninfer::models {

struct LoadOptions {
    EnginePurpose purpose            = EnginePurpose::Generation;
    bool vision                      = false;
    std::uint32_t vision_max_tokens  = 8192;
    SpeculativeBackend speculative   = SpeculativeBackend::None;
    ProposalHead proposal_head       = ProposalHead::Full;

    bool operator==(const LoadOptions&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool dflash2() const noexcept {
        return speculative == SpeculativeBackend::DFlash2;
    }

    [[nodiscard]] bool masked_draft() const noexcept { return dflash() || dflash2(); }

    [[nodiscard]] bool proposal_enabled() const noexcept {
        return purpose == EnginePurpose::Generation && speculative != SpeculativeBackend::None &&
               proposal_head == ProposalHead::Optimized;
    }

    [[nodiscard]] std::string_view speculative_component() const noexcept {
        switch (speculative) {
        case SpeculativeBackend::None:
            return {};
        case SpeculativeBackend::Mtp:
            return "mtp";
        case SpeculativeBackend::DFlash:
            return "dflash";
        case SpeculativeBackend::DFlash2:
            return "dflash2";
        }
        return {};
    }
};

[[nodiscard]] constexpr bool is_masked_draft_backend(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
}

// Whether a real --kvmem-budget turns the Device KV tier into a resident staging window for this
// backend. The cache-position pipeline (verify cache rows vs absolute RoPE positions) is what a
// windowed backend needs; MTP owns it (mtp-kvmem-plan.md stage 6a), masked draft backends read
// absolute positions and keep the identity window until stage 6b.
[[nodiscard]] constexpr bool kvmem_resident_staging(SpeculativeBackend backend) noexcept {
    return backend == SpeculativeBackend::None || backend == SpeculativeBackend::Mtp;
}

[[nodiscard]] inline LoadOptions load_options(const EngineOptions& options) noexcept {
    return {.purpose           = options.purpose,
            .vision            = options.enable_vision,
            .vision_max_tokens = options.vision_max_tokens > 0 ? options.vision_max_tokens : 8192,
            .speculative       = options.speculative.backend,
            .proposal_head     = options.speculative.proposal_head};
}

} // namespace ninfer::models
