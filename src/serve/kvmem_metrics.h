#pragma once

#include "serve/generation_service.h"
#include "product/speculative_options.h"
#include <nlohmann/json.hpp>

namespace ninfer::serve {

// Additive diagnostics; the native usage, timings and Prometheus series stay intact.
inline nlohmann::json kvmem_generation_metrics(const GenerationOutcome& outcome) {
    using Json = nlohmann::json;
    const auto& m = outcome.metrics;
    const auto rounds = m.engine_timing.decode_rounds;
    const auto tokens = outcome.completion_tokens > 0 ? outcome.completion_tokens - 1 : 0;
    Json spec{{"backend", product::speculative_backend_name(m.speculative_backend)},
              {"drafted_tokens", m.speculative_draft_tokens},
              {"accepted_tokens", m.speculative_accepted_tokens},
              {"verification_steps", m.speculative_rounds},
              {"fallback_rounds", m.speculative_fallback_steps},
              {"accepted_per_position", m.speculative_accepted_per_position},
              {"generation_rounds", rounds}, {"output_tokens", tokens},
              {"wall_ms", m.generation_wall_seconds * 1000.0},
              {"timing_scope", "first_to_last_token"},
              {"acceptance_rate", m.speculative_draft_tokens
                   ? Json(double(m.speculative_accepted_tokens) / m.speculative_draft_tokens) : Json(nullptr)},
              {"tokens_per_round", rounds ? Json(double(tokens) / rounds) : Json(nullptr)},
              {"ms_per_round", rounds ? Json(m.generation_wall_seconds * 1000.0 / rounds) : Json(nullptr)}};
    if (!rounds) spec["unavailable_reason"] = "no_decode_rounds_observed";
    return {{"speculative", std::move(spec)}};
}

} // namespace ninfer::serve
