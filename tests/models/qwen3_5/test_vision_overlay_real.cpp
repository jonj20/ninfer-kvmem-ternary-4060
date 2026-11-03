#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Two image requests admitted together under the overlay Vision residency. A device holds one
// Vision window, and one lane's item submitted ahead of its prefill unit can hold it when the other
// lane encodes its own item synchronously: that encode finishes the submitted item first. Both
// requests must complete, each reading its own image -- the first answer token (the color's name)
// of each concurrent request agrees with the one its image gives alone -- under both context caches.
namespace {

// A disc or a square of one color on a gray ground: something a model can name.
std::vector<std::uint8_t> test_image(bool disc, std::uint8_t red, std::uint8_t green,
                                     std::uint8_t blue) {
    constexpr int kSide       = 448;
    const std::string header  = "P6\n" + std::to_string(kSide) + ' ' + std::to_string(kSide) +
                               "\n255\n";
    std::vector<std::uint8_t> ppm(header.begin(), header.end());
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            const int dx = x - kSide / 2, dy = y - kSide / 2;
            const bool inside = disc ? dx * dx + dy * dy < (kSide / 4) * (kSide / 4)
                                     : std::abs(dx) < kSide / 4 && std::abs(dy) < kSide / 4;
            ppm.push_back(inside ? red : 128);
            ppm.push_back(inside ? green : 128);
            ppm.push_back(inside ? blue : 128);
        }
    }
    return ppm;
}

ninfer::PromptInput image_prompt(std::vector<std::uint8_t> image) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    ninfer::MessagePart part;
    part.kind              = ninfer::MessagePartKind::Media;
    part.media.kind        = ninfer::MediaKind::Image;
    part.media.bytes       = std::move(image);
    part.media.media_type  = "image/x-portable-pixmap";
    part.media.source_name = "shape.ppm";
    message.parts.push_back(std::move(part));
    message.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                .text  = "What color is the shape in the middle? "
                                                         "Answer with one word.",
                                                .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

ninfer::RequestOptions request_options() {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 8;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.execution.logprobs                = true;
    request.stop.include_model_defaults       = false;
    return request;
}

// The concurrent run's first token against the same image alone: most of the eight likeliest
// alternatives coincide and the two likeliest agree within a small margin of log probability. A
// shared prompt pass may round differently from a lone one, so equality is not expected.
bool agrees(const ninfer::GenerationResult& alone, const ninfer::GenerationResult& together) {
    if (alone.content_logprobs.empty() || together.content_logprobs.empty() ||
        !alone.prompt.has_media || !together.prompt.has_media) {
        return false;
    }
    const ninfer::TokenLogprob& a = alone.content_logprobs.front();
    const ninfer::TokenLogprob& b = together.content_logprobs.front();
    constexpr std::size_t kTop    = 8;
    const auto logprob_in         = [&](const ninfer::TokenLogprob& run, ninfer::TokenId token) {
        for (std::size_t i = 0; i < kTop; ++i) {
            if (run.top_ids[i] == token) { return run.top_values[i]; }
        }
        return -INFINITY;
    };
    std::size_t shared = 0;
    float margin       = 0.0F;
    for (std::size_t i = 0; i < kTop; ++i) {
        shared += std::isfinite(logprob_in(b, a.top_ids[i])) ? 1U : 0U;
    }
    for (std::size_t i = 0; i < 2; ++i) {
        margin = std::max({margin, std::abs(a.top_values[i] - logprob_in(b, a.top_ids[i])),
                           std::abs(b.top_values[i] - logprob_in(a, b.top_ids[i]))});
    }
    return shared >= 6 && margin <= 0.25F;
}

int run_with(const char* artifact, ninfer::ContextCacheMode mode) {
    ninfer::EngineOptions options;
    options.artifact_path            = artifact;
    // Free KV beyond the two requests funds the window an item submitted ahead takes; the legacy
    // cache accepts at most max_context tokens per lane.
    options.max_context              = 8192;
    options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(16384);
    options.prefill_chunk            = 512;
    options.enable_vision            = true;
    options.vision_residency         = ninfer::VisionResidency::Overlay;
    options.vision_max_merged_tokens = 256;
    options.max_concurrency          = 2;
    options.max_pending_requests     = 2;
    options.concurrent_prefill       = true;
    options.context_cache.mode       = mode;
    ninfer::Engine engine(std::move(options));

    const std::vector<std::vector<std::uint8_t>> images = {test_image(true, 230, 30, 30),
                                                           test_image(false, 30, 60, 230)};
    std::vector<ninfer::GenerationResult> alone;
    for (const auto& image : images) {
        alone.push_back(engine.generate(engine.prepare(image_prompt(image)), request_options()));
    }
    std::vector<ninfer::GenerationSubmission> group;
    for (const auto& image : images) {
        group.push_back({.prompt = engine.prepare(image_prompt(image)), .options = request_options()});
    }
    auto handles = engine.submit_group(std::move(group));
    std::vector<ninfer::GenerationResult> together;
    for (auto& handle : handles) { together.push_back(handle.wait()); }

    const char* name = mode == ninfer::ContextCacheMode::Hybrid ? "hybrid" : "legacy";
    int failures     = 0;
    if (alone[0].content_logprobs.empty() || alone[1].content_logprobs.empty() ||
        alone[0].content_logprobs.front().id == alone[1].content_logprobs.front().id) {
        std::cerr << name << ": the two images do not give two different first tokens alone\n";
        return 1;
    }
    for (std::size_t i = 0; i < images.size(); ++i) {
        std::cout << name << " image " << i << " alone: " << alone[i].content
                  << " | together: " << together[i].content << '\n';
        if (!agrees(alone[i], together[i])) {
            std::cerr << name << " image " << i
                      << ": the concurrent request did not read its own image\n";
            ++failures;
        }
    }
    return failures;
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') { return 77; }
    int failures = 0;
    for (const auto mode : {ninfer::ContextCacheMode::Hybrid, ninfer::ContextCacheMode::Legacy}) {
        failures += run_with(artifact, mode);
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
