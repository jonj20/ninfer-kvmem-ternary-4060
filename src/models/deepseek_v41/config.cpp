#include "models/deepseek_v41/config.h"

#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::deepseek_v41 {
namespace {

using artifact::ArtifactError;
using artifact::Json;

const Json& member(const Json& config, const char* name) {
    if (!config.contains(name)) {
        throw ArtifactError(std::string("DeepSeek-V4.1 config: missing ") + name);
    }
    return config.at(name);
}

std::uint32_t integer(const Json& config, const char* name, bool positive = true) {
    const auto n = artifact::require_u64(member(config, name), name, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(name) + ": exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

float real(const Json& config, const char* name) {
    const Json& value = member(config, name);
    if (!value.is_number()) {
        throw ArtifactError(std::string("DeepSeek-V4.1 config: ") + name + " must be a real value");
    }
    const float out = value.get<float>();
    if (!std::isfinite(out) || out <= 0) {
        throw ArtifactError(std::string("DeepSeek-V4.1 config: ") + name + " is out of range");
    }
    return out;
}

std::vector<std::uint64_t> u64_list(const Json& value, const char* name) {
    if (!value.is_array()) {
        throw ArtifactError(std::string("DeepSeek-V4.1 config: ") + name + " must be a list");
    }
    std::vector<std::uint64_t> out;
    for (const auto& item : value) { out.push_back(artifact::require_u64(item, name)); }
    return out;
}

std::vector<bool> layer_set(const Json& config, const char* name, std::uint32_t layers) {
    std::vector<bool> out(layers, false);
    for (const auto id : u64_list(member(config, name), name)) {
        if (id >= layers || out[id]) {
            throw ArtifactError(std::string("DeepSeek-V4.1 config: ") + name +
                                " must list distinct blocks");
        }
        out[id] = true;
    }
    return out;
}

} // namespace

TextConfig parse_text_config(const Json& value) {
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "DeepseekV41ForCausalLM" ||
        value.value("model_type", "") != "deepseek_v41") {
        throw ArtifactError("DeepSeek-V4.1 text config: architecture/model_type mismatch");
    }
    TextConfig out;
    out.hidden_size             = integer(value, "hidden_size");
    out.vocab_size              = integer(value, "vocab_size");
    out.num_hidden_layers       = integer(value, "num_hidden_layers");
    out.max_position_embeddings = integer(value, "max_position_embeddings");
    out.rms_norm_eps            = real(value, "rms_norm_eps");
    out.hc_mult                 = integer(value, "hc_mult");
    out.hc_sinkhorn_iters       = integer(value, "hc_sinkhorn_iters");
    out.hc_eps                  = real(value, "hc_eps");
    out.num_attention_heads     = integer(value, "num_attention_heads");
    out.head_dim                = integer(value, "head_dim");
    out.rope_dim                = integer(value, "qk_rope_head_dim");
    out.q_lora_rank             = integer(value, "q_lora_rank");
    out.o_lora_rank             = integer(value, "o_lora_rank");
    out.o_groups                = integer(value, "o_groups");
    out.sliding_window          = integer(value, "sliding_window");
    out.rope_theta              = real(value, "rope_theta");
    out.compress_rope_theta     = real(value, "compress_rope_theta");
    const Json& yarn            = member(value, "rope_scaling");
    out.yarn_factor             = real(yarn, "factor");
    out.yarn_beta_fast          = real(yarn, "beta_fast");
    out.yarn_beta_slow          = real(yarn, "beta_slow");
    out.yarn_original           = integer(yarn, "original_max_position_embeddings");
    const auto layers           = out.num_hidden_layers;
    for (const auto ratio : u64_list(member(value, "compress_ratios"), "compress_ratios")) {
        out.compress_ratios.push_back(static_cast<std::uint32_t>(ratio));
    }
    out.kv_source    = layer_set(value, "kv_source_layer_ids", layers);
    out.index_source = layer_set(value, "index_source_layer_ids", layers);
    if (!member(value, "candidate_source_layer_id").is_number_integer()) {
        throw ArtifactError("DeepSeek-V4.1 config: candidate_source_layer_id must be an integer");
    }
    out.candidate_source     = value.at("candidate_source_layer_id").get<std::int32_t>();
    out.index_heads          = integer(value, "index_n_heads");
    out.index_head_dim       = integer(value, "index_head_dim");
    out.index_topk           = integer(value, "index_topk");
    out.candidate_blocks     = integer(value, "candidate_topk_blocks");
    out.candidate_block_size = integer(value, "candidate_block_size");
    out.num_experts           = integer(value, "n_routed_experts");
    out.top_k_experts         = integer(value, "num_experts_per_tok");
    out.moe_intermediate_size = integer(value, "moe_intermediate_size");
    out.routed_scaling_factor = real(value, "routed_scaling_factor");
    out.swiglu_limit          = real(value, "swiglu_limit");

    auto& engram = out.engram;
    for (const auto id : u64_list(member(value, "engram_layer_ids"), "engram_layer_ids")) {
        if (id >= layers) { throw ArtifactError("DeepSeek-V4.1 config: engram block out of range"); }
        engram.layers.push_back(static_cast<std::uint32_t>(id));
    }
    engram.layer_rows = u64_list(member(value, "engram_num_embeddings"), "engram_num_embeddings");
    engram.ngram_size = integer(value, "engram_max_ngram_size");
    engram.heads      = integer(value, "engram_n_heads");
    engram.head_dim   = integer(value, "engram_head_dim");
    const Json& hash  = member(value, "engram");
    engram.pad_id     = static_cast<std::int32_t>(integer(hash, "pad_id", false));
    for (const char* name : {"multipliers", "primes", "offsets"}) {
        const Json& rows = member(hash, name);
        if (!rows.is_array() || rows.size() != engram.layers.size()) {
            throw ArtifactError(std::string("DeepSeek-V4.1 config: engram.") + name +
                                " must have a row per Engram block");
        }
        auto& target = std::string(name) == "multipliers" ? engram.multipliers
                       : std::string(name) == "primes"    ? engram.primes
                                                          : engram.offsets;
        for (const auto& row : rows) { target.push_back(u64_list(row, name)); }
    }

    // The kernels implement this release's geometry: 512-value latents with 64 rotated values,
    // eight output groups of eight heads, four streams, at most 32 indexer heads of 128 values, a
    // window of at most 128 positions, at most eight of 512 experts, and Engram rows of 256 values.
    bool ok = out.head_dim == 512 && out.rope_dim == 64 && out.hc_mult == 4 &&
              out.num_attention_heads == 64 && out.o_groups == 8 &&
              out.hidden_size % 256 == 0 && out.q_lora_rank % 256 == 0 &&
              out.index_head_dim == 128 && out.index_heads <= 32 && out.index_topk <= 1024 &&
              out.index_topk > 0 && out.sliding_window > 0 && out.sliding_window <= 128 &&
              out.num_experts <= 512 && out.top_k_experts > 0 && out.top_k_experts <= 8 &&
              out.top_k_experts <= out.num_experts && out.moe_intermediate_size % 256 == 0 &&
              out.candidate_block_size > 0 && out.compress_ratios.size() == layers &&
              out.kv_source.size() == layers && out.index_source.size() == layers;
    for (std::uint32_t i = 0; ok && i < layers; ++i) {
        const auto ratio = out.compress_ratios[i];
        ok = ratio <= 2;
        // A source block compresses (or selects for) its own ratio; a reading block takes the
        // latest source's cache and selection, of the same ratio.
        if (ratio != 0) {
            const int kv = out.kv_source_of(i), ix = out.index_source_of(i);
            ok = ok && kv >= 0 && ix >= kv && out.compress_ratios[std::size_t(kv)] == ratio &&
                 out.compress_ratios[std::size_t(ix)] == ratio;
        } else {
            ok = ok && !out.kv_source[i] && !out.index_source[i];
        }
    }
    if (out.candidate_source >= 0) {
        ok = ok && std::uint32_t(out.candidate_source) < layers &&
             out.index_source[std::size_t(out.candidate_source)];
    }
    ok = ok && engram.head_dim == 256 && engram.ngram_size >= 2 && engram.heads > 0 &&
         engram.layer_rows.size() == engram.layers.size();
    for (std::size_t l = 0; ok && l < engram.layers.size(); ++l) {
        ok = engram.multipliers[l].size() == engram.ngram_size &&
             engram.primes[l].size() == engram.columns() &&
             engram.offsets[l].size() == engram.columns();
        for (std::uint32_t c = 0; ok && c < engram.columns(); ++c) {
            ok = engram.primes[l][c] > 0 &&
                 engram.offsets[l][c] + engram.primes[l][c] <= engram.layer_rows[l];
        }
    }
    if (!ok) { throw ArtifactError("DeepSeek-V4.1 config: geometry outside the implemented kernels"); }
    return out;
}

DSparkConfig parse_dspark_config(const Json& value, const TextConfig& text) {
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "DeepseekV41DSpark" ||
        value.value("model_type", "") != "deepseek_v41_dspark") {
        throw ArtifactError("DeepSeek-V4.1 DSpark config: architecture/model_type mismatch");
    }
    DSparkConfig out;
    out.layers      = integer(value, "num_hidden_layers");
    out.block_size  = integer(value, "block_size");
    out.markov_rank = integer(value, "markov_rank");
    out.experts     = integer(value, "n_routed_experts");
    out.top_k       = integer(value, "num_experts_per_tok");
    out.noise_token = static_cast<std::int32_t>(integer(value, "noise_token_id", false));
    for (const auto id : u64_list(member(value, "target_layer_ids"), "target_layer_ids")) {
        if (id >= text.num_hidden_layers ||
            (!out.target_layers.empty() && id <= out.target_layers.back())) {
            throw ArtifactError("DeepSeek-V4.1 DSpark config: target_layer_ids must name backbone "
                                "blocks in order");
        }
        out.target_layers.push_back(static_cast<std::uint32_t>(id));
    }
    if (value.contains("stq")) {
        const auto& stq = value.at("stq");
        if (!stq.is_object() || stq.size() != 2 || stq.value("trunk", "") != "spec1_w15_n64" ||
            stq.value("experts", "") != "pair1_w16_n32") {
            throw ArtifactError("DeepSeek-V4.1 DSpark config: STQ drafters store SPEC1 (64 words) "
                                "trunk and PAIR1 (32 words) expert tiles");
        }
        out.stq = true;
    }
    // A draft's rows are one decode pass of the vector kernels (at most eight), its Markov rank
    // a warp's eight values per lane.
    const bool ok = out.block_size >= 1 && out.block_size <= 8 && out.markov_rank == 256 &&
                    out.experts <= 512 && out.top_k >= 1 && out.top_k <= 8 &&
                    out.top_k <= out.experts && !out.target_layers.empty() &&
                    std::uint32_t(out.noise_token) < text.vocab_size;
    if (!ok) {
        throw ArtifactError("DeepSeek-V4.1 DSpark config: geometry outside the implemented kernels");
    }
    return out;
}

VisionConfig parse_vision_config(const Json& value, const TextConfig& text) {
    const auto& architectures = member(value, "architectures");
    if (!architectures.is_array() || architectures.size() != 1 ||
        architectures[0] != "DeepseekV41VisionModel" ||
        value.value("model_type", "") != "deepseek_v41_vision") {
        throw ArtifactError("DeepSeek-V4.1 vision config: architecture/model_type mismatch");
    }
    VisionConfig out;
    out.num_hidden_layers   = integer(value, "num_hidden_layers");
    out.hidden_size         = integer(value, "hidden_size");
    out.num_attention_heads = integer(value, "num_attention_heads");
    out.intermediate_size   = integer(value, "intermediate_size");
    out.patch_size          = integer(value, "patch_size");
    out.downsample_ratio    = integer(value, "downsample_ratio");
    out.max_image_tokens    = integer(value, "max_image_tokens");
    out.min_pixels          = integer(value, "min_pixels");
    out.projection_dim      = integer(value, "projection_dim");
    out.rope_theta          = real(value, "rope_theta");
    out.rms_norm_eps        = real(value, "rms_norm_eps");
    out.image_token_id      = static_cast<std::int32_t>(integer(value, "image_token_id", false));
    // The tower's kernels: heads of 64 values (16 per rotation axis), rows of whole 8-value
    // vectors; a span of at least one aligner row.
    const bool ok = out.hidden_size % out.num_attention_heads == 0 && out.head_dim() == 64 &&
                    out.hidden_size % 8 == 0 && out.intermediate_size % 8 == 0 &&
                    out.projection_dim == text.hidden_size &&
                    std::uint32_t(out.image_token_id) < text.vocab_size &&
                    out.max_image_tokens >= VisionConfig::span_tokens(1, 1) &&
                    value.contains("max_wh_ratio") && value.at("max_wh_ratio").is_null();
    if (!ok) {
        throw ArtifactError("DeepSeek-V4.1 vision config: geometry outside the implemented kernels");
    }
    return out;
}

} // namespace ninfer::models::deepseek_v41
