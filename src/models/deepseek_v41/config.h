#pragma once

// The text config of DeepSeek-V4.1-Flash (DeepseekV41ForCausalLM) as tools/convert/deepseek_v41.py
// writes it into the artifact. The residual stream is hc_mult copies mixed by hyper-connections
// (Sinkhorn-normalized combinations). Each block's attention is one latent of head_dim values
// shared as key and value by every query head, its last rope_dim values rotated (adjacent pairs):
// a sliding window of the last `sliding_window` positions, and in the blocks with a compression
// ratio, compressed latents that an indexer selects (top index_topk). A kv source block compresses
// its input (ratio 1: one latent per position; ratio 2: a softmax-gated pool of two) into the
// compressed cache and the indexer's keys that the blocks after it read; an index source block
// selects the positions the blocks after it attend to, and the candidate source block's selection
// also bounds the later index sources to its candidate blocks. The FFN is a routed MoE (sqrt-softplus
// scores, chosen with a bias, renormalized and scaled) beside one shared expert, both SwiGLU with
// the gate clamped above and the up projection both ways at swiglu_limit. Engram blocks add rows of
// an n-gram hash table to the stream before their attention, each copy gated by its match.

#include "artifact/schema.h"

#include <array>
#include <cstdint>
#include <vector>

namespace ninfer::models::deepseek_v41 {

// The Engram hash of the release: per Engram block a multiplier per look-back (the token itself
// first), and per column (n-gram of n tokens and head: column (n - 2) * heads + head) the prime
// that bounds its hash and the row its range starts at in the block's table.
struct EngramHash {
    std::vector<std::uint32_t> layers;         // blocks with an Engram
    std::vector<std::uint64_t> layer_rows;     // table rows of each
    std::uint32_t ngram_size = 0, heads = 0, head_dim = 0;
    std::int32_t pad_id      = 0;             // compressed id of the pad token
    std::vector<std::vector<std::uint64_t>> multipliers, primes, offsets; // [engram layer][...]

    [[nodiscard]] std::uint32_t columns() const noexcept { return (ngram_size - 1) * heads; }
    // Index of `layer` among the Engram blocks, or -1.
    [[nodiscard]] int index(std::uint32_t layer) const noexcept {
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (layers[i] == layer) { return int(i); }
        }
        return -1;
    }
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    std::uint32_t hc_mult = 0, hc_sinkhorn_iters = 0;
    float hc_eps = 0;
    // Attention.
    std::uint32_t num_attention_heads = 0, head_dim = 0, rope_dim = 0, q_lora_rank = 0,
                  o_lora_rank = 0, o_groups = 0, sliding_window = 0;
    float rope_theta = 0, compress_rope_theta = 0;
    float yarn_factor = 0, yarn_beta_fast = 0, yarn_beta_slow = 0;
    std::uint32_t yarn_original = 0;
    std::vector<std::uint32_t> compress_ratios; // per block: 0 (window only), 1 or 2
    std::vector<bool> kv_source, index_source;  // per block
    std::int32_t candidate_source = -1;
    std::uint32_t index_heads = 0, index_head_dim = 0, index_topk = 0, candidate_blocks = 0,
                  candidate_block_size = 0;
    // MoE.
    std::uint32_t num_experts = 0, top_k_experts = 0, moe_intermediate_size = 0;
    float routed_scaling_factor = 0, swiglu_limit = 0;
    EngramHash engram;

    // The latest kv source / index source at or before `layer`, or -1.
    [[nodiscard]] int kv_source_of(std::uint32_t layer) const noexcept {
        for (int i = int(layer); i >= 0; --i) {
            if (kv_source[std::size_t(i)]) { return i; }
        }
        return -1;
    }
    [[nodiscard]] int index_source_of(std::uint32_t layer) const noexcept {
        for (int i = int(layer); i >= 0; --i) {
            if (index_source[std::size_t(i)]) { return i; }
        }
        return -1;
    }
    [[nodiscard]] std::uint32_t hc_mixes() const noexcept { return (2 + hc_mult) * hc_mult; }
};

// The DSpark drafter (the `dspark` component): `layers` blocks of the backbone's shape (window
// attention only) whose rings hold the backbone's streams entering target_layers, averaged, through
// a projection and its norm. A draft runs a block of block_size rows, the anchor token then the
// noise token, at the positions after the ring's; each row's logits are the backbone's head over
// the final norm of its collapsed streams plus a Markov bias of the token before it (rank
// markov_rank), and a confidence head reads the collapsed row and that token's Markov embedding.
// Its MoE routes to top_k of `experts`. Its matrices are the GGUF blocks of a llama.cpp drafter, or
// with `stq` the STQ matrices of an STQ pack's drafter: SPEC1 (64 words a tile) in the projection and
// the blocks' trunk, PAIR1 (32 words) in their routed experts.
struct DSparkConfig {
    std::uint32_t layers = 0, block_size = 0, markov_rank = 0, experts = 0, top_k = 0;
    std::int32_t noise_token = 0;
    std::vector<std::uint32_t> target_layers;
    bool stq = false;
};

// The vision tower (the `vision` component): a ViT of num_hidden_layers blocks of hidden_size over
// patches of patch_size pixels (2D-rotated heads of hidden_size / num_attention_heads values, a
// SiLU-gated MLP of intermediate_size), then an aligner over downsample_ratio x downsample_ratio
// cells of the patch grid into the text model's width. An image of n_h x n_w aligner rows takes
// span_tokens(n_h, n_w) prompt positions: the start row, each grid row's rows and a newline row,
// the end row, every one carrying image_token_id. max_image_tokens bounds a span and min_pixels
// the image's area before planning (image_processor.py).
struct VisionConfig {
    std::uint32_t num_hidden_layers = 0, hidden_size = 0, num_attention_heads = 0,
                  intermediate_size = 0, patch_size = 0, downsample_ratio = 0,
                  max_image_tokens = 0, min_pixels = 0, projection_dim = 0;
    float rope_theta = 0, rms_norm_eps = 0;
    std::int32_t image_token_id = 0;

    [[nodiscard]] std::uint32_t head_dim() const noexcept { return hidden_size / num_attention_heads; }
    [[nodiscard]] std::uint32_t patch_features() const noexcept { return 3 * patch_size * patch_size; }
    [[nodiscard]] std::uint32_t merged_features() const noexcept {
        return hidden_size * downsample_ratio * downsample_ratio;
    }
    [[nodiscard]] static std::uint32_t span_tokens(std::uint32_t rows, std::uint32_t columns) noexcept {
        return rows * (columns + 1) + 2;
    }
};

// Throws artifact::ArtifactError naming the field that is missing, malformed or outside the
// implemented geometry.
[[nodiscard]] TextConfig parse_text_config(const artifact::Json& value);
[[nodiscard]] DSparkConfig parse_dspark_config(const artifact::Json& value, const TextConfig& text);
[[nodiscard]] VisionConfig parse_vision_config(const artifact::Json& value, const TextConfig& text);

} // namespace ninfer::models::deepseek_v41
