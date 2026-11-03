#pragma once

// The Engram of DeepSeek-V4.1-Flash: an n-gram hash of each position's last tokens addresses rows
// of a table per Engram block, which the block's projection turns into keys and a value added to
// the stream. The tables are described by the artifact's `engram` component (their rows, format and
// the SHA-256 of their bytes) and stored either in the same artifact or in a separate table
// artifact (an `engram` component alone) that any model naming the same digest can use; one table
// holds every Engram block's rows, block after block. Nothing reads the rows at load: the executor
// reads the rows a pass addresses through qwen4_exp::NgramTableReader, from the files or from RAM.

#include "artifact/reader.h"
#include "models/deepseek_v41/config.h"
#include "models/qwen4_exp/ngram_table.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ninfer::models::deepseek_v41 {

inline constexpr std::uint32_t kEngramRowBytes = 144; // 256 IQ4_NL values

// The table rows of the model artifact `reader` opened at `artifact`: the rows of the table
// artifact at `table` when one is given, otherwise the model's own. Refuses a table whose digest,
// format or geometry differ from the model's, and a model that stores no rows when `table` is empty.
[[nodiscard]] qwen4_exp::NgramTableLayout engram_table_layout(const artifact::Reader& reader,
                                                              const std::filesystem::path& artifact,
                                                              const TextConfig& config,
                                                              const std::filesystem::path& table = {});

// True when the artifact holds an Engram table and no model.
[[nodiscard]] bool is_engram_table_artifact(const artifact::Reader& reader);

// The hash ids of a sequence's positions. Each position's n-gram of n tokens (n = 2 ..
// ngram_size) is the compressed ids of the token and the n - 1 before it (the pad id before the
// sequence starts), multiplied by the block's multipliers and folded by XOR; a column's row is that
// value modulo the column's prime, past the column's offset and the block's first row in the table.
// A dead position (an image's) takes no part in an n-gram: its own n-grams are the pad id's, and
// a later position's look-back stops at it, the pad id from there back (engram.py's DEAD).
class EngramHasher {
public:
    static constexpr std::int32_t kDead = -1;

    EngramHasher(const EngramHash& hash, std::vector<std::int32_t> token_map);

    [[nodiscard]] const EngramHash& hash() const noexcept { return hash_; }
    [[nodiscard]] std::uint32_t columns() const noexcept { return hash_.columns(); }
    // Rows of every Engram block for one position: blocks * columns.
    [[nodiscard]] std::uint32_t rows_per_token() const noexcept {
        return std::uint32_t(hash_.layers.size()) * hash_.columns();
    }
    [[nodiscard]] std::int32_t compressed(std::int32_t token) const;

    // The compressed ids of the last ngram_size - 1 tokens a sequence holds (kDead for a dead
    // position), the latest last, and how many there are (fewer at its start).
    struct History {
        std::int32_t ids[8] = {};
        std::uint32_t count = 0;
        void push(std::int32_t compressed_id, std::uint32_t keep);
    };

    // The table rows of `tokens` continuing `history`: out[t][block * columns + column] (table
    // rows, each block's first row added). `history` advances past the tokens when `advance`.
    // dead[t] nonzero marks token t dead (empty: none is).
    void rows(History& history, std::span<const std::int32_t> tokens, std::uint64_t* out,
              bool advance = true, std::span<const std::uint8_t> dead = {}) const;

private:
    EngramHash hash_;
    std::vector<std::int32_t> token_map_;
    std::vector<std::uint64_t> layer_base_; // first table row of each block
};

} // namespace ninfer::models::deepseek_v41
