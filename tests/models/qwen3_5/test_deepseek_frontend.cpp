// DeepSeek-V4.1 in the shared frontend.
//
// Pre-tokenizer: the word ends of DeepSeek's three splits (Flint's digits and CJK runs, then its
// own words) against Hugging Face tokenizers' pre-tokenization of the same texts
// (tests/fixtures/deepseek_v41/pretokenizer_cases.json, generated from the release's
// tokenizer.json, whose pre_tokenizer is tests/fixtures/deepseek_v41/pre_tokenizer.json). A byte
// vocabulary without merges makes a token of every byte, so the stable frontier of a boundary at
// every byte is the last word end at or before it.
//
// Chat template: tools/convert/templates/deepseek_v41/chat_template.jinja renders every
// conversation of tests/fixtures/deepseek_v41/chat_cases.json as the release's
// encoding/encoding.py encodes it (the fixture's `expected`, written by that script), thinking or
// not, at each reasoning effort, with and without preserved reasoning, with images between text
// blocks; a thinking prompt opens
// reasoning, and a forced call opens its DSML invoke after a closed reasoning block.
//
// Prompt and output: a Frontend over a byte vocabulary with DeepSeek's marker tokens tokenizes the
// prompt with those tokens, routes the text before the model's </think> token to reasoning and
// the rest to content, and returns DSML calls as structured calls.
//
// Images: the processor plans an image's grid, resizes it and writes its patches as the release's
// image_processor.py does (tests/fixtures/deepseek_v41/image_cases.json, written from that file and
// Pillow: plans, Pillow's bicubic resize and ImageOps.pad of a synthetic picture, and the patches
// of examples/cli/media/natural_scene.png as tools/reference/deepseek_v41_vision.py derives them),
// and a prompt's image becomes its span of <｜deepseek_image｜> tokens.
//
// Tool calls: DeepSeekToolCallDecoder pulls DSML blocks out of streamed content (markers split at
// every byte), types values by their string flag, returns an unreadable or incomplete block to
// content, and starts a forced call inside its invoke.

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/deepseek_output.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/processor.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/test_access.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"
#include "models/registry.h"
#include "text/structured_output.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#    error "NINFER_SOURCE_DIR is required"
#endif

namespace {

namespace fi = ninfer::models::qwen3_5::frontend;
using Json   = nlohmann::json;
// The fixture's conversations keep their key order, as the renderer keeps a request's.
using OrderedJson = nlohmann::ordered_json;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

std::string read_file(const std::string& relative) {
    std::ifstream file(std::string(NINFER_SOURCE_DIR) + "/" + relative, std::ios::binary);
    if (!file) { throw std::runtime_error("missing " + relative); }
    std::ostringstream out;
    out << file.rdbuf();
    return out.str();
}

std::string read_fixture(const std::string& name) {
    return read_file("tests/fixtures/deepseek_v41/" + name);
}

std::string chat_template() {
    return read_file("tools/convert/templates/deepseek_v41/chat_template.jinja");
}

constexpr std::string_view kBos = "<｜begin▁of▁sentence｜>";
constexpr std::string_view kEos = "<｜end▁of▁sentence｜>";

// The marker tokens after the 256 bytes: the release's specials (BOS, EOS) and its plain added
// tokens (roles and reasoning).
struct Marker {
    std::string_view content;
    bool special;
    bool normalized = !special;
};

constexpr Marker kMarkers[] = {{kBos, true},
                               {kEos, true},
                               {"<｜System｜>", false},
                               {"<｜User｜>", false},
                               {"<｜Assistant｜>", false},
                               {"<think>", false},
                               {"</think>", false},
                               {"<｜deepseek_image｜>", false, false}};
constexpr int kBosId = 256, kEosId = 257, kSystemId = 258, kUserId = 259, kAssistantId = 260,
              kThinkOpenId = 261, kThinkCloseId = 262, kImageId = 263;

// GPT-2's byte-to-symbol table, which byte-level BPE vocabularies are written in.
std::string byte_level_symbol(std::uint8_t byte) {
    const auto visible = [](int b) {
        return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    };
    int code = byte;
    if (!visible(byte)) {
        int next = 256;
        for (int b = 0; b < byte; ++b) {
            if (!visible(b)) { ++next; }
        }
        code = next;
    }
    std::string out;
    if (code < 0x80) {
        out.push_back(static_cast<char>(code));
    } else {
        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
    return out;
}

struct TokenizerFiles {
    std::string tokenizer_json, tokenizer_config_json, generation_config_json;
};

// A byte vocabulary without merges behind the release's pipeline (an empty normalizer Sequence,
// its pre-tokenizer, ByteLevel decoding), with or without its marker tokens. As in the release,
// the specials are also vocabulary entries and the plain markers are matched in normalized text.
TokenizerFiles byte_tokenizer_files(bool markers) {
    Json vocab = Json::object();
    for (int byte = 0; byte < 256; ++byte) {
        vocab[byte_level_symbol(static_cast<std::uint8_t>(byte))] = byte;
    }
    Json added   = Json::array();
    Json decoder = Json::object();
    if (markers) {
        int id = 256;
        for (const Marker& marker : kMarkers) {
            const Json entry = {{"content", std::string(marker.content)},
                                {"single_word", false},
                                {"lstrip", false},
                                {"rstrip", false},
                                {"normalized", marker.normalized},
                                {"special", marker.special}};
            Json token       = entry;
            token["id"]      = id;
            added.push_back(token);
            if (marker.special) { vocab[std::string(marker.content)] = id; }
            decoder[std::to_string(id)] = entry;
            ++id;
        }
    }
    TokenizerFiles files;
    files.tokenizer_json = Json{
        {"normalizer", {{"type", "Sequence"}, {"normalizers", Json::array()}}},
        {"pre_tokenizer", Json::parse(read_fixture("pre_tokenizer.json"))},
        {"decoder",
         {{"type", "ByteLevel"},
          {"add_prefix_space", true},
          {"trim_offsets", true},
          {"use_regex", true}}},
        {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", Json::array()}}},
        {"added_tokens",
         added}}.dump();
    files.tokenizer_config_json = Json{
        {"added_tokens_decoder", decoder},
        {"add_bos_token", false},
        {"bos_token", std::string(kBos)},
        {"eos_token",
         std::string(kEos)}}.dump();
    files.generation_config_json =
        markers ? Json{{"bos_token_id", kBosId}, {"eos_token_id", kEosId}}.dump()
                : std::string(R"({"eos_token_id":0})");
    return files;
}

fi::Tokenizer byte_tokenizer() {
    const TokenizerFiles files = byte_tokenizer_files(false);
    return fi::Tokenizer({.tokenizer_json         = files.tokenizer_json,
                          .tokenizer_config_json  = files.tokenizer_config_json,
                          .generation_config_json = files.generation_config_json});
}

// The word ends of `text`: the distinct stable frontiers of a boundary at every byte.
std::vector<std::size_t> word_ends(const fi::Tokenizer& tokenizer, const std::string& text) {
    std::vector<std::size_t> boundaries(text.size() + 1);
    for (std::size_t i = 0; i <= text.size(); ++i) { boundaries[i] = i; }
    const fi::BoundaryEncodedText encoded = tokenizer.encode_with_boundaries(text, boundaries);
    std::set<std::size_t> ends;
    for (const auto& boundary : encoded.boundaries) {
        if (boundary.stable_frontier != 0) { ends.insert(boundary.stable_frontier); }
    }
    return {ends.begin(), ends.end()};
}

// The release's added tokens load: specials that repeat their vocabulary entries, plain markers
// matched in normalized text under a normalizer that changes nothing. A pipeline that normalizes
// keeps refusing the latter, and an added token on another token's id stays refused.
void test_added_tokens() {
    const TokenizerFiles files = byte_tokenizer_files(true);
    const auto load            = [&](const std::string& tokenizer_json) {
        return fi::Tokenizer({.tokenizer_json         = tokenizer_json,
                              .tokenizer_config_json  = files.tokenizer_config_json,
                              .generation_config_json = files.generation_config_json});
    };
    const fi::Tokenizer tokenizer = load(files.tokenizer_json);
    expect(tokenizer.encode(std::string(kBos) + "a</think>") ==
               std::vector<int>{kBosId, 'a', kThinkCloseId},
           "markers encode as their tokens");
    expect(tokenizer.decode(std::vector<int>{kBosId, 'a', kThinkCloseId}) ==
               std::string(kBos) + "a</think>",
           "markers decode as their text");
    const auto refused = [&](Json root, const std::string& what) {
        try {
            (void)load(root.dump());
            expect(false, what);
        } catch (const std::invalid_argument&) {}
    };
    Json nfc          = Json::parse(files.tokenizer_json);
    nfc["normalizer"] = {{"type", "NFC"}};
    refused(nfc, "a normalized added token under NFC");
    Json clash = Json::parse(files.tokenizer_json);
    clash["model"]["vocab"].erase(std::string(kBos));
    clash["model"]["vocab"]["clash"] = kBosId;
    refused(clash, "an added token on another vocabulary entry's id");
}

void test_pretokenizer() {
    const fi::Tokenizer tokenizer = byte_tokenizer();
    const Json cases              = Json::parse(read_fixture("pretokenizer_cases.json"));
    for (const Json& item : cases) {
        const std::string text = item.at("text").get<std::string>();
        const auto expected    = item.at("word_ends").get<std::vector<std::size_t>>();
        const auto actual      = word_ends(tokenizer, text);
        if (actual != expected) {
            std::ostringstream got;
            for (const std::size_t end : actual) { got << end << ' '; }
            expect(false, "word ends of " + Json(text).dump() + ": " + got.str());
        }
    }
    // The empty normalizer Sequence normalizes nothing: a decomposed accent stays decomposed.
    const std::string decomposed = "e\xcc\x81";
    expect(tokenizer.encode(decomposed).size() == decomposed.size(),
           "no NFC before DeepSeek's BPE");
}

void test_run_reuse() {
    // A long run that continues a kept one encodes as a fresh tokenizer encodes it: reused word
    // ends are those that a number, CJK, punctuation or a symbol after them decides.
    std::string prefix;
    while (prefix.size() < 6000) { prefix += "word, more.text; _x -y "; }
    prefix += "\xe0\xb9\x84\xe0\xb8\x97\xe0\xb8\xa2"
              "ABCdef";
    const fi::Tokenizer warm = byte_tokenizer();
    (void)word_ends(warm, prefix + " tail, end.");
    const fi::Tokenizer cold = byte_tokenizer();
    for (const std::string& next :
         {prefix + "k tail, end.", prefix + ".tail", prefix.substr(0, 3000) + "x continuation."}) {
        expect(word_ends(warm, next) == word_ends(cold, next), "reused word ends of a kept run");
    }
}

std::optional<ninfer::ReasoningEffort> effort(const OrderedJson& item) {
    if (!item.contains("reasoning_effort")) { return std::nullopt; }
    const std::string name = item["reasoning_effort"];
    if (name == "low") { return ninfer::ReasoningEffort::Low; }
    if (name == "high") { return ninfer::ReasoningEffort::High; }
    if (name == "max") { return ninfer::ReasoningEffort::Max; }
    throw std::runtime_error("fixture effort " + name);
}

std::vector<fi::ChatMessage> messages_of(const OrderedJson& item) {
    std::vector<fi::ChatMessage> messages;
    for (const OrderedJson& source : item.at("messages")) {
        fi::ChatMessage message;
        const std::string role    = source.at("role");
        message.role              = role == "system"      ? ninfer::ChatRole::System
                                    : role == "user"      ? ninfer::ChatRole::User
                                    : role == "assistant" ? ninfer::ChatRole::Assistant
                                                          : ninfer::ChatRole::Tool;
        const OrderedJson content = source.value("content", OrderedJson(""));
        if (content.is_array()) { // text and image blocks
            for (const OrderedJson& block : content) {
                message.parts.push_back(block.at("type") == "image"
                                            ? fi::ChatPart::image(fi::MediaData{})
                                            : fi::ChatPart::text_part(block.at("text")));
            }
        } else if (!content.get<std::string>().empty()) {
            message.parts.push_back(fi::ChatPart::text_part(content.get<std::string>()));
        }
        message.reasoning_content = source.value("reasoning_content", "");
        message.tool_call_id      = source.value("tool_call_id", "");
        for (const OrderedJson& call : source.value("tool_calls", OrderedJson::array())) {
            message.tool_calls.push_back({.id             = call.at("id"),
                                          .name           = call.at("name"),
                                          .arguments_json = call.at("arguments").dump()});
        }
        messages.push_back(std::move(message));
    }
    return messages;
}

fi::CompiledChatTemplate compiled_template() {
    return fi::CompiledChatTemplate::resolve(
        chat_template(), "deepseek_v41",
        Json{{"bos_token", std::string(kBos)}, {"eos_token", std::string(kEos)}});
}

void test_chat_template() {
    const auto compiled     = compiled_template();
    const OrderedJson cases = OrderedJson::parse(read_fixture("chat_cases.json"));
    for (const OrderedJson& item : cases) {
        fi::ChatRenderOptions options;
        options.tool_call_syntax  = fi::ToolCallSyntax::DeepSeek; // the frontend's choice
        options.enable_thinking   = item.at("thinking").get<bool>();
        options.reasoning_effort  = effort(item);
        options.preserve_thinking = item.value("preserve_thinking", false);
        for (const OrderedJson& tool : item.value("tools", OrderedJson::array())) {
            options.tool_jsons.push_back(tool.dump());
        }
        const std::string label =
            item.at("name").get<std::string>() +
            (*options.enable_thinking ? " thinking" : " chat") +
            (item.contains("reasoning_effort") ? " " + item["reasoning_effort"].get<std::string>()
                                               : "") +
            (*options.preserve_thinking ? " preserved" : "");
        try {
            const fi::RenderedChat rendered = compiled.render(messages_of(item), options);
            const std::string expected      = item.at("expected");
            expect(rendered.text == expected,
                   label + " renders\n" + rendered.text + "\ninstead of\n" + expected);
            // A thinking generation prompt is a reasoning block the model continues.
            const bool opens = std::string_view(expected).ends_with("<｜Assistant｜><think>");
            expect(rendered.starts_in_reasoning == opens, label + ": starts_in_reasoning");
        } catch (const std::exception& error) { expect(false, label + ": " + error.what()); }
    }
}

void test_efforts_and_forced_call() {
    const auto compiled = compiled_template();
    std::vector<fi::ChatMessage> messages(1);
    messages[0].role = ninfer::ChatRole::User;
    messages[0].parts.push_back(fi::ChatPart::text_part("Hi"));
    fi::ChatRenderOptions options;
    options.tool_call_syntax = fi::ToolCallSyntax::DeepSeek;
    const auto budget        = [&](ninfer::ReasoningEffort value) {
        options.reasoning_effort = value;
        const std::string text   = compiled.render(messages, options).text;
        const std::size_t at     = text.find("Reasoning Effort: ");
        return at == std::string::npos ? std::string()
                                       : text.substr(at + 18, text.find(' ', at + 18) - at - 18);
    };
    // The protocol's efforts become the template's nearest, ties rounding up.
    expect(budget(ninfer::ReasoningEffort::Minimal) == "50", "minimal effort renders low");
    expect(budget(ninfer::ReasoningEffort::Medium) == "75", "medium effort renders high");
    expect(budget(ninfer::ReasoningEffort::XHigh) == "100", "xhigh effort renders max");
    expect(budget(ninfer::ReasoningEffort::None).empty(), "effort none disables thinking");
    options.reasoning_effort = std::nullopt;

    // A forced call: thinking off, the calls block and the invoke tag after the closed reasoning.
    options.enable_thinking = false;
    options.tool_jsons      = {
        R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}})"};
    const std::string free        = compiled.render(messages, options).text;
    options.forced_tool_name      = "get_weather";
    const fi::RenderedChat forced = compiled.render(messages, options);
    expect(std::string_view(free).ends_with("<｜Assistant｜></think>"),
           "chat-mode prompt: " + free);
    expect(forced.text == free + "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n",
           "forced call opener: " + forced.text);
    expect(!forced.starts_in_reasoning, "a forced call does not reason");
}

std::shared_ptr<const fi::ToolCallOutputContract> weather_contract(std::string_view forced = {}) {
    const std::vector<std::string> tools = {
        R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"options":{"type":"object"}}}}})"};
    return fi::build_tool_call_output_contract(tools, true, forced);
}

fi::DeepSeekToolCallDecoder::Terminal run(const std::string& text, std::string& streamed,
                                          std::string_view forced = {}) {
    fi::DeepSeekToolCallDecoder decoder(weather_contract(forced));
    for (const char byte : text) { streamed += decoder.feed(std::string(1, byte)); }
    auto terminal = decoder.finish();
    streamed += terminal.content;
    return terminal;
}

const std::string kWeatherBlock =
    "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
    "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n"
    "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n"
    "<｜DSML｜ parameter name=\"options\" string=\"false\">{\"units\": \"metric\"}"
    "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

void test_tool_calls() {
    {
        // The separator before the block belongs to it; values typed by their string flag.
        std::string streamed;
        const auto terminal = run("Checking." + kWeatherBlock, streamed);
        expect(streamed == "Checking.", "text before a call: " + streamed);
        expect(terminal.tool_calls.size() == 1, "one call");
        if (terminal.tool_calls.size() == 1) {
            expect(terminal.tool_calls[0].name == "get_weather", "call name");
            expect(Json::parse(terminal.tool_calls[0].arguments_json) ==
                       Json({{"city", "Paris"}, {"days", 3}, {"options", {{"units", "metric"}}}}),
                   "typed arguments: " + terminal.tool_calls[0].arguments_json);
        }
        expect(terminal.diagnostics.structured_call_count == 1, "call count");
    }
    {
        // Two invokes in one block, one without parameters; a string value kept verbatim, and
        // text promised as JSON that is not stays a string.
        std::string streamed;
        const auto terminal =
            run("<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter "
                "name=\"city\" string=\"true\">\"Oslo\" & <b></｜DSML｜ parameter>\n<｜DSML｜ "
                "parameter name=\"days\" string=\"false\">many</｜DSML｜ parameter>\n</｜DSML｜ "
                "invoke>\n<｜DSML｜ invoke name=\"get_weather\">\n</｜DSML｜ invoke>\n</｜DSML｜ "
                "calls>",
                streamed);
        expect(streamed.empty(), "nothing but calls: " + streamed);
        expect(terminal.tool_calls.size() == 2, "two calls");
        if (terminal.tool_calls.size() == 2) {
            expect(Json::parse(terminal.tool_calls[0].arguments_json) ==
                       Json({{"city", "\"Oslo\" & <b>"}, {"days", "many"}}),
                   "first call arguments: " + terminal.tool_calls[0].arguments_json);
            expect(terminal.tool_calls[1].arguments_json == "{}", "a call without arguments");
        }
        expect(terminal.diagnostics.schema_mismatch_arguments == 1,
               "unreadable JSON value counted");
    }
    {
        // An unreadable block returns to content as it was written, separator included.
        std::string streamed;
        const std::string broken =
            "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter "
            "name=\"city\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        const auto terminal = run("a" + broken + " b", streamed);
        expect(terminal.tool_calls.empty(), "no call from broken markup");
        expect(streamed == "a" + broken + " b", "broken block returns to content: " + streamed);
        expect(terminal.diagnostics.fallback_reason ==
                   ninfer::ToolCallParseFallbackReason::MalformedStructure,
               "broken block diagnostics");
    }
    {
        // A block that ends after its last invoke is a call; a block cut inside one is content.
        std::string streamed;
        const std::string unterminated =
            kWeatherBlock.substr(0, kWeatherBlock.rfind("</｜DSML｜ calls>"));
        auto terminal = run(unterminated, streamed);
        expect(terminal.tool_calls.size() == 1 && streamed.empty(), "complete unterminated block");
        streamed.clear();
        const std::string cut = kWeatherBlock.substr(0, kWeatherBlock.find("Paris"));
        terminal              = run("x" + cut, streamed);
        expect(terminal.tool_calls.empty() && streamed == "x" + cut,
               "incomplete block returns to content: " + streamed);
        // A text that only looks like the opener's beginning is content once it diverges.
        streamed.clear();
        terminal = run("a\n\n<｜DSML｜ call me", streamed);
        expect(terminal.tool_calls.empty() && streamed == "a\n\n<｜DSML｜ call me",
               "a diverging opener prefix: " + streamed);
    }
    {
        // A forced call: the prompt holds the block's opener and the invoke tag.
        std::string streamed;
        auto terminal = run("<｜DSML｜ parameter name=\"city\" string=\"true\">Lima</｜DSML｜ "
                            "parameter>\n</｜DSML｜ "
                            "invoke>\n</｜DSML｜ calls>",
                            streamed, "get_weather");
        expect(terminal.tool_calls.size() == 1 && streamed.empty(), "forced call");
        if (terminal.tool_calls.size() == 1) {
            expect(terminal.tool_calls[0].name == "get_weather" &&
                       terminal.tool_calls[0].arguments_json == R"({"city":"Lima"})",
                   "forced call arguments: " + terminal.tool_calls[0].arguments_json);
        }
        streamed.clear();
        terminal = run("<｜DSML｜ parameter name=\"city", streamed, "get_weather");
        expect(terminal.tool_calls.empty() && streamed == "<｜DSML｜ parameter name=\"city",
               "a failed forced call returns only the model's bytes: " + streamed);
    }
}

// The converter's processor_config.json of the release's image processing.
const std::string kProcessorConfig =
    R"({"image_processor":{"patch_size":14,"downsample_ratio":3,"min_pixels":295936,)"
    R"("max_image_tokens":1024,"max_wh_ratio":null,"resample":3,"pad_color":[127,127,127],)"
    R"("image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"image_token":"<｜deepseek_image｜>"}})";

ninfer::models::qwen3_5::Frontend deepseek_frontend(bool vision = false) {
    // The resources are views the Frontend may keep.
    static const TokenizerFiles files = byte_tokenizer_files(true);
    static const std::string chat     = chat_template();
    ninfer::models::qwen3_5::FrontendResources resources{
        .tokenizer_json         = files.tokenizer_json,
        .tokenizer_config_json  = files.tokenizer_config_json,
        .chat_template_jinja    = chat,
        .generation_config_json = files.generation_config_json};
    if (vision) { resources.processor_config_json = kProcessorConfig; }
    resources.tokenizer          = std::make_shared<const fi::Tokenizer>(fi::TokenizerResources{
        files.tokenizer_json, files.tokenizer_config_json, files.generation_config_json});
    resources.public_token_count = static_cast<std::uint32_t>(resources.tokenizer->vocab_size());
    ninfer::models::qwen3_5::FrontendOptions options;
    options.architecture   = ninfer::models::Architecture::DeepSeekV41;
    options.vision_enabled = vision;
    options.max_context    = std::numeric_limits<std::uint32_t>::max();
    return ninfer::models::qwen3_5::make_frontend(resources, options);
}

// The byte tokens of `text`.
std::vector<ninfer::TokenId> bytes(std::string_view text) {
    std::vector<ninfer::TokenId> out;
    for (const char byte : text) { out.push_back(static_cast<unsigned char>(byte)); }
    return out;
}

std::vector<ninfer::TokenId> joined(std::initializer_list<std::vector<ninfer::TokenId>> parts) {
    std::vector<ninfer::TokenId> out;
    for (const auto& part : parts) { out.insert(out.end(), part.begin(), part.end()); }
    return out;
}

struct Turn {
    std::string reasoning, content;
    std::vector<ninfer::GeneratedToolCall> calls;
};

// Generates `tokens` and EOS, one round or one token per round.
Turn generate(const ninfer::models::qwen3_5::Frontend& frontend,
              const ninfer::models::qwen3_5::PreparedPrompt& prompt,
              std::vector<ninfer::TokenId> tokens, bool streamed) {
    auto session =
        frontend.make_output_session(prompt, {}, ninfer::OutputOptions{.tool_name_max_length = 64});
    tokens.push_back(kEosId);
    Turn turn;
    const auto commit = [&] {
        for (const ninfer::OutputDelta& delta : session.commit_preview()) {
            (delta.channel == ninfer::OutputChannel::Reasoning ? turn.reasoning : turn.content) +=
                delta.text;
        }
    };
    const auto budget = static_cast<std::uint32_t>(tokens.size() + 8);
    if (streamed) {
        for (const ninfer::TokenId token : tokens) {
            (void)session.preview_model(std::span(&token, 1), budget,
                                        ninfer::FinishReason::OutputLimit);
            commit();
        }
    } else {
        (void)session.preview_model(tokens, budget, ninfer::FinishReason::OutputLimit);
        commit();
    }
    turn.calls = session.take_tool_calls();
    return turn;
}

ninfer::PromptInput user_prompt(std::string text) {
    ninfer::PromptInput input;
    input.messages.push_back(
        {.role  = ninfer::ChatRole::User,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = std::move(text)}}});
    input.options.continuation = ninfer::PromptContinuationMode::NewAssistantTurn;
    return input;
}

void test_prompt_and_output() {
    const auto frontend = deepseek_frontend();
    const std::string effort =
        "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the "
        "reasoning)\n\n";
    {
        // A thinking prompt: the markers are their tokens, the rest bytes; reasoning is open.
        const auto prompt                           = frontend.prepare(user_prompt("Hi"));
        const std::vector<ninfer::TokenId> expected = joined({{kBosId, kSystemId},
                                                              bytes(effort),
                                                              {kUserId},
                                                              bytes("Hi"),
                                                              {kAssistantId, kThinkOpenId}});
        const auto ids                              = prompt.token_ids();
        expect(std::vector<ninfer::TokenId>(ids.begin(), ids.end()) == expected,
               "thinking prompt tokens");
        expect(prompt.summary().starts_in_reasoning, "a thinking prompt starts in reasoning");
        // </think> closes the reasoning; the content follows it directly.
        for (const bool streamed : {false, true}) {
            const Turn turn =
                generate(frontend, prompt,
                         joined({bytes("Let me think.\nDone."), {kThinkCloseId}, bytes("Hello!")}),
                         streamed);
            expect(turn.reasoning == "Let me think.\nDone." && turn.content == "Hello!",
                   "reasoning then content: [" + turn.reasoning + "] [" + turn.content + "]");
        }
    }
    {
        // A thinking budget: once the reasoning reaches it, the control span (guidance and the
        // </think> token) closes it and the model's next tokens are content.
        const auto prompt = frontend.prepare(user_prompt("Hi"));
        auto session      = frontend.make_output_session(prompt, {}, {},
                                                         ninfer::ThinkingControlOptions{.budget = 2});
        Turn turn;
        const auto commit = [&] {
            for (const ninfer::OutputDelta& delta : session.commit_preview()) {
                (delta.channel == ninfer::OutputChannel::Reasoning ? turn.reasoning
                                                                   : turn.content) += delta.text;
            }
        };
        const std::vector<ninfer::TokenId> thought = bytes("ab");
        const auto reached = session.preview_model(thought, 1024, ninfer::FinishReason::OutputLimit);
        commit();
        expect(reached.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl,
               "a reached thinking budget applies the control span");
        const auto span = session.pending_control_tokens();
        const std::vector<ninfer::TokenId> control(span.begin(), span.end());
        expect(control.size() > 1 && control.back() == kThinkCloseId &&
                   frontend.thinking_control_token_count() == control.size(),
               "the control span ends with the </think> token");
        const auto forced = session.preview_control(control, 1022);
        expect(forced.accepted_tokens == control.size() && !forced.finished(),
               "the control span is accepted whole");
        commit();
        std::vector<ninfer::TokenId> answer = bytes("Hello!");
        answer.push_back(kEosId);
        (void)session.preview_model(answer, 512, ninfer::FinishReason::OutputLimit);
        commit();
        expect(turn.reasoning.starts_with("ab\n\nConsidering the limited time") &&
                   turn.reasoning.ends_with("directly now.") && turn.content == "Hello!",
               "budget-closed reasoning then content: [" + turn.reasoning + "] [" + turn.content +
                   "]");
    }
    {
        // The reasoning-loop guard under conclude: reasoning that keeps rewriting one check gets
        // the same control span, and the model's next tokens are content.
        std::string looping;
        for (int step = 0; step < 120; ++step) {
            looping += "step " + std::to_string(step) + " checks value " +
                       std::to_string(step * 7919 % 9973) + ". ";
        }
        for (int i = 0; i < 60; ++i) {
            looping += "Wait, let me double-check the case where the list is empty. Then the "
                       "function returns zero, which matches the expected output, so that case is "
                       "fine. ";
        }
        const auto prompt = frontend.prepare(user_prompt("Hi"));
        auto session      = frontend.make_output_session(
            prompt, {}, {},
            ninfer::ThinkingControlOptions{.loop = ninfer::ReasoningLoopAction::Conclude});
        Turn turn;
        const auto commit = [&] {
            for (const ninfer::OutputDelta& delta : session.commit_preview()) {
                (delta.channel == ninfer::OutputChannel::Reasoning ? turn.reasoning
                                                                   : turn.content) += delta.text;
            }
        };
        const std::vector<ninfer::TokenId> thought = bytes(looping);
        constexpr std::uint32_t kLarge             = 1'000'000;
        ninfer::runtime::OutputDecision last{};
        for (std::size_t at = 0; at < thought.size(); at += 200) {
            const std::size_t n = std::min<std::size_t>(200, thought.size() - at);
            last = session.preview_model(std::span(thought.data() + at, n), kLarge,
                                         ninfer::FinishReason::OutputLimit);
            commit();
            if (last.finished() ||
                last.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl) {
                break;
            }
        }
        expect(last.continuation == ninfer::runtime::ContinuationAction::ApplyTargetControl &&
                   session.thinking_stats().loop_detected,
               "a looping reasoning under conclude applies the control span");
        const auto span = session.pending_control_tokens();
        const std::vector<ninfer::TokenId> control(span.begin(), span.end());
        expect(!control.empty() && control.back() == kThinkCloseId,
               "the loop guard's control span ends with the </think> token");
        (void)session.preview_control(control, kLarge);
        commit();
        std::vector<ninfer::TokenId> answer = bytes("Hello!");
        answer.push_back(kEosId);
        (void)session.preview_model(answer, kLarge, ninfer::FinishReason::OutputLimit);
        commit();
        expect(turn.reasoning.ends_with("directly now.") && turn.content == "Hello!",
               "loop-concluded reasoning then content: [" + turn.content + "]");
    }
    {
        // Thinking off: the prompt closes an empty reasoning block, and everything is content.
        auto input                    = user_prompt("Hi");
        input.options.enable_thinking = false;
        const auto prompt             = frontend.prepare(std::move(input));
        const auto ids                = prompt.token_ids();
        expect(std::vector<ninfer::TokenId>(ids.begin(), ids.end()) ==
                   joined({{kBosId, kUserId}, bytes("Hi"), {kAssistantId, kThinkCloseId}}),
               "chat-mode prompt tokens");
        expect(!prompt.summary().starts_in_reasoning, "a chat-mode prompt does not reason");
        const Turn turn = generate(frontend, prompt, bytes("Hello!"), false);
        expect(turn.reasoning.empty() && turn.content == "Hello!", "chat-mode content");
    }
    const std::string tool =
        R"({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"options":{"type":"object"}}}}})";
    {
        // A call after the reasoning: the content before its block, the call structured.
        auto input = user_prompt("Weather in Paris?");
        input.options.tool_jsons.push_back(tool);
        const auto prompt = frontend.prepare(std::move(input));
        for (const bool streamed : {false, true}) {
            const Turn turn = generate(
                frontend, prompt,
                joined(
                    {bytes("Use the tool."), {kThinkCloseId}, bytes("Checking." + kWeatherBlock)}),
                streamed);
            expect(
                turn.reasoning == "Use the tool." && turn.content == "Checking." &&
                    turn.calls.size() == 1 && turn.calls[0].name == "get_weather" &&
                    Json::parse(turn.calls[0].arguments_json) ==
                        Json({{"city", "Paris"}, {"days", 3}, {"options", {{"units", "metric"}}}}),
                "tool call output: [" + turn.reasoning + "] [" + turn.content + "]");
        }
    }
    {
        // A forced call: the prompt ends inside the invoke, the model writes its parameters.
        auto input                     = user_prompt("Weather in Paris?");
        input.options.enable_thinking  = false;
        input.options.forced_tool_name = "get_weather";
        input.options.tool_jsons.push_back(tool);
        const auto prompt        = frontend.prepare(std::move(input));
        const auto ids           = prompt.token_ids();
        const std::string opener = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n";
        expect(ids.size() > opener.size() + 1 &&
                   std::vector<ninfer::TokenId>(
                       ids.end() - static_cast<std::ptrdiff_t>(opener.size()) - 1, ids.end()) ==
                       joined({{kThinkCloseId}, bytes(opener)}),
               "forced call prompt ends with the opener");
        const Turn turn = generate(
            frontend, prompt,
            bytes("<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n"
                  "</｜DSML｜ invoke>\n</｜DSML｜ calls>"),
            false);
        expect(turn.content.empty() && turn.calls.size() == 1 &&
                   turn.calls[0].arguments_json == R"({"city":"Paris"})",
               "forced call output: [" + turn.content + "]");
    }
    // A strict tool: the grammar admits its parameters under the schema in declaration order, each
    // value's string flag set by its type, and the call reads back as the typed arguments.
    const std::string strict =
        R"({"type":"function","function":{"name":"get_weather","strict":true,"parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city","days"],"additionalProperties":false}}})";
    // Whether the grammar takes `prefix` and then `next`.
    const auto admits = [](const ninfer::text::GrammarState& grammar, std::string_view prefix,
                           char next) {
        auto state = grammar.fork();
        try {
            state->accept(bytes(prefix));
        } catch (const std::exception&) { return false; }
        std::vector<std::uint32_t> mask(9); // 263 tokens
        state->fill_masks(mask, {});
        const auto byte = static_cast<unsigned char>(next);
        return (mask[byte / 32] & (1U << (byte % 32))) != 0;
    };
    const std::string parameter = "<｜DSML｜ parameter name=\"";
    const std::string city      = parameter + "city\" string=\"";
    const std::string days =
        city + "true\">Paris</｜DSML｜ parameter>\n" + parameter + "days\" string=\"";
    {
        auto input                    = user_prompt("Weather in Paris?");
        input.options.enable_thinking = false;
        input.options.tool_jsons.push_back(strict);
        const auto prompt = frontend.prepare(std::move(input));
        const auto session =
            frontend.make_output_session(prompt, {}, ninfer::OutputOptions{.tool_name_max_length = 64});
        const auto grammar = session.grammar_state();
        expect(grammar != nullptr, "a strict tool constrains the output");
        const std::string head =
            "Checking.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n";
        expect(admits(*grammar, head + parameter, 'c') && !admits(*grammar, head + parameter, 'd') &&
                   !admits(*grammar, head + "<｜DSML｜ invoke", ' '),
               "the declared parameters in order, the required ones present");
        expect(admits(*grammar, head + city, 't') && !admits(*grammar, head + city, 'f'),
               "a string value is string=\"true\"");
        expect(admits(*grammar, head + days, 'f') && !admits(*grammar, head + days, 't') &&
                   admits(*grammar, head + days + "false\">", '3') &&
                   !admits(*grammar, head + days + "false\">", 'x'),
               "an integer value is JSON under string=\"false\"");
        const std::string block =
            "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n" + city +
            "true\">Paris</｜DSML｜ parameter>\n" + parameter +
            "days\" string=\"false\">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        const Turn turn = generate(frontend, prompt, bytes("Checking." + block), false);
        expect(turn.content == "Checking." && turn.calls.size() == 1 &&
                   Json::parse(turn.calls[0].arguments_json) ==
                       Json({{"city", "Paris"}, {"days", 3}}),
               "strict call output: [" + turn.content + "]");
    }
    {
        // A forced strict call: the grammar starts in its parameters.
        auto input                     = user_prompt("Weather in Paris?");
        input.options.enable_thinking  = false;
        input.options.forced_tool_name = "get_weather";
        input.options.tool_jsons.push_back(strict);
        const auto session = frontend.make_output_session(
            frontend.prepare(std::move(input)), {},
            ninfer::OutputOptions{.tool_name_max_length = 64});
        const auto grammar = session.grammar_state();
        expect(grammar != nullptr && admits(*grammar, city, 't') && !admits(*grammar, "", 'x'),
               "a forced strict call starts in its parameters");
    }
    {
        // A structured response beside tools: the JSON or a calls block.
        auto input                    = user_prompt("Weather in Paris?");
        input.options.enable_thinking = false;
        input.options.tool_jsons.push_back(tool);
        const auto session = frontend.make_output_session(
            frontend.prepare(std::move(input)), {}, {}, {},
            ninfer::StructuredOutputOptions{.kind = ninfer::StructuredOutputKind::JsonObject});
        const auto grammar = session.grammar_state();
        expect(grammar != nullptr && admits(*grammar, "", '{') && admits(*grammar, "\n\n", '<') &&
                   admits(*grammar,
                          "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n" + parameter,
                          'd') &&
                   !admits(*grammar, "", 'x'),
               "structured output beside tools admits the JSON and a calls block");
    }
}

std::vector<std::uint8_t> synthetic_image(int width, int height) {
    std::vector<std::uint8_t> rgb(std::size_t(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                rgb[(std::size_t(y) * width + x) * 3 + c] =
                    std::uint8_t((x * 7 + y * 13 + c * 101 + (x * y) % 17 * 5) % 256);
            }
        }
    }
    return rgb;
}

void test_image_processing() {
    const OrderedJson cases = OrderedJson::parse(read_fixture("image_cases.json"));
    fi::DeepSeekMediaOptions options;
    for (const OrderedJson& c : cases.at("plans")) {
        const auto plan = fi::deepseek_image_plan(c[0], c[1], options, c[2]);
        expect(plan.height == c[3] && plan.width == c[4] && plan.rows == c[5] &&
                   plan.columns == c[6] && plan.span_tokens() <= c[2].get<std::uint32_t>(),
               "image plan of " + c.dump());
    }
    const int width = cases["synthetic"]["width"], height = cases["synthetic"]["height"];
    const std::vector<std::uint8_t> rgb = synthetic_image(width, height);
    const auto digest = [](const std::vector<std::uint8_t>& bytes) {
        return fi::sha256_hex(fi::sha256(std::span<const std::uint8_t>(bytes)));
    };
    for (const OrderedJson& c : cases.at("resizes")) {
        expect(digest(fi::pil_bicubic_resize(rgb, width, height, c["out"][0], c["out"][1])) ==
                   c["sha256"].get<std::string>(),
               "Pillow's bicubic resize to " + c["out"].dump());
    }
    for (const OrderedJson& c : cases.at("pads")) {
        expect(digest(fi::deepseek_pad(rgb, width, height, c["out"][0], c["out"][1])) ==
                   c["sha256"].get<std::string>(),
               "ImageOps.pad to " + c["out"].dump());
    }
}

// A picture between two texts: its patches as the reference derives them, its span of image
// tokens where the template wrote its placeholder.
void test_image_prompt() {
    const OrderedJson cases = OrderedJson::parse(read_fixture("image_cases.json"));
    const OrderedJson& image = cases.at("image");
    std::ifstream file(std::string(NINFER_SOURCE_DIR) + "/" + image["path"].get<std::string>(),
                       std::ios::binary);
    std::vector<std::uint8_t> png((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
    expect(!png.empty(), "the example picture reads");
    const auto frontend = deepseek_frontend(true);
    ninfer::PromptInput input;
    ninfer::MessagePart picture;
    picture.kind             = ninfer::MessagePartKind::Media;
    picture.media.kind       = ninfer::MediaKind::Image;
    picture.media.bytes      = std::move(png);
    picture.media.media_type = "image/png";
    input.messages.push_back(
        {.role  = ninfer::ChatRole::User,
         .parts = {{.kind = ninfer::MessagePartKind::Text, .text = "A"},
                   std::move(picture),
                   {.kind = ninfer::MessagePartKind::Text, .text = "B"}}});
    input.options.continuation    = ninfer::PromptContinuationMode::NewAssistantTurn;
    input.options.enable_thinking = false;
    const auto prompt = frontend.prepare(std::move(input));
    const auto& data  = ninfer::models::qwen3_5::FrontendTestAccess::inspect(prompt);
    const int gw = image["grid"][0], gh = image["grid"][1];
    const std::size_t span = std::size_t((gh + 2) / 3) * ((gw + 2) / 3 + 1) + 2;
    expect(data.vision_items.size() == 1 && data.media_payloads.size() == 1, "one image");
    const auto& item = data.vision_items.front();
    expect(item.grid.height == gh && item.grid.width == gw && item.token_spans.size() == 1 &&
               item.token_spans.front().count == span,
           "the image's grid and span");
    const auto words = data.media_payloads.front()->span();
    const auto patches = fi::sha256_hex(fi::sha256(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(words.data()), words.size() * 2)));
    expect(patches == image["patches_sha256"].get<std::string>(), "the image's patches");
    std::vector<ninfer::TokenId> expected = joined({{kBosId, kUserId}, bytes("A\n\n")});
    const std::size_t begin = expected.size();
    expected.insert(expected.end(), span, kImageId);
    const auto tail = joined({bytes("\n\nB"), {kAssistantId, kThinkCloseId}});
    expected.insert(expected.end(), tail.begin(), tail.end());
    expect(data.token_ids == expected, "the image's span among the prompt's tokens");
    expect(item.token_spans.front().begin == begin, "the span starts after the first text block");
}

} // namespace

int main() {
    try {
        test_added_tokens();
        test_pretokenizer();
        test_run_reuse();
        test_chat_template();
        test_image_processing();
        test_image_prompt();
        test_efforts_and_forced_call();
        test_tool_calls();
        test_prompt_and_output();
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
