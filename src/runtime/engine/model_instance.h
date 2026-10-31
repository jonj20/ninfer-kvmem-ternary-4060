#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<models::qwen3_5::Model> model;
    const models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    // Whether the Device KV tier is a bounded working set (a real --kvmem-budget with a backend
    // the window can serve: plain decoding, or MTP with its cache-position pipeline). One active
    // sequence then needs only its window resident, so the resident bound
    // below stops limiting prompts and equals the logical capacity.
    const bool resident_staging;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::unique_ptr<models::qwen3_5::Model> model, const EngineOptions& options);
    ~ModelInstance();
    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;

    // One active sequence has to be device-resident end to end: materialization maps every page in
    // [0, frontier] and attention reads all of them each step, so a ring-widened logical address
    // space (--max-context) cannot be reached by a single request. Dense plans resolve kv_capacity
    // >= max_context, which leaves this equal to the logical capacity. Under --kvmem-budget the KV
    // window stages the working set instead, so the resident bound lifts; the Program keeps the pool
    // bound for prefix reuse, whose restore materializes the retained base whole.
    [[nodiscard]] std::uint32_t active_context_capacity() const noexcept;
};

struct ConstructedModel {
    std::unique_ptr<ModelInstance> instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
