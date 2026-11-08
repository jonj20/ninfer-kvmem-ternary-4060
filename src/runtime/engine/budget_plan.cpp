#include "ninfer/engine.h"
#include "core/device.h"
#include "core/startup.h"
#include "runtime/engine/model_instance.h"
#include "runtime/engine/qwen4_exp_core.h"
#include <stdexcept>
#include <utility>

namespace ninfer {
KvmemBudgetPlan Engine::plan_kvmem_budget(EngineOptions options) {
    if (options.devices.size() > 1 || options.tensor_parallel ||
        options.purpose != EnginePurpose::Generation || !options.kvmem.selected_tokens ||
        options.cuda_memory_policy != CudaMemoryPolicy::DriverDefault ||
        options.expert_residency != ExpertResidency::Device ||
        options.suspend.enabled || !options.grafts.empty() ||
        runtime::is_qwen4_exp_artifact(options.artifact_path)) {
        throw std::invalid_argument("Budget planning requires single-GPU Qwen3.5-family KVMem with device-resident weights and ordinary CUDA memory");
    }
    options = runtime::normalize_engine_options(std::move(options));
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext device(options.devices.empty() ? options.device : options.devices.front());
    phase.complete();
    return runtime::plan_model_budget(options, device);
}
} // namespace ninfer
