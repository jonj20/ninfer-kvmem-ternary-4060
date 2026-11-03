// The launch helpers a kernel needs from the device layer, in a unit of their own: a module that
// carries kernels outside the runtime (the Windows kernel DLL) links its own copy of them without
// the rest of device.cu. Each copy calls its own module's CUDA runtime, which registered that
// module's kernels.
#include "core/device.h"
#ifdef NINFER_RESIDENT_MEMORY
#include "core/resident_memory.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
#ifdef NINFER_RESIDENT_MEMORY
    if (core::current_resident_memory()) {
        throw CudaError(err, std::string(file) + ":" + std::to_string(line) +
            ": CUDA_CHECK(" + expr + ") failed: " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err));
    }
#endif
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

void raise_dynamic_shared_memory(const void* kernel, int bytes) {
    static std::mutex mutex;
    static std::map<std::pair<int, const void*>, int> raised;
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::scoped_lock lock(mutex);
    int& limit = raised[{device, kernel}];
    if (limit >= bytes) { return; }
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    limit = bytes;
}

} // namespace ninfer
