// ============================================================================
// YGGDRASIL L2 Runtime — device_cuda.cpp
// CUDA device backend — stub implementation that compiles without CUDA SDK.
// All operations throw an informative error if actually called at runtime.
// When YGGDRASIL_HAS_CUDA is defined, replace stubs with real CUDA calls.
// ============================================================================

#include "device.h"
#include <stdexcept>
#include <string>

namespace yggdrasil {

#ifdef YGGDRASIL_HAS_CUDA

// ===== REAL CUDA IMPLEMENTATION (requires nvcc) =============================
// This section would contain cudaMalloc, cudaFree, cudaMemcpy, etc.
// Placeholder for when CUDA toolkit is available.

#else

// ===== STUB IMPLEMENTATION ==================================================

static void cuda_not_available(const std::string& fn) {
    throw std::runtime_error(
        "CUDA device: " + fn + "() called but YGGDRASIL_HAS_CUDA is not "
        "defined. Rebuild with CUDA support or use DeviceType::CPU.");
}

CudaDevice& CudaDevice::instance() {
    static CudaDevice dev;
    return dev;
}

void* CudaDevice::allocate(size_t bytes) {
    (void)bytes;
    cuda_not_available("allocate");
    return nullptr; // unreachable
}

void CudaDevice::deallocate(void* ptr, size_t bytes) {
    (void)ptr; (void)bytes;
    cuda_not_available("deallocate");
}

void CudaDevice::copy(void* dst, const void* src, size_t bytes) {
    (void)dst; (void)src; (void)bytes;
    cuda_not_available("copy");
}

void CudaDevice::zero(void* ptr, size_t bytes) {
    (void)ptr; (void)bytes;
    cuda_not_available("zero");
}

#endif // YGGDRASIL_HAS_CUDA

} // namespace yggdrasil
