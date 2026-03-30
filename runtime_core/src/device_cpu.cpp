// ============================================================================
// YGGDRASIL L2 Runtime — device_cpu.cpp
// CPU device backend implementation.
// ============================================================================

#include "device.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace yggdrasil {

// ---- CpuDevice -------------------------------------------------------------

CpuDevice& CpuDevice::instance() {
    static CpuDevice dev;
    return dev;
}

void* CpuDevice::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    // Use aligned allocation for SIMD friendliness (64-byte alignment)
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&ptr, 64, bytes) != 0) ptr = nullptr;
#endif
    if (!ptr) throw std::bad_alloc();
    return ptr;
}

void CpuDevice::deallocate(void* ptr, size_t /*bytes*/) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void CpuDevice::copy(void* dst, const void* src, size_t bytes) {
    if (bytes > 0 && dst && src) {
        std::memcpy(dst, src, bytes);
    }
}

void CpuDevice::zero(void* ptr, size_t bytes) {
    if (bytes > 0 && ptr) {
        std::memset(ptr, 0, bytes);
    }
}

// ---- GradeBuffer -----------------------------------------------------------

void GradeBuffer::allocate(Device& dev, size_t n, GradeMask mask) {
    free();
    device = &dev;
    num_multivectors = n;
    for (int g = 0; g < NUM_GRADES; ++g) {
        if (mask & (1 << g)) {
            counts[g] = n * static_cast<size_t>(GRADE_DIM[g]);
            ptrs[g] = static_cast<float*>(
                dev.allocate(counts[g] * sizeof(float)));
            dev.zero(ptrs[g], counts[g] * sizeof(float));
        } else {
            counts[g] = 0;
            ptrs[g] = nullptr;
        }
    }
}

void GradeBuffer::free() {
    if (!device) return;
    for (int g = 0; g < NUM_GRADES; ++g) {
        if (ptrs[g]) {
            device->deallocate(ptrs[g], counts[g] * sizeof(float));
            ptrs[g] = nullptr;
            counts[g] = 0;
        }
    }
    num_multivectors = 0;
}

void GradeBuffer::zero_all() {
    if (!device) return;
    for (int g = 0; g < NUM_GRADES; ++g) {
        if (ptrs[g] && counts[g] > 0) {
            device->zero(ptrs[g], counts[g] * sizeof(float));
        }
    }
}

size_t GradeBuffer::total_floats() const {
    size_t total = 0;
    for (int g = 0; g < NUM_GRADES; ++g) total += counts[g];
    return total;
}

void GradeBuffer::copy(GradeBuffer& dst, const GradeBuffer& src) {
    for (int g = 0; g < NUM_GRADES; ++g) {
        if (src.ptrs[g] && dst.ptrs[g] && src.counts[g] > 0) {
            if (dst.device == src.device) {
                dst.device->copy(dst.ptrs[g], src.ptrs[g],
                                 src.counts[g] * sizeof(float));
            } else {
                transfer(*dst.device, dst.ptrs[g],
                         *src.device, src.ptrs[g],
                         src.counts[g] * sizeof(float));
            }
        }
    }
}

GradeBuffer::GradeBuffer(GradeBuffer&& o) noexcept
    : device(o.device), ptrs(o.ptrs), counts(o.counts),
      num_multivectors(o.num_multivectors) {
    o.device = nullptr;
    o.ptrs.fill(nullptr);
    o.counts.fill(0);
    o.num_multivectors = 0;
}

GradeBuffer& GradeBuffer::operator=(GradeBuffer&& o) noexcept {
    if (this != &o) {
        free();
        device = o.device;
        ptrs = o.ptrs;
        counts = o.counts;
        num_multivectors = o.num_multivectors;
        o.device = nullptr;
        o.ptrs.fill(nullptr);
        o.counts.fill(0);
        o.num_multivectors = 0;
    }
    return *this;
}

// ---- Inter-device transfer -------------------------------------------------

void transfer(Device& dst_dev, void* dst_ptr,
              Device& src_dev, const void* src_ptr,
              size_t bytes) {
    if (bytes == 0) return;

    if (dst_dev.type() == DeviceType::CPU &&
        src_dev.type() == DeviceType::CPU) {
        // CPU → CPU
        std::memcpy(dst_ptr, src_ptr, bytes);
    } else {
        // For CPU↔CUDA transfers, we go through a CPU staging buffer.
        // In the stub build this path won't execute, but the logic is correct.
        if (src_dev.type() == DeviceType::CUDA &&
            dst_dev.type() == DeviceType::CPU) {
            // CUDA → CPU: the CUDA device's copy pulls to host
            src_dev.copy(dst_ptr, src_ptr, bytes);
        } else if (src_dev.type() == DeviceType::CPU &&
                   dst_dev.type() == DeviceType::CUDA) {
            // CPU → CUDA: the CUDA device's copy pushes to device
            dst_dev.copy(dst_ptr, src_ptr, bytes);
        } else {
            // CUDA → CUDA (same or different GPU): stage through CPU
            auto& cpu = CpuDevice::instance();
            void* staging = cpu.allocate(bytes);
            src_dev.copy(staging, src_ptr, bytes);
            dst_dev.copy(dst_ptr, staging, bytes);
            cpu.deallocate(staging, bytes);
        }
    }
}

} // namespace yggdrasil
