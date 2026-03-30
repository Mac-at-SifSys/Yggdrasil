#pragma once
// ============================================================================
// YGGDRASIL L2 Runtime — device.h
// Device abstraction for CPU and CUDA backends with grade-stratified memory.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>

namespace yggdrasil {

// ---- Grade constants for Cl(3,0) -------------------------------------------
// Grade 0: 1 scalar            (1 component)
// Grade 1: 3 vectors           (3 components)
// Grade 2: 3 bivectors         (3 components)
// Grade 3: 1 trivector/pseudo  (1 component)
// Total: 8 components per multivector
constexpr int NUM_GRADES   = 4;
constexpr int GRADE_DIM[4] = {1, 3, 3, 1};
constexpr int MV_DIM       = 8;   // sum of GRADE_DIM

// Bitmask for selecting grades
using GradeMask = uint8_t;
constexpr GradeMask GRADE_0 = 0x01;
constexpr GradeMask GRADE_1 = 0x02;
constexpr GradeMask GRADE_2 = 0x04;
constexpr GradeMask GRADE_3 = 0x08;
constexpr GradeMask ALL_GRADES = 0x0F;

// Return the number of float components for a given grade
inline int grade_dim(int grade) {
    if (grade < 0 || grade >= NUM_GRADES)
        throw std::out_of_range("grade must be 0..3");
    return GRADE_DIM[grade];
}

// Return offset into the 8-component MV layout for a grade
inline int grade_offset(int grade) {
    int off = 0;
    for (int g = 0; g < grade; ++g) off += GRADE_DIM[g];
    return off;
}

// ---- Device enumeration ----------------------------------------------------
enum class DeviceType : int {
    CPU  = 0,
    CUDA = 1,
};

inline std::string device_name(DeviceType d) {
    switch (d) {
        case DeviceType::CPU:  return "cpu";
        case DeviceType::CUDA: return "cuda";
    }
    return "unknown";
}

// ---- Abstract device interface ---------------------------------------------
class Device {
public:
    virtual ~Device() = default;

    virtual DeviceType type() const = 0;

    // Raw allocation / free (bytes)
    virtual void* allocate(size_t bytes) = 0;
    virtual void  deallocate(void* ptr, size_t bytes) = 0;

    // memcpy within same device
    virtual void copy(void* dst, const void* src, size_t bytes) = 0;

    // zero memory
    virtual void zero(void* ptr, size_t bytes) = 0;

    // Human-readable name
    virtual std::string name() const = 0;
};

// ---- CPU device ------------------------------------------------------------
class CpuDevice : public Device {
public:
    DeviceType type() const override { return DeviceType::CPU; }

    void* allocate(size_t bytes) override;
    void  deallocate(void* ptr, size_t bytes) override;
    void  copy(void* dst, const void* src, size_t bytes) override;
    void  zero(void* ptr, size_t bytes) override;

    std::string name() const override { return "cpu"; }

    static CpuDevice& instance();
};

// ---- CUDA device (stub) ----------------------------------------------------
class CudaDevice : public Device {
public:
    DeviceType type() const override { return DeviceType::CUDA; }

    void* allocate(size_t bytes) override;
    void  deallocate(void* ptr, size_t bytes) override;
    void  copy(void* dst, const void* src, size_t bytes) override;
    void  zero(void* ptr, size_t bytes) override;

    std::string name() const override { return "cuda:0"; }

    static CudaDevice& instance();
};

// ---- Inter-device transfer -------------------------------------------------
// Copy `bytes` from src_dev/src_ptr  →  dst_dev/dst_ptr.
void transfer(Device& dst_dev, void* dst_ptr,
              Device& src_dev, const void* src_ptr,
              size_t bytes);

// ---- Grade-stratified device buffer ----------------------------------------
// Holds pointers to per-grade storage on a single device.
// Each grade's buffer is independently allocated so that grade-projection
// operations touch only the grades they need.
struct GradeBuffer {
    Device* device = nullptr;
    std::array<float*, NUM_GRADES> ptrs = {nullptr, nullptr, nullptr, nullptr};
    std::array<size_t, NUM_GRADES> counts = {0, 0, 0, 0}; // #floats per grade
    size_t num_multivectors = 0;

    GradeBuffer() = default;

    // Allocate storage for `n` multivectors on `dev`, restricted to `mask`.
    void allocate(Device& dev, size_t n, GradeMask mask = ALL_GRADES);

    // Free all grade storage
    void free();

    // Zero all allocated grades
    void zero_all();

    // Total floats allocated
    size_t total_floats() const;

    // Copy data between two GradeBuffers (same shape required)
    static void copy(GradeBuffer& dst, const GradeBuffer& src);

    ~GradeBuffer() { free(); }

    // Move semantics
    GradeBuffer(GradeBuffer&& o) noexcept;
    GradeBuffer& operator=(GradeBuffer&& o) noexcept;
    GradeBuffer(const GradeBuffer&) = delete;
    GradeBuffer& operator=(const GradeBuffer&) = delete;
};

// ---- Helper: get default device --------------------------------------------
inline Device& default_device() { return CpuDevice::instance(); }

} // namespace yggdrasil
