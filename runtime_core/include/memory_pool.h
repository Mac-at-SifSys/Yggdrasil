#pragma once
// ============================================================================
// YGGDRASIL L2 Runtime — memory_pool.h
// Grade-stratified memory allocator that pre-allocates pools per grade to
// avoid fragmentation and accelerate multivector allocation.
// ============================================================================

#include "device.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <mutex>
#include <unordered_map>
#include <memory>

namespace yggdrasil {

// ---- GradePool -------------------------------------------------------------
// A free-list pool that hands out fixed-size blocks for a single grade.
// Block size = grade_dim(g) * sizeof(float) * block_multivectors.
class GradePool {
public:
    // `grade`         — which Clifford grade (0..3)
    // `block_mv`      — how many multivectors each block can store
    // `num_blocks`    — how many blocks to pre-allocate
    // `dev`           — device on which memory lives
    GradePool(int grade, size_t block_mv, size_t num_blocks, Device& dev);
    ~GradePool();

    GradePool(const GradePool&) = delete;
    GradePool& operator=(const GradePool&) = delete;
    GradePool(GradePool&&) noexcept;
    GradePool& operator=(GradePool&&) noexcept;

    // Allocate one block; returns nullptr if pool exhausted.
    float* allocate();

    // Return a block to the pool.
    void deallocate(float* ptr);

    // Stats
    int    grade()            const { return grade_; }
    size_t block_floats()     const { return block_floats_; }
    size_t block_bytes()      const { return block_floats_ * sizeof(float); }
    size_t total_blocks()     const { return total_blocks_; }
    size_t free_blocks()      const { return free_list_.size(); }
    size_t used_blocks()      const { return total_blocks_ - free_list_.size(); }
    Device& device()          const { return *dev_; }

private:
    int     grade_;
    size_t  block_floats_;    // floats per block
    size_t  total_blocks_;
    Device* dev_;
    void*   arena_;           // raw allocation from device
    size_t  arena_bytes_;
    std::vector<float*> free_list_;
};

// ---- MemoryManager ---------------------------------------------------------
// Owns a GradePool for each grade (0..3) and provides a unified allocation
// interface for multivector storage.
class MemoryManager {
public:
    struct Config {
        size_t block_mv   = 256;     // multivectors per block
        size_t num_blocks = 64;      // blocks per grade
        DeviceType device = DeviceType::CPU;
    };

    explicit MemoryManager(Config cfg = {});
    ~MemoryManager() = default;

    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // Allocate a block for a specific grade. Thread-safe.
    float* allocate_grade(int grade);

    // Free a grade block. Thread-safe.
    void free_grade(int grade, float* ptr);

    // Allocate a full GradeBuffer for `n` multivectors using pooled memory
    // where possible, falling back to direct device allocation for large n.
    GradeBuffer allocate_buffer(size_t n, GradeMask mask = ALL_GRADES);

    // Statistics
    struct Stats {
        std::array<size_t, NUM_GRADES> allocated_blocks;
        std::array<size_t, NUM_GRADES> free_blocks;
        std::array<size_t, NUM_GRADES> total_blocks;
    };
    Stats stats() const;

    Device& device() const { return *dev_; }

private:
    Config cfg_;
    Device* dev_;
    std::array<std::unique_ptr<GradePool>, NUM_GRADES> pools_;
    std::mutex mu_;
};

} // namespace yggdrasil
