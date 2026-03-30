// ============================================================================
// YGGDRASIL L2 Runtime — memory_pool.cpp
// Grade-stratified memory allocator implementation.
// ============================================================================

#include "memory_pool.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace yggdrasil {

// ---- GradePool -------------------------------------------------------------

GradePool::GradePool(int grade, size_t block_mv, size_t num_blocks, Device& dev)
    : grade_(grade),
      block_floats_(block_mv * static_cast<size_t>(GRADE_DIM[grade])),
      total_blocks_(num_blocks),
      dev_(&dev),
      arena_(nullptr),
      arena_bytes_(0) {
    if (grade < 0 || grade >= NUM_GRADES)
        throw std::invalid_argument("GradePool: grade must be 0..3");
    if (block_floats_ == 0 || num_blocks == 0)
        throw std::invalid_argument("GradePool: block_mv and num_blocks must be > 0");

    arena_bytes_ = block_floats_ * sizeof(float) * num_blocks;
    arena_ = dev.allocate(arena_bytes_);
    dev.zero(arena_, arena_bytes_);

    // Build free list (all blocks free initially)
    free_list_.reserve(num_blocks);
    auto* base = static_cast<float*>(arena_);
    for (size_t i = 0; i < num_blocks; ++i) {
        free_list_.push_back(base + i * block_floats_);
    }
}

GradePool::~GradePool() {
    if (arena_ && dev_) {
        dev_->deallocate(arena_, arena_bytes_);
        arena_ = nullptr;
    }
}

GradePool::GradePool(GradePool&& o) noexcept
    : grade_(o.grade_), block_floats_(o.block_floats_),
      total_blocks_(o.total_blocks_), dev_(o.dev_),
      arena_(o.arena_), arena_bytes_(o.arena_bytes_),
      free_list_(std::move(o.free_list_)) {
    o.arena_ = nullptr;
    o.arena_bytes_ = 0;
    o.dev_ = nullptr;
}

GradePool& GradePool::operator=(GradePool&& o) noexcept {
    if (this != &o) {
        if (arena_ && dev_) dev_->deallocate(arena_, arena_bytes_);
        grade_ = o.grade_;
        block_floats_ = o.block_floats_;
        total_blocks_ = o.total_blocks_;
        dev_ = o.dev_;
        arena_ = o.arena_;
        arena_bytes_ = o.arena_bytes_;
        free_list_ = std::move(o.free_list_);
        o.arena_ = nullptr;
        o.arena_bytes_ = 0;
        o.dev_ = nullptr;
    }
    return *this;
}

float* GradePool::allocate() {
    if (free_list_.empty()) return nullptr;
    float* ptr = free_list_.back();
    free_list_.pop_back();
    return ptr;
}

void GradePool::deallocate(float* ptr) {
    if (!ptr) return;
    // Basic bounds check
    auto* base = static_cast<float*>(arena_);
    auto* end  = base + block_floats_ * total_blocks_;
    if (ptr < base || ptr >= end) {
        throw std::invalid_argument("GradePool::deallocate: pointer not from this pool");
    }
    free_list_.push_back(ptr);
}

// ---- MemoryManager ---------------------------------------------------------

MemoryManager::MemoryManager(Config cfg)
    : cfg_(cfg) {
    if (cfg.device == DeviceType::CPU)
        dev_ = &CpuDevice::instance();
    else
        dev_ = &CudaDevice::instance();

    for (int g = 0; g < NUM_GRADES; ++g) {
        pools_[g] = std::make_unique<GradePool>(
            g, cfg.block_mv, cfg.num_blocks, *dev_);
    }
}

float* MemoryManager::allocate_grade(int grade) {
    if (grade < 0 || grade >= NUM_GRADES)
        throw std::out_of_range("MemoryManager: grade must be 0..3");
    std::lock_guard<std::mutex> lock(mu_);
    float* ptr = pools_[grade]->allocate();
    if (!ptr) {
        throw std::runtime_error(
            "MemoryManager: grade-" + std::to_string(grade) +
            " pool exhausted (" + std::to_string(pools_[grade]->total_blocks()) +
            " blocks)");
    }
    return ptr;
}

void MemoryManager::free_grade(int grade, float* ptr) {
    if (grade < 0 || grade >= NUM_GRADES)
        throw std::out_of_range("MemoryManager: grade must be 0..3");
    std::lock_guard<std::mutex> lock(mu_);
    pools_[grade]->deallocate(ptr);
}

GradeBuffer MemoryManager::allocate_buffer(size_t n, GradeMask mask) {
    // For small allocations that fit in a single pool block, use the pool.
    // For large allocations, go directly to the device.
    GradeBuffer buf;
    buf.device = dev_;
    buf.num_multivectors = n;

    for (int g = 0; g < NUM_GRADES; ++g) {
        if (!(mask & (1 << g))) {
            buf.ptrs[g] = nullptr;
            buf.counts[g] = 0;
            continue;
        }

        size_t need_floats = n * static_cast<size_t>(GRADE_DIM[g]);
        buf.counts[g] = need_floats;

        // Try pool if it fits in one block
        if (need_floats <= pools_[g]->block_floats()) {
            std::lock_guard<std::mutex> lock(mu_);
            float* ptr = pools_[g]->allocate();
            if (ptr) {
                buf.ptrs[g] = ptr;
                continue;
            }
        }

        // Fallback: direct device allocation
        buf.ptrs[g] = static_cast<float*>(
            dev_->allocate(need_floats * sizeof(float)));
        dev_->zero(buf.ptrs[g], need_floats * sizeof(float));
    }

    return buf;
}

MemoryManager::Stats MemoryManager::stats() const {
    Stats s;
    for (int g = 0; g < NUM_GRADES; ++g) {
        s.total_blocks[g] = pools_[g]->total_blocks();
        s.free_blocks[g]  = pools_[g]->free_blocks();
        s.allocated_blocks[g] = pools_[g]->used_blocks();
    }
    return s;
}

} // namespace yggdrasil
