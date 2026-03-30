// ============================================================================
// YGGDRASIL L2 Runtime — test_memory.cpp
// Test grade-stratified memory allocation.
// ============================================================================

#include "memory_pool.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <set>

using namespace yggdrasil;

// ---- Test GradePool basic alloc/free ---------------------------------------

void test_grade_pool_basic() {
    auto& cpu = CpuDevice::instance();
    GradePool pool(1, 16, 4, cpu);  // grade 1, 16 MVs/block, 4 blocks

    assert(pool.grade() == 1);
    assert(pool.block_floats() == 16 * 3);  // grade 1 has 3 components
    assert(pool.total_blocks() == 4);
    assert(pool.free_blocks() == 4);
    assert(pool.used_blocks() == 0);

    // Allocate all 4 blocks
    std::vector<float*> ptrs;
    for (int i = 0; i < 4; ++i) {
        float* p = pool.allocate();
        assert(p != nullptr);
        ptrs.push_back(p);
    }
    assert(pool.free_blocks() == 0);
    assert(pool.used_blocks() == 4);

    // Next allocation should fail
    assert(pool.allocate() == nullptr);

    // Free one and re-allocate
    pool.deallocate(ptrs[2]);
    assert(pool.free_blocks() == 1);
    float* p = pool.allocate();
    assert(p != nullptr);
    assert(pool.free_blocks() == 0);

    // Free all
    for (auto* ptr : ptrs) {
        if (ptr != p) pool.deallocate(ptr);
    }
    pool.deallocate(p);
    assert(pool.free_blocks() == 4);

    std::cout << "  [PASS] test_grade_pool_basic\n";
}

// ---- Test unique pointers --------------------------------------------------

void test_grade_pool_unique_ptrs() {
    auto& cpu = CpuDevice::instance();
    GradePool pool(0, 8, 10, cpu);  // grade 0, 8 MVs/block, 10 blocks

    std::set<float*> all_ptrs;
    for (int i = 0; i < 10; ++i) {
        float* p = pool.allocate();
        assert(p != nullptr);
        // All pointers should be unique
        assert(all_ptrs.find(p) == all_ptrs.end());
        all_ptrs.insert(p);
    }

    std::cout << "  [PASS] test_grade_pool_unique_ptrs\n";
}

// ---- Test all grade pools --------------------------------------------------

void test_all_grade_pools() {
    auto& cpu = CpuDevice::instance();

    for (int g = 0; g < NUM_GRADES; ++g) {
        GradePool pool(g, 32, 8, cpu);
        assert(pool.grade() == g);
        assert(pool.block_floats() == 32 * static_cast<size_t>(GRADE_DIM[g]));

        float* p = pool.allocate();
        assert(p != nullptr);

        // Write to the buffer to verify it's usable
        for (size_t i = 0; i < pool.block_floats(); ++i) {
            p[i] = static_cast<float>(i);
        }
        // Read back
        for (size_t i = 0; i < pool.block_floats(); ++i) {
            assert(p[i] == static_cast<float>(i));
        }

        pool.deallocate(p);
    }

    std::cout << "  [PASS] test_all_grade_pools\n";
}

// ---- Test MemoryManager ----------------------------------------------------

void test_memory_manager_basic() {
    MemoryManager::Config cfg;
    cfg.block_mv = 64;
    cfg.num_blocks = 8;
    cfg.device = DeviceType::CPU;

    MemoryManager mgr(cfg);

    auto s = mgr.stats();
    for (int g = 0; g < NUM_GRADES; ++g) {
        assert(s.total_blocks[g] == 8);
        assert(s.free_blocks[g] == 8);
        assert(s.allocated_blocks[g] == 0);
    }

    // Allocate one block per grade
    std::vector<float*> ptrs;
    for (int g = 0; g < NUM_GRADES; ++g) {
        ptrs.push_back(mgr.allocate_grade(g));
    }

    s = mgr.stats();
    for (int g = 0; g < NUM_GRADES; ++g) {
        assert(s.allocated_blocks[g] == 1);
        assert(s.free_blocks[g] == 7);
    }

    // Free them
    for (int g = 0; g < NUM_GRADES; ++g) {
        mgr.free_grade(g, ptrs[g]);
    }

    s = mgr.stats();
    for (int g = 0; g < NUM_GRADES; ++g) {
        assert(s.free_blocks[g] == 8);
    }

    std::cout << "  [PASS] test_memory_manager_basic\n";
}

// ---- Test GradeBuffer allocation -------------------------------------------

void test_grade_buffer() {
    auto& cpu = CpuDevice::instance();

    GradeBuffer buf;
    buf.allocate(cpu, 10, ALL_GRADES);

    assert(buf.num_multivectors == 10);
    assert(buf.total_floats() == 10 * MV_DIM);

    // Check per-grade counts
    for (int g = 0; g < NUM_GRADES; ++g) {
        assert(buf.counts[g] == 10 * static_cast<size_t>(GRADE_DIM[g]));
        assert(buf.ptrs[g] != nullptr);
    }

    // Write and verify
    for (int g = 0; g < NUM_GRADES; ++g) {
        for (size_t i = 0; i < buf.counts[g]; ++i) {
            buf.ptrs[g][i] = static_cast<float>(g * 100 + i);
        }
    }
    for (int g = 0; g < NUM_GRADES; ++g) {
        for (size_t i = 0; i < buf.counts[g]; ++i) {
            assert(buf.ptrs[g][i] == static_cast<float>(g * 100 + i));
        }
    }

    // Selective grade allocation
    GradeBuffer buf2;
    buf2.allocate(cpu, 5, GRADE_1 | GRADE_2);  // only grades 1 and 2
    assert(buf2.ptrs[0] == nullptr);  // grade 0 not allocated
    assert(buf2.ptrs[1] != nullptr);
    assert(buf2.ptrs[2] != nullptr);
    assert(buf2.ptrs[3] == nullptr);  // grade 3 not allocated

    std::cout << "  [PASS] test_grade_buffer\n";
}

// ---- Test GradeBuffer copy -------------------------------------------------

void test_grade_buffer_copy() {
    auto& cpu = CpuDevice::instance();

    GradeBuffer src;
    src.allocate(cpu, 4, ALL_GRADES);

    // Fill with test data
    for (int g = 0; g < NUM_GRADES; ++g) {
        for (size_t i = 0; i < src.counts[g]; ++i) {
            src.ptrs[g][i] = static_cast<float>(g + i * 0.1f);
        }
    }

    GradeBuffer dst;
    dst.allocate(cpu, 4, ALL_GRADES);
    GradeBuffer::copy(dst, src);

    // Verify copy
    for (int g = 0; g < NUM_GRADES; ++g) {
        for (size_t i = 0; i < src.counts[g]; ++i) {
            assert(dst.ptrs[g][i] == src.ptrs[g][i]);
        }
    }

    std::cout << "  [PASS] test_grade_buffer_copy\n";
}

// ---- Main ------------------------------------------------------------------

int main() {
    std::cout << "=== test_memory ===\n";
    test_grade_pool_basic();
    test_grade_pool_unique_ptrs();
    test_all_grade_pools();
    test_memory_manager_basic();
    test_grade_buffer();
    test_grade_buffer_copy();
    std::cout << "All memory tests passed.\n";
    return 0;
}
