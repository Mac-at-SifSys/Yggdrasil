// ============================================================================
// YGGDRASIL L2 Runtime — test_fusion.cpp
// Test operation fusion patterns:
//   1. grade_project(geometric_product(a,b)) → FUSED_GP_GRADE
//   2. GP chain fusion
//   3. Sandwich detection: R * x * R† → FUSED_SANDWICH
// ============================================================================

#include "graph.h"
#include <cassert>
#include <iostream>
#include <algorithm>

// Forward declarations for fusion functions (defined in fusion.cpp)
namespace yggdrasil {
    int fuse_gp_grade(ComputationGraph& graph);
    int fuse_gp_chain(ComputationGraph& graph);
    int fuse_sandwich(ComputationGraph& graph);
    struct FusionResult {
        int gp_grade_fused;
        int gp_chain_fused;
        int sandwich_fused;
        int total() const { return gp_grade_fused + gp_chain_fused + sandwich_fused; }
    };
    FusionResult run_fusion_passes(ComputationGraph& graph);
}

using namespace yggdrasil;

// ---- Test GP + grade_project fusion ----------------------------------------

void test_fuse_gp_grade() {
    ComputationGraph g;

    int a = g.add_input("a");
    int b = g.add_input("b");
    int gp = g.add_binary(OpType::GEOMETRIC_PROD, a, b);
    int proj = g.add_unary(OpType::GRADE_PROJECT, gp, GRADE_0);

    assert(g.live_node_ids().size() == 4);

    int fused = fuse_gp_grade(g);
    assert(fused == 1);

    auto live = g.live_node_ids();
    assert(live.size() == 3);  // a, b, fused_node

    // The projection node should now be FUSED_GP_GRADE
    const auto& fused_node = g.node(proj);
    assert(fused_node.op == OpType::FUSED_GP_GRADE);
    assert(fused_node.inputs.size() == 2);
    assert(fused_node.inputs[0] == a);
    assert(fused_node.inputs[1] == b);
    assert(fused_node.grade_mask == GRADE_0);

    // GP node should be dead
    assert(std::find(live.begin(), live.end(), gp) == live.end());

    std::cout << "  [PASS] test_fuse_gp_grade\n";
}

// ---- Test that GP with multiple consumers is NOT fused ---------------------

void test_fuse_gp_grade_multi_consumer() {
    ComputationGraph g;

    int a = g.add_input("a");
    int b = g.add_input("b");
    int gp = g.add_binary(OpType::GEOMETRIC_PROD, a, b);
    int proj = g.add_unary(OpType::GRADE_PROJECT, gp, GRADE_0);
    int neg  = g.add_unary(OpType::NEGATE, gp);  // second consumer of GP

    int fused = fuse_gp_grade(g);
    assert(fused == 0);  // should NOT fuse because GP has 2 consumers

    assert(g.node(gp).op == OpType::GEOMETRIC_PROD);  // unchanged
    assert(g.live_node_ids().size() == 5);

    std::cout << "  [PASS] test_fuse_gp_grade_multi_consumer\n";
}

// ---- Test GP chain fusion --------------------------------------------------

void test_fuse_gp_chain() {
    // GP(GP(a, b), c) → FUSED_GP_CHAIN(a, b, c)
    ComputationGraph g;

    int a = g.add_input("a");
    int b = g.add_input("b");
    int c = g.add_input("c");
    int gp1 = g.add_binary(OpType::GEOMETRIC_PROD, a, b);
    int gp2 = g.add_binary(OpType::GEOMETRIC_PROD, gp1, c);

    int fused = fuse_gp_chain(g);
    assert(fused == 1);

    auto live = g.live_node_ids();
    assert(live.size() == 4);  // a, b, c, fused

    const auto& fused_node = g.node(gp2);
    assert(fused_node.op == OpType::FUSED_GP_CHAIN);
    assert(fused_node.inputs.size() == 3);
    assert(fused_node.inputs[0] == a);
    assert(fused_node.inputs[1] == b);
    assert(fused_node.inputs[2] == c);

    std::cout << "  [PASS] test_fuse_gp_chain\n";
}

// ---- Test sandwich fusion --------------------------------------------------

void test_fuse_sandwich() {
    // Pattern: GP(GP(r, x), reverse(r)) → FUSED_SANDWICH(r, x)
    ComputationGraph g;

    int r = g.add_input("r");      // rotor
    int x = g.add_input("x");      // operand
    int rev_r = g.add_unary(OpType::REVERSE, r);
    int inner = g.add_binary(OpType::GEOMETRIC_PROD, r, x);
    int outer = g.add_binary(OpType::GEOMETRIC_PROD, inner, rev_r);

    assert(g.live_node_ids().size() == 5);

    int fused = fuse_sandwich(g);
    assert(fused == 1);

    auto live = g.live_node_ids();
    // r, x, and the fused sandwich node should remain
    assert(live.size() == 3);

    const auto& fused_node = g.node(outer);
    assert(fused_node.op == OpType::FUSED_SANDWICH);
    assert(fused_node.inputs.size() == 2);
    assert(fused_node.inputs[0] == r);
    assert(fused_node.inputs[1] == x);

    // inner GP and reverse should be dead
    assert(std::find(live.begin(), live.end(), inner) == live.end());
    assert(std::find(live.begin(), live.end(), rev_r) == live.end());

    std::cout << "  [PASS] test_fuse_sandwich\n";
}

// ---- Test sandwich with wrong rotor (should NOT fuse) ----------------------

void test_fuse_sandwich_wrong_rotor() {
    ComputationGraph g;

    int r1 = g.add_input("r1");
    int r2 = g.add_input("r2");  // different rotor
    int x  = g.add_input("x");
    int rev_r2 = g.add_unary(OpType::REVERSE, r2);
    int inner = g.add_binary(OpType::GEOMETRIC_PROD, r1, x);
    int outer = g.add_binary(OpType::GEOMETRIC_PROD, inner, rev_r2);

    int fused = fuse_sandwich(g);
    assert(fused == 0);  // rotors don't match → no fusion

    assert(g.node(outer).op == OpType::GEOMETRIC_PROD);

    std::cout << "  [PASS] test_fuse_sandwich_wrong_rotor\n";
}

// ---- Test run_fusion_passes ------------------------------------------------

void test_run_all_passes() {
    // Build a graph with a sandwich AND a GP+grade_project
    ComputationGraph g;

    // Sandwich: r * x * r†
    int r = g.add_input("r");
    int x = g.add_input("x");
    int rev_r = g.add_unary(OpType::REVERSE, r);
    int inner = g.add_binary(OpType::GEOMETRIC_PROD, r, x);
    int sw = g.add_binary(OpType::GEOMETRIC_PROD, inner, rev_r);

    // GP + grade_project on different inputs
    int a = g.add_input("a");
    int b = g.add_input("b");
    int gp = g.add_binary(OpType::GEOMETRIC_PROD, a, b);
    int proj = g.add_unary(OpType::GRADE_PROJECT, gp, GRADE_1);

    auto result = run_fusion_passes(g);
    assert(result.sandwich_fused == 1);
    assert(result.gp_grade_fused == 1);
    assert(result.total() >= 2);

    // Check final ops
    assert(g.node(sw).op == OpType::FUSED_SANDWICH);
    assert(g.node(proj).op == OpType::FUSED_GP_GRADE);

    std::cout << "  [PASS] test_run_all_passes\n";
}

// ---- Test topological sort still works after fusion ------------------------

void test_sort_after_fusion() {
    ComputationGraph g;

    int a = g.add_input("a");
    int b = g.add_input("b");
    int gp = g.add_binary(OpType::GEOMETRIC_PROD, a, b);
    int proj = g.add_unary(OpType::GRADE_PROJECT, gp, GRADE_0);

    fuse_gp_grade(g);

    // Should still be sortable
    auto order = g.topological_sort();
    auto live = g.live_node_ids();
    assert(order.size() == live.size());

    // Inputs should come before the fused node
    int a_pos = -1, b_pos = -1, f_pos = -1;
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        if (order[i] == a) a_pos = i;
        if (order[i] == b) b_pos = i;
        if (order[i] == proj) f_pos = i;  // proj was replaced by fused
    }
    assert(a_pos < f_pos);
    assert(b_pos < f_pos);

    std::cout << "  [PASS] test_sort_after_fusion\n";
}

// ---- Main ------------------------------------------------------------------

int main() {
    std::cout << "=== test_fusion ===\n";
    test_fuse_gp_grade();
    test_fuse_gp_grade_multi_consumer();
    test_fuse_gp_chain();
    test_fuse_sandwich();
    test_fuse_sandwich_wrong_rotor();
    test_run_all_passes();
    test_sort_after_fusion();
    std::cout << "All fusion tests passed.\n";
    return 0;
}
