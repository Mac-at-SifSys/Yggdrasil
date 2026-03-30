// ============================================================================
// YGGDRASIL L2 Runtime — test_graph.cpp
// Test computation graph construction, topological sort, and execution order.
// ============================================================================

#include "graph.h"
#include "scheduler.h"
#include <cassert>
#include <iostream>
#include <algorithm>

using namespace yggdrasil;

// ---- Test basic graph construction -----------------------------------------

void test_add_nodes() {
    ComputationGraph g;

    int a = g.add_input("a", 1, ALL_GRADES);
    int b = g.add_input("b", 1, ALL_GRADES);
    int c = g.add_binary(OpType::GEOMETRIC_PROD, a, b);

    assert(g.size() == 3);
    assert(g.node(a).op == OpType::INPUT);
    assert(g.node(b).op == OpType::INPUT);
    assert(g.node(c).op == OpType::GEOMETRIC_PROD);
    assert(g.node(c).inputs.size() == 2);
    assert(g.node(c).inputs[0] == a);
    assert(g.node(c).inputs[1] == b);

    std::cout << "  [PASS] test_add_nodes\n";
}

// ---- Test topological sort -------------------------------------------------

void test_topological_sort_linear() {
    // a -> negate -> grade_project
    ComputationGraph g;
    int a = g.add_input("a");
    int neg = g.add_unary(OpType::NEGATE, a);
    int proj = g.add_unary(OpType::GRADE_PROJECT, neg, GRADE_1);

    auto order = g.topological_sort();
    assert(order.size() == 3);
    assert(order[0] == a);
    assert(order[1] == neg);
    assert(order[2] == proj);

    // Check topo_order field
    assert(g.node(a).topo_order == 0);
    assert(g.node(neg).topo_order == 1);
    assert(g.node(proj).topo_order == 2);

    std::cout << "  [PASS] test_topological_sort_linear\n";
}

void test_topological_sort_diamond() {
    //     a
    //    / \  .
    //   b   c
    //    \ /
    //     d
    ComputationGraph g;
    int a = g.add_input("a");
    int b = g.add_unary(OpType::NEGATE, a);
    int c = g.add_unary(OpType::REVERSE, a);
    int d = g.add_binary(OpType::ADD, b, c);

    auto order = g.topological_sort();
    assert(order.size() == 4);

    // a must come first
    assert(order[0] == a);
    // d must come last
    assert(order[3] == d);
    // b and c can be in either order, but both before d
    assert((order[1] == b && order[2] == c) ||
           (order[1] == c && order[2] == b));

    std::cout << "  [PASS] test_topological_sort_diamond\n";
}

// ---- Test remove and live nodes --------------------------------------------

void test_remove_node() {
    ComputationGraph g;
    int a = g.add_input("a");
    int b = g.add_input("b");
    int c = g.add_binary(OpType::GEOMETRIC_PROD, a, b);
    int d = g.add_unary(OpType::GRADE_PROJECT, c, GRADE_0);

    g.remove_node(c);  // mark GP as dead

    auto live = g.live_node_ids();
    assert(live.size() == 3);
    assert(std::find(live.begin(), live.end(), c) == live.end());

    std::cout << "  [PASS] test_remove_node\n";
}

// ---- Test graph with constants ---------------------------------------------

void test_constant_node() {
    ComputationGraph g;
    std::vector<float> data = {1.0f, 0, 0, 0, 0, 0, 0, 0}; // scalar 1
    int c = g.add_constant(data, 1, GRADE_0);

    assert(g.node(c).op == OpType::CONSTANT);
    assert(g.node(c).const_data.size() == 8);
    assert(g.node(c).const_data[0] == 1.0f);

    std::cout << "  [PASS] test_constant_node\n";
}

// ---- Test execution order with scheduler -----------------------------------

void test_scheduler_batching() {
    // Two independent paths: a->b  and  c->d
    ComputationGraph g;
    int a = g.add_input("a", 1, GRADE_1);
    int b = g.add_unary(OpType::NEGATE, a, GRADE_1);
    int c = g.add_input("c", 1, GRADE_2);
    int d = g.add_unary(OpType::REVERSE, c, GRADE_2);
    int e = g.add_binary(OpType::ADD, b, d, ALL_GRADES);

    Scheduler sched;
    auto plan = sched.build_plan(g);

    // Should have at least 2 batches:
    //   Batch 0: inputs a, c
    //   Batch 1: negate(a), reverse(c)   [can be parallel — disjoint grades]
    //   Batch 2: add(b, d)
    assert(plan.batches.size() >= 2);
    assert(plan.total_ops == 5);

    // Verify inputs are in the first batch
    bool a_in_first = false, c_in_first = false;
    for (int nid : plan.batches[0].node_ids) {
        if (nid == a) a_in_first = true;
        if (nid == c) c_in_first = true;
    }
    assert(a_in_first && c_in_first);

    std::cout << "  [PASS] test_scheduler_batching\n";
}

// ---- Test can_parallelize --------------------------------------------------

void test_can_parallelize() {
    ComputationGraph g;
    int a = g.add_input("a", 1, GRADE_1);
    int b = g.add_input("b", 1, GRADE_2);
    int op_a = g.add_unary(OpType::NEGATE, a, GRADE_1);
    int op_b = g.add_unary(OpType::REVERSE, b, GRADE_2);

    // op_a and op_b touch disjoint grades and have no dependency
    assert(Scheduler::can_parallelize(g.node(op_a), g.node(op_b), g));

    // op_a depends on a, so a and op_a cannot be parallelized
    assert(!Scheduler::can_parallelize(g.node(a), g.node(op_a), g));

    std::cout << "  [PASS] test_can_parallelize\n";
}

// ---- Main ------------------------------------------------------------------

int main() {
    std::cout << "=== test_graph ===\n";
    test_add_nodes();
    test_topological_sort_linear();
    test_topological_sort_diamond();
    test_remove_node();
    test_constant_node();
    test_scheduler_batching();
    test_can_parallelize();
    std::cout << "All graph tests passed.\n";
    return 0;
}
