// ============================================================================
// YGGDRASIL L2 Runtime — fusion.cpp
// Operation fusion passes for the computation graph:
//   1. grade_project(geometric_product(a,b)) → FUSED_GP_GRADE
//   2. Adjacent geometric products sharing operands → FUSED_GP_CHAIN
//   3. Sandwich detection: a * x * reverse(a) → FUSED_SANDWICH
// ============================================================================

#include "graph.h"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace yggdrasil {

// ---- Helper: collect all consumers of a node --------------------------------
static std::unordered_map<int, std::vector<int>>
build_consumer_map(const ComputationGraph& graph) {
    std::unordered_map<int, std::vector<int>> consumers;
    auto live = graph.live_node_ids();
    for (int nid : live) {
        const auto& n = graph.node(nid);
        for (int inp : n.inputs) {
            consumers[inp].push_back(nid);
        }
    }
    return consumers;
}

// ============================================================================
// Pass 1: Fuse grade_project(geometric_product(a, b)) → FUSED_GP_GRADE
// ============================================================================
// Pattern: node N is GRADE_PROJECT with one input M that is GEOMETRIC_PROD.
// If M has only one consumer (N), fuse them.

int fuse_gp_grade(ComputationGraph& graph) {
    auto consumers = build_consumer_map(graph);
    auto live = graph.live_node_ids();
    int fused_count = 0;

    for (int nid : live) {
        const auto& n = graph.node(nid);
        if (n.op != OpType::GRADE_PROJECT) continue;
        if (n.inputs.size() != 1) continue;

        int gp_id = n.inputs[0];
        const auto& gp_node = graph.node(gp_id);
        if (gp_node.op != OpType::GEOMETRIC_PROD) continue;

        // Check that the GP node has only this one consumer
        auto it = consumers.find(gp_id);
        if (it != consumers.end() && it->second.size() > 1) continue;

        // Fuse: replace node N with FUSED_GP_GRADE that takes GP's inputs
        GraphNode fused;
        fused.op = OpType::FUSED_GP_GRADE;
        fused.inputs = gp_node.inputs;  // a, b
        fused.grade_mask = n.grade_mask; // the projection mask
        fused.shape = n.shape;
        fused.device = n.device;
        fused.label = "fused_gp_grade";
        fused.fused_ops = {OpType::GEOMETRIC_PROD, OpType::GRADE_PROJECT};

        graph.replace_node(nid, std::move(fused));
        graph.remove_node(gp_id);
        fused_count++;
    }

    return fused_count;
}

// ============================================================================
// Pass 2: Fuse adjacent geometric products sharing operands → FUSED_GP_CHAIN
// ============================================================================
// Pattern: node B = GP(A_out, c) where A = GP(a, b)
// If A has only one consumer (B), fuse into GP_CHAIN(a, b, c).

int fuse_gp_chain(ComputationGraph& graph) {
    auto consumers = build_consumer_map(graph);
    auto live = graph.live_node_ids();
    int fused_count = 0;

    for (int nid : live) {
        const auto& n = graph.node(nid);
        if (n.op != OpType::GEOMETRIC_PROD) continue;
        if (n.inputs.size() != 2) continue;

        // Check if the first input is also a GP
        int first_id = n.inputs[0];
        const auto& first = graph.node(first_id);
        if (first.op != OpType::GEOMETRIC_PROD) continue;
        if (first.inputs.size() != 2) continue;

        // Check single consumer
        auto it = consumers.find(first_id);
        if (it != consumers.end() && it->second.size() > 1) continue;

        // Fuse: GP(GP(a,b), c) → FUSED_GP_CHAIN(a, b, c)
        GraphNode fused;
        fused.op = OpType::FUSED_GP_CHAIN;
        fused.inputs = {first.inputs[0], first.inputs[1], n.inputs[1]};
        fused.grade_mask = n.grade_mask;
        fused.shape = n.shape;
        fused.device = n.device;
        fused.label = "fused_gp_chain";
        fused.fused_ops = {OpType::GEOMETRIC_PROD, OpType::GEOMETRIC_PROD};

        graph.replace_node(nid, std::move(fused));
        graph.remove_node(first_id);
        fused_count++;
    }

    return fused_count;
}

// ============================================================================
// Pass 3: Detect sandwich pattern: GP(GP(r, x), reverse(r)) → FUSED_SANDWICH
// ============================================================================
// Pattern:
//   rev_node = REVERSE(r)
//   inner_gp = GEOMETRIC_PROD(r, x)
//   outer_gp = GEOMETRIC_PROD(inner_gp, rev_node)
// Fuse into FUSED_SANDWICH(r, x)

int fuse_sandwich(ComputationGraph& graph) {
    auto consumers = build_consumer_map(graph);
    auto live = graph.live_node_ids();
    int fused_count = 0;

    for (int nid : live) {
        const auto& n = graph.node(nid);
        if (n.op != OpType::GEOMETRIC_PROD) continue;
        if (n.inputs.size() != 2) continue;

        int inner_id = n.inputs[0];
        int rev_id   = n.inputs[1];

        const auto& inner = graph.node(inner_id);
        const auto& rev_node = graph.node(rev_id);

        // Check pattern: inner is GP, rev_node is REVERSE
        if (inner.op != OpType::GEOMETRIC_PROD) continue;
        if (rev_node.op != OpType::REVERSE) continue;
        if (inner.inputs.size() != 2) continue;
        if (rev_node.inputs.size() != 1) continue;

        // Check that the reverse is applied to the same rotor as the inner GP's first arg
        int r_in_inner = inner.inputs[0];
        int r_in_rev   = rev_node.inputs[0];
        if (r_in_inner != r_in_rev) continue;

        int x_id = inner.inputs[1];

        // Check single consumers for the nodes being fused away
        {
            auto it1 = consumers.find(inner_id);
            if (it1 != consumers.end() && it1->second.size() > 1) continue;
            auto it2 = consumers.find(rev_id);
            if (it2 != consumers.end() && it2->second.size() > 1) continue;
        }

        // Fuse: GP(GP(r, x), reverse(r)) → FUSED_SANDWICH(r, x)
        GraphNode fused;
        fused.op = OpType::FUSED_SANDWICH;
        fused.inputs = {r_in_inner, x_id};
        fused.grade_mask = n.grade_mask;
        fused.shape = n.shape;
        fused.device = n.device;
        fused.label = "fused_sandwich";
        fused.fused_ops = {OpType::GEOMETRIC_PROD, OpType::REVERSE,
                           OpType::GEOMETRIC_PROD};

        graph.replace_node(nid, std::move(fused));
        graph.remove_node(inner_id);
        graph.remove_node(rev_id);
        fused_count++;
    }

    return fused_count;
}

// ============================================================================
// Run all fusion passes
// ============================================================================

struct FusionResult {
    int gp_grade_fused;
    int gp_chain_fused;
    int sandwich_fused;
    int total() const { return gp_grade_fused + gp_chain_fused + sandwich_fused; }
};

FusionResult run_fusion_passes(ComputationGraph& graph) {
    FusionResult result;

    // Run sandwich first (most specific pattern)
    result.sandwich_fused = fuse_sandwich(graph);

    // Then GP+grade projection
    result.gp_grade_fused = fuse_gp_grade(graph);

    // Then GP chains
    result.gp_chain_fused = fuse_gp_chain(graph);

    return result;
}

} // namespace yggdrasil
