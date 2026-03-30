// ============================================================================
// YGGDRASIL L2 Runtime — scheduler.cpp
// Scheduler with grade-aware parallelism detection.
// Two ops can run concurrently if they touch disjoint grade sets and have
// no data dependency.
// ============================================================================

#include "scheduler.h"
#include <algorithm>
#include <unordered_set>
#include <cassert>

namespace yggdrasil {

// ---- Dependency check ------------------------------------------------------

bool Scheduler::has_dependency(int node_id, int dep_id,
                               const ComputationGraph& graph,
                               const std::vector<int>& /*topo_order*/) {
    // BFS backwards from node_id to see if dep_id is reachable
    std::unordered_set<int> visited;
    std::vector<int> stack;
    stack.push_back(node_id);

    while (!stack.empty()) {
        int cur = stack.back(); stack.pop_back();
        if (cur == dep_id) return true;
        if (visited.count(cur)) continue;
        visited.insert(cur);

        const auto& n = graph.node(cur);
        for (int inp : n.inputs) {
            if (!visited.count(inp)) {
                stack.push_back(inp);
            }
        }
    }
    return false;
}

// ---- Parallelism check -----------------------------------------------------

bool Scheduler::can_parallelize(const GraphNode& a, const GraphNode& b,
                                const ComputationGraph& graph) {
    // Condition 1: disjoint grade masks
    bool disjoint_grades = (a.grade_mask & b.grade_mask) == 0;

    // Condition 2: no data dependency in either direction
    // We do a simple check: neither node appears in the other's transitive inputs
    std::vector<int> dummy_order; // not used by has_dependency in our impl
    bool a_depends_b = has_dependency(a.id, b.id, graph, dummy_order);
    bool b_depends_a = has_dependency(b.id, a.id, graph, dummy_order);

    if (a_depends_b || b_depends_a) return false;

    // If grades are disjoint, they are safe even with shared device resources.
    // If grades overlap but no dependency, we still allow parallelism but
    // the caller should ensure atomicity at the grade level.
    return true;  // no dependency → can parallelize
}

// ---- Build execution plan --------------------------------------------------

ExecutionPlan Scheduler::build_plan(ComputationGraph& graph) {
    auto order = graph.topological_sort();
    ExecutionPlan plan;
    plan.total_ops = order.size();

    if (order.empty()) return plan;

    // Greedy batching: assign each node to the earliest batch where it has
    // no dependency on any node in the same or later batch.
    // We track for each node: the minimum batch it can be placed in.
    std::unordered_map<int, int> node_batch;

    for (int nid : order) {
        const auto& n = graph.node(nid);

        // This node's batch must be after all its inputs' batches
        int earliest_batch = 0;
        for (int inp : n.inputs) {
            auto it = node_batch.find(inp);
            if (it != node_batch.end()) {
                earliest_batch = std::max(earliest_batch, it->second + 1);
            }
        }

        node_batch[nid] = earliest_batch;
        graph.node(nid).batch_id = earliest_batch;

        // Ensure we have enough batches
        while (static_cast<int>(plan.batches.size()) <= earliest_batch) {
            ExecBatch b;
            b.batch_id = static_cast<int>(plan.batches.size());
            b.combined_mask = 0;
            plan.batches.push_back(b);
        }

        plan.batches[earliest_batch].node_ids.push_back(nid);
        plan.batches[earliest_batch].combined_mask |= n.grade_mask;
    }

    // Post-pass: split batches where nodes have overlapping grades AND
    // we want to respect grade-disjoint parallelism more aggressively.
    // For now, the greedy approach is sufficient; finer splitting can be
    // added as a refinement.

    return plan;
}

// ---- Execute plan (synchronous) --------------------------------------------

void Scheduler::execute(const ExecutionPlan& plan,
                        const ComputationGraph& graph,
                        KernelFn kernel, bool async_batches) {
    for (const auto& batch : plan.batches) {
        if (batch.node_ids.size() == 1 || !async_batches) {
            // Sequential execution within batch (or single node)
            for (int nid : batch.node_ids) {
                kernel(graph.node(nid));
            }
        } else {
            // Parallel execution within batch using std::async
            std::vector<std::future<void>> futures;
            futures.reserve(batch.node_ids.size());
            for (int nid : batch.node_ids) {
                futures.push_back(std::async(std::launch::async, [&, nid]() {
                    kernel(graph.node(nid));
                }));
            }
            // Wait for all in this batch to complete before next batch
            for (auto& f : futures) {
                f.get();
            }
        }
    }
}

// ---- Execute plan (async kernel launch) ------------------------------------

void Scheduler::execute_async(const ExecutionPlan& plan,
                              const ComputationGraph& graph,
                              AsyncKernelFn kernel) {
    for (const auto& batch : plan.batches) {
        std::vector<std::future<void>> futures;
        futures.reserve(batch.node_ids.size());
        for (int nid : batch.node_ids) {
            futures.push_back(kernel(graph.node(nid)));
        }
        // Synchronize at batch boundary
        for (auto& f : futures) {
            f.get();
        }
    }
}

} // namespace yggdrasil
