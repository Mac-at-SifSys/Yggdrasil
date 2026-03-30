#pragma once
// ============================================================================
// YGGDRASIL L2 Runtime — scheduler.h
// Operation scheduling with grade-aware parallelism detection.
// If two operations touch disjoint grade sets they can run concurrently.
// ============================================================================

#include "graph.h"
#include "device.h"
#include <vector>
#include <functional>
#include <future>
#include <thread>

namespace yggdrasil {

// ---- Execution batch -------------------------------------------------------
// A batch contains operations that are safe to execute concurrently.
struct ExecBatch {
    int                batch_id = 0;
    std::vector<int>   node_ids;       // nodes in this batch
    GradeMask          combined_mask;  // union of all grade masks in batch
};

// ---- Execution plan --------------------------------------------------------
struct ExecutionPlan {
    std::vector<ExecBatch> batches;
    size_t total_ops = 0;

    void clear() { batches.clear(); total_ops = 0; }
};

// ---- Scheduler -------------------------------------------------------------
class Scheduler {
public:
    Scheduler() = default;

    // Build an execution plan from a sorted computation graph.
    // The plan groups independent nodes into parallel batches.
    ExecutionPlan build_plan(ComputationGraph& graph);

    // Check whether two nodes can run in parallel:
    // (1) no data dependency, and (2) disjoint grade masks
    static bool can_parallelize(const GraphNode& a, const GraphNode& b,
                                const ComputationGraph& graph);

    // Execute a plan.
    // `kernel` is called for each node; the scheduler manages concurrency.
    using KernelFn = std::function<void(const GraphNode& node)>;
    void execute(const ExecutionPlan& plan, const ComputationGraph& graph,
                 KernelFn kernel, bool async = false);

    // Execute a plan using an async launcher (returns futures per batch).
    using AsyncKernelFn = std::function<std::future<void>(const GraphNode& node)>;
    void execute_async(const ExecutionPlan& plan, const ComputationGraph& graph,
                       AsyncKernelFn kernel);

private:
    // Grade-aware dependency analysis: returns true if `node` depends on `dep`
    // through any data path.
    static bool has_dependency(int node_id, int dep_id,
                               const ComputationGraph& graph,
                               const std::vector<int>& topo_order);
};

} // namespace yggdrasil
