// ============================================================================
// YGGDRASIL L2 Runtime — graph.cpp
// Full implementation of computation graph with Clifford-typed nodes.
// ============================================================================

#include "graph.h"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace yggdrasil {

// ---- ComputationGraph ------------------------------------------------------

ComputationGraph::ComputationGraph(ExecMode mode)
    : mode_(mode) {}

int ComputationGraph::add_node(GraphNode node) {
    int id = static_cast<int>(nodes_.size());
    node.id = id;
    nodes_.push_back(std::move(node));
    alive_.push_back(true);
    sorted_ = false;
    return id;
}

int ComputationGraph::add_input(const std::string& label, size_t num_mv,
                                 GradeMask mask, DeviceType dev) {
    GraphNode n;
    n.op = OpType::INPUT;
    n.label = label;
    n.shape.num_mv = num_mv;
    n.shape.grade_mask = mask;
    n.grade_mask = mask;
    n.device = dev;
    return add_node(std::move(n));
}

int ComputationGraph::add_constant(const std::vector<float>& data,
                                    size_t num_mv, GradeMask mask) {
    GraphNode n;
    n.op = OpType::CONSTANT;
    n.const_data = data;
    n.shape.num_mv = num_mv;
    n.shape.grade_mask = mask;
    n.grade_mask = mask;
    return add_node(std::move(n));
}

int ComputationGraph::add_unary(OpType op, int input, GradeMask mask) {
    if (input < 0 || input >= static_cast<int>(nodes_.size()))
        throw std::out_of_range("add_unary: invalid input index");

    GraphNode n;
    n.op = op;
    n.inputs = {input};
    n.grade_mask = mask;
    n.shape = nodes_[input].shape;
    n.shape.grade_mask = mask;
    n.device = nodes_[input].device;
    return add_node(std::move(n));
}

int ComputationGraph::add_binary(OpType op, int lhs, int rhs, GradeMask mask) {
    if (lhs < 0 || lhs >= static_cast<int>(nodes_.size()) ||
        rhs < 0 || rhs >= static_cast<int>(nodes_.size()))
        throw std::out_of_range("add_binary: invalid input index");

    GraphNode n;
    n.op = op;
    n.inputs = {lhs, rhs};
    n.grade_mask = mask;

    // Output shape: take max batch, union of grade masks
    n.shape.num_mv = std::max(nodes_[lhs].shape.num_mv,
                               nodes_[rhs].shape.num_mv);
    // For products, all grades may be produced; mask restricts what we keep
    n.shape.grade_mask = mask;
    n.device = nodes_[lhs].device;
    return add_node(std::move(n));
}

const GraphNode& ComputationGraph::node(int id) const {
    if (id < 0 || id >= static_cast<int>(nodes_.size()))
        throw std::out_of_range("node: invalid id");
    return nodes_[id];
}

GraphNode& ComputationGraph::node(int id) {
    if (id < 0 || id >= static_cast<int>(nodes_.size()))
        throw std::out_of_range("node: invalid id");
    return nodes_[id];
}

std::vector<int> ComputationGraph::topological_sort() {
    int n = static_cast<int>(nodes_.size());

    // Compute in-degree (only counting live nodes and live edges)
    std::vector<int> in_degree(n, 0);
    for (int i = 0; i < n; ++i) {
        if (!alive_[i]) continue;
        for (int inp : nodes_[i].inputs) {
            if (inp >= 0 && inp < n && alive_[inp]) {
                in_degree[i]++;
            }
        }
    }

    // Kahn's algorithm
    std::queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (alive_[i] && in_degree[i] == 0) q.push(i);
    }

    exec_order_.clear();
    exec_order_.reserve(n);
    int order = 0;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        nodes_[u].topo_order = order++;
        exec_order_.push_back(u);

        // For each node v that uses u as input
        for (int v = 0; v < n; ++v) {
            if (!alive_[v]) continue;
            for (int inp : nodes_[v].inputs) {
                if (inp == u) {
                    in_degree[v]--;
                    if (in_degree[v] == 0) q.push(v);
                    break; // only count once per edge u→v
                }
            }
        }
    }

    // Check for cycles
    int live_count = 0;
    for (int i = 0; i < n; ++i) {
        if (alive_[i]) live_count++;
    }
    if (static_cast<int>(exec_order_.size()) != live_count) {
        throw std::runtime_error("topological_sort: cycle detected in graph");
    }

    sorted_ = true;
    return exec_order_;
}

void ComputationGraph::replace_node(int id, GraphNode new_node) {
    if (id < 0 || id >= static_cast<int>(nodes_.size()))
        throw std::out_of_range("replace_node: invalid id");
    new_node.id = id;
    nodes_[id] = std::move(new_node);
    sorted_ = false;
}

void ComputationGraph::remove_node(int id) {
    if (id < 0 || id >= static_cast<int>(nodes_.size()))
        throw std::out_of_range("remove_node: invalid id");
    alive_[id] = false;
    sorted_ = false;
}

std::vector<int> ComputationGraph::live_node_ids() const {
    std::vector<int> ids;
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        if (alive_[i]) ids.push_back(i);
    }
    return ids;
}

} // namespace yggdrasil
