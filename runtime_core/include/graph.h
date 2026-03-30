#pragma once
// ============================================================================
// YGGDRASIL L2 Runtime — graph.h
// Computation graph with Clifford-typed nodes for the geometric algebra HLM.
// Supports both eager and deferred (graph) execution modes.
// ============================================================================

#include "device.h"
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>

namespace yggdrasil {

// ---- Clifford operation types ----------------------------------------------
enum class OpType : int {
    // Leaf / input
    INPUT           = 0,
    CONSTANT        = 1,

    // Unary
    NEGATE          = 10,
    REVERSE         = 11,
    GRADE_INVOLUTION= 12,
    CLIFFORD_CONJ   = 13,
    DUAL            = 14,
    GRADE_PROJECT   = 15,   // parameterised by grade_mask
    NORM            = 16,

    // Binary products
    GEOMETRIC_PROD  = 20,
    OUTER_PROD      = 21,
    INNER_PROD      = 22,   // left contraction
    SCALAR_PROD     = 23,
    REGRESSIVE_PROD = 24,
    SANDWICH        = 25,   // R x R†

    // Scalar-MV
    SCALAR_MUL      = 30,
    SCALAR_ADD      = 31,

    // Element-wise
    ADD             = 40,
    SUB             = 41,

    // Special
    EXP             = 50,   // multivector exponential (bivector)
    LOG             = 51,

    // Fused (created by fusion pass)
    FUSED_GP_GRADE  = 100,  // geometric_product + grade_project
    FUSED_SANDWICH  = 101,  // R x R†  (explicit fusion)
    FUSED_GP_CHAIN  = 102,  // chain of GPs sharing operands
};

inline std::string op_name(OpType op) {
    switch (op) {
        case OpType::INPUT:            return "Input";
        case OpType::CONSTANT:         return "Constant";
        case OpType::NEGATE:           return "Negate";
        case OpType::REVERSE:          return "Reverse";
        case OpType::GRADE_INVOLUTION: return "GradeInvolution";
        case OpType::CLIFFORD_CONJ:    return "CliffordConj";
        case OpType::DUAL:             return "Dual";
        case OpType::GRADE_PROJECT:    return "GradeProject";
        case OpType::NORM:             return "Norm";
        case OpType::GEOMETRIC_PROD:   return "GeometricProduct";
        case OpType::OUTER_PROD:       return "OuterProduct";
        case OpType::INNER_PROD:       return "InnerProduct";
        case OpType::SCALAR_PROD:      return "ScalarProduct";
        case OpType::REGRESSIVE_PROD:  return "RegressiveProduct";
        case OpType::SANDWICH:         return "Sandwich";
        case OpType::SCALAR_MUL:       return "ScalarMul";
        case OpType::SCALAR_ADD:       return "ScalarAdd";
        case OpType::ADD:              return "Add";
        case OpType::SUB:              return "Sub";
        case OpType::EXP:              return "Exp";
        case OpType::LOG:              return "Log";
        case OpType::FUSED_GP_GRADE:   return "FusedGPGrade";
        case OpType::FUSED_SANDWICH:   return "FusedSandwich";
        case OpType::FUSED_GP_CHAIN:   return "FusedGPChain";
        default:                       return "Unknown";
    }
}

// ---- Tensor shape descriptor -----------------------------------------------
struct TensorShape {
    size_t num_mv = 1;        // batch size (number of multivectors)
    GradeMask grade_mask = ALL_GRADES;  // which grades are live

    bool operator==(const TensorShape& o) const {
        return num_mv == o.num_mv && grade_mask == o.grade_mask;
    }
};

// ---- Graph node ------------------------------------------------------------
struct GraphNode {
    int                 id = -1;
    OpType              op = OpType::INPUT;
    std::vector<int>    inputs;         // indices of input nodes
    TensorShape         shape;
    GradeMask           grade_mask = ALL_GRADES;  // op-level grade restriction
    DeviceType          device = DeviceType::CPU;
    std::string         label;          // optional human-readable name

    // For CONSTANT nodes: inline data (small multivectors)
    std::vector<float>  const_data;

    // For fused ops: the constituent ops
    std::vector<OpType> fused_ops;

    // Metadata written during scheduling
    int                 topo_order = -1;
    int                 batch_id   = -1;  // parallel batch assignment
};

// ---- Execution mode --------------------------------------------------------
enum class ExecMode {
    EAGER,   // execute immediately on add
    GRAPH,   // build graph, then optimize & execute
};

// ---- ComputationGraph ------------------------------------------------------
class ComputationGraph {
public:
    explicit ComputationGraph(ExecMode mode = ExecMode::GRAPH);

    // Add a node; returns its id.
    int add_node(GraphNode node);

    // Add typed helper shortcuts
    int add_input(const std::string& label, size_t num_mv = 1,
                  GradeMask mask = ALL_GRADES,
                  DeviceType dev = DeviceType::CPU);

    int add_constant(const std::vector<float>& data, size_t num_mv = 1,
                     GradeMask mask = ALL_GRADES);

    int add_unary(OpType op, int input, GradeMask mask = ALL_GRADES);

    int add_binary(OpType op, int lhs, int rhs, GradeMask mask = ALL_GRADES);

    // Access nodes
    const GraphNode& node(int id) const;
    GraphNode&       node(int id);
    size_t           size() const { return nodes_.size(); }

    // Topological sort — writes topo_order into each node.
    // Returns ordered list of node ids.
    std::vector<int> topological_sort();

    // Get previously computed execution order (call topological_sort first).
    const std::vector<int>& execution_order() const { return exec_order_; }

    // Replace a node (used by fusion passes)
    void replace_node(int id, GraphNode new_node);

    // Mark a node as dead (fused away)
    void remove_node(int id);

    // Iterate live nodes
    std::vector<int> live_node_ids() const;

    ExecMode mode() const { return mode_; }

private:
    ExecMode mode_;
    std::vector<GraphNode> nodes_;
    std::vector<bool>      alive_;
    std::vector<int>       exec_order_;
    bool                   sorted_ = false;
};

} // namespace yggdrasil
