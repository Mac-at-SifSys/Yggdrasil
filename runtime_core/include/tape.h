#pragma once
// ============================================================================
// YGGDRASIL L2 Runtime — tape.h
// Autodiff tape that records multivector-level operations (not scalar).
// Stores forward values for backward pass and integrates with the Clifford
// derivative rules from L1.
// ============================================================================

#include "device.h"
#include "graph.h"
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <string>

namespace yggdrasil {

// ---- Multivector value (8 floats for Cl(3,0)) ------------------------------
struct MVValue {
    std::array<float, MV_DIM> data = {};

    float& operator[](int i)       { return data[i]; }
    float  operator[](int i) const { return data[i]; }

    // Grade accessors (views into the 8-component layout)
    float  scalar()   const { return data[0]; }
    float& scalar()         { return data[0]; }

    // Grade-k slice pointer (non-owning)
    const float* grade_ptr(int g) const { return data.data() + grade_offset(g); }
    float*       grade_ptr(int g)       { return data.data() + grade_offset(g); }

    static MVValue zero() { MVValue v; v.data.fill(0.0f); return v; }
    static MVValue scalar_mv(float s) { MVValue v = zero(); v.data[0] = s; return v; }
};

// Arithmetic helpers
MVValue mv_add(const MVValue& a, const MVValue& b);
MVValue mv_sub(const MVValue& a, const MVValue& b);
MVValue mv_negate(const MVValue& a);
MVValue mv_scale(const MVValue& a, float s);
MVValue mv_reverse(const MVValue& a);
MVValue mv_grade_involution(const MVValue& a);
MVValue mv_grade_project(const MVValue& a, GradeMask mask);
MVValue mv_geometric_product(const MVValue& a, const MVValue& b);
MVValue mv_outer_product(const MVValue& a, const MVValue& b);
MVValue mv_inner_product(const MVValue& a, const MVValue& b);
MVValue mv_sandwich(const MVValue& r, const MVValue& x);

// ---- Tape entry ------------------------------------------------------------
struct TapeEntry {
    int          id = -1;
    OpType       op = OpType::INPUT;
    std::vector<int> input_ids;        // tape ids of inputs
    MVValue      forward_value;        // stored for backward pass
    std::vector<MVValue> saved_inputs; // input values cached for backward
    GradeMask    grade_mask = ALL_GRADES;
};

// ---- Backward function signature -------------------------------------------
// Given: the tape entry and the adjoint of its output,
// returns adjoints for each input (same order as input_ids).
using BackwardFn = std::function<std::vector<MVValue>(
    const TapeEntry& entry, const MVValue& output_adjoint)>;

// ---- AutodiffTape ----------------------------------------------------------
class AutodiffTape {
public:
    AutodiffTape();
    ~AutodiffTape() = default;

    // ---- Recording API (forward pass) ----

    // Record a new variable (leaf)
    int record_input(const MVValue& value, const std::string& label = "");

    // Record a unary operation
    int record_unary(OpType op, int input_id, const MVValue& result,
                     GradeMask mask = ALL_GRADES);

    // Record a binary operation
    int record_binary(OpType op, int lhs_id, int rhs_id, const MVValue& result,
                      GradeMask mask = ALL_GRADES);

    // ---- Backward pass ----

    // Compute gradients from `output_id` back to all recorded leaves.
    // `seed` is the adjoint of the output (default: scalar 1).
    void backward(int output_id, const MVValue& seed = MVValue::scalar_mv(1.0f));

    // Retrieve the adjoint (gradient) for a recorded entry.
    MVValue grad(int id) const;

    // ---- Utilities ----

    const TapeEntry& entry(int id) const;
    size_t size() const { return entries_.size(); }
    void clear();

    // Is this entry a leaf (input)?
    bool is_leaf(int id) const;

    // Get forward value
    MVValue value(int id) const;

private:
    std::vector<TapeEntry> entries_;
    std::unordered_map<int, MVValue> adjoints_;

    // Built-in backward rules for each OpType
    static std::vector<MVValue> backward_rule(const TapeEntry& entry,
                                              const MVValue& out_adj);
};

} // namespace yggdrasil
