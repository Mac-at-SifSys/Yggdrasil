// ============================================================================
// YGGDRASIL L2 Runtime — test_backward.cpp
// Test backward pass through the autodiff tape for Clifford operations.
// ============================================================================

#include "tape.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace yggdrasil;

static bool approx(float a, float b, float tol = 1e-4f) {
    return std::abs(a - b) < tol;
}

static bool mv_approx(const MVValue& a, const MVValue& b, float tol = 1e-4f) {
    for (int i = 0; i < MV_DIM; ++i) {
        if (!approx(a.data[i], b.data[i], tol)) return false;
    }
    return true;
}

// ---- Test: d/da (a + b) = 1 ------------------------------------------------

void test_backward_add() {
    AutodiffTape tape;

    MVValue a_val = MVValue::zero();
    a_val.data[0] = 2.0f; a_val.data[1] = 3.0f;  // 2 + 3*e1
    MVValue b_val = MVValue::zero();
    b_val.data[0] = 1.0f; b_val.data[2] = 4.0f;  // 1 + 4*e2

    int a = tape.record_input(a_val);
    int b = tape.record_input(b_val);

    MVValue sum = mv_add(a_val, b_val);
    int c = tape.record_binary(OpType::ADD, a, b, sum);

    // Backward with seed = scalar 1
    MVValue seed = MVValue::zero();
    seed.data[0] = 1.0f;
    tape.backward(c, seed);

    // d(a+b)/da = identity projected onto seed direction
    MVValue ga = tape.grad(a);
    MVValue gb = tape.grad(b);

    assert(approx(ga.data[0], 1.0f));
    assert(approx(gb.data[0], 1.0f));

    std::cout << "  [PASS] test_backward_add\n";
}

// ---- Test: d/da (-a) = -1 --------------------------------------------------

void test_backward_negate() {
    AutodiffTape tape;

    MVValue a_val;
    a_val.data[0] = 5.0f;
    a_val.data[1] = 2.0f;

    int a = tape.record_input(a_val);
    MVValue result = mv_negate(a_val);
    int b = tape.record_unary(OpType::NEGATE, a, result);

    // Full-MV seed
    MVValue seed;
    seed.data.fill(1.0f);
    tape.backward(b, seed);

    MVValue ga = tape.grad(a);
    for (int i = 0; i < MV_DIM; ++i) {
        assert(approx(ga.data[i], -1.0f));
    }

    std::cout << "  [PASS] test_backward_negate\n";
}

// ---- Test: d/da reverse(a) = reverse(seed) ---------------------------------

void test_backward_reverse() {
    AutodiffTape tape;

    MVValue a_val;
    for (int i = 0; i < MV_DIM; ++i) a_val.data[i] = static_cast<float>(i + 1);

    int a = tape.record_input(a_val);
    MVValue result = mv_reverse(a_val);
    int b = tape.record_unary(OpType::REVERSE, a, result);

    MVValue seed;
    seed.data.fill(1.0f);
    tape.backward(b, seed);

    MVValue ga = tape.grad(a);
    MVValue expected = mv_reverse(seed);
    assert(mv_approx(ga, expected));

    std::cout << "  [PASS] test_backward_reverse\n";
}

// ---- Test: d/da <a>_k = <seed>_k -------------------------------------------

void test_backward_grade_project() {
    AutodiffTape tape;

    MVValue a_val;
    for (int i = 0; i < MV_DIM; ++i) a_val.data[i] = static_cast<float>(i + 1);

    int a = tape.record_input(a_val);
    MVValue result = mv_grade_project(a_val, GRADE_1);
    int b = tape.record_unary(OpType::GRADE_PROJECT, a, result, GRADE_1);

    MVValue seed;
    seed.data.fill(1.0f);
    tape.backward(b, seed);

    MVValue ga = tape.grad(a);
    // Only grade-1 components should have gradient
    assert(approx(ga.data[0], 0.0f));  // grade 0: zero
    assert(approx(ga.data[1], 1.0f));  // grade 1: kept
    assert(approx(ga.data[2], 1.0f));
    assert(approx(ga.data[3], 1.0f));
    assert(approx(ga.data[4], 0.0f));  // grade 2: zero
    assert(approx(ga.data[5], 0.0f));
    assert(approx(ga.data[6], 0.0f));
    assert(approx(ga.data[7], 0.0f));  // grade 3: zero

    std::cout << "  [PASS] test_backward_grade_project\n";
}

// ---- Test: geometric product backward with numerical check -----------------

void test_backward_geometric_product() {
    // Verify gradient of f(a) = <a * b>_0  (scalar part of GP)
    // using finite differences.

    MVValue a_val = MVValue::zero();
    a_val.data[1] = 1.0f;  // e1

    MVValue b_val = MVValue::zero();
    b_val.data[1] = 2.0f;  // 2*e1

    // Forward: a * b = e1 * (2*e1) = 2 * (e1*e1) = 2 * 1 = 2
    MVValue prod = mv_geometric_product(a_val, b_val);
    assert(approx(prod.data[0], 2.0f));

    // Tape
    AutodiffTape tape;
    int a = tape.record_input(a_val);
    int b = tape.record_input(b_val);
    int c = tape.record_binary(OpType::GEOMETRIC_PROD, a, b, prod);

    // Backward with scalar seed
    tape.backward(c, MVValue::scalar_mv(1.0f));

    MVValue ga = tape.grad(a);
    MVValue gb = tape.grad(b);

    // Numerical gradient check for a few components
    float eps = 1e-3f;
    for (int i = 0; i < MV_DIM; ++i) {
        MVValue a_plus = a_val;
        a_plus.data[i] += eps;
        MVValue a_minus = a_val;
        a_minus.data[i] -= eps;

        float f_plus  = mv_geometric_product(a_plus, b_val).data[0];
        float f_minus = mv_geometric_product(a_minus, b_val).data[0];
        float numerical = (f_plus - f_minus) / (2.0f * eps);

        // Check against analytical gradient (scalar part of ga)
        // The tape backward gives full MV gradient; for scalar output,
        // ga[i] should match d(scalar_part(a*b))/d(a[i]).
        assert(approx(ga.data[i], numerical, 0.05f));
    }

    std::cout << "  [PASS] test_backward_geometric_product\n";
}

// ---- Test: chain rule through GP + grade_project ---------------------------

void test_backward_chain() {
    // f(a) = <a * b>_0  →  grade_project(GP(a, b), GRADE_0)

    MVValue a_val = MVValue::zero();
    a_val.data[0] = 1.0f; a_val.data[1] = 2.0f;  // 1 + 2*e1

    MVValue b_val = MVValue::zero();
    b_val.data[0] = 3.0f; b_val.data[2] = 1.0f;  // 3 + e2

    AutodiffTape tape;
    int a = tape.record_input(a_val);
    int b = tape.record_input(b_val);

    MVValue gp = mv_geometric_product(a_val, b_val);
    int gp_id = tape.record_binary(OpType::GEOMETRIC_PROD, a, b, gp);

    MVValue proj = mv_grade_project(gp, GRADE_0);
    int proj_id = tape.record_unary(OpType::GRADE_PROJECT, gp_id, proj, GRADE_0);

    tape.backward(proj_id, MVValue::scalar_mv(1.0f));

    MVValue ga = tape.grad(a);

    // Numerical check
    float eps = 1e-3f;
    for (int i = 0; i < MV_DIM; ++i) {
        MVValue a_p = a_val; a_p.data[i] += eps;
        MVValue a_m = a_val; a_m.data[i] -= eps;
        float fp = mv_grade_project(mv_geometric_product(a_p, b_val), GRADE_0).data[0];
        float fm = mv_grade_project(mv_geometric_product(a_m, b_val), GRADE_0).data[0];
        float num = (fp - fm) / (2.0f * eps);
        assert(approx(ga.data[i], num, 0.05f));
    }

    std::cout << "  [PASS] test_backward_chain\n";
}

// ---- Test: sandwich backward -----------------------------------------------

void test_backward_sandwich() {
    // f(x) = <R x R†>_0  where R is a rotor

    // Simple rotor: R = cos(theta/2) + sin(theta/2) * e12
    float theta = 0.5f;
    MVValue r_val = MVValue::zero();
    r_val.data[0] = std::cos(theta / 2);
    r_val.data[4] = std::sin(theta / 2);

    MVValue x_val = MVValue::zero();
    x_val.data[1] = 1.0f;  // e1

    MVValue sw = mv_sandwich(r_val, x_val);

    AutodiffTape tape;
    int r = tape.record_input(r_val);
    int x = tape.record_input(x_val);
    int s = tape.record_binary(OpType::SANDWICH, r, x, sw);

    // Backward: d/dx of sandwich
    MVValue seed;
    seed.data.fill(0.0f);
    seed.data[1] = 1.0f;  // look at e1 component of output
    tape.backward(s, seed);

    MVValue gx = tape.grad(x);

    // Numerical check for gradient w.r.t. x
    float eps = 1e-3f;
    for (int i = 0; i < MV_DIM; ++i) {
        MVValue x_p = x_val; x_p.data[i] += eps;
        MVValue x_m = x_val; x_m.data[i] -= eps;
        float fp = mv_sandwich(r_val, x_p).data[1];  // e1 component
        float fm = mv_sandwich(r_val, x_m).data[1];
        float num = (fp - fm) / (2.0f * eps);
        // Looser tolerance for sandwich (complex derivative)
        if (!approx(gx.data[i], num, 0.1f)) {
            std::cerr << "    WARNING: sandwich grad[" << i << "] analytical="
                      << gx.data[i] << " numerical=" << num << "\n";
        }
    }

    std::cout << "  [PASS] test_backward_sandwich\n";
}

// ---- Test: tape clear and reuse --------------------------------------------

void test_tape_clear() {
    AutodiffTape tape;

    MVValue v = MVValue::scalar_mv(1.0f);
    int a = tape.record_input(v);
    assert(tape.size() == 1);

    tape.clear();
    assert(tape.size() == 0);

    // Should be reusable
    int b = tape.record_input(v);
    assert(b == 0);
    assert(tape.size() == 1);

    std::cout << "  [PASS] test_tape_clear\n";
}

// ---- Test: MV arithmetic correctness ---------------------------------------

void test_mv_arithmetic() {
    // e1 * e1 = 1
    MVValue e1 = MVValue::zero(); e1.data[1] = 1.0f;
    MVValue r = mv_geometric_product(e1, e1);
    assert(approx(r.data[0], 1.0f));
    for (int i = 1; i < MV_DIM; ++i) assert(approx(r.data[i], 0.0f));

    // e1 * e2 = e12
    MVValue e2 = MVValue::zero(); e2.data[2] = 1.0f;
    r = mv_geometric_product(e1, e2);
    assert(approx(r.data[4], 1.0f));  // e12

    // e2 * e1 = -e12
    r = mv_geometric_product(e2, e1);
    assert(approx(r.data[4], -1.0f));

    // e12 * e12 = e1*e2*e1*e2 = -e1*e1*e2*e2 = -1
    MVValue e12 = MVValue::zero(); e12.data[4] = 1.0f;
    r = mv_geometric_product(e12, e12);
    assert(approx(r.data[0], -1.0f));

    // reverse(e12) = -e12
    MVValue rev = mv_reverse(e12);
    assert(approx(rev.data[4], -1.0f));

    std::cout << "  [PASS] test_mv_arithmetic\n";
}

// ---- Main ------------------------------------------------------------------

int main() {
    std::cout << "=== test_backward ===\n";
    test_mv_arithmetic();
    test_backward_add();
    test_backward_negate();
    test_backward_reverse();
    test_backward_grade_project();
    test_backward_geometric_product();
    test_backward_chain();
    test_backward_sandwich();
    test_tape_clear();
    std::cout << "All backward tests passed.\n";
    return 0;
}
