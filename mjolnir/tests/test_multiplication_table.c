/**
 * test_multiplication_table.c — Verify all 64 products in the Cl(3,0) table
 *
 * Tests every basis element product against the known Cl(3,0) algebra.
 */

#include <stdio.h>
#include <math.h>
#include "../include/cl3_types.h"
#include "../include/cl3_tables.h"
#include "../include/cl3_ops.h"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_FLOAT_EQ(a, b, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) < 1e-6f) { tests_passed++; } \
    else { printf("FAIL: %s (expected %f, got %f)\n", msg, (float)(b), (float)(a)); } \
} while(0)

/* Create a basis element multivector: only component idx is 1.0 */
static Cl3Multivector basis(int idx) {
    Cl3Multivector m = cl3_zero();
    cl3_set(&m, idx, 1.0f);
    return m;
}

static void test_all_basis_products(void) {
    printf("Testing all 64 basis element products...\n");

    /* Expected results from the Cl(3,0) multiplication table */
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            Cl3Multivector bi = basis(i);
            Cl3Multivector bj = basis(j);
            Cl3Multivector result = cl3_geometric_product(&bi, &bj);

            int expected_idx = CL3_PRODUCT_IDX[i][j];
            int expected_sign = CL3_PRODUCT_SIGN[i][j];

            /* Check that only the expected component is nonzero */
            for (int k = 0; k < 8; k++) {
                float expected_val = (k == expected_idx) ? (float)expected_sign : 0.0f;
                char msg[128];
                snprintf(msg, sizeof(msg), "basis[%d]*basis[%d] component[%d]", i, j, k);
                ASSERT_FLOAT_EQ(cl3_get(&result, k), expected_val, msg);
            }
        }
    }
}

static void test_vector_squares(void) {
    printf("Testing e_i^2 = +1 for i=1,2,3 (signature (3,0))...\n");

    /* In Cl(3,0): e1^2 = e2^2 = e3^2 = +1 */
    for (int i = 1; i <= 3; i++) {
        Cl3Multivector v = basis(i);
        Cl3Multivector sq = cl3_geometric_product(&v, &v);
        char msg[64];
        snprintf(msg, sizeof(msg), "e%d^2 = +1", i);
        ASSERT_FLOAT_EQ(sq.s, 1.0f, msg);
        /* All other components should be zero */
        ASSERT_FLOAT_EQ(cl3_norm(&sq) - fabsf(sq.s), 0.0f, "no other components");
    }
}

static void test_anticommutation(void) {
    printf("Testing anti-commutation: e_i * e_j = -e_j * e_i for i!=j...\n");

    for (int i = 1; i <= 3; i++) {
        for (int j = i + 1; j <= 3; j++) {
            Cl3Multivector vi = basis(i);
            Cl3Multivector vj = basis(j);
            Cl3Multivector ij = cl3_geometric_product(&vi, &vj);
            Cl3Multivector ji = cl3_geometric_product(&vj, &vi);
            Cl3Multivector sum = cl3_add(&ij, &ji);

            char msg[64];
            for (int k = 0; k < 8; k++) {
                snprintf(msg, sizeof(msg), "e%d*e%d + e%d*e%d component[%d] = 0", i, j, j, i, k);
                ASSERT_FLOAT_EQ(cl3_get(&sum, k), 0.0f, msg);
            }
        }
    }
}

static void test_pseudoscalar(void) {
    printf("Testing pseudoscalar: e123^2 = -1...\n");

    Cl3Multivector ps = basis(7);  /* e123 */
    Cl3Multivector sq = cl3_geometric_product(&ps, &ps);
    ASSERT_FLOAT_EQ(sq.s, -1.0f, "e123^2 scalar = -1");
}

int main(void) {
    printf("=== Cl(3,0) Multiplication Table Tests ===\n\n");

    test_all_basis_products();
    test_vector_squares();
    test_anticommutation();
    test_pseudoscalar();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
