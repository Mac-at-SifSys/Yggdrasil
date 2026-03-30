/**
 * test_geometric_product.c — Tests for geometric product correctness
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../include/cl3_types.h"
#include "../include/cl3_ops.h"

static int tests_run = 0;
static int tests_passed = 0;
static const float EPS = 1e-5f;

#define ASSERT_FLOAT_EQ(a, b, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) < EPS) { tests_passed++; } \
    else { printf("FAIL: %s (expected %f, got %f)\n", msg, (float)(b), (float)(a)); } \
} while(0)

#define ASSERT_MV_EQ(a, b, msg) do { \
    for (int _k = 0; _k < 8; _k++) { \
        char _msg[256]; snprintf(_msg, sizeof(_msg), "%s component[%d]", msg, _k); \
        ASSERT_FLOAT_EQ(cl3_get(&(a), _k), cl3_get(&(b), _k), _msg); \
    } \
} while(0)

static Cl3Multivector random_mv(void) {
    Cl3Multivector m;
    m.s = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < 3; i++) m.v[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < 3; i++) m.b[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    m.t = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    return m;
}

static void test_identity(void) {
    printf("Testing: 1 * a = a * 1 = a...\n");
    Cl3Multivector one = cl3_scalar_mv(1.0f);
    Cl3Multivector a = cl3_from_components(2.0f, 3.0f, -1.0f, 0.5f, 1.5f, -0.5f, 2.0f, -1.0f);

    Cl3Multivector r1 = cl3_geometric_product(&one, &a);
    Cl3Multivector r2 = cl3_geometric_product(&a, &one);

    ASSERT_MV_EQ(r1, a, "1*a = a");
    ASSERT_MV_EQ(r2, a, "a*1 = a");
}

static void test_scalar_multiplication(void) {
    printf("Testing: scalar * a = a * scalar = scale(a, scalar)...\n");
    Cl3Multivector s = cl3_scalar_mv(3.0f);
    Cl3Multivector a = cl3_from_components(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);

    Cl3Multivector r = cl3_geometric_product(&s, &a);
    Cl3Multivector expected = cl3_scale(&a, 3.0f);
    ASSERT_MV_EQ(r, expected, "3 * a");
}

static void test_associativity_fuzz(void) {
    printf("Testing associativity: (a*b)*c = a*(b*c) (100 random trials)...\n");
    srand(42);

    for (int trial = 0; trial < 100; trial++) {
        Cl3Multivector a = random_mv();
        Cl3Multivector b = random_mv();
        Cl3Multivector c = random_mv();

        Cl3Multivector ab = cl3_geometric_product(&a, &b);
        Cl3Multivector ab_c = cl3_geometric_product(&ab, &c);

        Cl3Multivector bc = cl3_geometric_product(&b, &c);
        Cl3Multivector a_bc = cl3_geometric_product(&a, &bc);

        char msg[64];
        for (int k = 0; k < 8; k++) {
            snprintf(msg, sizeof(msg), "assoc trial %d comp[%d]", trial, k);
            ASSERT_FLOAT_EQ(cl3_get(&ab_c, k), cl3_get(&a_bc, k), msg);
        }
    }
}

static void test_distributivity(void) {
    printf("Testing distributivity: a*(b+c) = a*b + a*c...\n");
    srand(123);

    for (int trial = 0; trial < 50; trial++) {
        Cl3Multivector a = random_mv();
        Cl3Multivector b = random_mv();
        Cl3Multivector c = random_mv();

        Cl3Multivector bc_sum = cl3_add(&b, &c);
        Cl3Multivector lhs = cl3_geometric_product(&a, &bc_sum);

        Cl3Multivector ab = cl3_geometric_product(&a, &b);
        Cl3Multivector ac = cl3_geometric_product(&a, &c);
        Cl3Multivector rhs = cl3_add(&ab, &ac);

        char msg[64];
        for (int k = 0; k < 8; k++) {
            snprintf(msg, sizeof(msg), "distrib trial %d comp[%d]", trial, k);
            ASSERT_FLOAT_EQ(cl3_get(&lhs, k), cl3_get(&rhs, k), msg);
        }
    }
}

static void test_reverse_anti_automorphism(void) {
    printf("Testing reverse anti-automorphism: ~(a*b) = ~b * ~a...\n");
    srand(777);

    for (int trial = 0; trial < 50; trial++) {
        Cl3Multivector a = random_mv();
        Cl3Multivector b = random_mv();

        Cl3Multivector ab = cl3_geometric_product(&a, &b);
        Cl3Multivector rev_ab = cl3_reverse(&ab);

        Cl3Multivector rev_a = cl3_reverse(&a);
        Cl3Multivector rev_b = cl3_reverse(&b);
        Cl3Multivector rev_b_rev_a = cl3_geometric_product(&rev_b, &rev_a);

        char msg[64];
        for (int k = 0; k < 8; k++) {
            snprintf(msg, sizeof(msg), "rev anti-auto trial %d comp[%d]", trial, k);
            ASSERT_FLOAT_EQ(cl3_get(&rev_ab, k), cl3_get(&rev_b_rev_a, k), msg);
        }
    }
}

static void test_grade_projection_completeness(void) {
    printf("Testing grade projection completeness: sum(grade_k(x)) = x...\n");
    srand(999);

    for (int trial = 0; trial < 50; trial++) {
        Cl3Multivector x = random_mv();
        Cl3Multivector sum = cl3_zero();

        for (int k = 0; k <= 3; k++) {
            Cl3Multivector gk = cl3_grade_project(&x, k);
            sum = cl3_add(&sum, &gk);
        }

        char msg[64];
        for (int k = 0; k < 8; k++) {
            snprintf(msg, sizeof(msg), "grade completeness trial %d comp[%d]", trial, k);
            ASSERT_FLOAT_EQ(cl3_get(&sum, k), cl3_get(&x, k), msg);
        }
    }
}

int main(void) {
    printf("=== Geometric Product Tests ===\n\n");

    test_identity();
    test_scalar_multiplication();
    test_associativity_fuzz();
    test_distributivity();
    test_reverse_anti_automorphism();
    test_grade_projection_completeness();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
