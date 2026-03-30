/**
 * test_associativity.c — Fuzz test for associativity with 1000+ random triples
 *
 * Verifies (AB)C == A(BC) for random multivectors.
 * Also tests: ~(AB) == ~B * ~A, norm positivity, grade completeness.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../include/cl3_types.h"
#include "../include/cl3_ops.h"

static int tests_run = 0;
static int tests_passed = 0;
static const float EPS = 1e-4f;

#define ASSERT_FLOAT_NEAR(a, b, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) < EPS) { tests_passed++; } \
    else { printf("FAIL: %s (expected %f, got %f, diff=%e)\n", msg, (float)(b), (float)(a), fabsf((a)-(b))); } \
} while(0)

static Cl3Multivector random_mv(void) {
    Cl3Multivector m;
    m.s = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < 3; i++) m.v[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < 3; i++) m.b[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    m.t = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    return m;
}

static void test_associativity_1000(void) {
    printf("Testing associativity: (a*b)*c = a*(b*c) (1000 random triples)...\n");
    srand(12345);

    int failures = 0;
    for (int trial = 0; trial < 1000; trial++) {
        Cl3Multivector a = random_mv();
        Cl3Multivector b = random_mv();
        Cl3Multivector c = random_mv();

        Cl3Multivector ab = cl3_geometric_product(&a, &b);
        Cl3Multivector ab_c = cl3_geometric_product(&ab, &c);

        Cl3Multivector bc = cl3_geometric_product(&b, &c);
        Cl3Multivector a_bc = cl3_geometric_product(&a, &bc);

        for (int k = 0; k < 8; k++) {
            float diff = fabsf(cl3_get(&ab_c, k) - cl3_get(&a_bc, k));
            tests_run++;
            if (diff < EPS) {
                tests_passed++;
            } else {
                failures++;
                if (failures <= 5) {
                    printf("  FAIL trial %d comp[%d]: (ab)c=%f, a(bc)=%f, diff=%e\n",
                           trial, k, cl3_get(&ab_c, k), cl3_get(&a_bc, k), diff);
                }
            }
        }
    }
    printf("  Associativity: %d/%d component checks passed (%d failures)\n",
           tests_passed, tests_run, failures);
}

static void test_reverse_anti_auto_1000(void) {
    printf("Testing ~(a*b) = ~b * ~a (1000 random pairs)...\n");
    srand(67890);

    int start_pass = tests_passed;
    int start_run = tests_run;
    for (int trial = 0; trial < 1000; trial++) {
        Cl3Multivector a = random_mv();
        Cl3Multivector b = random_mv();

        Cl3Multivector ab = cl3_geometric_product(&a, &b);
        Cl3Multivector rev_ab = cl3_reverse(&ab);

        Cl3Multivector rev_a = cl3_reverse(&a);
        Cl3Multivector rev_b = cl3_reverse(&b);
        Cl3Multivector rev_b_rev_a = cl3_geometric_product(&rev_b, &rev_a);

        for (int k = 0; k < 8; k++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "rev trial %d comp[%d]", trial, k);
            ASSERT_FLOAT_NEAR(cl3_get(&rev_ab, k), cl3_get(&rev_b_rev_a, k), msg);
        }
    }
    printf("  Reverse anti-automorphism: %d/%d passed\n",
           tests_passed - start_pass, tests_run - start_run);
}

static void test_norm_positivity_1000(void) {
    printf("Testing norm(x) >= 0 and norm(0) == 0 (1000 random)...\n");
    srand(11111);

    Cl3Multivector zero = cl3_zero();
    float zero_norm = cl3_norm(&zero);
    tests_run++;
    if (zero_norm < 1e-10f) { tests_passed++; }
    else { printf("  FAIL: norm(0) = %f, expected 0\n", zero_norm); }

    int start_pass = tests_passed;
    int start_run = tests_run;
    for (int trial = 0; trial < 1000; trial++) {
        Cl3Multivector x = random_mv();
        float n = cl3_norm(&x);
        tests_run++;
        if (n >= -1e-10f) { tests_passed++; }
        else { printf("  FAIL trial %d: norm = %f < 0\n", trial, n); }
    }
    printf("  Norm positivity: %d/%d passed\n",
           tests_passed - start_pass, tests_run - start_run);
}

static void test_grade_completeness_1000(void) {
    printf("Testing sum(grade_k(x)) = x (1000 random)...\n");
    srand(22222);

    int start_pass = tests_passed;
    int start_run = tests_run;
    for (int trial = 0; trial < 1000; trial++) {
        Cl3Multivector x = random_mv();
        Cl3Multivector sum = cl3_zero();

        for (int k = 0; k <= 3; k++) {
            Cl3Multivector gk = cl3_grade_project(&x, k);
            sum = cl3_add(&sum, &gk);
        }

        for (int k = 0; k < 8; k++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "grade complete trial %d comp[%d]", trial, k);
            ASSERT_FLOAT_NEAR(cl3_get(&sum, k), cl3_get(&x, k), msg);
        }
    }
    printf("  Grade completeness: %d/%d passed\n",
           tests_passed - start_pass, tests_run - start_run);
}

static void test_sandwich_preserves_grade(void) {
    printf("Testing sandwich R*v*~R preserves vector grade (100 random)...\n");
    srand(33333);

    int start_pass = tests_passed;
    int start_run = tests_run;
    for (int trial = 0; trial < 100; trial++) {
        /* Random bivector for rotor */
        Cl3Multivector bv = cl3_zero();
        bv.b[0] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        bv.b[1] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        bv.b[2] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        Cl3Multivector R = cl3_bivector_exp(&bv);

        /* Random pure vector */
        Cl3Multivector v = cl3_zero();
        v.v[0] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v.v[1] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v.v[2] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        Cl3Multivector rotated = cl3_sandwich(&R, &v);

        /* Result should be pure vector: scalar, bivector, trivector all ~0 */
        char msg[64];
        snprintf(msg, sizeof(msg), "sandwich grade trial %d scalar", trial);
        ASSERT_FLOAT_NEAR(rotated.s, 0.0f, msg);
        snprintf(msg, sizeof(msg), "sandwich grade trial %d bv0", trial);
        ASSERT_FLOAT_NEAR(rotated.b[0], 0.0f, msg);
        snprintf(msg, sizeof(msg), "sandwich grade trial %d bv1", trial);
        ASSERT_FLOAT_NEAR(rotated.b[1], 0.0f, msg);
        snprintf(msg, sizeof(msg), "sandwich grade trial %d bv2", trial);
        ASSERT_FLOAT_NEAR(rotated.b[2], 0.0f, msg);
        snprintf(msg, sizeof(msg), "sandwich grade trial %d tri", trial);
        ASSERT_FLOAT_NEAR(rotated.t, 0.0f, msg);
    }
    printf("  Sandwich grade preservation: %d/%d passed\n",
           tests_passed - start_pass, tests_run - start_run);
}

int main(void) {
    printf("=== Cl(3,0) Extended Fuzz Tests (1000+ trials) ===\n\n");

    test_associativity_1000();
    test_reverse_anti_auto_1000();
    test_norm_positivity_1000();
    test_grade_completeness_1000();
    test_sandwich_preserves_grade();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
