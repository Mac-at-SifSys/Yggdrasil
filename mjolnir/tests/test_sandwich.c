/**
 * test_sandwich.c — Tests for sandwich product and rotor operations
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../include/cl3_types.h"
#include "../include/cl3_ops.h"

static int tests_run = 0;
static int tests_passed = 0;
static const float EPS = 1e-4f;

#define ASSERT_FLOAT_EQ(a, b, msg) do { \
    tests_run++; \
    if (fabsf((a) - (b)) < EPS) { tests_passed++; } \
    else { printf("FAIL: %s (expected %f, got %f)\n", msg, (float)(b), (float)(a)); } \
} while(0)

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static void test_rotor_rotation_90deg(void) {
    printf("Testing 90-degree rotation via rotor sandwich...\n");

    /* Rotation by angle theta in the e1-e2 plane:
     * R = exp(-theta/2 * e12) = cos(theta/2) - sin(theta/2) * e12
     * For theta = pi/2: R = cos(pi/4) - sin(pi/4) * e12 */
    float half_angle = (float)M_PI / 4.0f;
    Cl3Multivector bv = cl3_zero();
    bv.b[0] = -half_angle;  /* -theta/2 in e12 plane */

    Cl3Multivector R = cl3_bivector_exp(&bv);

    /* Apply to e1: should get e2 */
    Cl3Multivector e1 = cl3_zero(); e1.v[0] = 1.0f;
    Cl3Multivector rotated = cl3_sandwich(&R, &e1);

    ASSERT_FLOAT_EQ(rotated.v[0], 0.0f, "Rotated e1: x component = 0");
    ASSERT_FLOAT_EQ(rotated.v[1], 1.0f, "Rotated e1: y component = 1");
    ASSERT_FLOAT_EQ(rotated.v[2], 0.0f, "Rotated e1: z component = 0");
}

static void test_rotor_preserves_norm(void) {
    printf("Testing that rotor sandwich preserves norm...\n");
    srand(42);

    for (int trial = 0; trial < 50; trial++) {
        /* Random bivector for rotation */
        Cl3Multivector bv = cl3_zero();
        bv.b[0] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        bv.b[1] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        bv.b[2] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        Cl3Multivector R = cl3_bivector_exp(&bv);

        /* Random vector to rotate */
        Cl3Multivector v = cl3_zero();
        v.v[0] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v.v[1] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        v.v[2] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        Cl3Multivector rotated = cl3_sandwich(&R, &v);

        float norm_before = cl3_norm(&v);
        float norm_after = cl3_norm(&rotated);

        char msg[64];
        snprintf(msg, sizeof(msg), "norm preserved trial %d", trial);
        ASSERT_FLOAT_EQ(norm_after, norm_before, msg);
    }
}

static void test_rotor_unit_norm(void) {
    printf("Testing that exp(bivector) produces unit-norm rotors...\n");
    srand(123);

    for (int trial = 0; trial < 50; trial++) {
        Cl3Multivector bv = cl3_zero();
        bv.b[0] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
        bv.b[1] = (float)rand() / RAND_MAX * 4.0f - 2.0f;
        bv.b[2] = (float)rand() / RAND_MAX * 4.0f - 2.0f;

        Cl3Multivector R = cl3_bivector_exp(&bv);
        float n = cl3_norm(&R);

        char msg[64];
        snprintf(msg, sizeof(msg), "rotor unit norm trial %d", trial);
        ASSERT_FLOAT_EQ(n, 1.0f, msg);
    }
}

static void test_exp_log_roundtrip(void) {
    printf("Testing exp(log(R)) = R roundtrip...\n");
    srand(456);

    for (int trial = 0; trial < 50; trial++) {
        Cl3Multivector bv = cl3_zero();
        /* Keep angle small enough to avoid branch cuts */
        bv.b[0] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        bv.b[1] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        bv.b[2] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

        Cl3Multivector R = cl3_bivector_exp(&bv);
        Cl3Multivector log_R = cl3_rotor_log(&R);
        Cl3Multivector R2 = cl3_bivector_exp(&log_R);

        char msg[64];
        for (int k = 0; k < 8; k++) {
            snprintf(msg, sizeof(msg), "exp-log roundtrip trial %d comp[%d]", trial, k);
            ASSERT_FLOAT_EQ(cl3_get(&R2, k), cl3_get(&R, k), msg);
        }
    }
}

static void test_double_rotation(void) {
    printf("Testing R1 * R2 composition = sequential rotation...\n");

    /* Two 90-degree rotations in e12 plane should give 180-degree rotation */
    Cl3Multivector bv = cl3_zero();
    bv.b[0] = -(float)M_PI / 4.0f;  /* 90 degrees / 2 */

    Cl3Multivector R = cl3_bivector_exp(&bv);
    Cl3Multivector R2 = cl3_geometric_product(&R, &R);  /* R * R = 180 degrees */

    /* Apply to e1: should get -e1 */
    Cl3Multivector e1 = cl3_zero(); e1.v[0] = 1.0f;
    Cl3Multivector rotated = cl3_sandwich(&R2, &e1);

    ASSERT_FLOAT_EQ(rotated.v[0], -1.0f, "180deg rotation: x = -1");
    ASSERT_FLOAT_EQ(rotated.v[1], 0.0f, "180deg rotation: y = 0");
    ASSERT_FLOAT_EQ(rotated.v[2], 0.0f, "180deg rotation: z = 0");
}

int main(void) {
    printf("=== Sandwich Product & Rotor Tests ===\n\n");

    test_rotor_rotation_90deg();
    test_rotor_preserves_norm();
    test_rotor_unit_norm();
    test_exp_log_roundtrip();
    test_double_rotation();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
