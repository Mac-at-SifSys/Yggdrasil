/**
 * batch_ops.c — Batched Clifford algebra operations
 */

#include "cl3_types.h"
#include "cl3_ops.h"
#include <stdlib.h>

void cl3_batch_geometric_product(
    const Cl3Batch *a, const Cl3Batch *b, Cl3Batch *out, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        Cl3Multivector ma = cl3_batch_get(a, i);
        Cl3Multivector mb = cl3_batch_get(b, i);
        Cl3Multivector mc = cl3_geometric_product(&ma, &mb);
        cl3_batch_set(out, i, &mc);
    }
}

void cl3_batch_sandwich(
    const Cl3Batch *r, const Cl3Batch *x, Cl3Batch *out, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        Cl3Multivector mr = cl3_batch_get(r, i);
        Cl3Multivector mx = cl3_batch_get(x, i);
        Cl3Multivector mc = cl3_sandwich(&mr, &mx);
        cl3_batch_set(out, i, &mc);
    }
}

void cl3_batch_bivector_exp(const Cl3Batch *bv, Cl3Batch *out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        Cl3Multivector mb = cl3_batch_get(bv, i);
        Cl3Multivector mr = cl3_bivector_exp(&mb);
        cl3_batch_set(out, i, &mr);
    }
}

void cl3_batch_grade_project(
    const Cl3Batch *in, Cl3Batch *out, int grade, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        Cl3Multivector mi = cl3_batch_get(in, i);
        Cl3Multivector mo = cl3_grade_project(&mi, grade);
        cl3_batch_set(out, i, &mo);
    }
}

void cl3_batch_norm(const Cl3Batch *in, cl3_scalar *out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        Cl3Multivector mi = cl3_batch_get(in, i);
        out[i] = cl3_norm(&mi);
    }
}

void cl3_batch_geom_matmul(
    const Cl3Multivector *a, const Cl3Multivector *b, Cl3Multivector *out,
    size_t M, size_t K, size_t N)
{
    /* out[i,j] = sum_k a[i,k] * b[k,j] (geometric product) */
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            Cl3Multivector acc = cl3_zero();
            for (size_t k = 0; k < K; k++) {
                Cl3Multivector prod = cl3_geometric_product(&a[i * K + k], &b[k * N + j]);
                acc = cl3_add(&acc, &prod);
            }
            out[i * N + j] = acc;
        }
    }
}

/* --- Flat-layout batch operations (for Python FFI) ---
 * These take contiguous float arrays where each multivector is 8 consecutive floats.
 * The Cl3Multivector struct has the same memory layout as float[8], so we can
 * simply cast and call the existing per-element functions.
 */

void cl3_batch_gp_flat(const float *a, const float *b, float *out, size_t count) {
    const Cl3Multivector *ma = (const Cl3Multivector *)a;
    const Cl3Multivector *mb = (const Cl3Multivector *)b;
    Cl3Multivector *mo = (Cl3Multivector *)out;
    for (size_t i = 0; i < count; i++) {
        mo[i] = cl3_geometric_product(&ma[i], &mb[i]);
    }
}

void cl3_batch_reverse_flat(const float *a, float *out, size_t count) {
    const Cl3Multivector *ma = (const Cl3Multivector *)a;
    Cl3Multivector *mo = (Cl3Multivector *)out;
    for (size_t i = 0; i < count; i++) {
        mo[i] = cl3_reverse(&ma[i]);
    }
}

void cl3_batch_sandwich_flat(const float *r, const float *x, float *out, size_t count) {
    const Cl3Multivector *mr = (const Cl3Multivector *)r;
    const Cl3Multivector *mx = (const Cl3Multivector *)x;
    Cl3Multivector *mo = (Cl3Multivector *)out;
    for (size_t i = 0; i < count; i++) {
        mo[i] = cl3_sandwich(&mr[i], &mx[i]);
    }
}

void cl3_batch_bivector_exp_flat(const float *bv, float *out, size_t count) {
    const Cl3Multivector *mb = (const Cl3Multivector *)bv;
    Cl3Multivector *mo = (Cl3Multivector *)out;
    for (size_t i = 0; i < count; i++) {
        mo[i] = cl3_bivector_exp(&mb[i]);
    }
}

void cl3_batch_norm_flat(const float *a, float *out, size_t count) {
    const Cl3Multivector *ma = (const Cl3Multivector *)a;
    for (size_t i = 0; i < count; i++) {
        out[i] = cl3_norm(&ma[i]);
    }
}

void cl3_batch_normalize_flat(const float *a, float *out, size_t count) {
    const Cl3Multivector *ma = (const Cl3Multivector *)a;
    Cl3Multivector *mo = (Cl3Multivector *)out;
    for (size_t i = 0; i < count; i++) {
        mo[i] = cl3_normalize(&ma[i]);
    }
}

void cl3_batch_add_flat(const float *a, const float *b, float *out, size_t count) {
    const Cl3Multivector *ma = (const Cl3Multivector *)a;
    const Cl3Multivector *mb = (const Cl3Multivector *)b;
    Cl3Multivector *mo = (Cl3Multivector *)out;
    for (size_t i = 0; i < count; i++) {
        mo[i] = cl3_add(&ma[i], &mb[i]);
    }
}

void cl3_batch_scale_flat(const float *a, float s, float *out, size_t count) {
    const Cl3Multivector *ma = (const Cl3Multivector *)a;
    Cl3Multivector *mo = (Cl3Multivector *)out;
    for (size_t i = 0; i < count; i++) {
        mo[i] = cl3_scale(&ma[i], s);
    }
}

void cl3_batch_grade_project_flat(const float *a, float *out, int grade, size_t count) {
    const Cl3Multivector *ma = (const Cl3Multivector *)a;
    Cl3Multivector *mo = (Cl3Multivector *)out;
    for (size_t i = 0; i < count; i++) {
        mo[i] = cl3_grade_project(&ma[i], grade);
    }
}

/* Scalar part of geometric product only — much cheaper than full GP */
void cl3_batch_scalar_product_flat(const float *a, const float *b, float *out, size_t count) {
    const Cl3Multivector *ma = (const Cl3Multivector *)a;
    const Cl3Multivector *mb = (const Cl3Multivector *)b;
    for (size_t i = 0; i < count; i++) {
        out[i] = cl3_scalar_product(&ma[i], &mb[i]);
    }
}

/* Geometric matrix multiply: out[i,j] = sum_k gp(a[i,k], b[k,j])
 * All arrays are contiguous Cl3Multivector (float[8]) with row-major order.
 * a: M x K, b: K x N, out: M x N */
void cl3_geom_matmul_flat(const float *a, const float *b, float *out,
                           size_t M, size_t K, size_t N) {
    cl3_batch_geom_matmul(
        (const Cl3Multivector *)a,
        (const Cl3Multivector *)b,
        (Cl3Multivector *)out,
        M, K, N);
}
