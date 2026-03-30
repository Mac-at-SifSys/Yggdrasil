/**
 * cl3_ops.h — Cl(3,0) Operation Signatures
 *
 * All fundamental operations on Cl(3,0) multivectors.
 * CPU reference implementations in src/cpu/.
 */

#ifndef CL3_OPS_H
#define CL3_OPS_H

#include "cl3_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * FUNDAMENTAL OPERATIONS (single multivector pairs)
 * ============================================================ */

/**
 * Geometric product: c = a * b
 * The fundamental operation of the algebra. 64 multiply-adds.
 */
Cl3Multivector cl3_geometric_product(const Cl3Multivector *a, const Cl3Multivector *b);

/**
 * Outer (wedge) product: c = a ^ b
 * Takes only the grade-raising part of the geometric product.
 */
Cl3Multivector cl3_outer_product(const Cl3Multivector *a, const Cl3Multivector *b);

/**
 * Inner (dot/contraction) product: c = a | b
 * Takes only the grade-lowering part of the geometric product.
 * Uses left contraction convention.
 */
Cl3Multivector cl3_inner_product(const Cl3Multivector *a, const Cl3Multivector *b);

/**
 * Scalar product: takes grade-0 part of a * b
 */
cl3_scalar cl3_scalar_product(const Cl3Multivector *a, const Cl3Multivector *b);

/* ============================================================
 * GRADE OPERATIONS
 * ============================================================ */

/**
 * Grade projection: extract grade-k component.
 * Returns a multivector with only grade-k nonzero.
 */
Cl3Multivector cl3_grade_project(const Cl3Multivector *m, int grade);

/**
 * Grade projection with mask: extract multiple grades.
 */
Cl3Multivector cl3_grade_mask_project(const Cl3Multivector *m, Cl3GradeMask mask);

/**
 * Even projection: grade 0 + grade 2 (rotor sub-algebra).
 */
Cl3Even cl3_even_part(const Cl3Multivector *m);

/**
 * Odd projection: grade 1 + grade 3.
 */
Cl3Odd cl3_odd_part(const Cl3Multivector *m);

/* ============================================================
 * INVOLUTIONS
 * ============================================================ */

/**
 * Reverse (reversion): ~a
 * Reverses the order of basis vectors in each blade.
 * Grade k picks up sign (-1)^(k*(k-1)/2).
 */
Cl3Multivector cl3_reverse(const Cl3Multivector *m);

/**
 * Grade involution (hat): ^a
 * Grade k picks up sign (-1)^k.
 */
Cl3Multivector cl3_involution(const Cl3Multivector *m);

/**
 * Clifford conjugate: bar(a) = reverse(involution(a))
 */
Cl3Multivector cl3_conjugate(const Cl3Multivector *m);

/* ============================================================
 * NORMS
 * ============================================================ */

/**
 * Squared norm: a * ~a, take scalar part.
 * For Cl(3,0), this is always non-negative for versors.
 */
cl3_scalar cl3_norm_squared(const Cl3Multivector *m);

/**
 * Norm: sqrt(|norm_squared(a)|)
 */
cl3_scalar cl3_norm(const Cl3Multivector *m);

/**
 * Normalize to unit norm.
 */
Cl3Multivector cl3_normalize(const Cl3Multivector *m);

/* ============================================================
 * SANDWICH PRODUCT
 * ============================================================ */

/**
 * Sandwich product: R * x * ~R
 * Used for rotations/reflections.
 * Fused kernel: computes in two geometric products but should be a single call.
 */
Cl3Multivector cl3_sandwich(const Cl3Multivector *r, const Cl3Multivector *x);

/* ============================================================
 * EXPONENTIAL / LOGARITHM
 * ============================================================ */

/**
 * Exponential of a pure bivector: exp(B)
 * For Cl(3,0), this is closed-form:
 *   exp(B) = cos(|B|) + sin(|B|)/|B| * B
 * where |B| is the bivector magnitude.
 * Result is always an even multivector (rotor).
 */
Cl3Multivector cl3_bivector_exp(const Cl3Multivector *bv);

/**
 * Logarithm of a rotor: log(R)
 * Inverse of bivector_exp. Returns a pure bivector.
 */
Cl3Multivector cl3_rotor_log(const Cl3Multivector *rotor);

/* ============================================================
 * ARITHMETIC
 * ============================================================ */

/** Addition: c = a + b (component-wise) */
Cl3Multivector cl3_add(const Cl3Multivector *a, const Cl3Multivector *b);

/** Subtraction: c = a - b (component-wise) */
Cl3Multivector cl3_sub(const Cl3Multivector *a, const Cl3Multivector *b);

/** Scalar multiplication: c = s * a */
Cl3Multivector cl3_scale(const Cl3Multivector *m, cl3_scalar s);

/** Negation: c = -a */
Cl3Multivector cl3_negate(const Cl3Multivector *m);

/* ============================================================
 * BATCH OPERATIONS
 * ============================================================ */

/**
 * Batched geometric product: out[i] = a[i] * b[i]
 */
void cl3_batch_geometric_product(
    const Cl3Batch *a, const Cl3Batch *b, Cl3Batch *out, size_t count);

/**
 * Batched sandwich product: out[i] = r[i] * x[i] * ~r[i]
 */
void cl3_batch_sandwich(
    const Cl3Batch *r, const Cl3Batch *x, Cl3Batch *out, size_t count);

/**
 * Batched bivector exponential: out[i] = exp(bv[i])
 */
void cl3_batch_bivector_exp(const Cl3Batch *bv, Cl3Batch *out, size_t count);

/**
 * Batched grade projection: out[i] = grade_k(in[i])
 */
void cl3_batch_grade_project(
    const Cl3Batch *in, Cl3Batch *out, int grade, size_t count);

/**
 * Batched norm: out[i] = norm(in[i])
 */
void cl3_batch_norm(const Cl3Batch *in, cl3_scalar *out, size_t count);

/**
 * "Matrix multiply" in the algebra: out[i,j] = sum_k a[i,k] * b[k,j]
 * where * is the geometric product.
 * a: [M x K] batch, b: [K x N] batch, out: [M x N] batch
 */
void cl3_batch_geom_matmul(
    const Cl3Multivector *a, const Cl3Multivector *b, Cl3Multivector *out,
    size_t M, size_t K, size_t N);

/* --- Flat-layout batch operations (for FFI) --- */
void cl3_batch_gp_flat(const float *a, const float *b, float *out, size_t count);
void cl3_batch_reverse_flat(const float *a, float *out, size_t count);
void cl3_batch_sandwich_flat(const float *r, const float *x, float *out, size_t count);
void cl3_batch_bivector_exp_flat(const float *bv, float *out, size_t count);
void cl3_batch_norm_flat(const float *a, float *out, size_t count);
void cl3_batch_normalize_flat(const float *a, float *out, size_t count);
void cl3_batch_add_flat(const float *a, const float *b, float *out, size_t count);
void cl3_batch_scale_flat(const float *a, float s, float *out, size_t count);
void cl3_batch_grade_project_flat(const float *a, float *out, int grade, size_t count);
void cl3_batch_scalar_product_flat(const float *a, const float *b, float *out, size_t count);
void cl3_geom_matmul_flat(const float *a, const float *b, float *out, size_t M, size_t K, size_t N);

#ifdef __cplusplus
}
#endif

#endif /* CL3_OPS_H */
