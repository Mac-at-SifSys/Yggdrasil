/**
 * grade_ops.c — Grade projection, involutions, norms, arithmetic
 */

#include "cl3_types.h"
#include "cl3_tables.h"
#include "cl3_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * GRADE PROJECTION
 * ============================================================ */

Cl3Multivector cl3_grade_project(const Cl3Multivector *m, int grade) {
    Cl3Multivector r = cl3_zero();
    switch (grade) {
        case 0: r.s = m->s; break;
        case 1: r.v[0] = m->v[0]; r.v[1] = m->v[1]; r.v[2] = m->v[2]; break;
        case 2: r.b[0] = m->b[0]; r.b[1] = m->b[1]; r.b[2] = m->b[2]; break;
        case 3: r.t = m->t; break;
    }
    return r;
}

Cl3Multivector cl3_grade_mask_project(const Cl3Multivector *m, Cl3GradeMask mask) {
    Cl3Multivector r = cl3_zero();
    if (mask & CL3_GRADE0_MASK) r.s = m->s;
    if (mask & CL3_GRADE1_MASK) { r.v[0] = m->v[0]; r.v[1] = m->v[1]; r.v[2] = m->v[2]; }
    if (mask & CL3_GRADE2_MASK) { r.b[0] = m->b[0]; r.b[1] = m->b[1]; r.b[2] = m->b[2]; }
    if (mask & CL3_GRADE3_MASK) r.t = m->t;
    return r;
}

Cl3Even cl3_even_part(const Cl3Multivector *m) {
    Cl3Even e;
    e.s = m->s;
    e.b[0] = m->b[0]; e.b[1] = m->b[1]; e.b[2] = m->b[2];
    return e;
}

Cl3Odd cl3_odd_part(const Cl3Multivector *m) {
    Cl3Odd o;
    o.v[0] = m->v[0]; o.v[1] = m->v[1]; o.v[2] = m->v[2];
    o.t = m->t;
    return o;
}

/* ============================================================
 * INVOLUTIONS
 * ============================================================ */

Cl3Multivector cl3_reverse(const Cl3Multivector *m) {
    /* Reverse: grade k -> (-1)^(k*(k-1)/2) * grade k
     * grade 0: +1, grade 1: +1, grade 2: -1, grade 3: -1 */
    Cl3Multivector r;
    r.s    =  m->s;
    r.v[0] =  m->v[0]; r.v[1] =  m->v[1]; r.v[2] =  m->v[2];
    r.b[0] = -m->b[0]; r.b[1] = -m->b[1]; r.b[2] = -m->b[2];
    r.t    = -m->t;
    return r;
}

Cl3Multivector cl3_involution(const Cl3Multivector *m) {
    /* Grade involution: grade k -> (-1)^k * grade k
     * grade 0: +1, grade 1: -1, grade 2: +1, grade 3: -1 */
    Cl3Multivector r;
    r.s    =  m->s;
    r.v[0] = -m->v[0]; r.v[1] = -m->v[1]; r.v[2] = -m->v[2];
    r.b[0] =  m->b[0]; r.b[1] =  m->b[1]; r.b[2] =  m->b[2];
    r.t    = -m->t;
    return r;
}

Cl3Multivector cl3_conjugate(const Cl3Multivector *m) {
    /* Clifford conjugate = reverse(involution(m))
     * grade 0: +1, grade 1: -1, grade 2: -1, grade 3: +1 */
    Cl3Multivector r;
    r.s    =  m->s;
    r.v[0] = -m->v[0]; r.v[1] = -m->v[1]; r.v[2] = -m->v[2];
    r.b[0] = -m->b[0]; r.b[1] = -m->b[1]; r.b[2] = -m->b[2];
    r.t    =  m->t;
    return r;
}

/* ============================================================
 * NORMS
 * ============================================================ */

cl3_scalar cl3_norm_squared(const Cl3Multivector *m) {
    /* norm^2 = scalar_part(m * ~m) */
    Cl3Multivector rev = cl3_reverse(m);
    return cl3_scalar_product(m, &rev);
}

cl3_scalar cl3_norm(const Cl3Multivector *m) {
    cl3_scalar ns = cl3_norm_squared(m);
    return sqrtf(fabsf(ns));
}

Cl3Multivector cl3_normalize(const Cl3Multivector *m) {
    cl3_scalar n = cl3_norm(m);
    if (n < 1e-12f) return cl3_zero();
    return cl3_scale(m, 1.0f / n);
}

/* ============================================================
 * ARITHMETIC
 * ============================================================ */

Cl3Multivector cl3_add(const Cl3Multivector *a, const Cl3Multivector *b) {
    Cl3Multivector r;
    r.s = a->s + b->s;
    r.v[0] = a->v[0] + b->v[0]; r.v[1] = a->v[1] + b->v[1]; r.v[2] = a->v[2] + b->v[2];
    r.b[0] = a->b[0] + b->b[0]; r.b[1] = a->b[1] + b->b[1]; r.b[2] = a->b[2] + b->b[2];
    r.t = a->t + b->t;
    return r;
}

Cl3Multivector cl3_sub(const Cl3Multivector *a, const Cl3Multivector *b) {
    Cl3Multivector r;
    r.s = a->s - b->s;
    r.v[0] = a->v[0] - b->v[0]; r.v[1] = a->v[1] - b->v[1]; r.v[2] = a->v[2] - b->v[2];
    r.b[0] = a->b[0] - b->b[0]; r.b[1] = a->b[1] - b->b[1]; r.b[2] = a->b[2] - b->b[2];
    r.t = a->t - b->t;
    return r;
}

Cl3Multivector cl3_scale(const Cl3Multivector *m, cl3_scalar s) {
    Cl3Multivector r;
    r.s = m->s * s;
    r.v[0] = m->v[0] * s; r.v[1] = m->v[1] * s; r.v[2] = m->v[2] * s;
    r.b[0] = m->b[0] * s; r.b[1] = m->b[1] * s; r.b[2] = m->b[2] * s;
    r.t = m->t * s;
    return r;
}

Cl3Multivector cl3_negate(const Cl3Multivector *m) {
    return cl3_scale(m, -1.0f);
}

/* ============================================================
 * BATCH ALLOCATION
 * ============================================================ */

Cl3Batch cl3_batch_alloc(size_t count) {
    Cl3Batch b;
    b.count = count;
    b.scalars    = (cl3_scalar *)calloc(count, sizeof(cl3_scalar));
    b.vectors    = (cl3_scalar *)calloc(count * 3, sizeof(cl3_scalar));
    b.bivectors  = (cl3_scalar *)calloc(count * 3, sizeof(cl3_scalar));
    b.trivectors = (cl3_scalar *)calloc(count, sizeof(cl3_scalar));
    return b;
}

void cl3_batch_free(Cl3Batch *batch) {
    free(batch->scalars);
    free(batch->vectors);
    free(batch->bivectors);
    free(batch->trivectors);
    batch->scalars = batch->vectors = batch->bivectors = batch->trivectors = NULL;
    batch->count = 0;
}

void cl3_batch_set(Cl3Batch *batch, size_t idx, const Cl3Multivector *mv) {
    batch->scalars[idx] = mv->s;
    batch->vectors[idx * 3 + 0] = mv->v[0];
    batch->vectors[idx * 3 + 1] = mv->v[1];
    batch->vectors[idx * 3 + 2] = mv->v[2];
    batch->bivectors[idx * 3 + 0] = mv->b[0];
    batch->bivectors[idx * 3 + 1] = mv->b[1];
    batch->bivectors[idx * 3 + 2] = mv->b[2];
    batch->trivectors[idx] = mv->t;
}

Cl3Multivector cl3_batch_get(const Cl3Batch *batch, size_t idx) {
    Cl3Multivector mv;
    mv.s = batch->scalars[idx];
    mv.v[0] = batch->vectors[idx * 3 + 0];
    mv.v[1] = batch->vectors[idx * 3 + 1];
    mv.v[2] = batch->vectors[idx * 3 + 2];
    mv.b[0] = batch->bivectors[idx * 3 + 0];
    mv.b[1] = batch->bivectors[idx * 3 + 1];
    mv.b[2] = batch->bivectors[idx * 3 + 2];
    mv.t = batch->trivectors[idx];
    return mv;
}
