/**
 * geometric_product.c — CPU reference implementation of the Cl(3,0) geometric product
 *
 * The geometric product is the fundamental operation.
 * For two multivectors A and B, computes C = A*B using the full
 * Cl(3,0) multiplication table. 64 multiply-adds, fully expanded
 * for clarity and correctness (this is the reference, not the fast path).
 *
 * Basis ordering: {1, e1, e2, e3, e12, e13, e23, e123}
 * Stored as:      {s, v[0], v[1], v[2], b[0], b[1], b[2], t}
 */

#include "cl3_types.h"
#include "cl3_tables.h"
#include "cl3_ops.h"

Cl3Multivector cl3_geometric_product(const Cl3Multivector *a, const Cl3Multivector *b) {
    /*
     * Direct expansion from the Cl(3,0) multiplication table.
     * Each output component is a sum of 8 terms (one from each input component of a
     * multiplied by the corresponding input component of b).
     *
     * We read a and b as flat arrays of 8 components:
     *   a[0]=a.s, a[1]=a.v[0](e1), a[2]=a.v[1](e2), a[3]=a.v[2](e3),
     *   a[4]=a.b[0](e12), a[5]=a.b[1](e13), a[6]=a.b[2](e23), a[7]=a.t(e123)
     */
    cl3_scalar af[8], bf[8], cf[8];

    /* Flatten inputs */
    af[0] = a->s;    af[1] = a->v[0]; af[2] = a->v[1]; af[3] = a->v[2];
    af[4] = a->b[0]; af[5] = a->b[1]; af[6] = a->b[2]; af[7] = a->t;

    bf[0] = b->s;    bf[1] = b->v[0]; bf[2] = b->v[1]; bf[3] = b->v[2];
    bf[4] = b->b[0]; bf[5] = b->b[1]; bf[6] = b->b[2]; bf[7] = b->t;

    /* Zero output */
    for (int i = 0; i < 8; i++) cf[i] = 0.0f;

    /* Full 8x8 product using the multiplication table */
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int out_idx = CL3_PRODUCT_IDX[i][j];
            int sign = CL3_PRODUCT_SIGN[i][j];
            cf[out_idx] += sign * af[i] * bf[j];
        }
    }

    /* Unflatten output */
    Cl3Multivector c;
    c.s    = cf[0];
    c.v[0] = cf[1]; c.v[1] = cf[2]; c.v[2] = cf[3];
    c.b[0] = cf[4]; c.b[1] = cf[5]; c.b[2] = cf[6];
    c.t    = cf[7];

    return c;
}

Cl3Multivector cl3_outer_product(const Cl3Multivector *a, const Cl3Multivector *b) {
    /*
     * Outer product: keep only terms where grade(result) = grade(a_i) + grade(b_j).
     */
    cl3_scalar af[8], bf[8], cf[8];

    af[0] = a->s;    af[1] = a->v[0]; af[2] = a->v[1]; af[3] = a->v[2];
    af[4] = a->b[0]; af[5] = a->b[1]; af[6] = a->b[2]; af[7] = a->t;

    bf[0] = b->s;    bf[1] = b->v[0]; bf[2] = b->v[1]; bf[3] = b->v[2];
    bf[4] = b->b[0]; bf[5] = b->b[1]; bf[6] = b->b[2]; bf[7] = b->t;

    for (int i = 0; i < 8; i++) cf[i] = 0.0f;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int out_idx = CL3_PRODUCT_IDX[i][j];
            int sign = CL3_PRODUCT_SIGN[i][j];
            /* Outer product: only keep if grade increases */
            if (CL3_BASIS_GRADE[out_idx] == CL3_BASIS_GRADE[i] + CL3_BASIS_GRADE[j]) {
                cf[out_idx] += sign * af[i] * bf[j];
            }
        }
    }

    Cl3Multivector c;
    c.s    = cf[0];
    c.v[0] = cf[1]; c.v[1] = cf[2]; c.v[2] = cf[3];
    c.b[0] = cf[4]; c.b[1] = cf[5]; c.b[2] = cf[6];
    c.t    = cf[7];
    return c;
}

Cl3Multivector cl3_inner_product(const Cl3Multivector *a, const Cl3Multivector *b) {
    /*
     * Left contraction: keep only terms where
     * grade(result) = grade(b_j) - grade(a_i) AND grade(a_i) <= grade(b_j).
     */
    cl3_scalar af[8], bf[8], cf[8];

    af[0] = a->s;    af[1] = a->v[0]; af[2] = a->v[1]; af[3] = a->v[2];
    af[4] = a->b[0]; af[5] = a->b[1]; af[6] = a->b[2]; af[7] = a->t;

    bf[0] = b->s;    bf[1] = b->v[0]; bf[2] = b->v[1]; bf[3] = b->v[2];
    bf[4] = b->b[0]; bf[5] = b->b[1]; bf[6] = b->b[2]; bf[7] = b->t;

    for (int i = 0; i < 8; i++) cf[i] = 0.0f;

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int out_idx = CL3_PRODUCT_IDX[i][j];
            int sign = CL3_PRODUCT_SIGN[i][j];
            int ga = CL3_BASIS_GRADE[i];
            int gb = CL3_BASIS_GRADE[j];
            int gc = CL3_BASIS_GRADE[out_idx];
            /* Left contraction: grade decreases by grade of a */
            if (ga <= gb && gc == gb - ga) {
                cf[out_idx] += sign * af[i] * bf[j];
            }
        }
    }

    Cl3Multivector c;
    c.s    = cf[0];
    c.v[0] = cf[1]; c.v[1] = cf[2]; c.v[2] = cf[3];
    c.b[0] = cf[4]; c.b[1] = cf[5]; c.b[2] = cf[6];
    c.t    = cf[7];
    return c;
}

cl3_scalar cl3_scalar_product(const Cl3Multivector *a, const Cl3Multivector *b) {
    /* Grade-0 part of a * b */
    Cl3Multivector p = cl3_geometric_product(a, b);
    return p.s;
}
