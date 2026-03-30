/**
 * cl3_tables.h — Precomputed Cl(3,0) Multiplication Table
 *
 * The Cl(3,0) multiplication table for basis elements.
 * Basis ordering: {1, e1, e2, e3, e12, e13, e23, e123}
 *                  0   1   2   3   4    5    6    7
 *
 * For product e_i * e_j: result = sign[i][j] * basis[product_idx[i][j]]
 *
 * This is the SINGLE SOURCE OF TRUTH for the entire stack.
 */

#ifndef CL3_TABLES_H
#define CL3_TABLES_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Product index table: (basis_i * basis_j) yields basis element at this index.
 * CL3_PRODUCT_IDX[i][j] gives the index of the resulting basis element.
 */
static const int CL3_PRODUCT_IDX[8][8] = {
/*           1   e1  e2  e3  e12 e13 e23 e123 */
/* 1    */ { 0,  1,  2,  3,  4,  5,  6,  7 },
/* e1   */ { 1,  0,  4,  5,  2,  3,  7,  6 },
/* e2   */ { 2,  4,  0,  6,  1,  7,  3,  5 },
/* e3   */ { 3,  5,  6,  0,  7,  1,  2,  4 },
/* e12  */ { 4,  2,  1,  7,  0,  6,  5,  3 },
/* e13  */ { 5,  3,  7,  1,  6,  0,  4,  2 },
/* e23  */ { 6,  7,  3,  2,  5,  4,  0,  1 },
/* e123 */ { 7,  6,  5,  4,  3,  2,  1,  0 },
};

/**
 * Sign table: (basis_i * basis_j) = CL3_PRODUCT_SIGN[i][j] * basis[CL3_PRODUCT_IDX[i][j]]
 *
 * Derived from the Cl(3,0) multiplication rules:
 *   e1*e1 = e2*e2 = e3*e3 = +1  (signature (3,0))
 *   e_i * e_j = -e_j * e_i  for i != j  (anti-commutation of vectors)
 */
static const int CL3_PRODUCT_SIGN[8][8] = {
/*           1   e1  e2  e3  e12 e13 e23 e123 */
/* 1    */ {+1, +1, +1, +1, +1, +1, +1, +1 },
/* e1   */ {+1, +1, +1, +1, +1, +1, +1, +1 },
/* e2   */ {+1, -1, +1, +1, -1, -1, +1, -1 },
/* e3   */ {+1, -1, -1, +1, +1, -1, -1, +1 },
/* e12  */ {+1, -1, +1, +1, -1, -1, +1, -1 },
/* e13  */ {+1, -1, -1, +1, +1, -1, -1, +1 },
/* e23  */ {+1, +1, -1, +1, -1, +1, -1, -1 },
/* e123 */ {+1, +1, -1, +1, -1, +1, -1, -1 },
};

/**
 * Grade of each basis element.
 */
static const int CL3_BASIS_GRADE[8] = {
    0,  /* 1    */
    1,  /* e1   */
    1,  /* e2   */
    1,  /* e3   */
    2,  /* e12  */
    2,  /* e13  */
    2,  /* e23  */
    3,  /* e123 */
};

/**
 * Reverse sign for each basis element.
 * ~(blade of grade k) = (-1)^(k*(k-1)/2) * blade
 */
static const int CL3_REVERSE_SIGN[8] = {
    +1,  /* grade 0: (-1)^0 = +1 */
    +1,  /* grade 1: (-1)^0 = +1 */
    +1,  /* grade 1: (-1)^0 = +1 */
    +1,  /* grade 1: (-1)^0 = +1 */
    -1,  /* grade 2: (-1)^1 = -1 */
    -1,  /* grade 2: (-1)^1 = -1 */
    -1,  /* grade 2: (-1)^1 = -1 */
    -1,  /* grade 3: (-1)^3 = -1 */
};

/**
 * Grade involution sign for each basis element.
 * hat(blade of grade k) = (-1)^k * blade
 */
static const int CL3_INVOLUTION_SIGN[8] = {
    +1,  /* grade 0 */
    -1,  /* grade 1 */
    -1,  /* grade 1 */
    -1,  /* grade 1 */
    +1,  /* grade 2 */
    +1,  /* grade 2 */
    +1,  /* grade 2 */
    -1,  /* grade 3 */
};

/**
 * Conjugate sign = reverse_sign * involution_sign
 */
static const int CL3_CONJUGATE_SIGN[8] = {
    +1,  /* grade 0: +1 * +1 */
    -1,  /* grade 1: +1 * -1 */
    -1,  /* grade 1: +1 * -1 */
    -1,  /* grade 1: +1 * -1 */
    -1,  /* grade 2: -1 * +1 */
    -1,  /* grade 2: -1 * +1 */
    -1,  /* grade 2: -1 * +1 */
    +1,  /* grade 3: -1 * -1 */
};

#ifdef __cplusplus
}
#endif

#endif /* CL3_TABLES_H */
