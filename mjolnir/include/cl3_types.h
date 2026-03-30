/**
 * cl3_types.h — Cl(3,0) Clifford Algebra Type Definitions
 *
 * Grade-stratified storage for multivectors in the Clifford algebra Cl(3,0).
 * 8 basis elements: {1, e1, e2, e3, e12, e13, e23, e123}
 *
 * Memory layout: grade-stratified, NOT flat 8-element arrays.
 * Grade 0 (scalar):    1 float   — index 0
 * Grade 1 (vectors):   3 floats  — indices 1-3  (e1, e2, e3)
 * Grade 2 (bivectors): 3 floats  — indices 4-6  (e12, e13, e23)
 * Grade 3 (trivector): 1 float   — index 7      (e123)
 */

#ifndef CL3_TYPES_H
#define CL3_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Scalar type (configurable precision) --- */
typedef float cl3_scalar;

/* --- Grade component counts --- */
#define CL3_GRADE0_SIZE 1
#define CL3_GRADE1_SIZE 3
#define CL3_GRADE2_SIZE 3
#define CL3_GRADE3_SIZE 1
#define CL3_TOTAL_SIZE  8
#define CL3_NUM_GRADES  4

/* --- Basis element indices within a flat multivector --- */
#define CL3_IDX_ONE   0   /* 1     (scalar)    */
#define CL3_IDX_E1    1   /* e1    (vector)    */
#define CL3_IDX_E2    2   /* e2    (vector)    */
#define CL3_IDX_E3    3   /* e3    (vector)    */
#define CL3_IDX_E12   4   /* e12   (bivector)  */
#define CL3_IDX_E13   5   /* e13   (bivector)  */
#define CL3_IDX_E23   6   /* e23   (bivector)  */
#define CL3_IDX_E123  7   /* e123  (trivector) */

/* --- Single multivector (interleaved, for convenience/testing) --- */
typedef struct {
    cl3_scalar s;          /* grade 0: scalar */
    cl3_scalar v[3];       /* grade 1: e1, e2, e3 */
    cl3_scalar b[3];       /* grade 2: e12, e13, e23 */
    cl3_scalar t;          /* grade 3: e123 */
} Cl3Multivector;

/* --- Grade-stratified batch storage --- */
typedef struct {
    cl3_scalar *scalars;     /* float[N]     grade 0 */
    cl3_scalar *vectors;     /* float[N*3]   grade 1 (e1,e2,e3 interleaved per element) */
    cl3_scalar *bivectors;   /* float[N*3]   grade 2 (e12,e13,e23 interleaved per element) */
    cl3_scalar *trivectors;  /* float[N]     grade 3 */
    size_t count;            /* number of multivectors in the batch */
} Cl3Batch;

/* --- Even sub-algebra (scalar + bivector) = rotor/spinor space --- */
typedef struct {
    cl3_scalar s;      /* grade 0 */
    cl3_scalar b[3];   /* grade 2: e12, e13, e23 */
} Cl3Even;

/* --- Odd sub-algebra (vector + trivector) --- */
typedef struct {
    cl3_scalar v[3];   /* grade 1: e1, e2, e3 */
    cl3_scalar t;      /* grade 3: e123 */
} Cl3Odd;

/* --- Grade mask for projection --- */
typedef uint8_t Cl3GradeMask;
#define CL3_GRADE0_MASK  0x01
#define CL3_GRADE1_MASK  0x02
#define CL3_GRADE2_MASK  0x04
#define CL3_GRADE3_MASK  0x08
#define CL3_ALL_GRADES   0x0F
#define CL3_EVEN_GRADES  0x05  /* grade 0 + grade 2 */
#define CL3_ODD_GRADES   0x0A  /* grade 1 + grade 3 */

/* --- Utility: zero-initialize a multivector --- */
static inline Cl3Multivector cl3_zero(void) {
    Cl3Multivector m = {0.0f, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, 0.0f};
    return m;
}

/* --- Utility: create a scalar multivector --- */
static inline Cl3Multivector cl3_scalar_mv(cl3_scalar s) {
    Cl3Multivector m = cl3_zero();
    m.s = s;
    return m;
}

/* --- Utility: create from all 8 components --- */
static inline Cl3Multivector cl3_from_components(
    cl3_scalar s,
    cl3_scalar e1, cl3_scalar e2, cl3_scalar e3,
    cl3_scalar e12, cl3_scalar e13, cl3_scalar e23,
    cl3_scalar e123)
{
    Cl3Multivector m;
    m.s = s;
    m.v[0] = e1;  m.v[1] = e2;  m.v[2] = e3;
    m.b[0] = e12; m.b[1] = e13; m.b[2] = e23;
    m.t = e123;
    return m;
}

/* --- Utility: access flat component by index (0-7) --- */
static inline cl3_scalar cl3_get(const Cl3Multivector *m, int idx) {
    switch (idx) {
        case 0: return m->s;
        case 1: return m->v[0];
        case 2: return m->v[1];
        case 3: return m->v[2];
        case 4: return m->b[0];
        case 5: return m->b[1];
        case 6: return m->b[2];
        case 7: return m->t;
        default: return 0.0f;
    }
}

static inline void cl3_set(Cl3Multivector *m, int idx, cl3_scalar val) {
    switch (idx) {
        case 0: m->s = val; break;
        case 1: m->v[0] = val; break;
        case 2: m->v[1] = val; break;
        case 3: m->v[2] = val; break;
        case 4: m->b[0] = val; break;
        case 5: m->b[1] = val; break;
        case 6: m->b[2] = val; break;
        case 7: m->t = val; break;
    }
}

/* --- Batch allocation/deallocation --- */
Cl3Batch cl3_batch_alloc(size_t count);
void cl3_batch_free(Cl3Batch *batch);
void cl3_batch_set(Cl3Batch *batch, size_t idx, const Cl3Multivector *mv);
Cl3Multivector cl3_batch_get(const Cl3Batch *batch, size_t idx);

#ifdef __cplusplus
}
#endif

#endif /* CL3_TYPES_H */
