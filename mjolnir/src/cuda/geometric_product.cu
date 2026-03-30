/**
 * geometric_product.cu -- Batched Cl(3,0) Geometric Product CUDA Kernels
 *
 * YGGDRASIL Clifford Algebra Stack -- mjolnir L0
 *
 * Each thread computes one geometric product of two multivectors.
 * The full 8x8 multiplication table is hardcoded with unrolled
 * multiply-adds for maximum throughput. Supports both flat-array
 * (8 floats per MV) and grade-stratified (Cl3Batch) layouts.
 *
 * Basis ordering: {1, e1, e2, e3, e12, e13, e23, e123}
 *                  0   1   2   3   4    5    6    7
 */

#include "../../include/cl3_types.h"
#include "../../include/cl3_tables.h"
#include "../../include/cl3_ops.h"

#include <cuda_runtime.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/*  Constant memory: multiplication tables                            */
/* ------------------------------------------------------------------ */
__constant__ int d_product_idx[8][8];
__constant__ int d_product_sign[8][8];

#define CL3_BLOCK_SIZE 256

/* Flag to avoid re-uploading tables every launch */
static int s_tables_uploaded = 0;

static void ensure_tables_uploaded(void)
{
    if (s_tables_uploaded) return;
    cudaMemcpyToSymbol(d_product_idx,  CL3_PRODUCT_IDX,  sizeof(CL3_PRODUCT_IDX));
    cudaMemcpyToSymbol(d_product_sign, CL3_PRODUCT_SIGN, sizeof(CL3_PRODUCT_SIGN));
    s_tables_uploaded = 1;
}

/* ================================================================== */
/*  Device helper: fully-unrolled geometric product                   */
/* ================================================================== */

/**
 * Compute c = a * b using the Cl(3,0) geometric product.
 *
 * Fully unrolled: 64 fmads, no loops, no table lookups at runtime.
 * The signs and index mappings are baked in from CL3_PRODUCT_SIGN
 * and CL3_PRODUCT_IDX.
 *
 * Row i of the multiplication table tells us:
 *   a[i] * b[j]  contributes  sign[i][j] * a[i]*b[j]  to  c[idx[i][j]]
 */
__device__ __forceinline__
void cl3_gp_device(const float *__restrict__ a,
                   const float *__restrict__ b,
                   float *__restrict__ c)
{
    /* Zero the output */
    c[0] = 0.0f; c[1] = 0.0f; c[2] = 0.0f; c[3] = 0.0f;
    c[4] = 0.0f; c[5] = 0.0f; c[6] = 0.0f; c[7] = 0.0f;

    /* ---- Row 0: basis = 1 ----
     *  1*1=+1       1*e1=+e1     1*e2=+e2     1*e3=+e3
     *  1*e12=+e12   1*e13=+e13   1*e23=+e23   1*e123=+e123 */
    float a0 = a[0];
    c[0] += a0 * b[0];
    c[1] += a0 * b[1];
    c[2] += a0 * b[2];
    c[3] += a0 * b[3];
    c[4] += a0 * b[4];
    c[5] += a0 * b[5];
    c[6] += a0 * b[6];
    c[7] += a0 * b[7];

    /* ---- Row 1: basis = e1 ----
     *  e1*1=+e1     e1*e1=+1     e1*e2=+e12   e1*e3=+e13
     *  e1*e12=+e2   e1*e13=+e3   e1*e23=+e123 e1*e123=+e23 */
    float a1 = a[1];
    c[1] += a1 * b[0];
    c[0] += a1 * b[1];
    c[4] += a1 * b[2];
    c[5] += a1 * b[3];
    c[2] += a1 * b[4];
    c[3] += a1 * b[5];
    c[7] += a1 * b[6];
    c[6] += a1 * b[7];

    /* ---- Row 2: basis = e2 ----
     *  e2*1=+e2     e2*e1=-e12   e2*e2=+1     e2*e3=+e23
     *  e2*e12=-e1   e2*e13=-e123 e2*e23=+e3   e2*e123=-e13 */
    float a2 = a[2];
    c[2] += a2 * b[0];
    c[4] -= a2 * b[1];
    c[0] += a2 * b[2];
    c[6] += a2 * b[3];
    c[1] -= a2 * b[4];
    c[7] -= a2 * b[5];
    c[3] += a2 * b[6];
    c[5] -= a2 * b[7];

    /* ---- Row 3: basis = e3 ----
     *  e3*1=+e3     e3*e1=-e13   e3*e2=-e23   e3*e3=+1
     *  e3*e12=+e123 e3*e13=-e1   e3*e23=-e2   e3*e123=+e12 */
    float a3 = a[3];
    c[3] += a3 * b[0];
    c[5] -= a3 * b[1];
    c[6] -= a3 * b[2];
    c[0] += a3 * b[3];
    c[7] += a3 * b[4];
    c[1] -= a3 * b[5];
    c[2] -= a3 * b[6];
    c[4] += a3 * b[7];

    /* ---- Row 4: basis = e12 ----
     *  e12*1=+e12   e12*e1=-e2   e12*e2=+e1   e12*e3=+e123
     *  e12*e12=-1   e12*e13=-e23 e12*e23=+e13  e12*e123=-e3 */
    float a4 = a[4];
    c[4] += a4 * b[0];
    c[2] -= a4 * b[1];
    c[1] += a4 * b[2];
    c[7] += a4 * b[3];
    c[0] -= a4 * b[4];
    c[6] -= a4 * b[5];
    c[5] += a4 * b[6];
    c[3] -= a4 * b[7];

    /* ---- Row 5: basis = e13 ----
     *  e13*1=+e13   e13*e1=-e3   e13*e2=-e123  e13*e3=+e1
     *  e13*e12=+e23 e13*e13=-1   e13*e23=-e12  e13*e123=+e2 */
    float a5 = a[5];
    c[5] += a5 * b[0];
    c[3] -= a5 * b[1];
    c[7] -= a5 * b[2];
    c[1] += a5 * b[3];
    c[6] += a5 * b[4];
    c[0] -= a5 * b[5];
    c[4] -= a5 * b[6];
    c[2] += a5 * b[7];

    /* ---- Row 6: basis = e23 ----
     *  e23*1=+e23   e23*e1=+e123 e23*e2=-e3   e23*e3=+e2
     *  e23*e12=-e13 e23*e13=+e12 e23*e23=-1   e23*e123=-e1 */
    float a6 = a[6];
    c[6] += a6 * b[0];
    c[7] += a6 * b[1];
    c[3] -= a6 * b[2];
    c[2] += a6 * b[3];
    c[5] -= a6 * b[4];
    c[4] += a6 * b[5];
    c[0] -= a6 * b[6];
    c[1] -= a6 * b[7];

    /* ---- Row 7: basis = e123 ----
     *  e123*1=+e123  e123*e1=+e23  e123*e2=-e13  e123*e3=+e12
     *  e123*e12=-e3  e123*e13=+e2  e123*e23=-e1  e123*e123=-1 */
    float a7 = a[7];
    c[7] += a7 * b[0];
    c[6] += a7 * b[1];
    c[5] -= a7 * b[2];
    c[4] += a7 * b[3];
    c[3] -= a7 * b[4];
    c[2] += a7 * b[5];
    c[1] -= a7 * b[6];
    c[0] -= a7 * b[7];
}

/* ================================================================== */
/*  Kernel: batched geometric product (flat layout)                   */
/* ================================================================== */

/**
 * Flat layout: a and b are arrays of N multivectors, each 8 contiguous
 * floats.  a[i*8 .. i*8+7], etc.
 */
__global__
void cl3_geometric_product_flat_kernel(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float *__restrict__ out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float la[8], lb[8], lc[8];

    /* Load a and b into registers */
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        la[k] = a[off + k];
        lb[k] = b[off + k];
    }

    cl3_gp_device(la, lb, lc);

    /* Store result */
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off + k] = lc[k];
    }
}

/* ================================================================== */
/*  Kernel: batched geometric product (grade-stratified layout)       */
/* ================================================================== */

/**
 * Grade-stratified: components stored in separate arrays per grade.
 *   scalars[i]            -> flat index 0
 *   vectors[i*3+0..2]     -> flat indices 1,2,3
 *   bivectors[i*3+0..2]   -> flat indices 4,5,6
 *   trivectors[i]         -> flat index 7
 */
__global__
void cl3_geometric_product_stratified_kernel(
    const float *__restrict__ a_scalars,
    const float *__restrict__ a_vectors,
    const float *__restrict__ a_bivectors,
    const float *__restrict__ a_trivectors,
    const float *__restrict__ b_scalars,
    const float *__restrict__ b_vectors,
    const float *__restrict__ b_bivectors,
    const float *__restrict__ b_trivectors,
    float *__restrict__ out_scalars,
    float *__restrict__ out_vectors,
    float *__restrict__ out_bivectors,
    float *__restrict__ out_trivectors,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float la[8], lb[8], lc[8];

    /* Gather from stratified layout */
    la[0] = a_scalars[tid];
    la[1] = a_vectors[tid * 3 + 0];
    la[2] = a_vectors[tid * 3 + 1];
    la[3] = a_vectors[tid * 3 + 2];
    la[4] = a_bivectors[tid * 3 + 0];
    la[5] = a_bivectors[tid * 3 + 1];
    la[6] = a_bivectors[tid * 3 + 2];
    la[7] = a_trivectors[tid];

    lb[0] = b_scalars[tid];
    lb[1] = b_vectors[tid * 3 + 0];
    lb[2] = b_vectors[tid * 3 + 1];
    lb[3] = b_vectors[tid * 3 + 2];
    lb[4] = b_bivectors[tid * 3 + 0];
    lb[5] = b_bivectors[tid * 3 + 1];
    lb[6] = b_bivectors[tid * 3 + 2];
    lb[7] = b_trivectors[tid];

    cl3_gp_device(la, lb, lc);

    /* Scatter to stratified layout */
    out_scalars[tid]          = lc[0];
    out_vectors[tid * 3 + 0]  = lc[1];
    out_vectors[tid * 3 + 1]  = lc[2];
    out_vectors[tid * 3 + 2]  = lc[3];
    out_bivectors[tid * 3 + 0] = lc[4];
    out_bivectors[tid * 3 + 1] = lc[5];
    out_bivectors[tid * 3 + 2] = lc[6];
    out_trivectors[tid]        = lc[7];
}

/* ================================================================== */
/*  Launch wrappers with C linkage                                    */
/* ================================================================== */

extern "C" {

void cl3_geometric_product_flat_launch(
    const float *d_a,
    const float *d_b,
    float       *d_out,
    int          N,
    cudaStream_t stream)
{
    ensure_tables_uploaded();
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_geometric_product_flat_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_a, d_b, d_out, N);
}

void cl3_geometric_product_stratified_launch(
    const Cl3Batch *d_a,
    const Cl3Batch *d_b,
    Cl3Batch       *d_out,
    int             N,
    cudaStream_t    stream)
{
    ensure_tables_uploaded();
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_geometric_product_stratified_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_a->scalars, d_a->vectors, d_a->bivectors, d_a->trivectors,
        d_b->scalars, d_b->vectors, d_b->bivectors, d_b->trivectors,
        d_out->scalars, d_out->vectors, d_out->bivectors, d_out->trivectors,
        N);
}

} /* extern "C" */
