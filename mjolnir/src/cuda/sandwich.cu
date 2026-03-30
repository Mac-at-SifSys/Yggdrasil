/**
 * sandwich.cu -- Fused Sandwich Product Kernel: R * x * ~R
 *
 * YGGDRASIL Clifford Algebra Stack -- mjolnir L0
 *
 * Computes both geometric products in registers without writing
 * the intermediate result to global memory.
 *
 * Basis ordering: {1, e1, e2, e3, e12, e13, e23, e123}
 *                  0   1   2   3   4    5    6    7
 */

#include "../../include/cl3_types.h"
#include "../../include/cl3_tables.h"
#include "../../include/cl3_ops.h"

#include <cuda_runtime.h>

#define CL3_BLOCK_SIZE 256

/* ------------------------------------------------------------------ */
/*  Inline device geometric product (same as geometric_product.cu)    */
/* ------------------------------------------------------------------ */

__device__ __forceinline__
void cl3_gp_dev(const float *__restrict__ a,
                const float *__restrict__ b,
                float *__restrict__ c)
{
    c[0] = 0.0f; c[1] = 0.0f; c[2] = 0.0f; c[3] = 0.0f;
    c[4] = 0.0f; c[5] = 0.0f; c[6] = 0.0f; c[7] = 0.0f;

    /* Row 0: 1 */
    float a0 = a[0];
    c[0] += a0*b[0]; c[1] += a0*b[1]; c[2] += a0*b[2]; c[3] += a0*b[3];
    c[4] += a0*b[4]; c[5] += a0*b[5]; c[6] += a0*b[6]; c[7] += a0*b[7];

    /* Row 1: e1 */
    float a1 = a[1];
    c[1] += a1*b[0]; c[0] += a1*b[1]; c[4] += a1*b[2]; c[5] += a1*b[3];
    c[2] += a1*b[4]; c[3] += a1*b[5]; c[7] += a1*b[6]; c[6] += a1*b[7];

    /* Row 2: e2 */
    float a2 = a[2];
    c[2] += a2*b[0]; c[4] -= a2*b[1]; c[0] += a2*b[2]; c[6] += a2*b[3];
    c[1] -= a2*b[4]; c[7] -= a2*b[5]; c[3] += a2*b[6]; c[5] -= a2*b[7];

    /* Row 3: e3 */
    float a3 = a[3];
    c[3] += a3*b[0]; c[5] -= a3*b[1]; c[6] -= a3*b[2]; c[0] += a3*b[3];
    c[7] += a3*b[4]; c[1] -= a3*b[5]; c[2] -= a3*b[6]; c[4] += a3*b[7];

    /* Row 4: e12 */
    float a4 = a[4];
    c[4] += a4*b[0]; c[2] -= a4*b[1]; c[1] += a4*b[2]; c[7] += a4*b[3];
    c[0] -= a4*b[4]; c[6] -= a4*b[5]; c[5] += a4*b[6]; c[3] -= a4*b[7];

    /* Row 5: e13 */
    float a5 = a[5];
    c[5] += a5*b[0]; c[3] -= a5*b[1]; c[7] -= a5*b[2]; c[1] += a5*b[3];
    c[6] += a5*b[4]; c[0] -= a5*b[5]; c[4] -= a5*b[6]; c[2] += a5*b[7];

    /* Row 6: e23 */
    float a6 = a[6];
    c[6] += a6*b[0]; c[7] += a6*b[1]; c[3] -= a6*b[2]; c[2] += a6*b[3];
    c[5] -= a6*b[4]; c[4] += a6*b[5]; c[0] -= a6*b[6]; c[1] -= a6*b[7];

    /* Row 7: e123 */
    float a7 = a[7];
    c[7] += a7*b[0]; c[6] += a7*b[1]; c[5] -= a7*b[2]; c[4] += a7*b[3];
    c[3] -= a7*b[4]; c[2] += a7*b[5]; c[1] -= a7*b[6]; c[0] -= a7*b[7];
}

/* ================================================================== */
/*  Fused sandwich kernel: out = R * x * ~R                           */
/* ================================================================== */

/**
 * Flat layout.
 * r[i*8..], x[i*8..], out[i*8..]
 *
 * Step 1: tmp = R * x          (in registers)
 * Step 2: ~R = reverse of R    (in registers)
 * Step 3: out = tmp * ~R       (in registers, then write)
 */
__global__
void cl3_sandwich_flat_kernel(
    const float *__restrict__ r,
    const float *__restrict__ x,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float lr[8], lx[8], tmp[8], rev_r[8], lout[8];

    /* Load R and x */
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        lr[k] = r[off + k];
        lx[k] = x[off + k];
    }

    /* Step 1: tmp = R * x */
    cl3_gp_dev(lr, lx, tmp);

    /* Step 2: ~R (reverse) */
    rev_r[0] =  lr[0];
    rev_r[1] =  lr[1];
    rev_r[2] =  lr[2];
    rev_r[3] =  lr[3];
    rev_r[4] = -lr[4];
    rev_r[5] = -lr[5];
    rev_r[6] = -lr[6];
    rev_r[7] = -lr[7];

    /* Step 3: out = tmp * ~R */
    cl3_gp_dev(tmp, rev_r, lout);

    /* Store */
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off + k] = lout[k];
    }
}

/* ================================================================== */
/*  Sandwich with stratified layout                                   */
/* ================================================================== */

__global__
void cl3_sandwich_stratified_kernel(
    const float *__restrict__ r_scalars,
    const float *__restrict__ r_vectors,
    const float *__restrict__ r_bivectors,
    const float *__restrict__ r_trivectors,
    const float *__restrict__ x_scalars,
    const float *__restrict__ x_vectors,
    const float *__restrict__ x_bivectors,
    const float *__restrict__ x_trivectors,
    float *__restrict__ out_scalars,
    float *__restrict__ out_vectors,
    float *__restrict__ out_bivectors,
    float *__restrict__ out_trivectors,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float lr[8], lx[8], tmp[8], rev_r[8], lout[8];

    /* Gather R */
    lr[0] = r_scalars[tid];
    lr[1] = r_vectors[tid * 3 + 0];
    lr[2] = r_vectors[tid * 3 + 1];
    lr[3] = r_vectors[tid * 3 + 2];
    lr[4] = r_bivectors[tid * 3 + 0];
    lr[5] = r_bivectors[tid * 3 + 1];
    lr[6] = r_bivectors[tid * 3 + 2];
    lr[7] = r_trivectors[tid];

    /* Gather x */
    lx[0] = x_scalars[tid];
    lx[1] = x_vectors[tid * 3 + 0];
    lx[2] = x_vectors[tid * 3 + 1];
    lx[3] = x_vectors[tid * 3 + 2];
    lx[4] = x_bivectors[tid * 3 + 0];
    lx[5] = x_bivectors[tid * 3 + 1];
    lx[6] = x_bivectors[tid * 3 + 2];
    lx[7] = x_trivectors[tid];

    /* tmp = R * x */
    cl3_gp_dev(lr, lx, tmp);

    /* ~R */
    rev_r[0] =  lr[0]; rev_r[1] =  lr[1]; rev_r[2] =  lr[2]; rev_r[3] =  lr[3];
    rev_r[4] = -lr[4]; rev_r[5] = -lr[5]; rev_r[6] = -lr[6]; rev_r[7] = -lr[7];

    /* out = tmp * ~R */
    cl3_gp_dev(tmp, rev_r, lout);

    /* Scatter */
    out_scalars[tid]           = lout[0];
    out_vectors[tid * 3 + 0]   = lout[1];
    out_vectors[tid * 3 + 1]   = lout[2];
    out_vectors[tid * 3 + 2]   = lout[3];
    out_bivectors[tid * 3 + 0] = lout[4];
    out_bivectors[tid * 3 + 1] = lout[5];
    out_bivectors[tid * 3 + 2] = lout[6];
    out_trivectors[tid]        = lout[7];
}

/* ================================================================== */
/*  Launch wrappers (C linkage)                                       */
/* ================================================================== */

extern "C" {

void cl3_sandwich_flat_launch(
    const float *d_r,
    const float *d_x,
    float       *d_out,
    int          N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_sandwich_flat_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_r, d_x, d_out, N);
}

void cl3_sandwich_stratified_launch(
    const Cl3Batch *d_r,
    const Cl3Batch *d_x,
    Cl3Batch       *d_out,
    int             N,
    cudaStream_t    stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_sandwich_stratified_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_r->scalars, d_r->vectors, d_r->bivectors, d_r->trivectors,
        d_x->scalars, d_x->vectors, d_x->bivectors, d_x->trivectors,
        d_out->scalars, d_out->vectors, d_out->bivectors, d_out->trivectors,
        N);
}

} /* extern "C" */
