/**
 * fused_kernels.cu -- Fused Operation Kernels
 *
 * YGGDRASIL Clifford Algebra Stack -- mjolnir L0
 *
 * Fused kernels that combine two operations in a single pass to avoid
 * intermediate global memory traffic:
 *
 *   1. grade_project(geometric_product(a, b))
 *      - Only compute the output components belonging to the target grade.
 *
 *   2. grade_project(sandwich(r, x))
 *      - Sandwich + grade projection in one kernel.
 *
 *   3. normalize(x) = x / norm(x)
 *      - Compute norm and divide in one pass.
 *
 * Basis ordering: {1, e1, e2, e3, e12, e13, e23, e123}
 *                  0   1   2   3   4    5    6    7
 */

#include "../../include/cl3_types.h"
#include "../../include/cl3_tables.h"
#include "../../include/cl3_ops.h"

#include <cuda_runtime.h>
#include <math.h>

#define CL3_BLOCK_SIZE 256

/* ------------------------------------------------------------------ */
/*  Device helpers                                                    */
/* ------------------------------------------------------------------ */

/* Full geometric product (inline, same as other files) */
__device__ __forceinline__
void cl3_gp_dev(const float *__restrict__ a,
                const float *__restrict__ b,
                float *__restrict__ c)
{
    c[0] = 0.0f; c[1] = 0.0f; c[2] = 0.0f; c[3] = 0.0f;
    c[4] = 0.0f; c[5] = 0.0f; c[6] = 0.0f; c[7] = 0.0f;

    float a0 = a[0];
    c[0] += a0*b[0]; c[1] += a0*b[1]; c[2] += a0*b[2]; c[3] += a0*b[3];
    c[4] += a0*b[4]; c[5] += a0*b[5]; c[6] += a0*b[6]; c[7] += a0*b[7];

    float a1 = a[1];
    c[1] += a1*b[0]; c[0] += a1*b[1]; c[4] += a1*b[2]; c[5] += a1*b[3];
    c[2] += a1*b[4]; c[3] += a1*b[5]; c[7] += a1*b[6]; c[6] += a1*b[7];

    float a2 = a[2];
    c[2] += a2*b[0]; c[4] -= a2*b[1]; c[0] += a2*b[2]; c[6] += a2*b[3];
    c[1] -= a2*b[4]; c[7] -= a2*b[5]; c[3] += a2*b[6]; c[5] -= a2*b[7];

    float a3 = a[3];
    c[3] += a3*b[0]; c[5] -= a3*b[1]; c[6] -= a3*b[2]; c[0] += a3*b[3];
    c[7] += a3*b[4]; c[1] -= a3*b[5]; c[2] -= a3*b[6]; c[4] += a3*b[7];

    float a4 = a[4];
    c[4] += a4*b[0]; c[2] -= a4*b[1]; c[1] += a4*b[2]; c[7] += a4*b[3];
    c[0] -= a4*b[4]; c[6] -= a4*b[5]; c[5] += a4*b[6]; c[3] -= a4*b[7];

    float a5 = a[5];
    c[5] += a5*b[0]; c[3] -= a5*b[1]; c[7] -= a5*b[2]; c[1] += a5*b[3];
    c[6] += a5*b[4]; c[0] -= a5*b[5]; c[4] -= a5*b[6]; c[2] += a5*b[7];

    float a6 = a[6];
    c[6] += a6*b[0]; c[7] += a6*b[1]; c[3] -= a6*b[2]; c[2] += a6*b[3];
    c[5] -= a6*b[4]; c[4] += a6*b[5]; c[0] -= a6*b[6]; c[1] -= a6*b[7];

    float a7 = a[7];
    c[7] += a7*b[0]; c[6] += a7*b[1]; c[5] -= a7*b[2]; c[4] += a7*b[3];
    c[3] -= a7*b[4]; c[2] += a7*b[5]; c[1] -= a7*b[6]; c[0] -= a7*b[7];
}

/* ================================================================== */
/*  Fused: grade_project(geometric_product(a, b))                     */
/* ================================================================== */

/**
 * Compute only the grade-k components of a*b.
 *
 * For small target grades this is significantly cheaper than full GP:
 *   Grade 0 (scalar): 1 output component
 *   Grade 1 (vector): 3 output components
 *   Grade 2 (bivector): 3 output components
 *   Grade 3 (trivector): 1 output component
 *
 * We still compute partial sums but only for the relevant output slots.
 * The output is a full 8-component MV with non-target grades zeroed.
 */

/*
 * Grade 0 only: compute just c[0]
 * c[0] = sum over all (i,j) where product_idx[i][j]==0 of sign*a[i]*b[j]
 * These pairs are: (0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)
 * with signs:       +1    +1    +1    +1    -1    -1    -1    -1
 */
__device__ __forceinline__
float cl3_gp_grade0_dev(const float *a, const float *b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
         - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
}

/*
 * Grade 1 only: compute c[1], c[2], c[3]
 *
 * c[1]: pairs mapping to index 1 with their signs:
 *   (0,1)+  (1,0)+  (2,4)-  (3,5)-  (4,2)+  (5,3)+  (6,7)-  (7,6)-
 * c[2]: pairs mapping to index 2:
 *   (0,2)+  (1,4)+  (2,0)+  (3,6)-  (4,1)-  (5,7)+  (6,3)-  (7,5)-  -- wait,
 *
 * Actually let me just compute full GP and mask. The partial computation
 * for grade 1 or grade 2 still touches all 8 input components of both
 * a and b, so the savings from skipping accumulation into unused slots
 * is modest. The real win is that we can do GP + project in one kernel
 * call, saving a kernel launch and global memory round-trip.
 */
__global__
void cl3_fused_gp_grade_kernel(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float       *__restrict__ out,
    int          grade,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float la[8], lb[8], lc[8];

    #pragma unroll
    for (int k = 0; k < 8; k++) { la[k] = a[off+k]; lb[k] = b[off+k]; }

    /* For grade 0 we can take the fast path */
    if (grade == 0) {
        float s = cl3_gp_grade0_dev(la, lb);
        out[off+0] = s;
        out[off+1] = 0.0f; out[off+2] = 0.0f; out[off+3] = 0.0f;
        out[off+4] = 0.0f; out[off+5] = 0.0f; out[off+6] = 0.0f;
        out[off+7] = 0.0f;
        return;
    }

    /* General case: full GP then mask */
    cl3_gp_dev(la, lb, lc);

    /* Grade index ranges: 0->{0}, 1->{1,2,3}, 2->{4,5,6}, 3->{7} */
    int gs, ge;
    switch (grade) {
        case 1: gs = 1; ge = 4; break;
        case 2: gs = 4; ge = 7; break;
        case 3: gs = 7; ge = 8; break;
        default: gs = 0; ge = 0; break;
    }

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off+k] = (k >= gs && k < ge) ? lc[k] : 0.0f;
    }
}

/**
 * Fused GP + grade mask (multiple grades at once).
 */
__global__
void cl3_fused_gp_grade_mask_kernel(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float       *__restrict__ out,
    int          mask,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float la[8], lb[8], lc[8];

    #pragma unroll
    for (int k = 0; k < 8; k++) { la[k] = a[off+k]; lb[k] = b[off+k]; }

    cl3_gp_dev(la, lb, lc);

    /* Basis grade table hardcoded: {0,1,1,1,2,2,2,3} */
    static const int bg[8] = {0,1,1,1,2,2,2,3};

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int keep = (mask >> bg[k]) & 1;
        out[off+k] = keep ? lc[k] : 0.0f;
    }
}

/* ================================================================== */
/*  Fused: grade_project(sandwich(r, x))                              */
/* ================================================================== */

__global__
void cl3_fused_sandwich_grade_kernel(
    const float *__restrict__ r,
    const float *__restrict__ x,
    float       *__restrict__ out,
    int          grade,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float lr[8], lx[8], tmp[8], rev[8], lc[8];

    #pragma unroll
    for (int k = 0; k < 8; k++) { lr[k] = r[off+k]; lx[k] = x[off+k]; }

    /* tmp = R * x */
    cl3_gp_dev(lr, lx, tmp);

    /* ~R */
    rev[0]= lr[0]; rev[1]= lr[1]; rev[2]= lr[2]; rev[3]= lr[3];
    rev[4]=-lr[4]; rev[5]=-lr[5]; rev[6]=-lr[6]; rev[7]=-lr[7];

    /* lc = tmp * ~R */
    cl3_gp_dev(tmp, rev, lc);

    /* Grade mask */
    int gs, ge;
    switch (grade) {
        case 0: gs = 0; ge = 1; break;
        case 1: gs = 1; ge = 4; break;
        case 2: gs = 4; ge = 7; break;
        case 3: gs = 7; ge = 8; break;
        default: gs = 0; ge = 0; break;
    }

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off+k] = (k >= gs && k < ge) ? lc[k] : 0.0f;
    }
}

/* ================================================================== */
/*  Fused: normalize(x) = x / norm(x)                                */
/* ================================================================== */

/**
 * Normalize each multivector in-place (or to a separate output).
 * Handles near-zero norms gracefully by clamping.
 */
__global__
void cl3_normalize_kernel(
    const float *__restrict__ in,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float sum = 0.0f;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        float v = in[off + k];
        sum += v * v;
    }

    /* Avoid division by zero */
    float inv_norm;
    if (sum > 1.0e-14f) {
        inv_norm = rsqrtf(sum);
    } else {
        inv_norm = 0.0f;
    }

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off + k] = in[off + k] * inv_norm;
    }
}

/**
 * Normalize using an externally-computed norm array.
 * Useful when the norm has already been computed for other purposes.
 */
__global__
void cl3_normalize_with_norm_kernel(
    const float *__restrict__ in,
    const float *__restrict__ norms,   /* [N] scalar norms */
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float n = norms[tid];
    float inv_n = (n > 1.0e-7f) ? (1.0f / n) : 0.0f;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off + k] = in[off + k] * inv_n;
    }
}

/* ================================================================== */
/*  Launch wrappers (C linkage)                                       */
/* ================================================================== */

extern "C" {

void cl3_fused_gp_grade_launch(
    const float *d_a,
    const float *d_b,
    float       *d_out,
    int          grade,
    int          N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_fused_gp_grade_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_a, d_b, d_out, grade, N);
}

void cl3_fused_gp_grade_mask_launch(
    const float *d_a,
    const float *d_b,
    float       *d_out,
    int          mask,
    int          N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_fused_gp_grade_mask_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_a, d_b, d_out, mask, N);
}

void cl3_fused_sandwich_grade_launch(
    const float *d_r,
    const float *d_x,
    float       *d_out,
    int          grade,
    int          N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_fused_sandwich_grade_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_r, d_x, d_out, grade, N);
}

void cl3_normalize_launch(
    const float *d_in,
    float       *d_out,
    int          N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_normalize_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_in, d_out, N);
}

void cl3_normalize_with_norm_launch(
    const float *d_in,
    const float *d_norms,
    float       *d_out,
    int          N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_normalize_with_norm_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_in, d_norms, d_out, N);
}

} /* extern "C" */
