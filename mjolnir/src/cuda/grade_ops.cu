/**
 * grade_ops.cu -- Grade Projection, Involutions, Norms, Arithmetic
 *
 * YGGDRASIL Clifford Algebra Stack -- mjolnir L0
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
/*  Constant memory: sign tables                                      */
/* ------------------------------------------------------------------ */
__constant__ int d_reverse_sign[8];
__constant__ int d_involution_sign[8];
__constant__ int d_conjugate_sign[8];
__constant__ int d_basis_grade[8];

static int s_grade_tables_uploaded = 0;

static void ensure_grade_tables_uploaded(void)
{
    if (s_grade_tables_uploaded) return;
    cudaMemcpyToSymbol(d_reverse_sign,    CL3_REVERSE_SIGN,    sizeof(CL3_REVERSE_SIGN));
    cudaMemcpyToSymbol(d_involution_sign,  CL3_INVOLUTION_SIGN, sizeof(CL3_INVOLUTION_SIGN));
    cudaMemcpyToSymbol(d_conjugate_sign,   CL3_CONJUGATE_SIGN,  sizeof(CL3_CONJUGATE_SIGN));
    cudaMemcpyToSymbol(d_basis_grade,      CL3_BASIS_GRADE,     sizeof(CL3_BASIS_GRADE));
    s_grade_tables_uploaded = 1;
}

/* ================================================================== */
/*  Grade projection kernel                                           */
/* ================================================================== */

/**
 * Extract grade-k from a batch of flat multivectors.
 * Components not belonging to grade k are zeroed.
 *
 * Grade  -> flat indices
 *   0    -> {0}
 *   1    -> {1,2,3}
 *   2    -> {4,5,6}
 *   3    -> {7}
 */
__global__
void cl3_grade_project_kernel(
    const float *__restrict__ in,
    float       *__restrict__ out,
    int          grade,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off + k] = (d_basis_grade[k] == grade) ? in[off + k] : 0.0f;
    }
}

/**
 * Grade projection with bitmask: keep multiple grades.
 * mask bit 0 = grade 0, bit 1 = grade 1, etc.
 */
__global__
void cl3_grade_mask_project_kernel(
    const float *__restrict__ in,
    float       *__restrict__ out,
    int          mask,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int keep = (mask >> d_basis_grade[k]) & 1;
        out[off + k] = keep ? in[off + k] : 0.0f;
    }
}

/* ================================================================== */
/*  Reverse kernel                                                    */
/* ================================================================== */

/**
 * Reversion: flip sign of grade 2 and grade 3 components.
 * ~x:  signs = {+1,+1,+1,+1,-1,-1,-1,-1}
 */
__global__
void cl3_reverse_kernel(
    const float *__restrict__ in,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    /* Hardcoded for speed -- matches CL3_REVERSE_SIGN */
    out[off + 0] =  in[off + 0];
    out[off + 1] =  in[off + 1];
    out[off + 2] =  in[off + 2];
    out[off + 3] =  in[off + 3];
    out[off + 4] = -in[off + 4];
    out[off + 5] = -in[off + 5];
    out[off + 6] = -in[off + 6];
    out[off + 7] = -in[off + 7];
}

/* ================================================================== */
/*  Involution kernel                                                 */
/* ================================================================== */

/**
 * Grade involution: signs = {+1,-1,-1,-1,+1,+1,+1,-1}
 */
__global__
void cl3_involution_kernel(
    const float *__restrict__ in,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    out[off + 0] =  in[off + 0];
    out[off + 1] = -in[off + 1];
    out[off + 2] = -in[off + 2];
    out[off + 3] = -in[off + 3];
    out[off + 4] =  in[off + 4];
    out[off + 5] =  in[off + 5];
    out[off + 6] =  in[off + 6];
    out[off + 7] = -in[off + 7];
}

/* ================================================================== */
/*  Conjugate kernel                                                  */
/* ================================================================== */

/**
 * Clifford conjugate = reverse * involution
 * signs = {+1,-1,-1,-1,-1,-1,-1,+1}
 */
__global__
void cl3_conjugate_kernel(
    const float *__restrict__ in,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    out[off + 0] =  in[off + 0];
    out[off + 1] = -in[off + 1];
    out[off + 2] = -in[off + 2];
    out[off + 3] = -in[off + 3];
    out[off + 4] = -in[off + 4];
    out[off + 5] = -in[off + 5];
    out[off + 6] = -in[off + 6];
    out[off + 7] =  in[off + 7];
}

/* ================================================================== */
/*  Norm-squared kernel                                               */
/* ================================================================== */

/**
 * norm_sq = scalar part of x * ~x
 *
 * For Cl(3,0) with signature (+,+,+):
 *   x * ~x scalar part = s^2 + v1^2 + v2^2 + v3^2
 *                       - b12^2 - b13^2 - b23^2 - t^2
 *
 * Wait -- let's be precise.  For Cl(3,0):
 *   <x ~x>_0 = x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2
 *             - x[4]^2 - x[5]^2 - x[6]^2 + x[7]^2
 *
 * Actually we need to compute this correctly from the table.
 * For x * ~x, the scalar part is sum_i sum_j x[i]*rev[j]*sign[i][j]*delta(idx[i][j],0)
 * where rev[j] = reverse_sign[j]*x[j].
 *
 * The pairs (i,j) that map to index 0 are:
 *   (0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)
 *
 * So <x*~x>_0 = sum_i x[i] * reverse_sign[i]*x[i] * sign[i][i]
 *
 * sign[i][i] from table: {+1,+1,+1,+1,-1,-1,-1,-1}
 * reverse_sign:           {+1,+1,+1,+1,-1,-1,-1,-1}
 * Product sign[i][i]*rev[i]: {+1,+1,+1,+1,+1,+1,+1,+1}
 *
 * So norm_sq = x[0]^2+x[1]^2+x[2]^2+x[3]^2+x[4]^2+x[5]^2+x[6]^2+x[7]^2
 *
 * That's just the sum of squares for Cl(3,0).  This is correct because
 * the metric is positive definite.
 */
__global__
void cl3_norm_squared_kernel(
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

    out[tid] = sum;
}

/**
 * Norm = sqrt(norm_squared)
 */
__global__
void cl3_norm_kernel(
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

    out[tid] = sqrtf(sum);
}

/* ================================================================== */
/*  Add / Subtract / Scale kernels                                    */
/* ================================================================== */

__global__
void cl3_add_kernel(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off + k] = a[off + k] + b[off + k];
    }
}

__global__
void cl3_sub_kernel(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off + k] = a[off + k] - b[off + k];
    }
}

__global__
void cl3_scale_kernel(
    const float *__restrict__ in,
    float        s,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off + k] = s * in[off + k];
    }
}

__global__
void cl3_negate_kernel(
    const float *__restrict__ in,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    #pragma unroll
    for (int k = 0; k < 8; k++) {
        out[off + k] = -in[off + k];
    }
}

/* ================================================================== */
/*  Launch wrappers (C linkage)                                       */
/* ================================================================== */

extern "C" {

void cl3_grade_project_launch(
    const float *d_in, float *d_out, int grade, int N, cudaStream_t stream)
{
    ensure_grade_tables_uploaded();
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_grade_project_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_in, d_out, grade, N);
}

void cl3_grade_mask_project_launch(
    const float *d_in, float *d_out, int mask, int N, cudaStream_t stream)
{
    ensure_grade_tables_uploaded();
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_grade_mask_project_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_in, d_out, mask, N);
}

void cl3_reverse_launch(
    const float *d_in, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_reverse_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(d_in, d_out, N);
}

void cl3_involution_launch(
    const float *d_in, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_involution_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(d_in, d_out, N);
}

void cl3_conjugate_launch(
    const float *d_in, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_conjugate_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(d_in, d_out, N);
}

void cl3_norm_squared_launch(
    const float *d_in, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_norm_squared_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_in, d_out, N);
}

void cl3_norm_launch(
    const float *d_in, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_norm_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(d_in, d_out, N);
}

void cl3_add_launch(
    const float *d_a, const float *d_b, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_add_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_out, N);
}

void cl3_sub_launch(
    const float *d_a, const float *d_b, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_sub_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_out, N);
}

void cl3_scale_launch(
    const float *d_in, float s, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_scale_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(d_in, s, d_out, N);
}

void cl3_negate_launch(
    const float *d_in, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_negate_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(d_in, d_out, N);
}

} /* extern "C" */
