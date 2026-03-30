/**
 * batch_ops.cu -- Batched Operations and Geometric "Matrix Multiply"
 *
 * YGGDRASIL Clifford Algebra Stack -- mjolnir L0
 *
 * Provides batched wrappers for all core operations and a geometric
 * algebra "matrix multiply":
 *   out[i,j] = sum_k  a[i,k] * b[k,j]
 * where * is the Cl(3,0) geometric product.
 *
 * The matmul kernel uses shared memory to tile the inner dimension K,
 * analogous to classical GEMM tiling but with 8-component MVs instead
 * of scalars.
 *
 * Basis ordering: {1, e1, e2, e3, e12, e13, e23, e123}
 */

#include "../../include/cl3_types.h"
#include "../../include/cl3_tables.h"
#include "../../include/cl3_ops.h"

#include <cuda_runtime.h>

#define CL3_BLOCK_SIZE 256

/* Tile size for the geometric matmul.
 * Each block handles a TILE_M x TILE_N tile of the output.
 * We iterate over the K dimension in chunks of TILE_K. */
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

/* ------------------------------------------------------------------ */
/*  Inline device geometric product                                   */
/* ------------------------------------------------------------------ */

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

/**
 * Accumulate: c += a * b  (geometric product, added to existing c)
 */
__device__ __forceinline__
void cl3_gp_acc_dev(const float *__restrict__ a,
                    const float *__restrict__ b,
                    float *__restrict__ c)
{
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
/*  Batched element-wise operations (flat layout)                     */
/* ================================================================== */

/* Batched geometric product: out[i] = a[i] * b[i] */
__global__
void cl3_batch_gp_kernel(
    const float *__restrict__ a,
    const float *__restrict__ b,
    float       *__restrict__ out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float la[8], lb[8], lc[8];

    #pragma unroll
    for (int k = 0; k < 8; k++) { la[k] = a[off+k]; lb[k] = b[off+k]; }

    cl3_gp_dev(la, lb, lc);

    #pragma unroll
    for (int k = 0; k < 8; k++) out[off+k] = lc[k];
}

/* Batched sandwich: out[i] = r[i] * x[i] * ~r[i] */
__global__
void cl3_batch_sandwich_kernel(
    const float *__restrict__ r,
    const float *__restrict__ x,
    float       *__restrict__ out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float lr[8], lx[8], tmp[8], rev[8], lo[8];

    #pragma unroll
    for (int k = 0; k < 8; k++) { lr[k] = r[off+k]; lx[k] = x[off+k]; }

    cl3_gp_dev(lr, lx, tmp);

    rev[0]= lr[0]; rev[1]= lr[1]; rev[2]= lr[2]; rev[3]= lr[3];
    rev[4]=-lr[4]; rev[5]=-lr[5]; rev[6]=-lr[6]; rev[7]=-lr[7];

    cl3_gp_dev(tmp, rev, lo);

    #pragma unroll
    for (int k = 0; k < 8; k++) out[off+k] = lo[k];
}

/* Batched bivector exp: out[i] = exp(bv[i]) */
__global__
void cl3_batch_bvexp_kernel(
    const float *__restrict__ bv,
    float       *__restrict__ out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float b12 = bv[off+4], b13 = bv[off+5], b23 = bv[off+6];
    float mag_sq = b12*b12 + b13*b13 + b23*b23;
    float mag = sqrtf(mag_sq);

    float cv, sv;
    if (mag > 1.0e-7f) {
        cv = cosf(mag);
        sv = sinf(mag) / mag;
    } else {
        cv = 1.0f - mag_sq * (0.5f - mag_sq / 24.0f);
        sv = 1.0f - mag_sq * (1.0f/6.0f - mag_sq / 120.0f);
    }

    out[off+0] = cv;    out[off+1] = 0.0f;  out[off+2] = 0.0f;  out[off+3] = 0.0f;
    out[off+4] = sv*b12; out[off+5] = sv*b13; out[off+6] = sv*b23; out[off+7] = 0.0f;
}

/* Batched grade projection */
__global__
void cl3_batch_grade_proj_kernel(
    const float *__restrict__ in,
    float       *__restrict__ out,
    int          grade,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    /* Grade -> index ranges: 0->{0}, 1->{1,2,3}, 2->{4,5,6}, 3->{7} */
    int off = tid * 8;

    /* Start/end indices for each grade */
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
        out[off+k] = (k >= gs && k < ge) ? in[off+k] : 0.0f;
    }
}

/* Batched norm: out_scalar[i] = norm(in[i]) */
__global__
void cl3_batch_norm_kernel(
    const float *__restrict__ in,
    float       *__restrict__ out,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++) { float v = in[off+k]; sum += v*v; }
    out[tid] = sqrtf(sum);
}

/* ================================================================== */
/*  Geometric "Matrix Multiply" with shared memory tiling             */
/* ================================================================== */

/**
 * out[i,j] = sum_k  a[i,k] * b[k,j]   (geometric product)
 *
 * a: [M x K] stored row-major as a[i*K + k], each element is 8 floats
 * b: [K x N] stored row-major as b[k*N + j], each element is 8 floats
 * out: [M x N]
 *
 * Block: TILE_M x TILE_N threads (each thread handles one (i,j) output).
 * Shared memory: TILE_M x TILE_K chunk of A and TILE_K x TILE_N chunk of B.
 * Each tile element is 8 floats = 32 bytes.
 */
__global__
void cl3_geom_matmul_kernel(
    const float *__restrict__ a,     /* [M*K*8] */
    const float *__restrict__ b,     /* [K*N*8] */
    float       *__restrict__ out,   /* [M*N*8] */
    int M, int K, int N)
{
    /* Thread coordinates within the output tile */
    int tx = threadIdx.x;  /* column within tile, 0..TILE_N-1 */
    int ty = threadIdx.y;  /* row within tile,    0..TILE_M-1 */

    int row = blockIdx.y * TILE_M + ty;  /* global output row */
    int col = blockIdx.x * TILE_N + tx;  /* global output col */

    /* Accumulator in registers */
    float acc[8];
    #pragma unroll
    for (int c = 0; c < 8; c++) acc[c] = 0.0f;

    /* Shared memory tiles */
    __shared__ float sA[TILE_M][TILE_K][8];
    __shared__ float sB[TILE_K][TILE_N][8];

    /* Iterate over K in tiles */
    int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * TILE_K;

        /* Load A tile: sA[ty][tx_k] for tx_k = tx (reusing tx as k-index loader) */
        /* Each thread loads one element if within bounds */
        /* We need TILE_M * TILE_K loads but have TILE_M * TILE_N threads.
         * Since TILE_K == TILE_N, each thread loads exactly one element. */
        {
            int k_idx = k_base + tx;
            if (row < M && k_idx < K) {
                int a_off = (row * K + k_idx) * 8;
                #pragma unroll
                for (int c = 0; c < 8; c++) sA[ty][tx][c] = a[a_off + c];
            } else {
                #pragma unroll
                for (int c = 0; c < 8; c++) sA[ty][tx][c] = 0.0f;
            }
        }

        /* Load B tile: sB[ty_k][tx] */
        {
            int k_idx = k_base + ty;
            if (k_idx < K && col < N) {
                int b_off = (k_idx * N + col) * 8;
                #pragma unroll
                for (int c = 0; c < 8; c++) sB[ty][tx][c] = b[b_off + c];
            } else {
                #pragma unroll
                for (int c = 0; c < 8; c++) sB[ty][tx][c] = 0.0f;
            }
        }

        __syncthreads();

        /* Accumulate: acc += sum over k in tile of sA[ty][k] * sB[k][tx] */
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            cl3_gp_acc_dev(sA[ty][kk], sB[kk][tx], acc);
        }

        __syncthreads();
    }

    /* Write output */
    if (row < M && col < N) {
        int out_off = (row * N + col) * 8;
        #pragma unroll
        for (int c = 0; c < 8; c++) out[out_off + c] = acc[c];
    }
}

/* ================================================================== */
/*  Launch wrappers (C linkage)                                       */
/* ================================================================== */

extern "C" {

void cl3_batch_gp_launch(
    const float *d_a, const float *d_b, float *d_out, int N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_batch_gp_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_a, d_b, d_out, N);
}

void cl3_batch_sandwich_launch(
    const float *d_r, const float *d_x, float *d_out, int N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_batch_sandwich_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_r, d_x, d_out, N);
}

void cl3_batch_bvexp_launch(
    const float *d_bv, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_batch_bvexp_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_bv, d_out, N);
}

void cl3_batch_grade_proj_launch(
    const float *d_in, float *d_out, int grade, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_batch_grade_proj_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_in, d_out, grade, N);
}

void cl3_batch_norm_launch(
    const float *d_in, float *d_out, int N, cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_batch_norm_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_in, d_out, N);
}

void cl3_geom_matmul_launch(
    const float *d_a,
    const float *d_b,
    float       *d_out,
    int M, int K, int N,
    cudaStream_t stream)
{
    dim3 block(TILE_N, TILE_M);   /* 16x16 = 256 threads */
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    cl3_geom_matmul_kernel<<<grid, block, 0, stream>>>(
        d_a, d_b, d_out, M, K, N);
}

} /* extern "C" */
