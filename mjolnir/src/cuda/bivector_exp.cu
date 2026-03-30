/**
 * bivector_exp.cu -- Bivector Exponential and Rotor Logarithm
 *
 * YGGDRASIL Clifford Algebra Stack -- mjolnir L0
 *
 * For Cl(3,0) with positive-definite signature, a pure bivector B
 * squares to a non-positive scalar:
 *   B^2 = -(b12^2 + b13^2 + b23^2) = -|B|^2
 *
 * Therefore:
 *   exp(B) = cos(|B|) + sin(|B|)/|B| * B
 *
 * This gives a unit rotor (even multivector with unit norm).
 *
 * The inverse (logarithm of a unit rotor R = s + B):
 *   log(R) = atan2(|B|, s) / |B| * B
 *
 * Basis ordering: {1, e1, e2, e3, e12, e13, e23, e123}
 * Bivector components are at indices 4, 5, 6.
 */

#include "../../include/cl3_types.h"
#include "../../include/cl3_tables.h"
#include "../../include/cl3_ops.h"

#include <cuda_runtime.h>
#include <math.h>

#define CL3_BLOCK_SIZE 256

/* Threshold below which we use Taylor expansions for numerical stability */
#define CL3_BIVEC_EPS 1.0e-7f

/* ================================================================== */
/*  Bivector exponential kernel                                       */
/* ================================================================== */

/**
 * Input: flat multivectors where only components [4],[5],[6] are used.
 *        (other components are ignored)
 * Output: full even multivector (rotor): components [0],[4],[5],[6]
 *         with [1],[2],[3],[7] = 0.
 */
__global__
void cl3_bivector_exp_kernel(
    const float *__restrict__ bv,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    float b12 = bv[off + 4];
    float b13 = bv[off + 5];
    float b23 = bv[off + 6];

    float mag_sq = b12 * b12 + b13 * b13 + b23 * b23;
    float mag    = sqrtf(mag_sq);

    float cos_val, sinc_val;

    if (mag > CL3_BIVEC_EPS) {
        cos_val  = cosf(mag);
        sinc_val = sinf(mag) / mag;
    } else {
        /* Taylor expansion for small magnitude:
         * cos(x)    ~ 1 - x^2/2 + x^4/24
         * sin(x)/x  ~ 1 - x^2/6 + x^4/120 */
        cos_val  = 1.0f - mag_sq * (0.5f - mag_sq * (1.0f / 24.0f));
        sinc_val = 1.0f - mag_sq * (1.0f / 6.0f - mag_sq * (1.0f / 120.0f));
    }

    /* exp(B) = cos(|B|) + sin(|B|)/|B| * B */
    out[off + 0] = cos_val;
    out[off + 1] = 0.0f;
    out[off + 2] = 0.0f;
    out[off + 3] = 0.0f;
    out[off + 4] = sinc_val * b12;
    out[off + 5] = sinc_val * b13;
    out[off + 6] = sinc_val * b23;
    out[off + 7] = 0.0f;
}

/* ================================================================== */
/*  Rotor logarithm kernel                                            */
/* ================================================================== */

/**
 * Input:  rotor R = s + b12*e12 + b13*e13 + b23*e23  (even MV)
 *         stored as flat 8-component MV, uses [0],[4],[5],[6]
 * Output: pure bivector B such that exp(B) = R
 *         stored in [4],[5],[6]; all others zero
 *
 * log(R) = atan2(|B|, s) / |B| * B
 *
 * For near-zero bivector part (R ~ +/-1), we handle gracefully.
 */
__global__
void cl3_rotor_log_kernel(
    const float *__restrict__ rotor,
    float       *__restrict__ out,
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    int off = tid * 8;

    float s   = rotor[off + 0];
    float b12 = rotor[off + 4];
    float b13 = rotor[off + 5];
    float b23 = rotor[off + 6];

    float biv_mag_sq = b12 * b12 + b13 * b13 + b23 * b23;
    float biv_mag    = sqrtf(biv_mag_sq);

    float scale;

    if (biv_mag > CL3_BIVEC_EPS) {
        float angle = atan2f(biv_mag, s);
        scale = angle / biv_mag;
    } else {
        /* Taylor expansion of atan2(x, s)/x around x=0:
         * atan2(x,s)/x ~ 1/s - x^2/(3*s^3) for small x, s > 0
         *
         * But we also need to handle s < 0 (rotation by ~pi).
         * For s > 0: angle ~ biv_mag/s, so scale ~ 1/s
         * For s < 0: angle ~ pi - biv_mag/|s|, scale ~ (pi - biv_mag/|s|)/biv_mag
         *   which diverges -- but that means the bivector part IS near zero
         *   and the rotor is close to -1 (rotation by pi with indeterminate axis).
         */
        if (s > 0.0f) {
            /* angle ~ biv_mag / s, so scale ~ 1/s */
            float inv_s = 1.0f / s;
            scale = inv_s * (1.0f - biv_mag_sq * inv_s * inv_s * (1.0f / 3.0f));
        } else if (s < -CL3_BIVEC_EPS) {
            /* R ~ -1.  This is a pi rotation.  The axis is indeterminate
             * when the bivector part is exactly zero.  Return pi along
             * the bivector direction if any, else default to e12. */
            float angle = atan2f(biv_mag, s);  /* ~ pi */
            if (biv_mag > 1.0e-12f) {
                scale = angle / biv_mag;
            } else {
                /* Truly degenerate: return pi * e12 as convention */
                out[off + 0] = 0.0f;
                out[off + 1] = 0.0f;
                out[off + 2] = 0.0f;
                out[off + 3] = 0.0f;
                out[off + 4] = 3.14159265358979323846f;
                out[off + 5] = 0.0f;
                out[off + 6] = 0.0f;
                out[off + 7] = 0.0f;
                return;
            }
        } else {
            /* s ~ 0 and biv_mag ~ 0: degenerate, return zero */
            out[off + 0] = 0.0f; out[off + 1] = 0.0f;
            out[off + 2] = 0.0f; out[off + 3] = 0.0f;
            out[off + 4] = 0.0f; out[off + 5] = 0.0f;
            out[off + 6] = 0.0f; out[off + 7] = 0.0f;
            return;
        }
    }

    out[off + 0] = 0.0f;
    out[off + 1] = 0.0f;
    out[off + 2] = 0.0f;
    out[off + 3] = 0.0f;
    out[off + 4] = scale * b12;
    out[off + 5] = scale * b13;
    out[off + 6] = scale * b23;
    out[off + 7] = 0.0f;
}

/* ================================================================== */
/*  Stratified layout variants                                        */
/* ================================================================== */

__global__
void cl3_bivector_exp_stratified_kernel(
    const float *__restrict__ bv_bivectors,   /* [N*3] */
    float       *__restrict__ out_scalars,    /* [N]   */
    float       *__restrict__ out_vectors,    /* [N*3] */
    float       *__restrict__ out_bivectors,  /* [N*3] */
    float       *__restrict__ out_trivectors, /* [N]   */
    int          N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float b12 = bv_bivectors[tid * 3 + 0];
    float b13 = bv_bivectors[tid * 3 + 1];
    float b23 = bv_bivectors[tid * 3 + 2];

    float mag_sq = b12 * b12 + b13 * b13 + b23 * b23;
    float mag    = sqrtf(mag_sq);

    float cos_val, sinc_val;

    if (mag > CL3_BIVEC_EPS) {
        cos_val  = cosf(mag);
        sinc_val = sinf(mag) / mag;
    } else {
        cos_val  = 1.0f - mag_sq * (0.5f - mag_sq * (1.0f / 24.0f));
        sinc_val = 1.0f - mag_sq * (1.0f / 6.0f - mag_sq * (1.0f / 120.0f));
    }

    out_scalars[tid]          = cos_val;
    out_vectors[tid * 3 + 0]  = 0.0f;
    out_vectors[tid * 3 + 1]  = 0.0f;
    out_vectors[tid * 3 + 2]  = 0.0f;
    out_bivectors[tid * 3 + 0] = sinc_val * b12;
    out_bivectors[tid * 3 + 1] = sinc_val * b13;
    out_bivectors[tid * 3 + 2] = sinc_val * b23;
    out_trivectors[tid]        = 0.0f;
}

/* ================================================================== */
/*  Launch wrappers (C linkage)                                       */
/* ================================================================== */

extern "C" {

void cl3_bivector_exp_flat_launch(
    const float *d_bv,
    float       *d_out,
    int          N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_bivector_exp_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_bv, d_out, N);
}

void cl3_bivector_exp_stratified_launch(
    const Cl3Batch *d_bv,
    Cl3Batch       *d_out,
    int             N,
    cudaStream_t    stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_bivector_exp_stratified_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_bv->bivectors,
        d_out->scalars, d_out->vectors, d_out->bivectors, d_out->trivectors,
        N);
}

void cl3_rotor_log_flat_launch(
    const float *d_rotor,
    float       *d_out,
    int          N,
    cudaStream_t stream)
{
    int grid = (N + CL3_BLOCK_SIZE - 1) / CL3_BLOCK_SIZE;
    cl3_rotor_log_kernel<<<grid, CL3_BLOCK_SIZE, 0, stream>>>(
        d_rotor, d_out, N);
}

} /* extern "C" */
