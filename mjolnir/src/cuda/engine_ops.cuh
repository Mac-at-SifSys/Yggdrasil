#pragma once
/**
 * engine_ops.cuh — Device functions for the persistent CUDA training engine.
 *
 * All Cl(3,0) operations as __device__ __forceinline__ functions.
 * Called from the persistent kernel (persistent_engine.cu).
 *
 * Basis ordering: {1, e1, e2, e3, e12, e13, e23, e123}
 *                  0   1   2   3   4    5    6    7
 *
 * The geometric product is fully unrolled and verified against
 * cl3_tables.h (the single source of truth).
 */

#include <math.h>

// ===== Cl(3,0) Multiplication Table (compile-time constants) =====
// These mirror cl3_tables.h exactly.

__constant__ int CL3_PROD_IDX[8][8] = {
    {0,1,2,3,4,5,6,7}, {1,0,4,5,2,3,7,6}, {2,4,0,6,1,7,3,5}, {3,5,6,0,7,1,2,4},
    {4,2,1,7,0,6,5,3}, {5,3,7,1,6,0,4,2}, {6,7,3,2,5,4,0,1}, {7,6,5,4,3,2,1,0}
};
__constant__ float CL3_PROD_SIGN[8][8] = {
    {1,1,1,1,1,1,1,1}, {1,1,1,1,1,1,1,1}, {1,-1,1,1,-1,-1,1,-1}, {1,-1,-1,1,1,-1,-1,1},
    {1,-1,1,1,-1,-1,1,-1}, {1,-1,-1,1,1,-1,-1,1}, {1,1,-1,1,-1,1,-1,-1}, {1,1,-1,1,-1,1,-1,-1}
};
__constant__ float CL3_REVERSE_SIGN[8] = {1,1,1,1,-1,-1,-1,-1};

// Grade slices: grade 0 = [0:1], grade 1 = [1:4], grade 2 = [4:7], grade 3 = [7:8]
__constant__ int GRADE_START[4] = {0, 1, 4, 7};
__constant__ int GRADE_END[4]   = {1, 4, 7, 8};

// ===== Core device functions =====

/**
 * Full Cl(3,0) geometric product, fully unrolled.
 *
 * Verified against cl3_tables.h row-by-row:
 *   Row i: for each j, c[IDX[i][j]] += SIGN[i][j] * a[i] * b[j]
 *
 * This matches the verified cl3_gp_device() from geometric_product.cu exactly.
 */
__device__ __forceinline__ void dev_gp(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ c) {
    c[0] = 0.0f; c[1] = 0.0f; c[2] = 0.0f; c[3] = 0.0f;
    c[4] = 0.0f; c[5] = 0.0f; c[6] = 0.0f; c[7] = 0.0f;

    // Row 0: basis = 1
    // 1*1=+1, 1*e1=+e1, 1*e2=+e2, 1*e3=+e3, 1*e12=+e12, 1*e13=+e13, 1*e23=+e23, 1*e123=+e123
    float a0 = a[0];
    c[0] += a0 * b[0];
    c[1] += a0 * b[1];
    c[2] += a0 * b[2];
    c[3] += a0 * b[3];
    c[4] += a0 * b[4];
    c[5] += a0 * b[5];
    c[6] += a0 * b[6];
    c[7] += a0 * b[7];

    // Row 1: basis = e1
    // e1*1=+e1, e1*e1=+1, e1*e2=+e12, e1*e3=+e13, e1*e12=+e2, e1*e13=+e3, e1*e23=+e123, e1*e123=+e23
    float a1 = a[1];
    c[1] += a1 * b[0];
    c[0] += a1 * b[1];
    c[4] += a1 * b[2];
    c[5] += a1 * b[3];
    c[2] += a1 * b[4];
    c[3] += a1 * b[5];
    c[7] += a1 * b[6];
    c[6] += a1 * b[7];

    // Row 2: basis = e2
    // e2*1=+e2, e2*e1=-e12, e2*e2=+1, e2*e3=+e23, e2*e12=-e1, e2*e13=-e123, e2*e23=+e3, e2*e123=-e13
    float a2 = a[2];
    c[2] += a2 * b[0];
    c[4] -= a2 * b[1];
    c[0] += a2 * b[2];
    c[6] += a2 * b[3];
    c[1] -= a2 * b[4];
    c[7] -= a2 * b[5];
    c[3] += a2 * b[6];
    c[5] -= a2 * b[7];

    // Row 3: basis = e3
    // e3*1=+e3, e3*e1=-e13, e3*e2=-e23, e3*e3=+1, e3*e12=+e123, e3*e13=-e1, e3*e23=-e2, e3*e123=+e12
    float a3 = a[3];
    c[3] += a3 * b[0];
    c[5] -= a3 * b[1];
    c[6] -= a3 * b[2];
    c[0] += a3 * b[3];
    c[7] += a3 * b[4];
    c[1] -= a3 * b[5];
    c[2] -= a3 * b[6];
    c[4] += a3 * b[7];

    // Row 4: basis = e12
    // e12*1=+e12, e12*e1=-e2, e12*e2=+e1, e12*e3=+e123, e12*e12=-1, e12*e13=-e23, e12*e23=+e13, e12*e123=-e3
    float a4 = a[4];
    c[4] += a4 * b[0];
    c[2] -= a4 * b[1];
    c[1] += a4 * b[2];
    c[7] += a4 * b[3];
    c[0] -= a4 * b[4];
    c[6] -= a4 * b[5];
    c[5] += a4 * b[6];
    c[3] -= a4 * b[7];

    // Row 5: basis = e13
    // e13*1=+e13, e13*e1=-e3, e13*e2=-e123, e13*e3=+e1, e13*e12=+e23, e13*e13=-1, e13*e23=-e12, e13*e123=+e2
    float a5 = a[5];
    c[5] += a5 * b[0];
    c[3] -= a5 * b[1];
    c[7] -= a5 * b[2];
    c[1] += a5 * b[3];
    c[6] += a5 * b[4];
    c[0] -= a5 * b[5];
    c[4] -= a5 * b[6];
    c[2] += a5 * b[7];

    // Row 6: basis = e23
    // e23*1=+e23, e23*e1=+e123, e23*e2=-e3, e23*e3=+e2, e23*e12=-e13, e23*e13=+e12, e23*e23=-1, e23*e123=-e1
    float a6 = a[6];
    c[6] += a6 * b[0];
    c[7] += a6 * b[1];
    c[3] -= a6 * b[2];
    c[2] += a6 * b[3];
    c[5] -= a6 * b[4];
    c[4] += a6 * b[5];
    c[0] -= a6 * b[6];
    c[1] -= a6 * b[7];

    // Row 7: basis = e123
    // e123*1=+e123, e123*e1=+e23, e123*e2=-e13, e123*e3=+e12, e123*e12=-e3, e123*e13=+e2, e123*e23=-e1, e123*e123=-1
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

__device__ __forceinline__ void dev_reverse(const float* __restrict__ a,
                                             float* __restrict__ out) {
    out[0] =  a[0]; out[1] =  a[1]; out[2] =  a[2]; out[3] =  a[3];
    out[4] = -a[4]; out[5] = -a[5]; out[6] = -a[6]; out[7] = -a[7];
}

__device__ __forceinline__ void dev_sandwich(const float* __restrict__ r,
                                              const float* __restrict__ x,
                                              float* __restrict__ out) {
    float tmp[8], r_rev[8];
    dev_gp(r, x, tmp);
    dev_reverse(r, r_rev);
    dev_gp(tmp, r_rev, out);
}

__device__ __forceinline__ void dev_add(const float* a, const float* b, float* out) {
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = a[i] + b[i];
}

__device__ __forceinline__ void dev_scale(const float* a, float s, float* out) {
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = a[i] * s;
}

__device__ __forceinline__ void dev_zero(float* out) {
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = 0.0f;
}

__device__ __forceinline__ void dev_copy(const float* src, float* dst) {
    #pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = src[i];
}

__device__ __forceinline__ float dev_norm_sq(const float* a) {
    // norm_sq = scalar_part(a * ~a)
    // For Cl(3,0): scalar part of a * reverse(a)
    float r[8];
    dev_reverse(a, r);
    float result = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            if (CL3_PROD_IDX[i][j] == 0)
                result += CL3_PROD_SIGN[i][j] * a[i] * r[j];
    return result;
}

__device__ __forceinline__ float dev_scalar_product(const float* a, const float* b) {
    // grade-0 part of GP(a, b)
    float result = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            if (CL3_PROD_IDX[i][j] == 0)
                result += CL3_PROD_SIGN[i][j] * a[i] * b[j];
    return result;
}

__device__ __forceinline__ void dev_bivector_exp(const float* bv, float* out) {
    // exp(B) for pure bivector B at indices [4],[5],[6]
    float mag_sq = bv[4]*bv[4] + bv[5]*bv[5] + bv[6]*bv[6];
    float mag = sqrtf(mag_sq + 1e-12f);
    float cos_m = cosf(mag);
    float sinc_m = (mag > 1e-7f) ? sinf(mag) / mag : 1.0f;
    dev_zero(out);
    out[0] = cos_m;
    out[4] = sinc_m * bv[4];
    out[5] = sinc_m * bv[5];
    out[6] = sinc_m * bv[6];
}

__device__ __forceinline__ void dev_bivector_exp_backward(
    const float* grad_output,
    const float* bv,
    float* grad_bv
) {
    float b0 = bv[4];
    float b1 = bv[5];
    float b2 = bv[6];
    float mag_sq = b0*b0 + b1*b1 + b2*b2;
    float mag = sqrtf(mag_sq + 1e-24f);
    float cos_mag = cosf(mag);
    float sin_mag = sinf(mag);
    float sinc = (mag > 1e-12f) ? (sin_mag / mag) : 1.0f;
    float d_sinc = (mag > 1e-12f) ? ((cos_mag - sinc) / mag) : 0.0f;
    float inv_mag = 1.0f / (mag + 1e-24f);

    dev_zero(grad_bv);

    const float grad_scalar = grad_output[0];
    grad_bv[4] += grad_scalar * (-sin_mag) * b0 * inv_mag;
    grad_bv[5] += grad_scalar * (-sin_mag) * b1 * inv_mag;
    grad_bv[6] += grad_scalar * (-sin_mag) * b2 * inv_mag;

    const float grad_b0 = grad_output[4];
    const float grad_b1 = grad_output[5];
    const float grad_b2 = grad_output[6];

    grad_bv[4] += grad_b0 * (sinc + d_sinc * b0 * b0 * inv_mag);
    grad_bv[5] += grad_b0 * (d_sinc * b0 * b1 * inv_mag);
    grad_bv[6] += grad_b0 * (d_sinc * b0 * b2 * inv_mag);

    grad_bv[4] += grad_b1 * (d_sinc * b1 * b0 * inv_mag);
    grad_bv[5] += grad_b1 * (sinc + d_sinc * b1 * b1 * inv_mag);
    grad_bv[6] += grad_b1 * (d_sinc * b1 * b2 * inv_mag);

    grad_bv[4] += grad_b2 * (d_sinc * b2 * b0 * inv_mag);
    grad_bv[5] += grad_b2 * (d_sinc * b2 * b1 * inv_mag);
    grad_bv[6] += grad_b2 * (sinc + d_sinc * b2 * b2 * inv_mag);
}

// Fused: GP then take only grade-0 (scalar part) — avoids full 64 mads
__device__ __forceinline__ float dev_gp_grade0(const float* a, const float* b) {
    return dev_scalar_product(a, b);
}

// Fused: GP then grade projection — full GP then zero non-grade components
__device__ __forceinline__ void dev_gp_grade(const float* a, const float* b,
                                              float* out, int grade) {
    dev_gp(a, b, out);
    int start = GRADE_START[grade];
    int end = GRADE_END[grade];
    #pragma unroll
    for (int k = 0; k < 8; k++)
        if (k < start || k >= end)
            out[k] = 0.0f;
}

// ===== Activation functions =====

__device__ __forceinline__ float dev_gelu_scalar(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float dev_gelu_grad(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float tanh_val = tanhf(inner);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float inner_grad = 0.7978845608f * (1.0f + 0.134145f * x * x);
    return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * inner_grad;
}

__device__ __forceinline__ float dev_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ void dev_magnitude_gate_gelu(const float* in3, float* out3) {
    float mag = sqrtf(in3[0]*in3[0] + in3[1]*in3[1] + in3[2]*in3[2] + 1e-12f);
    float gate = dev_gelu_scalar(mag);
    float inv_mag = 1.0f / mag;
    out3[0] = in3[0] * inv_mag * gate;
    out3[1] = in3[1] * inv_mag * gate;
    out3[2] = in3[2] * inv_mag * gate;
}

__device__ __forceinline__ void dev_clifford_gelu(const float* in, float* out) {
    // Grade 0: GELU
    out[0] = dev_gelu_scalar(in[0]);
    // Grade 1: magnitude-gated GELU
    dev_magnitude_gate_gelu(in + 1, out + 1);
    // Grade 2: magnitude-gated GELU
    dev_magnitude_gate_gelu(in + 4, out + 4);
    // Grade 3: tanh
    out[7] = tanhf(in[7]);
}

__device__ __forceinline__ void dev_magnitude_gate_gelu_backward(
    const float* in3, const float* grad3, float* out3
) {
    float mag = sqrtf(in3[0]*in3[0] + in3[1]*in3[1] + in3[2]*in3[2] + 1e-12f);
    float inv_mag = 1.0f / mag;
    float direction0 = in3[0] * inv_mag;
    float direction1 = in3[1] * inv_mag;
    float direction2 = in3[2] * inv_mag;
    float gate = dev_gelu_scalar(mag);
    float gate_grad = dev_gelu_grad(mag);
    float dot_dg = direction0 * grad3[0] + direction1 * grad3[1] + direction2 * grad3[2];
    float coeff = gate * inv_mag;
    float proj_coeff = gate_grad - coeff;
    out3[0] = coeff * grad3[0] + proj_coeff * direction0 * dot_dg;
    out3[1] = coeff * grad3[1] + proj_coeff * direction1 * dot_dg;
    out3[2] = coeff * grad3[2] + proj_coeff * direction2 * dot_dg;
}

__device__ __forceinline__ void dev_clifford_gelu_backward(
    const float* in, const float* grad_out, float* grad_in
) {
    grad_in[0] = grad_out[0] * dev_gelu_grad(in[0]);
    dev_magnitude_gate_gelu_backward(in + 1, grad_out + 1, grad_in + 1);
    dev_magnitude_gate_gelu_backward(in + 4, grad_out + 4, grad_in + 4);
    float t = tanhf(in[7]);
    grad_in[7] = grad_out[7] * (1.0f - t * t);
}
