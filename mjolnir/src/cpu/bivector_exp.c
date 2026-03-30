/**
 * bivector_exp.c — Exponential of a bivector and logarithm of a rotor
 *
 * For Cl(3,0), the exponential of a pure bivector B is:
 *   exp(B) = cos(|B|) + sin(|B|)/|B| * B
 *
 * where |B| = sqrt(b12^2 + b13^2 + b23^2) is the bivector magnitude.
 * The result is always an even multivector (rotor): scalar + bivector.
 *
 * Note: In Cl(3,0), bivectors square to negative scalars (like quaternions),
 * so B^2 = -(b12^2 + b13^2 + b23^2) which gives the cos/sin form.
 */

#include "cl3_types.h"
#include "cl3_ops.h"
#include <math.h>

Cl3Multivector cl3_bivector_exp(const Cl3Multivector *bv) {
    /* Extract bivector components */
    cl3_scalar b12 = bv->b[0];
    cl3_scalar b13 = bv->b[1];
    cl3_scalar b23 = bv->b[2];

    /* Bivector magnitude squared */
    cl3_scalar mag_sq = b12 * b12 + b13 * b13 + b23 * b23;
    cl3_scalar mag = sqrtf(mag_sq);

    Cl3Multivector r = cl3_zero();

    if (mag < 1e-12f) {
        /* Small angle: exp(B) ≈ 1 + B (first order Taylor) */
        r.s = 1.0f;
        r.b[0] = b12;
        r.b[1] = b13;
        r.b[2] = b23;
    } else {
        cl3_scalar c = cosf(mag);
        cl3_scalar sinc = sinf(mag) / mag;  /* sin(x)/x */

        r.s = c;
        r.b[0] = sinc * b12;
        r.b[1] = sinc * b13;
        r.b[2] = sinc * b23;
    }

    return r;
}

Cl3Multivector cl3_rotor_log(const Cl3Multivector *rotor) {
    /* Extract scalar and bivector parts of the rotor */
    cl3_scalar s = rotor->s;
    cl3_scalar b12 = rotor->b[0];
    cl3_scalar b13 = rotor->b[1];
    cl3_scalar b23 = rotor->b[2];

    /* Bivector magnitude of the rotor */
    cl3_scalar bv_mag = sqrtf(b12 * b12 + b13 * b13 + b23 * b23);

    Cl3Multivector r = cl3_zero();

    if (bv_mag < 1e-12f) {
        /* Nearly pure scalar rotor, log ≈ 0 */
        return r;
    }

    /* angle = atan2(|bv|, scalar) */
    cl3_scalar angle = atan2f(bv_mag, s);
    cl3_scalar scale = angle / bv_mag;

    r.b[0] = scale * b12;
    r.b[1] = scale * b13;
    r.b[2] = scale * b23;

    return r;
}
