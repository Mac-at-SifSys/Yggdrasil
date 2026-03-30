/**
 * sandwich.c — Sandwich product R * x * ~R
 */

#include "cl3_types.h"
#include "cl3_ops.h"

Cl3Multivector cl3_sandwich(const Cl3Multivector *r, const Cl3Multivector *x) {
    /* Sandwich product: r * x * ~r
     * Two geometric products fused. For a future optimized kernel,
     * this can be done more efficiently, but the reference is correct. */
    Cl3Multivector rev_r = cl3_reverse(r);
    Cl3Multivector rx = cl3_geometric_product(r, x);
    return cl3_geometric_product(&rx, &rev_r);
}
