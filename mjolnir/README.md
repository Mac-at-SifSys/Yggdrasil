# Mjolnir — L0 Compute Kernels

Bare-metal GPU and CPU kernels for Cl(3,0) Clifford algebra operations.

## Build

```bash
mkdir build && cd build
cmake ..
make
ctest --output-on-failure
```

## Operations

- **Geometric Product**: Two 8-vectors → one 8-vector (64 multiply-adds)
- **Grade Projection**: Extract grade-k components (zero-cost with grade-stratified layout)
- **Reverse / Conjugate / Norm**: Bitmask operations on grade structure
- **Sandwich Product**: R * x * ~R (fused two-product kernel)
- **Bivector Exponential**: exp(B) = cos(|B|) + sin(|B|)/|B| * B
- **Batched Operations**: All above, operating on batches

## Memory Layout

Grade-stratified, NOT flat 8-element arrays:
- Grade 0 (scalar): float[N]
- Grade 1 (vectors): float[N*3]
- Grade 2 (bivectors): float[N*3]
- Grade 3 (trivector): float[N]
