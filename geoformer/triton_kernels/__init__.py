"""HLM Triton Kernel Stack — H100 Optimized.

Custom CUDA kernels for Holographic Language Models, written in Triton.
Fuses the Clifford algebra geometric product, SwiGLU FFN, and MoE expert
dispatch into minimal kernel launches.

Kernels:
  1. hlm_geo_round: Fused geometric product + SwiGLU (6 launches → 1)
  2. hlm_expert_gemm: Grouped expert GEMM (8 launches → 1)
  3. hlm_moe_dispatch: Route + sort + scatter (16 launches → 3)
"""
