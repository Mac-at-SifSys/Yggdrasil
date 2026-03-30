# Yggdrasil

**A complete Clifford algebra machine learning stack — from CUDA kernels to serving layer.**

Yggdrasil is a ground-up ML training and inference framework where every operation — attention, autodiff, optimization, memory — runs natively in Clifford algebra Cl(3,0). This is not a wrapper around PyTorch or TensorFlow. Every layer was built from scratch on different math.

46,000 lines. Seven modules. One algebra.

\---

## Why Clifford Algebra?

Standard neural networks operate on flat tensors — arrays of numbers with no geometric meaning. Clifford algebra gives every value a geometric identity: scalars, vectors, bivectors (oriented planes), and trivectors (oriented volumes). Operations like the geometric product simultaneously encode rotation, scaling, and projection in a single multiply.

This matters for ML because:

* **Attention scores** become rotational similarity measurements (`grade\_0(Q \* \~K)`) instead of arbitrary dot products
* **Memory retrieval** uses the same algebraic operation as relevance scoring — no separate similarity function needed
* **Gradients** flow through algebraically correct derivative rules (`d(AB)/dA = grad \* \~B`), not generic chain rule approximations
* **Optimization** adapts per-grade, because scalar and bivector components have different learning dynamics
* **Dead computation is detectable at compile time** — if downstream consumers only need grade-0, the compiler can prune grades 1-3 from upstream producers

\---

## Architecture

Yggdrasil is named for the world tree of Norse mythology. The naming is both structural and decorative — modules map to their role in the cosmology.

```
                        ┌─────────────┐
                        │   Bifrost    │  Serving, export, quantization
                        │   (Bridge)   │  → The bridge to the outside world
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       ┌──────┴──────┐  ┌─────┴─────┐  ┌───────┴───────┐
       │  Holograph   │  │ Geoformer │  │     Forge     │
       │  (Models)    │  │ (Variants)│  │  (Training)   │
       │              │  │           │  │               │
       │ HLM, HLM     │  │ MoE-HLM,  │  │ CliffordAdam, │
       │ attention,   │  │ Triton    │  │ schedulers,   │
       │ memory bank  │  │ kernels   │  │ distributed   │
       └──────┬──────┘  └─────┬─────┘  └───────┬───────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                        ┌──────┴──────┐
                        │    Rune     │  Framework layer
                        │  (Language) │  → Types, autodiff, compiler, codegen
                        └──────┬──────┘
                               │
                  ┌────────────┼────────────┐
                  │                         │
           ┌──────┴──────┐          ┌───────┴───────┐
           │   Mjolnir   │          │ Runtime Core  │
           │  (Hammer)   │          │  (Trunk)      │
           │             │          │               │
           │ C/CUDA      │          │ C++ graph,    │
           │ Cl(3,0)     │          │ scheduler,    │
           │ kernels     │          │ memory pool,  │
           │             │          │ tape autodiff │
           └─────────────┘          └───────────────┘
```

\---

## Modules

### Mjolnir — Algebra Kernels

`mjolnir/`

C and CUDA implementations of Cl(3,0) operations. The foundation everything else is built on.

* Fully unrolled 8×8 geometric product (64 fused multiply-adds, no loops, no runtime table lookups)
* Sandwich product, bivector exponentiation, grade projection, batch operations
* Both flat (8 contiguous floats) and grade-stratified memory layouts
* Persistent cooperative CUDA kernel — runs an entire forward-backward-update cycle without returning to host, using a 47-opcode command buffer

### Rune — Differentiable Programming Framework

`rune/`

The framework layer. Rune is to Yggdrasil what PyTorch is to standard ML.

* **Type system**: Multivector and GradedTensor types with grade-aware shape inference
* **Operations**: Geometric product, reverse, grade projection, sandwich, bivector exp, norms — all with forward and backward implementations
* **Autodiff**: Clifford-specific derivative rules. Not generic autograd — uses the algebraic structure of Cl(3,0) to compute gradients analytically
* **Compiler**: Traces Python code into a typed IR with grade annotations, then runs optimization passes:

  * Grade pruning — backward dataflow analysis eliminates dead grade computation
  * Operator fusion — combines sequential Clifford ops into single kernel launches
  * Memory planning — allocates and reuses buffers across the execution graph
* **Codegen**: Lowers optimized IR to Mjolnir kernel launches
* **Bindings**: FFI bridge between Python and Mjolnir's C/CUDA kernels

### Holograph — Model Architectures

`holograph/`

HLM (Holographic Language Model) architecture and components.

* **CliffordAttention**: Multi-head attention where scores are computed via the scalar part of the geometric product with reverse — `grade\_0(Q \* \~K)` — measuring rotational similarity rather than arbitrary dot products
* **CliffordLinear**: Linear layers operating on multivectors, preserving algebraic structure through the transformation
* **CliffordRouter**: Routes inputs through grade-specific pathways
* **HolographicMemoryBank**: Long-range algebraic memory. 1M slots × 8 floats = 32 MB. Stores distilled multivector summaries of processed chunks. Temporal position encoded as rotors via bivector exponentiation. Retrieval, relevance scoring, and relationship computation happen in a single geometric product
* **DensityField**: Continuous density representation over the algebra
* **GradeDecay**: Energy-based memory eviction using per-grade decay rates

### Geoformer — Experimental Models \& Kernels

`geoformer/`

Model variants, training scripts, and performance-critical Triton kernels.

* HLM v9 (tokenized, raw, LoRA, SFT variants)
* MoE-HLM (Mixture of Experts with holographic expert modules and Clifford routing)
* Triton kernels: fused expert dispatch, expert GEMM, geometric rounding
* Cloud training configurations for Colab and Kaggle
* Chat inference with CPU and GPU paths
* Checkpoint evaluation and introspection tools

### Forge — Training Infrastructure

`forge/`

Everything needed to train Clifford-native models.

* **CliffordAdam**: Adam optimizer using the geometric self-product (`g \* \~g`) for second moments instead of element-wise squaring. The adaptive denominator respects algebraic magnitude. Per-grade learning rate scaling dampens noisy components (trivector default: 0.5×)
* **CliffordSGD**: SGD with grade-aware momentum
* **Algebraic consistency loss**: Penalizes outputs that violate Clifford algebraic identities
* **Grade entropy loss**: Encourages balanced grade utilization across the model
* **Distributed training**: Multi-GPU support with gradient synchronization
* **Checkpointing**: Save/restore model state, optimizer state, and training metadata

### Bifrost — Serving \& Deployment

`bifrost/`

The bridge from trained models to production.

* **CliffordKVCache**: KV cache adapted for multivector-valued keys and values
* **Quantization**: Grade-aware quantization (different precision per grade), mixed precision calibration
* **Export**: ONNX and TensorRT export paths
* **Serving**: FastAPI server with an OpenAI-compatible API endpoint

### Runtime Core — Execution Engine

`runtime\_core/`

C++ runtime providing the execution substrate.

* **Graph**: Operation graph construction and topological scheduling
* **Scheduler**: Dependency-aware operation dispatch
* **MemoryPool**: Arena-based GPU memory allocation with buffer reuse
* **Tape**: Tape-based autodiff with Cl(3,0) arithmetic (reverse, grade involution, grade projection)
* **Device**: CPU and CUDA device abstraction
* **Fusion**: Runtime operator fusion for sequential operations

\---

## Basis Ordering

All modules use the same convention throughout:

```
Index:  0    1    2    3    4     5     6     7
Basis:  1    e₁   e₂   e₃   e₁₂   e₁₃   e₂₃   e₁₂₃
Grade:  0    1    1    1    2     2     2     3
```

\---

## Status

Yggdrasil is research software. It has been used to train small Clifford-native language models up to a few hundred million parameters and powers the reasoning enrichment layer at [AINW.ai](https://ainw.ai). The architecture is stable and the algebra is verified, but this is not yet production-hardened at scale.

**What works:**

* Full forward and backward passes through all Clifford operations
* Training loop with CliffordAdam on GPU
* Persistent CUDA engine for single-kernel training steps
* Checkpoint save/restore
* Inference with chat interface
* Algebraic correctness verified by numerical gradient tests

**What's next:**

* Multi-node distributed training
* Larger model training runs (current constraint is compute, not architecture)
* Metal backend for Apple Silicon (directory exists, kernels not yet ported)
* Comprehensive benchmarking against standard transformer baselines at equivalent parameter counts

\---

## Requirements

* Python 3.10+
* NumPy (CPU path)
* CuPy (GPU path, optional)
* CUDA Toolkit 12.0+ (for Mjolnir CUDA kernels)
* CMake 3.20+ (for building Runtime Core and Mjolnir)
* A C++17 compiler (GCC 11+, MSVC 2022+, Clang 14+)

Optional:

* Triton (for Geoformer Triton kernels)
* FastAPI + Uvicorn (for Bifrost serving)
* ONNX Runtime / TensorRT (for export paths)

\---

## Building

```bash
# Build Mjolnir (CUDA kernels)
cd mjolnir
mkdir build \&\& cd build
cmake .. -DCMAKE\_BUILD\_TYPE=Release
make -j$(nproc)

# Build Runtime Core
cd ../../runtime\_core
mkdir build \&\& cd build
cmake .. -DCMAKE\_BUILD\_TYPE=Release
make -j$(nproc)

# Install Python packages
cd ../..
pip install -e .
```

\---

## Quick Start

```python
from rune.types.multivector import Multivector
from rune.ops.geometric\_product import geometric\_product
from holograph.models.hlm import HLM
from holograph.models.hlm\_config import HLMConfig

# Create two multivectors in Cl(3,0)
a = Multivector.from\_components(scalar=1.0, vector=\[0.5, 0.3, 0.1])
b = Multivector.from\_components(scalar=0.0, bivector=\[0.7, 0.2, 0.4])

# Geometric product
c = geometric\_product(a, b)

# Build a small HLM
config = HLMConfig(
    d\_model=64,
    n\_heads=4,
    n\_layers=6,
    vocab\_size=256,
)
model = HLM(config)
```

\---

## How This Was Built

Yggdrasil was built by one person — a health systems researcher in rural Arkansas who was teaching ML to grad students before ChatGPT existed — using [Claude Code](https://claude.ai) as a full engineering collaborator. Opus for architecture and theory decisions, Code for implementation, and occasional Codex searching for bugs and loose ends. The collaboration produced this stack over several months, from the first geometric product kernel to the serving layer.

The experience was less like using a code generation tool and more like pair programming with someone who could hold the full Clifford algebra in context while I held the vision for what the stack needed to become. The persistent CUDA engine, the grade pruning compiler pass, the algebraic autodiff rules — these required genuine back-and-forth between human intuition about what should exist and AI execution of the math and systems engineering to make it real, and to keep the AI coders from habitually reaching for standard stack tools over and over and over and over again.

\---

## Related

* [AINW.ai](https://ainw.ai) — The product built on this stack. Enriches any LLM's output using Clifford algebra reasoning primitives.
* [WZ1](https://ainw.ai) — a 1m parameter Clifford router for The Human Periodic Table. 1,400+ reasoning kernels derived from 5,000 years of human knowledge, from mathematics to world mythology.

\---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

\---

## Citation

If you use Yggdrasil in research, please cite:

```bibtex
@software{yggdrasil2026,
  author = {McNew, Mac},
  title = {Yggdrasil: A Clifford Algebra Machine Learning Stack},
  year = {2026},
  url = {https://github.com/SifSystems/yggdrasil}
}
```

\---

*"I know that I hung on that wind-battered tree, wounded by my own blade, nine long nights."
— Hávamál, Stanza 138*

