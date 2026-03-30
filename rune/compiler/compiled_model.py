"""
CompiledModel -- JIT-compiled HLM for maximum GPU throughput.

Usage:
    from rune.compiler import CompiledModel
    compiled = CompiledModel.from_model(model, batch_size=16, seq_len=2048)
    for step in range(total_steps):
        loss = compiled.train_step(input_ids, target_ids, lr=get_lr(step))

The compilation pipeline:
1. Trace the model to get an IR graph
2. Run optimization passes (grade pruning, fusion, memory planning)
3. Generate an execution plan (codegen)
4. The execution plan can be fed to the persistent CUDA engine
"""

from typing import Dict, List, Optional, Tuple
from rune.compiler.ir import IRGraph, OpCode
from rune.compiler.tracer import Tracer
from rune.compiler.passes.grade_pruning import GradePruningPass
from rune.compiler.passes.fusion import FusionPass
from rune.compiler.passes.lower_fused_ops import LowerFusedOpsPass
from rune.compiler.passes.memory_plan import MemoryPlanPass, MemoryPlan
from rune.compiler.codegen import Codegen, ExecutionPlan


class CompilationResult:
    """Result of compiling a model."""

    def __init__(self):
        self.ir_before: Optional[IRGraph] = None
        self.ir_after: Optional[IRGraph] = None
        self.memory_plan: Optional[MemoryPlan] = None
        self.execution_plan: Optional[ExecutionPlan] = None

        # Stats
        self.nodes_before: int = 0
        self.nodes_after: int = 0
        self.grade_pruning_stats: Dict = {}
        self.fusion_stats: Dict = {}
        self.memory_stats: Dict = {}

    @property
    def node_reduction(self) -> float:
        """Fraction of nodes eliminated."""
        if self.nodes_before == 0:
            return 0.0
        return 1.0 - (self.nodes_after / self.nodes_before)

    @property
    def command_count(self) -> int:
        return self.execution_plan.command_count if self.execution_plan else 0

    def summary(self) -> str:
        lines = [
            "=== Compilation Summary ===",
            f"IR nodes: {self.nodes_before} -> {self.nodes_after} "
            f"({self.node_reduction:.1%} reduction)",
        ]
        if self.grade_pruning_stats:
            lines.append(
                f"Grade pruning: {self.grade_pruning_stats.get('grades_eliminated', 0)} "
                f"grade components eliminated, "
                f"{self.grade_pruning_stats.get('floats_saved', 0)} floats saved"
            )
        if self.fusion_stats:
            lines.append(
                f"Fusion: {self.fusion_stats.get('total_nodes_fused', 0)} nodes fused "
                f"(gp_grade={self.fusion_stats.get('gp_grade_fusions', 0)}, "
                f"ffn={self.fusion_stats.get('ffn_fusions', 0)}, "
                f"attn={self.fusion_stats.get('attention_fusions', 0)})"
            )
        if self.memory_plan:
            lines.append(
                f"Memory: {self.memory_plan.n_buffers} buffers, "
                f"{self.memory_plan.total_bytes / 1024 / 1024:.1f} MB"
            )
        if self.execution_plan:
            lines.append(f"Commands: {self.command_count}")
        return '\n'.join(lines)


def compile_model(model, batch_size: int = 1, seq_len: int = 64,
                  verbose: bool = False, with_loss: bool = False,
                  with_training: bool = False,
                  include_memory: bool = False) -> CompilationResult:
    """
    Compile an HLM model through the full optimization pipeline.

    Args:
        model: HLM model instance
        batch_size: Batch dimension for the compiled graph
        seq_len: Sequence length for the compiled graph
        verbose: Print compilation progress
        with_loss: Include cross-entropy loss in the graph
        with_training: Include full backward + optimizer update in the graph
        include_memory: Include memory bank nodes in the graph

    Returns:
        CompilationResult with IR, memory plan, and execution plan
    """
    result = CompilationResult()

    # Step 1: Trace
    if verbose:
        print("Step 1: Tracing model...")
    tracer = Tracer()
    if with_training:
        ir = tracer.trace_training_step(model, batch_size, seq_len,
                                        include_memory=include_memory)
    elif with_loss:
        ir = tracer.trace_with_loss(model, batch_size, seq_len, include_memory=include_memory)
    else:
        ir = tracer.trace(model, batch_size, seq_len, include_memory=include_memory)

    result.nodes_before = ir.live_node_count()
    # Save a snapshot of node count before optimization
    ir_before_count = ir.live_node_count()

    if verbose:
        print(f"  Traced: {ir.live_node_count()} nodes")

    # Step 2: Grade pruning
    if verbose:
        print("Step 2: Grade pruning...")
    grade_pass = GradePruningPass(verbose=verbose)
    ir = grade_pass.run(ir)
    result.grade_pruning_stats = dict(grade_pass.stats)

    # Step 3: Fusion
    if verbose:
        print("Step 3: Fusion...")
    fusion_pass = FusionPass(verbose=verbose)
    ir = fusion_pass.run(ir)
    result.fusion_stats = dict(fusion_pass.stats)

    result.ir_after = ir
    result.nodes_after = ir.live_node_count()

    if verbose:
        print(f"  After optimization: {ir.live_node_count()} nodes "
              f"(was {ir_before_count})")

    # Step 4: Lower fused ops for runtimes that still execute primitive kernels.
    if verbose:
        print("Step 4: Lower fused ops...")
    lower_fused_pass = LowerFusedOpsPass(verbose=verbose)
    ir = lower_fused_pass.run(ir)

    # Step 5: Memory planning
    if verbose:
        print("Step 5: Memory planning...")
    mem_pass = MemoryPlanPass(verbose=verbose)
    memory_plan = mem_pass.run(ir)
    result.memory_plan = memory_plan
    result.memory_stats = dict(mem_pass.stats)

    # Step 6: Codegen
    if verbose:
        print("Step 6: Code generation...")
    codegen = Codegen(verbose=verbose)
    exec_plan = codegen.generate(ir, memory_plan)
    result.execution_plan = exec_plan

    if verbose:
        print(result.summary())

    return result


class CompiledModel:
    """
    Compiled HLM wrapper providing train_step() and inference.

    The compiled model holds:
    - The compilation result (IR, memory plan, execution plan)
    - A reference to the original model (for parameter access)
    - Engine state for execution
    """

    def __init__(self, model, compilation: CompilationResult, *, engine=None, engine_support=None):
        self.model = model
        self.compilation = compilation
        self._param_buffers: Dict[int, object] = {}  # param_id -> array
        self._engine = engine
        self._engine_support = engine_support
        self._engine_error: Optional[str] = None

    @classmethod
    def from_model(cls, model, batch_size: int = 1, seq_len: int = 64,
                   verbose: bool = False, use_persistent_engine: bool = True,
                   strict_engine: bool = False,
                   max_grad_norm: Optional[float] = None) -> 'CompiledModel':
        """
        Compile a model and return a CompiledModel.

        Args:
            model: HLM model instance
            batch_size: Batch size for compilation
            seq_len: Sequence length for compilation
            verbose: Print compilation progress
        """
        compilation = compile_model(
            model, batch_size=batch_size, seq_len=seq_len,
            verbose=verbose, with_training=True,
            include_memory=bool(getattr(getattr(model, "config", model), "memory_enabled", False)),
        )
        engine = None
        engine_support = None
        engine_error = None

        if use_persistent_engine:
            try:
                from rune.engine.hlm_adapter import (
                    can_attach,
                    persistent_engine_support_report,
                )
                from rune.engine.cuda_engine import CUDAEngine

                engine_support = persistent_engine_support_report(model)
                attach_report = can_attach(model)
                if attach_report.supported:
                    params_by_id = {
                        id(param): param
                        for _, param in model.named_parameters()
                    }
                    engine = CUDAEngine()
                    engine.compile_execution_plan(
                        compilation.execution_plan,
                        params_by_id,
                        max_grad_norm=max_grad_norm,
                    )
                elif strict_engine:
                    raise ValueError(attach_report.summary())
            except Exception as exc:
                engine_error = str(exc)
                if strict_engine:
                    raise

        compiled = cls(model, compilation, engine=engine, engine_support=engine_support)
        compiled._engine_error = engine_error
        return compiled

    def train_step(self, input_ids, target_ids, lr: float = 1e-4) -> float:
        """
        Execute one training step using the compiled execution plan.

        When a persistent engine is attached, delegates to the engine.
        Otherwise uses an interpreted fallback that performs forward + loss +
        backward + parameter update.

        Args:
            input_ids: (batch, seq) integer token IDs
            target_ids: (batch, seq) integer target IDs
            lr: learning rate

        Returns:
            loss value (float)
        """
        if self._engine is not None:
            from rune.backend import to_numpy
            import numpy as np

            input_ids_np = np.ascontiguousarray(to_numpy(input_ids), dtype=np.int32)
            target_ids_np = np.ascontiguousarray(to_numpy(target_ids), dtype=np.int32)
            return float(self._engine.run_step(input_ids_np, target_ids_np, lr=lr))

        from rune.backend import xp

        # --- Zero gradients ---
        try:
            from holograph.layers.clifford_linear import _zero_grad_params, _get_grad
            _zero_grad_params(self.model.parameters())
        except ImportError:
            _get_grad = None

        # --- Forward pass (interpreted, using original model) ---
        logits = self.model.forward(input_ids)

        # --- Cross-entropy loss ---
        batch, seq, vocab = logits.shape
        logits_flat = logits.reshape(-1, vocab)
        targets_flat = target_ids.reshape(-1)

        # Numerically stable softmax
        logits_max = xp.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = xp.exp(logits_flat - logits_max)
        probs = exp_logits / (xp.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

        # Cross entropy
        n = targets_flat.shape[0]
        log_probs = xp.log(probs[xp.arange(n), targets_flat] + 1e-12)
        loss = -xp.mean(log_probs)

        # --- Backward pass ---
        # Gradient of CE loss w.r.t. logits: probs - one_hot(targets)
        grad_logits = probs.copy()
        grad_logits[xp.arange(n), targets_flat] -= 1.0
        grad_logits /= n

        # Backward through model (if model supports backward)
        if hasattr(self.model, 'backward'):
            self.model.backward(grad_logits.reshape(batch, seq, vocab))
        else:
            # Minimal backward: propagate through lm_head
            # The lm_head expects multivector gradients (batch*seq, vocab, 8)
            # where only the scalar component is nonzero
            if hasattr(self.model, 'lm_head') and hasattr(self.model.lm_head, 'backward'):
                # Expand scalar grad_logits to multivector format for lm_head
                grad_logits_mv = xp.zeros((n, vocab, 8), dtype=grad_logits.dtype)
                grad_logits_mv[..., 0] = grad_logits
                try:
                    grad_hidden = self.model.lm_head.backward(grad_logits_mv)
                except Exception:
                    grad_hidden = None

                if grad_hidden is not None:
                    if hasattr(self.model, 'final_norm') and hasattr(self.model.final_norm, 'backward'):
                        try:
                            grad_hidden = self.model.final_norm.backward(grad_hidden)
                        except Exception:
                            pass
                    # Backward through blocks in reverse
                    for block in reversed(self.model.blocks):
                        if hasattr(block, 'backward'):
                            try:
                                grad_hidden = block.backward(grad_hidden)
                            except Exception:
                                break

        # --- Parameter update (SGD with learning rate) ---
        for p in self.model.parameters():
            if _get_grad is not None:
                g = _get_grad(p)
            else:
                g = getattr(p, 'grad', None)
            if g is not None:
                p -= lr * g

        return float(loss)

    def inference(self, input_ids) -> object:
        """
        Run inference using the compiled model.

        If the persistent engine is attached, uses the engine's forward path.
        Otherwise falls back to interpreted execution.
        """
        if self._engine is not None:
            try:
                from rune.backend import to_numpy
                import numpy as np

                input_ids_np = np.ascontiguousarray(to_numpy(input_ids), dtype=np.int32)
                # Use engine for forward-only by running with dummy targets and lr=0
                # to avoid parameter updates. This is a pragmatic approach until
                # a dedicated forward-only program is built.
                n = input_ids_np.size
                dummy_targets = np.zeros(n, dtype=np.int32)
                self._engine.run_step(input_ids_np, dummy_targets, lr=0.0)
                # Read logits from engine
                if self._engine.pool.has('act.logits'):
                    logits = self._engine.pool.download('act.logits')
                    batch = input_ids_np.shape[0] if input_ids_np.ndim > 1 else 1
                    seq = input_ids_np.shape[1] if input_ids_np.ndim > 1 else input_ids_np.shape[0]
                    vocab = logits.shape[-1] if logits.ndim > 1 else logits.size // (batch * seq)
                    return logits.reshape(batch, seq, vocab)
            except Exception:
                pass  # Fall back to interpreted
        return self.model.forward(input_ids)

    @property
    def uses_persistent_engine(self) -> bool:
        return self._engine is not None

    @property
    def engine_support(self):
        return self._engine_support

    @property
    def engine_error(self) -> Optional[str]:
        return self._engine_error

    def sync_model_from_engine(self) -> None:
        """Copy engine-owned parameters back into the wrapped model."""
        if self._engine is None:
            return
        if hasattr(self._engine, "get_param_by_id"):
            import numpy as np
            from rune.backend import xp
            for _, param in self.model.named_parameters():
                try:
                    updated = self._engine.get_param_by_id(id(param))
                except Exception:
                    continue
                if isinstance(param, np.ndarray):
                    param[...] = np.asarray(updated, dtype=param.dtype)
                else:
                    param[...] = xp.asarray(updated, dtype=param.dtype)
        else:
            from rune.engine.hlm_adapter import sync_hlm_core_params_from_engine
            sync_hlm_core_params_from_engine(self.model, self._engine)

        if getattr(self.model, "memory_bank", None) is not None and hasattr(self._engine, "get_memory_bank_state"):
            import numpy as np
            from rune.backend import xp

            try:
                state = self._engine.get_memory_bank_state()
            except Exception:
                return

            bank = self.model.memory_bank
            if isinstance(bank.bank, np.ndarray):
                bank_state = np.asarray(state, dtype=bank.bank.dtype)
            else:
                bank_state = xp.asarray(state, dtype=bank.bank.dtype)
            n_slots = min(bank.bank.shape[0], max(int(bank_state.shape[0] - 1), 0))
            if n_slots > 0:
                bank.bank[:n_slots] = bank_state[1:1 + n_slots]
            bank.n_valid = min(int(round(float(state[0, 0]))), bank.bank.shape[0])
            bank.write_head = int(round(float(state[0, 1])))
            if getattr(self.model, "memory_controller", None) is not None:
                self.model.memory_controller.chunk_count = bank.write_head

    def get_engine_param_grad(self, name: str):
        """Read a parameter gradient directly from the persistent engine."""
        if self._engine is None:
            raise RuntimeError("persistent engine is not attached")
        if not hasattr(self._engine, "get_param_grad"):
            raise RuntimeError("persistent engine does not expose gradient buffers")
        return self._engine.get_param_grad(name)

    def get_engine_param_grad_by_id(self, param_id: int):
        """Read a parameter gradient directly from the persistent engine by id."""
        if self._engine is None:
            raise RuntimeError("persistent engine is not attached")
        if not hasattr(self._engine, "get_param_grad_by_id"):
            raise RuntimeError("persistent engine does not expose gradient buffers")
        return self._engine.get_param_grad_by_id(param_id)

    @property
    def execution_plan(self) -> ExecutionPlan:
        return self.compilation.execution_plan

    @property
    def memory_plan(self) -> MemoryPlan:
        return self.compilation.memory_plan

    def summary(self) -> str:
        lines = [self.compilation.summary()]
        if self._engine_support is not None:
            status = "attached" if self._engine is not None else "not-attached"
            lines.append(f"Persistent engine: {status}")
            lines.extend(self._engine_support.summary_lines())
        if self._engine_error:
            lines.append(f"Persistent engine error: {self._engine_error}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"CompiledModel({self.compilation.nodes_before} -> "
                f"{self.compilation.nodes_after} nodes, "
                f"{self.compilation.command_count} commands)")
