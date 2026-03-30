"""
training_step.py — Builds the full command sequence for one training step.

Handles buffer allocation for all parameters, activations, and gradients,
then constructs the forward + loss + backward + update program for the
persistent engine.

The forward pass for a Clifford transformer:
  1. Embed lookup (token -> multivector)
  2. For each layer:
     a. LayerNorm (CliffordLayerNorm)
     b. Self-attention (Q/K/V projections, score, softmax, weighted sum, output proj)
     c. Residual add
     d. LayerNorm
     e. FFN (CliffordLinear -> GELU -> CliffordLinear)
     f. Residual add
  3. Final LayerNorm
  4. Project to logits (scalar part @ embedding^T)
  5. Cross-entropy loss

The current engine program still computes a real backward signal only
through the tied LM head. Unsupported parameter paths are zeroed so
their updates remain well-defined instead of consuming uninitialized
GPU memory.
"""

import math
import numpy as np
from typing import List, Optional, Dict

from rune.engine.program import ProgramBuilder, EngineOp
from rune.engine.memory_pool import MemoryPool


class TrainingStepBuilder:
    """
    Allocates buffers and builds the command program for a Clifford transformer.
    """

    def __init__(self, pool: MemoryPool,
                 vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int,
                 batch_size: int, seq_len: int,
                 use_positional_encoding: bool = False):
        self.pool = pool
        self.V = vocab_size
        self.D = d_model
        self.H = n_heads
        self.L = n_layers
        self.F = d_ff
        self.B = batch_size
        self.S = seq_len
        self.d_head = d_model // n_heads
        self.N = batch_size * seq_len  # total tokens per step
        self.use_positional_encoding = bool(use_positional_encoding)

    # ---- Buffer allocation ----

    def allocate_buffers(self):
        """Allocate all buffers in the memory pool."""
        p = self.pool
        V, D, H, L, F, B, S, N = (
            self.V, self.D, self.H, self.L, self.F, self.B, self.S, self.N
        )
        d_head = self.d_head

        # --- Input/output ---
        p.alloc_int('input_ids', (N,))
        p.alloc_int('target_ids', (N,))
        p.alloc_scalar('loss_out', 1)
        p.alloc_scalar('grad_norm_out', 1)

        # --- Embedding / positional encoding ---
        p.alloc('embed.weight', (V, D, 8))
        if self.use_positional_encoding:
            p.alloc('pos.rotors', (N, D, 8))

        # --- Per-layer parameters ---
        for layer in range(L):
            pfx = f'layer{layer}'
            # Attention LayerNorm
            p.alloc(f'{pfx}.ln1.gamma', (D, 8))
            p.alloc(f'{pfx}.ln1.beta', (D, 8))
            # Q, K, V, O projections
            p.alloc(f'{pfx}.attn.Wq', (D, D, 8))
            p.alloc(f'{pfx}.attn.Wk', (D, D, 8))
            p.alloc(f'{pfx}.attn.Wv', (D, D, 8))
            p.alloc(f'{pfx}.attn.Wo', (D, D, 8))
            p.alloc(f'{pfx}.attn.bq', (D, 8))
            p.alloc(f'{pfx}.attn.bk', (D, 8))
            p.alloc(f'{pfx}.attn.bv', (D, 8))
            p.alloc(f'{pfx}.attn.bo', (D, 8))
            # FFN LayerNorm
            p.alloc(f'{pfx}.ln2.gamma', (D, 8))
            p.alloc(f'{pfx}.ln2.beta', (D, 8))
            # FFN weights
            p.alloc(f'{pfx}.ffn.W1', (F, D, 8))
            p.alloc(f'{pfx}.ffn.b1', (F, 8))
            p.alloc(f'{pfx}.ffn.W2', (D, F, 8))
            p.alloc(f'{pfx}.ffn.b2', (D, 8))

        # Final LayerNorm
        p.alloc('final_ln.gamma', (D, 8))
        p.alloc('final_ln.beta', (D, 8))

        # --- Shared activation scratch ---
        p.alloc('act.embed_out', (N, D, 8))
        p.alloc('act.residual', (N, D, 8))
        p.alloc('act.ln1_out', (N, D, 8))
        p.alloc('act.q', (N, D, 8))
        p.alloc('act.k', (N, D, 8))
        p.alloc('act.v', (N, D, 8))
        p.alloc('act.scores', (B, H, S, S))
        p.alloc('act.attn_weights', (B, H, S, S))
        p.alloc('act.attn_out', (N, D, 8))
        p.alloc('act.o_proj', (N, D, 8))
        p.alloc('act.ln2_out', (N, D, 8))
        p.alloc('act.ffn_mid', (N, F, 8))
        p.alloc('act.ffn_act', (N, F, 8))
        p.alloc('act.ffn_out', (N, D, 8))
        p.alloc('act.final_ln_out', (N, D, 8))
        p.alloc('act.logits', (N, V))
        p.alloc('grad.logits', (N, V))

        # --- Gradient buffers for ALL parameters ---
        p.alloc('grad.embed.weight', (V, D, 8))

        if self.use_positional_encoding:
            p.alloc('grad.pos.rotors', (N, D, 8))

        for layer in range(L):
            pfx = f'layer{layer}'
            # Attention LayerNorm grads
            p.alloc(f'grad.{pfx}.ln1.gamma', (D, 8))
            p.alloc(f'grad.{pfx}.ln1.beta', (D, 8))
            # Q, K, V, O projection grads
            p.alloc(f'grad.{pfx}.attn.Wq', (D, D, 8))
            p.alloc(f'grad.{pfx}.attn.Wk', (D, D, 8))
            p.alloc(f'grad.{pfx}.attn.Wv', (D, D, 8))
            p.alloc(f'grad.{pfx}.attn.Wo', (D, D, 8))
            p.alloc(f'grad.{pfx}.attn.bq', (D, 8))
            p.alloc(f'grad.{pfx}.attn.bk', (D, 8))
            p.alloc(f'grad.{pfx}.attn.bv', (D, 8))
            p.alloc(f'grad.{pfx}.attn.bo', (D, 8))
            # FFN LayerNorm grads
            p.alloc(f'grad.{pfx}.ln2.gamma', (D, 8))
            p.alloc(f'grad.{pfx}.ln2.beta', (D, 8))
            # FFN weight grads
            p.alloc(f'grad.{pfx}.ffn.W1', (F, D, 8))
            p.alloc(f'grad.{pfx}.ffn.b1', (F, 8))
            p.alloc(f'grad.{pfx}.ffn.W2', (D, F, 8))
            p.alloc(f'grad.{pfx}.ffn.b2', (D, 8))

        # Final LayerNorm grads
        p.alloc('grad.final_ln.gamma', (D, 8))
        p.alloc('grad.final_ln.beta', (D, 8))

        # --- Hidden state gradient buffer (for backward propagation) ---
        p.alloc('grad.hidden', (N, D, 8))

        # --- Adam optimizer state buffers (m and v for each parameter) ---
        p.alloc('adam_m.embed.weight', (V, D, 8))
        p.alloc('adam_v.embed.weight', (V, D, 8))

        if self.use_positional_encoding:
            p.alloc('adam_m.pos.rotors', (N, D, 8))
            p.alloc('adam_v.pos.rotors', (N, D, 8))

        for layer in range(L):
            pfx = f'layer{layer}'
            for suffix, shape in [
                (f'{pfx}.ln1.gamma', (D, 8)),
                (f'{pfx}.ln1.beta', (D, 8)),
                (f'{pfx}.attn.Wq', (D, D, 8)),
                (f'{pfx}.attn.Wk', (D, D, 8)),
                (f'{pfx}.attn.Wv', (D, D, 8)),
                (f'{pfx}.attn.Wo', (D, D, 8)),
                (f'{pfx}.attn.bq', (D, 8)),
                (f'{pfx}.attn.bk', (D, 8)),
                (f'{pfx}.attn.bv', (D, 8)),
                (f'{pfx}.attn.bo', (D, 8)),
                (f'{pfx}.ln2.gamma', (D, 8)),
                (f'{pfx}.ln2.beta', (D, 8)),
                (f'{pfx}.ffn.W1', (F, D, 8)),
                (f'{pfx}.ffn.b1', (F, 8)),
                (f'{pfx}.ffn.W2', (D, F, 8)),
                (f'{pfx}.ffn.b2', (D, 8)),
            ]:
                p.alloc(f'adam_m.{suffix}', shape)
                p.alloc(f'adam_v.{suffix}', shape)

        p.alloc('adam_m.final_ln.gamma', (D, 8))
        p.alloc('adam_v.final_ln.gamma', (D, 8))
        p.alloc('adam_m.final_ln.beta', (D, 8))
        p.alloc('adam_v.final_ln.beta', (D, 8))

    # ---- Parameter initialization ----

    def init_params_random(self, seed: int = 42):
        """Initialize parameters with scaled random values."""
        rng = np.random.RandomState(seed)
        D = self.D
        scale = 1.0 / math.sqrt(D)

        for name in list(self.pool._buffers.keys()):
            info = self.pool._buffers[name]
            # Skip non-parameter buffers
            if (name.startswith('act.') or name.startswith('grad.') or
                    name.startswith('input_') or name.startswith('target_') or
                    name in ('loss_out', 'grad_norm_out')):
                continue

            shape = info.shape
            if 'gamma' in name:
                # LayerNorm gamma: scalar 1
                data = np.zeros(shape, dtype=np.float32)
                data[..., 0] = 1.0  # scalar part = 1
                self.pool.upload(name, data)
            elif 'beta' in name:
                # LayerNorm beta: zero
                self.pool.upload(name, np.zeros(shape, dtype=np.float32))
            elif name == 'embed.weight':
                data = rng.randn(*shape).astype(np.float32) * 0.02
                self.pool.upload(name, data)
            elif 'W' in name.split('.')[-1]:
                # Weight matrices: scaled normal
                data = rng.randn(*shape).astype(np.float32) * scale
                self.pool.upload(name, data)
            elif 'b' in name.split('.')[-1]:
                # Biases: zero
                self.pool.upload(name, np.zeros(shape, dtype=np.float32))

    # ---- Program construction ----

    def build_program(self, prog: ProgramBuilder) -> List[int]:
        """
        Build the full forward + loss + update command sequence.

        Returns list of command indices for adam_step commands
        (so the engine can patch the learning rate).
        """
        adam_indices = []

        # Forward pass
        self._build_forward(prog)

        # Loss computation
        self._build_loss(prog)

        # Parameter update
        adam_indices = self._build_update(prog)

        return adam_indices

    def _build_forward(self, prog: ProgramBuilder):
        """Build the forward pass commands."""
        p = self.pool
        N, D, H, L, F, B, S = self.N, self.D, self.H, self.L, self.F, self.B, self.S
        d_head = self.d_head

        # 1. Embedding lookup
        prog.embed_lookup(
            p.ptr('embed.weight'),
            p.ptr('input_ids'),
            p.ptr('act.embed_out'),
            N, D
        )

        if self.use_positional_encoding:
            prog.batch_gp(
                p.ptr('pos.rotors'),
                p.ptr('act.embed_out'),
                p.ptr('act.residual'),
                N * D,
            )
        else:
            prog.copy(p.ptr('act.embed_out'), p.ptr('act.residual'), N * D * 8)

        # 2. Transformer layers
        for layer in range(L):
            pfx = f'layer{layer}'
            # 2a. LayerNorm 1
            prog.norm_scale(
                p.ptr('act.residual'),
                p.ptr(f'{pfx}.ln1.gamma'),
                p.ptr(f'{pfx}.ln1.beta'),
                p.ptr('act.ln1_out'),
                N, D, eps=1e-6
            )

            # 2b. Q, K, V projections
            prog.linear_fwd(
                p.ptr(f'{pfx}.attn.Wq'), p.ptr('act.ln1_out'),
                p.ptr('act.q'), p.ptr(f'{pfx}.attn.bq'),
                N, D, D
            )
            prog.linear_fwd(
                p.ptr(f'{pfx}.attn.Wk'), p.ptr('act.ln1_out'),
                p.ptr('act.k'), p.ptr(f'{pfx}.attn.bk'),
                N, D, D
            )
            prog.linear_fwd(
                p.ptr(f'{pfx}.attn.Wv'), p.ptr('act.ln1_out'),
                p.ptr('act.v'), p.ptr(f'{pfx}.attn.bv'),
                N, D, D
            )

            # 2c. Attention scoring
            scale = 1.0 / math.sqrt(d_head)
            prog.attn_score(
                p.ptr('act.q'), p.ptr('act.k'),
                p.ptr('act.scores'),
                B, H, S, d_head,
                scale
            )

            # 2d. Softmax
            prog.softmax(
                p.ptr('act.scores'),
                p.ptr('act.attn_weights'),
                B * H * S, S
            )

            # 2e. Weighted sum
            prog.weighted_sum(
                p.ptr('act.attn_weights'),
                p.ptr('act.v'),
                p.ptr('act.attn_out'),
                B, H, S, d_head
            )

            # 2f. Output projection
            prog.linear_fwd(
                p.ptr(f'{pfx}.attn.Wo'), p.ptr('act.attn_out'),
                p.ptr('act.o_proj'), p.ptr(f'{pfx}.attn.bo'),
                N, D, D
            )

            # 2g. Residual add
            prog.batch_add(
                p.ptr('act.residual'), p.ptr('act.o_proj'),
                p.ptr('act.residual'), N * D
            )

            # 2h. LayerNorm 2
            prog.norm_scale(
                p.ptr('act.residual'),
                p.ptr(f'{pfx}.ln2.gamma'),
                p.ptr(f'{pfx}.ln2.beta'),
                p.ptr('act.ln2_out'),
                N, D, eps=1e-6
            )

            # 2i. FFN: W1 -> GELU -> W2
            prog.linear_fwd(
                p.ptr(f'{pfx}.ffn.W1'), p.ptr('act.ln2_out'),
                p.ptr('act.ffn_mid'), p.ptr(f'{pfx}.ffn.b1'),
                N, D, F
            )
            prog.batch_gelu(
                p.ptr('act.ffn_mid'),
                p.ptr('act.ffn_act'),
                N * F
            )
            prog.linear_fwd(
                p.ptr(f'{pfx}.ffn.W2'), p.ptr('act.ffn_act'),
                p.ptr('act.ffn_out'), p.ptr(f'{pfx}.ffn.b2'),
                N, F, D
            )

            # 2j. Residual add
            prog.batch_add(
                p.ptr('act.residual'), p.ptr('act.ffn_out'),
                p.ptr('act.residual'), N * D
            )

        # 3. Final LayerNorm
        prog.norm_scale(
            p.ptr('act.residual'),
            p.ptr('final_ln.gamma'),
            p.ptr('final_ln.beta'),
            p.ptr('act.final_ln_out'),
            N, D, eps=1e-6
        )

        prog.tied_lm_head(
            p.ptr('act.final_ln_out'),
            p.ptr('embed.weight'),
            p.ptr('act.logits'),
            N, D, self.V,
        )

    def _build_loss(self, prog: ProgramBuilder):
        """Build CE loss + gradient computation.

        The backward for tied LM head (logits = x_scalar @ emb_scalar.T) needs:
        - grad_x_scalar = grad_logits @ emb_scalar  (for hidden state gradient)
        - grad_emb_scalar = grad_logits.T @ x_scalar  (for embedding gradient)
        Both must be computed and propagated.
        """
        # Zero all gradient buffers before accumulation. The current engine
        # only writes a subset of them.
        for name in self.pool.names():
            if name.startswith('grad.'):
                prog.zero(self.pool.ptr(name), self.pool.info(name).n_floats)

        prog.ce_loss(
            self.pool.ptr('act.logits'),
            self.pool.ptr('target_ids'),
            self.pool.ptr('grad.logits'),
            self.N, self.V
        )

        # Backward through tied LM head:
        # grad_embed += grad_logits.T @ x_scalar (already computed by TIED_LM_HEAD_BWD)
        prog.tied_lm_head_bwd(
            self.pool.ptr('grad.logits'),
            self.pool.ptr('act.final_ln_out'),
            self.pool.ptr('embed.weight'),
            self.pool.ptr('grad.embed.weight'),
            self.N, self.D, self.V,
        )
        # Also compute grad_hidden = grad_logits @ embed_scalar for backprop
        # This is the hidden state gradient that should propagate back through
        # the rest of the network. We use matmul_scalar:
        # grad_hidden = grad_logits @ grade_0(embed.weight)
        # grad_logits: (N, V), embed.weight_scalar: (V, D) -> grad_hidden: (N, D)
        # Store in grad.hidden for backward propagation
        prog.matmul_scalar(
            self.pool.ptr('grad.logits'),
            self.pool.ptr('embed.weight'),
            self.pool.ptr('grad.hidden'),
            self.N, self.V, self.D, mode=1,
        )

    def _build_update(self, prog: ProgramBuilder) -> List[int]:
        """
        Build parameter update commands for EVERY parameter with real Adam.
        Returns indices of adam commands for LR patching.

        Real Adam for each parameter:
          m = beta1 * m + (1 - beta1) * grad
          v = beta2 * v + (1 - beta2) * grad^2
          m_hat = m / (1 - beta1^t)
          v_hat = v / (1 - beta2^t)
          param -= lr * m_hat / (sqrt(v_hat) + eps)
        """
        adam_indices = []
        p = self.pool

        # Collect all parameter names and their corresponding gradient / adam state names
        param_names = self._get_all_param_names()

        for param_name in param_names:
            grad_name = f'grad.{param_name}'
            m_name = f'adam_m.{param_name}'
            v_name = f'adam_v.{param_name}'

            # Only emit update if we have all buffers
            if not (p.has(param_name) and p.has(grad_name) and
                    p.has(m_name) and p.has(v_name)):
                continue

            n_floats = p.info(param_name).n_floats
            info = p.info(param_name)
            group_size = 8 if (info.shape and info.shape[-1] == 8) else (
                int(info.shape[-1]) if info.shape else 1
            )

            idx = prog.adam_full(
                p.ptr(param_name),
                p.ptr(grad_name),
                p.ptr(m_name),
                p.ptr(v_name),
                n_floats=n_floats,
                lr=1e-4,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                step=1,
                group_size=group_size,
            )
            adam_indices.append(idx)

        return adam_indices

    def _get_all_param_names(self) -> List[str]:
        """Return names of all trainable parameter buffers."""
        names = ['embed.weight']

        if self.use_positional_encoding:
            names.append('pos.rotors')

        for layer in range(self.L):
            pfx = f'layer{layer}'
            names.extend([
                f'{pfx}.ln1.gamma', f'{pfx}.ln1.beta',
                f'{pfx}.attn.Wq', f'{pfx}.attn.Wk',
                f'{pfx}.attn.Wv', f'{pfx}.attn.Wo',
                f'{pfx}.attn.bq', f'{pfx}.attn.bk',
                f'{pfx}.attn.bv', f'{pfx}.attn.bo',
                f'{pfx}.ln2.gamma', f'{pfx}.ln2.beta',
                f'{pfx}.ffn.W1', f'{pfx}.ffn.b1',
                f'{pfx}.ffn.W2', f'{pfx}.ffn.b2',
            ])

        names.extend(['final_ln.gamma', 'final_ln.beta'])
        return names

    # ---- Utility ----

    def param_count(self) -> int:
        """Count total parameters (in multivectors)."""
        total = 0
        for name in self.pool.names():
            if not (name.startswith('act.') or name.startswith('grad.') or
                    name.startswith('input_') or name.startswith('target_') or
                    name in ('loss_out', 'grad_norm_out')):
                total += self.pool.info(name).n_floats
        return total // 8  # Convert floats to multivectors

    def activation_bytes(self) -> int:
        """Count total activation memory."""
        total = 0
        for name in self.pool.names():
            if name.startswith('act.'):
                total += self.pool.info(name).size_bytes
        return total
