"""
program.py — Program builder for the persistent CUDA engine.

Converts a list of operation descriptors into a command buffer that
matches the EngineCommand struct in persistent_engine.cu exactly.
"""

import ctypes
import numpy as np
from typing import List, Optional


class EngineOp:
    """Operation codes matching the CUDA EngineOp enum exactly."""
    NOP = 0
    BATCH_GP = 1
    BATCH_REVERSE = 2
    BATCH_SANDWICH = 3
    BATCH_BVEXP = 4
    BATCH_ADD = 5
    BATCH_SCALE = 6
    BATCH_GRADE_PROJ = 7
    BATCH_SCALAR_PROD = 8
    BATCH_GELU = 9
    BATCH_NORM_SCALE = 10
    EMBED_LOOKUP = 11
    LINEAR_FWD = 12
    ATTN_SCORE = 13
    SOFTMAX = 14
    WEIGHTED_SUM = 15
    MATMUL_SCALAR = 16
    CE_LOSS_FWD = 17
    ADAM_STEP = 18
    GRAD_CLIP = 19
    COPY = 20
    ZERO = 21
    ACCUMULATE = 22
    BARRIER = 23
    DONE = 24
    TIED_LM_HEAD = 25
    TIED_LM_HEAD_BWD = 26
    ADAM_FULL = 27  # Real Adam: param, grad, m, v, lr, beta1, beta2, eps, step
    BACKWARD_CE = 28
    BACKWARD_LINEAR = 29
    BACKWARD_GP = 30
    BACKWARD_NORM = 31
    BACKWARD_GELU = 32
    BACKWARD_EMBED = 33
    BACKWARD_ADD = 34
    BACKWARD_MATMUL = 35
    BACKWARD_GRADE_PROJECT = 36
    BACKWARD_BVEXP = 37
    BACKWARD_SOFTMAX = 38
    BACKWARD_WEIGHTED_SUM = 39
    BACKWARD_COPY = 40
    BACKWARD_ATTENTION = 41
    BACKWARD_FFN = 42
    MEAN_POOL_SEQ = 43
    MEMORY_READ = 44
    MEMORY_WRITE = 45
    MEMORY_GATE = 46
    BACKWARD_MEMORY_GATE = 47

    _NAMES = {
        0: 'NOP', 1: 'BATCH_GP', 2: 'BATCH_REVERSE', 3: 'BATCH_SANDWICH',
        4: 'BATCH_BVEXP', 5: 'BATCH_ADD', 6: 'BATCH_SCALE', 7: 'BATCH_GRADE_PROJ',
        8: 'BATCH_SCALAR_PROD', 9: 'BATCH_GELU', 10: 'BATCH_NORM_SCALE',
        11: 'EMBED_LOOKUP', 12: 'LINEAR_FWD', 13: 'ATTN_SCORE', 14: 'SOFTMAX',
        15: 'WEIGHTED_SUM', 16: 'MATMUL_SCALAR', 17: 'CE_LOSS_FWD', 18: 'ADAM_STEP',
        19: 'GRAD_CLIP', 20: 'COPY', 21: 'ZERO', 22: 'ACCUMULATE', 23: 'BARRIER',
        24: 'DONE', 25: 'TIED_LM_HEAD', 26: 'TIED_LM_HEAD_BWD',
        27: 'ADAM_FULL', 28: 'BACKWARD_CE', 29: 'BACKWARD_LINEAR',
        30: 'BACKWARD_GP', 31: 'BACKWARD_NORM', 32: 'BACKWARD_GELU',
        33: 'BACKWARD_EMBED', 34: 'BACKWARD_ADD', 35: 'BACKWARD_MATMUL',
        36: 'BACKWARD_GRADE_PROJECT', 37: 'BACKWARD_BVEXP',
        38: 'BACKWARD_SOFTMAX', 39: 'BACKWARD_WEIGHTED_SUM',
        40: 'BACKWARD_COPY', 41: 'BACKWARD_ATTENTION', 42: 'BACKWARD_FFN',
        43: 'MEAN_POOL_SEQ', 44: 'MEMORY_READ', 45: 'MEMORY_WRITE',
        46: 'MEMORY_GATE', 47: 'BACKWARD_MEMORY_GATE',
    }

    @classmethod
    def name(cls, opcode: int) -> str:
        return cls._NAMES.get(opcode, f'UNKNOWN({opcode})')


class EngineCommand(ctypes.Structure):
    """
    Matches the CUDA EngineCommand struct.

    Layout:
        int opcode          (4 bytes)
        float* arg0         (8 bytes on 64-bit)
        float* arg1         (8 bytes)
        float* arg2         (8 bytes)
        float* arg3         (8 bytes)
        int dim0            (4 bytes)
        int dim1            (4 bytes)
        int dim2            (4 bytes)
        int dim3            (4 bytes)
        float scalar0       (4 bytes)
        float scalar1       (4 bytes)
        int pad[16]         (64 bytes)

    The CUDA side uses ``alignas(64)``, which makes the struct size 128 bytes
    on 64-bit platforms. Python has to match that exact stride or later
    commands in the uploaded buffer will be misread by the kernel.
    """
    _fields_ = [
        ("opcode", ctypes.c_int),
        ("arg0", ctypes.c_void_p),
        ("arg1", ctypes.c_void_p),
        ("arg2", ctypes.c_void_p),
        ("arg3", ctypes.c_void_p),
        ("dim0", ctypes.c_int),
        ("dim1", ctypes.c_int),
        ("dim2", ctypes.c_int),
        ("dim3", ctypes.c_int),
        ("scalar0", ctypes.c_float),
        ("scalar1", ctypes.c_float),
        ("pad", ctypes.c_int * 16),
    ]
    _pack_ = 8  # Match 64-bit pointer alignment


class ProgramBuilder:
    """
    Builds a sequence of commands for the persistent engine.

    Usage:
        prog = ProgramBuilder()
        prog.embed_lookup(table_ptr, ids_ptr, out_ptr, n_tokens, d_model)
        prog.linear_fwd(w_ptr, x_ptr, y_ptr, bias_ptr, batch, d_in, d_out)
        prog.batch_gelu(y_ptr, y_ptr, batch * d_out)
        prog.done()
        cmd_array, n_cmds = prog.build()
    """

    def __init__(self):
        self.commands: List[EngineCommand] = []

    def __len__(self):
        return len(self.commands)

    def add(self, opcode: int, arg0=0, arg1=0, arg2=0, arg3=0,
            dim0=0, dim1=0, dim2=0, dim3=0,
            scalar0=0.0, scalar1=0.0) -> int:
        """Add a raw command. Returns the command index."""
        cmd = EngineCommand()
        cmd.opcode = opcode
        cmd.arg0 = arg0 if arg0 else 0
        cmd.arg1 = arg1 if arg1 else 0
        cmd.arg2 = arg2 if arg2 else 0
        cmd.arg3 = arg3 if arg3 else 0
        cmd.dim0 = dim0
        cmd.dim1 = dim1
        cmd.dim2 = dim2
        cmd.dim3 = dim3
        cmd.scalar0 = scalar0
        cmd.scalar1 = scalar1
        self.commands.append(cmd)
        return len(self.commands) - 1

    # ---- Convenience methods ----

    def batch_gp(self, a_ptr, b_ptr, out_ptr, n):
        """Batched geometric product. n = number of multivectors."""
        return self.add(EngineOp.BATCH_GP, a_ptr, b_ptr, out_ptr, dim0=n)

    def batch_reverse(self, a_ptr, out_ptr, n):
        return self.add(EngineOp.BATCH_REVERSE, a_ptr, 0, out_ptr, dim0=n)

    def batch_sandwich(self, r_ptr, x_ptr, out_ptr, n):
        return self.add(EngineOp.BATCH_SANDWICH, r_ptr, x_ptr, out_ptr, dim0=n)

    def batch_bvexp(self, bv_ptr, out_ptr, n):
        return self.add(EngineOp.BATCH_BVEXP, bv_ptr, 0, out_ptr, dim0=n)

    def batch_add(self, a_ptr, b_ptr, out_ptr, n):
        return self.add(EngineOp.BATCH_ADD, a_ptr, b_ptr, out_ptr, dim0=n)

    def batch_scale(self, a_ptr, out_ptr, n, scale):
        return self.add(EngineOp.BATCH_SCALE, a_ptr, 0, out_ptr, dim0=n, scalar0=scale)

    def batch_grade_proj(self, a_ptr, out_ptr, n, grade, output_components=8):
        return self.add(
            EngineOp.BATCH_GRADE_PROJ,
            a_ptr,
            0,
            out_ptr,
            dim0=n,
            dim1=grade,
            dim2=output_components,
        )

    def batch_scalar_prod(self, a_ptr, b_ptr, out_ptr, n):
        return self.add(EngineOp.BATCH_SCALAR_PROD, a_ptr, b_ptr, out_ptr, dim0=n)

    def batch_gelu(self, a_ptr, out_ptr, n):
        return self.add(EngineOp.BATCH_GELU, a_ptr, 0, out_ptr, dim0=n)

    def linear_fwd(self, w_ptr, x_ptr, y_ptr, bias_ptr, batch, d_in, d_out):
        return self.add(EngineOp.LINEAR_FWD, w_ptr, x_ptr, y_ptr, bias_ptr,
                       dim0=batch, dim1=d_in, dim2=d_out)

    def norm_scale(self, x_ptr, gamma_ptr, beta_ptr, out_ptr, n, d_model, eps=1e-6):
        return self.add(EngineOp.BATCH_NORM_SCALE, x_ptr, gamma_ptr, out_ptr, beta_ptr,
                       dim0=n, dim1=d_model, scalar0=eps)

    def attn_score(self, q_ptr, k_ptr, scores_ptr, batch, n_heads, seq, d_head, scale):
        return self.add(EngineOp.ATTN_SCORE, q_ptr, k_ptr, scores_ptr,
                       dim0=batch, dim1=n_heads, dim2=seq, dim3=d_head, scalar0=scale)

    def softmax(self, in_ptr, out_ptr, n_rows, row_len):
        return self.add(EngineOp.SOFTMAX, in_ptr, 0, out_ptr, dim0=n_rows, dim1=row_len)

    def weighted_sum(self, weights_ptr, v_ptr, out_ptr, batch, n_heads, seq, d_head):
        return self.add(EngineOp.WEIGHTED_SUM, weights_ptr, v_ptr, out_ptr,
                       dim0=batch, dim1=n_heads, dim2=seq, dim3=d_head)

    def embed_lookup(self, table_ptr, ids_ptr, out_ptr, n_tokens, d_model):
        return self.add(EngineOp.EMBED_LOOKUP, table_ptr, ids_ptr, out_ptr,
                       dim0=n_tokens, dim1=d_model)

    def matmul_scalar(self, a_ptr, b_ptr, c_ptr, M, K, N, mode=0):
        """
        mode=0:
            A=(M,K), B=(N,K), C=(M,N) dense scalar matmul.
        mode=1:
            A=(M,K) scalar, B=(N,K,8) multivector, C=(M,N,8) multivector.
            Only the scalar lane of B participates and the result is written
            into the scalar lane of C.
        """
        return self.add(EngineOp.MATMUL_SCALAR, a_ptr, b_ptr, c_ptr,
                       dim0=M, dim1=K, dim2=N, dim3=mode)

    def tied_lm_head(self, hidden_ptr, embed_ptr, logits_ptr, n_tokens, d_model, vocab):
        return self.add(EngineOp.TIED_LM_HEAD, hidden_ptr, embed_ptr, logits_ptr,
                       dim0=n_tokens, dim1=d_model, dim2=vocab)

    def tied_lm_head_bwd(self, grad_logits_ptr, hidden_ptr, embed_ptr, grad_embed_ptr,
                         n_tokens, d_model, vocab):
        return self.add(
            EngineOp.TIED_LM_HEAD_BWD,
            grad_logits_ptr, hidden_ptr, embed_ptr, grad_embed_ptr,
            dim0=n_tokens, dim1=d_model, dim2=vocab
        )

    def ce_loss(self, logits_ptr, targets_ptr, grad_ptr, n, vocab):
        return self.add(EngineOp.CE_LOSS_FWD, logits_ptr, targets_ptr, grad_ptr,
                       dim0=n, dim1=vocab)

    def adam_step(self, param_ptr, grad_ptr, n_mvs, lr):
        return self.add(EngineOp.ADAM_STEP, param_ptr, grad_ptr, 0, 0,
                       dim0=n_mvs, scalar0=lr)

    def adam_full(self, param_ptr, grad_ptr, m_ptr, v_ptr, n_floats, lr,
                  beta1=0.9, beta2=0.999, eps=1e-8, step=1, group_size=8):
        """
        Real Adam optimizer step.

        For each element i:
          m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
          v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2
          m_hat = m[i] / (1 - beta1^step)
          v_hat = v[i] / (1 - beta2^step)
          param[i] -= lr * m_hat / (sqrt(v_hat) + eps)

        Args:
            param_ptr: Device pointer to parameter buffer.
            grad_ptr: Device pointer to gradient buffer.
            m_ptr: Device pointer to first moment (m) buffer.
            v_ptr: Device pointer to second moment (v) buffer.
            n_floats: Number of float32 elements.
            lr: Learning rate.
            beta1: First moment decay.
            beta2: Second moment decay.
            eps: Epsilon for numerical stability.
            step: Current training step (for bias correction).
        """
        # Pack beta1 and beta2 into scalar0 and scalar1 via a two-command encoding.
        # scalar0 = lr, scalar1 = eps. dim0 = n_floats, dim1 = step,
        # dim2 encodes beta1*1000, dim3 encodes beta2*1000 (integer encoding)
        idx = self.add(EngineOp.ADAM_FULL, param_ptr, grad_ptr, m_ptr, v_ptr,
                       dim0=n_floats, dim1=step,
                       dim2=int(beta1 * 10000), dim3=int(beta2 * 10000),
                       scalar0=lr, scalar1=eps)
        self.commands[idx].pad[0] = int(max(group_size, 1))
        return idx

    def grad_clip(self, grad_ptr, n_floats, max_norm, mode='full'):
        mode_map = {
            'accumulate': 0,
            'apply': 1,
            'full': 2,
        }
        mode_id = mode_map.get(mode, mode if isinstance(mode, int) else 2)
        return self.add(EngineOp.GRAD_CLIP, grad_ptr, 0, 0,
                       dim0=n_floats, dim1=mode_id, scalar0=max_norm)

    def zero(self, ptr, n_floats):
        return self.add(EngineOp.ZERO, 0, 0, ptr, dim0=n_floats)

    def accumulate(self, in_ptr, out_ptr, n_floats):
        return self.add(EngineOp.ACCUMULATE, in_ptr, 0, out_ptr, dim0=n_floats)

    def copy(self, src_ptr, dst_ptr, n_floats):
        return self.add(EngineOp.COPY, src_ptr, 0, dst_ptr, dim0=n_floats)

    def barrier(self):
        return self.add(EngineOp.BARRIER)

    def done(self):
        return self.add(EngineOp.DONE)

    def build(self):
        """
        Build the command buffer as a ctypes array.
        Returns (command_array, n_commands).
        """
        n = len(self.commands)
        arr = (EngineCommand * n)(*self.commands)
        return arr, n

    def build_bytes(self) -> bytes:
        """Return the command buffer as raw bytes (for uploading to GPU)."""
        arr, n = self.build()
        return bytes(arr)

    def dump(self) -> str:
        """Human-readable dump of the program."""
        lines = []
        for i, cmd in enumerate(self.commands):
            name = EngineOp.name(cmd.opcode)
            dims = f"dim=({cmd.dim0},{cmd.dim1},{cmd.dim2},{cmd.dim3})"
            scalars = f"s0={cmd.scalar0:.4f} s1={cmd.scalar1:.4f}"
            lines.append(f"  [{i:3d}] {name:20s} {dims}  {scalars}")
        return f"Program ({len(self.commands)} commands):\n" + "\n".join(lines)

    def patch_scalar(self, cmd_idx: int, field: str, value: float):
        """
        Patch a scalar field in an existing command.
        Used to update learning rate between steps without rebuilding.
        """
        if field == 'scalar0':
            self.commands[cmd_idx].scalar0 = value
        elif field == 'scalar1':
            self.commands[cmd_idx].scalar1 = value
        else:
            raise ValueError(f"Unknown scalar field: {field}")
