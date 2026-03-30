import math, time, os, sys, gc
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch.utils.data import Dataset, DataLoader

print(f"PyTorch {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

# ============================================================================
# Cl(3,0) Geometric Algebra Core
# ============================================================================

BLADE_NAMES = ["1", "e1", "e2", "e3", "e12", "e13", "e23", "e123"]
BLADE_GRADES = [0, 1, 1, 1, 2, 2, 2, 3]
BLADE_SEMANTIC = [
    "temporal", "narrative", "causation", "affect",
    "wisdom", "ecology", "relations", "epistemics"
]

_BASIS = [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
_BLADE_INDEX = {blade: idx for idx, blade in enumerate(_BASIS)}


def _multiply_basis(a_gens, b_gens):
    gens = list(a_gens) + list(b_gens)
    sign = 1
    n = len(gens)
    for i in range(n):
        for j in range(n - 1 - i):
            if gens[j] > gens[j + 1]:
                gens[j], gens[j + 1] = gens[j + 1], gens[j]
                sign *= -1
    result = []
    i = 0
    while i < len(gens):
        if i + 1 < len(gens) and gens[i] == gens[i + 1]:
            i += 2
        else:
            result.append(gens[i])
            i += 1
    return sign, tuple(result)


def build_cl300_table():
    gamma = torch.zeros(8, 8, 8, dtype=torch.float32)
    for i in range(8):
        for j in range(8):
            sign, result_gens = _multiply_basis(_BASIS[i], _BASIS[j])
            k = _BLADE_INDEX[result_gens]
            gamma[i, j, k] = float(sign)
    return gamma

_GAMMA = build_cl300_table()


def _cl300_expand_batched(w_all):
    """Hardcoded Cl(3,0) weight expansion.
    w_all: [F, P, 27, 8] → W_eff: [F, P, 8, 8, 3, 3, 3]
    64 sparse ±1 additions."""
    F_, P = w_all.shape[0], w_all.shape[1]
    W = torch.zeros(F_, P, 8, 8, 27, device=w_all.device, dtype=w_all.dtype)
    W[:,:,0,0]+=w_all[:,:,:,0]; W[:,:,0,1]+=w_all[:,:,:,1]
    W[:,:,0,2]+=w_all[:,:,:,2]; W[:,:,0,3]+=w_all[:,:,:,3]
    W[:,:,0,4]-=w_all[:,:,:,4]; W[:,:,0,5]-=w_all[:,:,:,5]
    W[:,:,0,6]-=w_all[:,:,:,6]; W[:,:,0,7]-=w_all[:,:,:,7]
    W[:,:,1,1]+=w_all[:,:,:,0]; W[:,:,1,0]+=w_all[:,:,:,1]
    W[:,:,1,4]-=w_all[:,:,:,2]; W[:,:,1,5]-=w_all[:,:,:,3]
    W[:,:,1,2]+=w_all[:,:,:,4]; W[:,:,1,3]+=w_all[:,:,:,5]
    W[:,:,1,7]-=w_all[:,:,:,6]; W[:,:,1,6]-=w_all[:,:,:,7]
    W[:,:,2,2]+=w_all[:,:,:,0]; W[:,:,2,4]+=w_all[:,:,:,1]
    W[:,:,2,0]+=w_all[:,:,:,2]; W[:,:,2,6]-=w_all[:,:,:,3]
    W[:,:,2,1]-=w_all[:,:,:,4]; W[:,:,2,7]+=w_all[:,:,:,5]
    W[:,:,2,3]+=w_all[:,:,:,6]; W[:,:,2,5]+=w_all[:,:,:,7]
    W[:,:,3,3]+=w_all[:,:,:,0]; W[:,:,3,5]+=w_all[:,:,:,1]
    W[:,:,3,6]+=w_all[:,:,:,2]; W[:,:,3,0]+=w_all[:,:,:,3]
    W[:,:,3,7]-=w_all[:,:,:,4]; W[:,:,3,1]-=w_all[:,:,:,5]
    W[:,:,3,2]-=w_all[:,:,:,6]; W[:,:,3,4]-=w_all[:,:,:,7]
    W[:,:,4,4]+=w_all[:,:,:,0]; W[:,:,4,2]+=w_all[:,:,:,1]
    W[:,:,4,1]-=w_all[:,:,:,2]; W[:,:,4,7]+=w_all[:,:,:,3]
    W[:,:,4,0]+=w_all[:,:,:,4]; W[:,:,4,6]-=w_all[:,:,:,5]
    W[:,:,4,5]+=w_all[:,:,:,6]; W[:,:,4,3]+=w_all[:,:,:,7]
    W[:,:,5,5]+=w_all[:,:,:,0]; W[:,:,5,3]+=w_all[:,:,:,1]
    W[:,:,5,7]-=w_all[:,:,:,2]; W[:,:,5,1]-=w_all[:,:,:,3]
    W[:,:,5,6]+=w_all[:,:,:,4]; W[:,:,5,0]+=w_all[:,:,:,5]
    W[:,:,5,4]-=w_all[:,:,:,6]; W[:,:,5,2]-=w_all[:,:,:,7]
    W[:,:,6,6]+=w_all[:,:,:,0]; W[:,:,6,7]+=w_all[:,:,:,1]
    W[:,:,6,3]+=w_all[:,:,:,2]; W[:,:,6,2]-=w_all[:,:,:,3]
    W[:,:,6,5]-=w_all[:,:,:,4]; W[:,:,6,4]+=w_all[:,:,:,5]
    W[:,:,6,0]+=w_all[:,:,:,6]; W[:,:,6,1]+=w_all[:,:,:,7]
    W[:,:,7,7]+=w_all[:,:,:,0]; W[:,:,7,6]+=w_all[:,:,:,1]
    W[:,:,7,5]-=w_all[:,:,:,2]; W[:,:,7,4]+=w_all[:,:,:,3]
    W[:,:,7,3]+=w_all[:,:,:,4]; W[:,:,7,2]-=w_all[:,:,:,5]
    W[:,:,7,1]+=w_all[:,:,:,6]; W[:,:,7,0]+=w_all[:,:,:,7]
    return W.reshape(F_, P, 8, 8, 3, 3, 3)


# ============================================================================
# Multi-Scale Memory Banks
# ============================================================================

class ByteMemoryBank(nn.Module):
    """Memory bank for n-byte n-grams. Buffers, not trainable."""

    def __init__(self, n_gram_size, d_state=8, n_slots=65536):
        super().__init__()
        self.n = n_gram_size
        self.d_state = d_state
        self.n_slots = n_slots
        self.register_buffer('bank', torch.zeros(n_slots, d_state))
        self.register_buffer('counts', torch.zeros(n_slots, dtype=torch.long))
        # Precompute multipliers for address computation (vectorized)
        multipliers = torch.tensor([256 ** (n_gram_size - 1 - i) for i in range(n_gram_size)],
                                   dtype=torch.long)
        self.register_buffer('_addr_multipliers', multipliers)

    def compute_address(self, byte_window):
        """Fully vectorized address computation — no Python loops."""
        if self.n <= 4:
            # Direct polynomial: dot product with [256^(n-1), 256^(n-2), ..., 1]
            addr = (byte_window.long() * self._addr_multipliers).sum(dim=-1)
            return addr % self.n_slots
        else:
            # Vectorized FNV-1a: use tensor ops instead of per-byte loop
            # Process all bytes simultaneously via cumulative XOR+multiply
            bw = byte_window.long()  # (..., n)
            h = torch.full(bw.shape[:-1], 2166136261, dtype=torch.long, device=bw.device)
            # Unrolled: process 4 bytes at a time for speed
            n = self.n
            i = 0
            while i + 3 < n:
                h = ((h ^ bw[..., i]) * 16777619) % (2**32)
                h = ((h ^ bw[..., i+1]) * 16777619) % (2**32)
                h = ((h ^ bw[..., i+2]) * 16777619) % (2**32)
                h = ((h ^ bw[..., i+3]) * 16777619) % (2**32)
                i += 4
            while i < n:
                h = ((h ^ bw[..., i]) * 16777619) % (2**32)
                i += 1
            return h % self.n_slots

    def read(self, byte_window):
        addr = self.compute_address(byte_window)
        return self.bank[addr]

    def write_batch(self, byte_window, states, momentum=0.9):
        """EMA write using CPU temporaries to avoid VRAM fragmentation."""
        with torch.no_grad():
            # Move to CPU for scatter work — avoids GPU memory fragmentation
            addr = self.compute_address(byte_window).reshape(-1).cpu()
            flat_states = states.reshape(-1, self.d_state).float().cpu()

            # 1. Count occurrences per address
            ones = torch.ones(addr.shape[0])
            hit_counts = torch.zeros(self.n_slots)
            hit_counts.scatter_add_(0, addr, ones)

            # 2. Sum states per address
            state_sums = torch.zeros(self.n_slots, self.d_state)
            addr_expanded = addr.unsqueeze(1).expand_as(flat_states)
            state_sums.scatter_add_(0, addr_expanded, flat_states)

            # 3. Compute mean for addresses that got hits
            hit_mask = hit_counts > 0
            if hit_mask.any():
                mean_states = state_sums[hit_mask] / hit_counts[hit_mask].unsqueeze(1)
                old_counts = self.counts[hit_mask].cpu()
                is_new = (old_counts == 0)
                alpha = torch.where(is_new, torch.zeros_like(mean_states[:, 0]),
                                    torch.full_like(mean_states[:, 0], momentum))
                new_vals = alpha.unsqueeze(1) * self.bank[hit_mask].cpu() + \
                           (1 - alpha.unsqueeze(1)) * mean_states
                self.bank[hit_mask] = new_vals.to(self.bank.device)
                self.counts[hit_mask] += hit_counts[hit_mask].long().to(self.counts.device)

    def coverage(self):
        return (self.counts > 0).float().mean().item()


class StackedBladeBank(nn.Module):
    """8 blade-specific banks stacked into a single tensor for parallel lookup.
    Instead of 8 separate ByteMemoryBank objects with 8 sequential reads,
    this uses a single (8, n_slots, d_state) tensor with one gather op."""

    def __init__(self, n_gram_size, d_state=8, n_slots=100_000):
        super().__init__()
        self.n = n_gram_size
        self.d_state = d_state
        self.n_slots = n_slots
        # All 8 blade banks in one tensor
        self.register_buffer('bank', torch.zeros(8, n_slots, d_state))
        self.register_buffer('counts', torch.zeros(8, n_slots, dtype=torch.long))

    def compute_address(self, byte_window):
        """Hash-based addressing for all n-gram sizes (16, 32 byte windows)."""
        bw = byte_window.long()
        h = torch.full(bw.shape[:-1], 2166136261, dtype=torch.long, device=bw.device)
        n = self.n
        i = 0
        while i + 3 < n:
            h = ((h ^ bw[..., i]) * 16777619) % (2**32)
            h = ((h ^ bw[..., i+1]) * 16777619) % (2**32)
            h = ((h ^ bw[..., i+2]) * 16777619) % (2**32)
            h = ((h ^ bw[..., i+3]) * 16777619) % (2**32)
            i += 4
        while i < n:
            h = ((h ^ bw[..., i]) * 16777619) % (2**32)
            i += 1
        return h % self.n_slots

    def read_all_blades(self, byte_window):
        """Read from ALL 8 blade banks in parallel. Returns (B, seq, 8, d_state)."""
        addr = self.compute_address(byte_window)  # (B, seq)
        # Gather from all 8 banks simultaneously
        # bank: (8, n_slots, d_state), addr: (B, seq)
        addr_flat = addr.reshape(-1)  # (B*seq,)
        # Expand addr for all 8 blades: gather from (8, n_slots, d_state)
        results = self.bank[:, addr_flat, :]  # (8, B*seq, d_state)
        B_seq = addr.shape
        return results.permute(1, 0, 2).reshape(*B_seq, 8, self.d_state)

    def write_by_blade(self, byte_window, states, dominant_blade, momentum=0.9):
        """Write routed by dominant blade — CPU temporaries to avoid VRAM leak."""
        with torch.no_grad():
            addr = self.compute_address(byte_window).reshape(-1).cpu()
            blades = dominant_blade.reshape(-1).cpu()
            flat_states = states.reshape(-1, self.d_state).float().cpu()

            for b in range(8):
                mask = (blades == b)
                if not mask.any():
                    continue
                b_addr = addr[mask]
                b_states = flat_states[mask]
                ones = torch.ones(b_addr.shape[0])
                hit_counts = torch.zeros(self.n_slots)
                hit_counts.scatter_add_(0, b_addr, ones)
                state_sums = torch.zeros(self.n_slots, self.d_state)
                state_sums.scatter_add_(0, b_addr.unsqueeze(1).expand_as(b_states), b_states)
                hit_mask = hit_counts > 0
                if hit_mask.any():
                    mean_states = state_sums[hit_mask] / hit_counts[hit_mask].unsqueeze(1)
                    old_counts = self.counts[b, hit_mask].cpu()
                    is_new = (old_counts == 0)
                    alpha = torch.where(is_new, torch.zeros_like(mean_states[:, 0]),
                                        torch.full_like(mean_states[:, 0], momentum))
                    new_vals = alpha.unsqueeze(1) * self.bank[b, hit_mask].cpu() + \
                               (1 - alpha.unsqueeze(1)) * mean_states
                    self.bank[b, hit_mask] = new_vals.to(self.bank.device)
                    self.counts[b, hit_mask] += hit_counts[hit_mask].long().to(self.counts.device)

    def coverage(self):
        return (self.counts > 0).float().mean().item()


class MultiScaleMemory(nn.Module):
    """Five memory banks: 2, 4, 8, 16 (per-blade), 32 (per-blade) bytes.
    Parallel window extraction + stacked blade banks for single-op lookups."""

    def __init__(self, d_state=8, global_slots=1_000_000, per_blade_slots=100_000):
        super().__init__()
        self.d_state = d_state
        self.bank_2 = ByteMemoryBank(2, d_state, 65_536)
        self.bank_4 = ByteMemoryBank(4, d_state, global_slots)
        self.bank_8 = ByteMemoryBank(8, d_state, global_slots)
        # Stacked blade banks: single tensor per scale, parallel 8-blade lookup
        self.blade_bank_16 = StackedBladeBank(16, d_state, per_blade_slots)
        self.blade_bank_32 = StackedBladeBank(32, d_state, per_blade_slots)
        self.scale_weights = nn.Parameter(torch.ones(5) / 5)

    def extract_all_windows(self, byte_seq):
        """Extract ALL windows at all 5 scales in one pass.
        Returns dict of scale → (B, seq, scale) tensors."""
        B, seq = byte_seq.shape
        # Pad once to max window size (32), extract all sizes from it
        max_n = 32
        padded = F_func.pad(byte_seq, (max_n - 1, 0), value=0)
        # All windows share the same padded sequence
        windows = {}
        for n in [2, 4, 8, 16, 32]:
            # Offset from padded start: extract windows of size n
            # padded has (max_n - 1) zeros prepended
            # For window size n, we need to start at offset (max_n - n)
            offset = max_n - n
            windows[n] = padded[:, offset:offset + seq + n - 1].unfold(1, n, 1)
        return windows

    def read(self, byte_seq):
        """Read all 5 scales in parallel where possible."""
        windows = self.extract_all_windows(byte_seq)

        # Global banks — 3 independent reads (could be concurrent on GPU)
        r2 = self.bank_2.read(windows[2])
        r4 = self.bank_4.read(windows[4])
        r8 = self.bank_8.read(windows[8])

        # Blade banks — single parallel lookup across all 8 blades
        r16 = self.blade_bank_16.read_all_blades(windows[16]).mean(dim=2)  # (B, seq, d_state)
        r32 = self.blade_bank_32.read_all_blades(windows[32]).mean(dim=2)

        # Stack and blend — single vectorized op
        stacked = torch.stack([r2, r4, r8, r16, r32], dim=2)  # (B, seq, 5, d_state)
        weights = F_func.softmax(self.scale_weights, dim=0)
        return (stacked * weights[None, None, :, None]).sum(dim=2)

    def write(self, byte_seq, states, blade_profiles=None):
        """Write to all banks — vectorized scatter for each."""
        states_proj = states[..., :self.d_state] if states.shape[-1] != self.d_state else states
        windows = self.extract_all_windows(byte_seq)

        # Global banks — 3 parallel writes
        self.bank_2.write_batch(windows[2], states_proj)
        self.bank_4.write_batch(windows[4], states_proj)
        self.bank_8.write_batch(windows[8], states_proj)

        # Blade banks — vectorized blade-routed write
        if blade_profiles is not None:
            dominant = blade_profiles.argmax(dim=-1)
            self.blade_bank_16.write_by_blade(windows[16], states_proj, dominant)
            self.blade_bank_32.write_by_blade(windows[32], states_proj, dominant)

    def coverage_stats(self):
        return {
            '2b': self.bank_2.coverage() * 100,
            '4b': self.bank_4.coverage() * 100,
            '8b': self.bank_8.coverage() * 100,
            '16b': self.blade_bank_16.coverage() * 100,
            '32b': self.blade_bank_32.coverage() * 100,
        }


# ============================================================================
# Clifford Engine
# ============================================================================

class CliffordEngine(nn.Module):
    """Cl(3,0) conv3d engine with cross-field mixing.
    CAUSAL via shifted convolution: asymmetric padding (2,0) per dimension
    shifts the 3x3x3 kernel so ALL 27 weights look at past/self positions.
    No future leakage, no wasted weights."""

    def __init__(self, n_fields, n_passes, grid_size):
        super().__init__()
        self.n_fields = n_fields
        self.n_passes = n_passes
        self.grid_size = grid_size
        F_ = n_fields

        self.all_weights = nn.Parameter(torch.empty(F_, n_passes, 27, 8))
        self.all_biases = nn.Parameter(torch.zeros(F_, n_passes, 8))
        self.field_mix_logits = nn.Parameter(torch.zeros(n_passes, F_, F_))
        for p in range(n_passes):
            self.field_mix_logits.data[p] = torch.eye(F_) * 3.0
        self.pass_alpha_logit = nn.Parameter(torch.full((n_passes,), -1.0))
        self.gate_weight = nn.Parameter(torch.full((F_, 8), 0.1))
        self.gate_bias = nn.Parameter(torch.full((F_, 8), 2.0))
        self._init_weights()

    def _init_weights(self):
        for f in range(self.n_fields):
            for p in range(self.n_passes):
                w = self.all_weights.data[f, p]
                nn.init.normal_(w, std=0.05 if p == 0 else 0.1)
                # Self-connection at kernel position (2,2,2) = index 26
                w[26, :] = 0.0
                if p == 0:
                    w[26, 0] = 0.8

    def forward(self, state):
        B = state.shape[0]
        F_ = self.n_fields
        G = self.grid_size
        G3 = G * G * G

        # Precompute ALL expanded weights and mixing matrices at once
        W_eff_all = _cl300_expand_batched(self.all_weights)  # (F, P, 8, 8, 3, 3, 3)
        mix_all = F_func.softmax(self.field_mix_logits, dim=2)  # (P, F, F)

        x = state
        old_state = x

        for p in range(self.n_passes):
            x_before = x
            W_p = W_eff_all[:, p]
            bias_p = self.all_biases[:, p]
            W_grouped = W_p.reshape(F_ * 8, 8, 3, 3, 3)
            bias_grouped = bias_p.reshape(F_ * 8)
            x_flat = x.reshape(B, F_ * 8, G, G, G)
            # Causal conv3d: pad 2 on past side, 0 on future side per dimension
            # Shifts the 3x3x3 kernel so output[z,y,x] reads input[z-2..z, y-2..y, x-2..x]
            # All 27 kernel positions are at linear offset <= 0 (past or self)
            x_flat = F_func.pad(x_flat, (2, 0, 2, 0, 2, 0))
            x_flat = torch.tanh(F_func.conv3d(x_flat, W_grouped, bias_grouped,
                                               padding=0, groups=F_))
            x = x_flat.reshape(B, F_, 8, G, G, G)

            # Cross-field mixing: bmm is faster than einsum for this shape
            # mix_all[p]: (F_out, F_in), x: (B, F_in, 8*G³) → (B, F_out, 8*G³)
            x_flat = x.reshape(B, F_, 8 * G3)  # (B, F, 8*G³)
            mix_p = mix_all[p].unsqueeze(0).expand(B, -1, -1)  # (B, F, F)
            x_flat = torch.bmm(mix_p, x_flat)  # (B, F, 8*G³)
            x = x_flat.reshape(B, F_, 8, G, G, G)

            alpha = torch.sigmoid(self.pass_alpha_logit[p])
            x = alpha * x_before + (1.0 - alpha) * x

        gw = self.gate_weight.reshape(1, F_, 8, 1, 1, 1)
        gb = self.gate_bias.reshape(1, F_, 8, 1, 1, 1)
        gate = torch.sigmoid(gw * old_state + gb)
        return gate * old_state + (1.0 - gate) * x


# ============================================================================
# Stacked Clifford Block
# ============================================================================

class CliffordBlock(nn.Module):
    """Engine + MLP block with pre-norm residuals."""

    def __init__(self, grid_size, n_fields, n_passes, d_model, mlp_hidden):
        super().__init__()
        self.grid_size = grid_size
        self.n_fields = n_fields
        self.n_voxels = grid_size ** 3
        self.d_model = d_model
        self.engine = CliffordEngine(n_fields, n_passes, grid_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x):
        B, seq, D = x.shape
        G = self.grid_size
        nv = self.n_voxels
        F_ = self.n_fields

        x_normed = self.norm1(x)
        n_chunks = (seq + nv - 1) // nv
        pad_len = n_chunks * nv - seq
        if pad_len > 0:
            x_padded = F_func.pad(x_normed, (0, 0, 0, pad_len))
        else:
            x_padded = x_normed

        x_chunked = x_padded.reshape(B * n_chunks, nv, F_, 8)
        field = x_chunked.permute(0, 2, 3, 1).reshape(B * n_chunks, F_, 8, G, G, G)
        field = self.engine(field)
        out = field.reshape(B * n_chunks, F_, 8, nv).permute(0, 3, 1, 2)
        out = out.reshape(B, n_chunks * nv, F_ * 8)
        if pad_len > 0:
            out = out[:, :seq, :]

        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# HLM v9 Full Model
# ============================================================================

class HLMv9(nn.Module):
    """Raw bytes → memory → Clifford blocks → byte prediction."""

    def __init__(self, grid_size=4, n_tracks=8, n_passes=2, n_blocks=24,
                 mlp_hidden=4096, global_mem_slots=1_000_000,
                 per_blade_mem_slots=100_000, mem_d_state=8):
        super().__init__()
        self.grid_size = grid_size
        self.n_fields = n_tracks * 8
        self.d_model = self.n_fields * 8
        self.n_blocks = n_blocks

        self.memory = MultiScaleMemory(mem_d_state, global_mem_slots, per_blade_mem_slots)
        self.mem_to_model = nn.Linear(mem_d_state, self.d_model)
        self.max_seq_len = 512
        self.pos_embed = nn.Parameter(torch.randn(self.max_seq_len, self.d_model) * 0.02)

        self.blocks = nn.ModuleList([
            CliffordBlock(grid_size, self.n_fields, n_passes, self.d_model, mlp_hidden)
            for _ in range(n_blocks)
        ])

        self.final_norm = nn.LayerNorm(self.d_model)
        self.output_head = nn.Linear(self.d_model, 256)
        self.blade_classifier = nn.Linear(self.d_model, 8)
        self.use_checkpoint = True  # Gradient checkpointing for memory efficiency

    def forward(self, byte_seq, labels=None, do_memory_write=False):
        """byte_seq: (B, seq) input bytes. labels: (B, seq) same sequence shifted by caller.
        do_memory_write: only write to memory banks when True (avoids VRAM fragmentation)."""
        B, seq = byte_seq.shape
        mem_return = self.memory.read(byte_seq)
        x = self.mem_to_model(mem_return)
        x = x + self.pos_embed[:seq]

        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.final_norm(x)
        logits = self.output_head(x)  # (B, seq, 256)
        result = {'logits': logits}

        if labels is not None:
            # logits: (B, seq, 256), labels: (B, seq)
            # Predict next byte: logits[t] predicts labels[t]
            result['loss'] = F_func.cross_entropy(
                logits.reshape(-1, 256),
                labels.reshape(-1).long()
            )

        # Memory writes only when requested — prevents VRAM leak from
        # scatter temporaries fragmenting GPU memory every step
        if do_memory_write:
            with torch.no_grad():
                blade_profiles = F_func.softmax(self.blade_classifier(x.detach()), dim=-1)
                write_states = x.detach()[:, :, :8]
                self.memory.write(byte_seq, write_states, blade_profiles)

        return result


# ============================================================================
# Dataset
# ============================================================================

class ByteDataset(Dataset):
    def __init__(self, data, seq_len=256):
        self.data = data
        self.seq_len = seq_len
        self.n_sequences = max(0, len(data) // seq_len - 1)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk)


# ============================================================================
# Data Loading — FineWeb-Edu
# ============================================================================

def load_fineweb(target_bytes=2_200_000_000):
    """Stream FineWeb-Edu and collect raw bytes."""
    cache_path = '/content/fineweb_bytes.bin'
    if os.path.exists(cache_path):
        data = np.fromfile(cache_path, dtype=np.uint8)
        if len(data) >= target_bytes:
            print(f"Loaded cached {len(data):,} bytes")
            return data[:target_bytes]

    from datasets import load_dataset
    print(f"Streaming FineWeb-Edu for {target_bytes/1e9:.1f}B bytes...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                       split="train", streaming=True)
    chunks = []
    total = 0
    for i, example in enumerate(ds):
        text = example.get('text', '')
        if text:
            raw = text.encode('utf-8')
            chunks.append(raw)
            total += len(raw)
            if i % 50000 == 0:
                print(f"  {total/1e9:.2f}B / {target_bytes/1e9:.1f}B", flush=True)
            if total >= target_bytes:
                break

    data = np.frombuffer(b''.join(chunks), dtype=np.uint8).copy()
    data.tofile(cache_path)
    print(f"Cached {len(data):,} bytes to {cache_path}")
    return data[:target_bytes]


# ============================================================================
# LR Schedule
# ============================================================================

def get_lr(step, max_steps, base_lr, warmup_steps=500):
    min_lr = base_lr * 0.1
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * coeff

