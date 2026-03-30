"""
holographic_memory_bank.py â€” Persistent algebraic memory for HLM.

1,000,000 slots Ã— 8 floats = 32 MB. Each slot stores one multivector
that is the distilled algebraic summary of a 2,048-token chunk.

This is NOT a KV cache. This is NOT RAG. The geometric product
simultaneously computes relevance, relationship, and retrieval.
"""

import numpy as np
from rune.backend import xp
from rune.ops.batched import (
    batched_geom_prod, batched_sandwich, batched_normalize, bivector_exp_from_components,
)
from holograph.memory.geometric_fold import geometric_fold
from holograph.memory.grade_decay import (
    compute_grade_energy, apply_grade_decay, DEFAULT_DECAY_RATES,
)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - xp.max(x, axis=axis, keepdims=True)
    exp_x = xp.exp(shifted)
    return exp_x / (xp.sum(exp_x, axis=axis, keepdims=True) + 1e-12)


_SCALAR_SIGNATURE = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)


def _position_to_rotor(chunk_position: int) -> np.ndarray:
    """
    Encode chunk position as a rotor (even sub-algebra multivector).

    Uses a fixed set of bivector frequencies so that relative temporal
    distance between memories is recoverable via GP.

    Args:
        chunk_position: integer chunk index
    Returns:
        (8,) multivector â€” rotor encoding the position
    """
    # Three bivector frequencies (one per plane)
    freqs = xp.array([0.001, 0.01, 0.1], dtype=xp.float32)
    bv = chunk_position * freqs  # (3,)
    rotor = bivector_exp_from_components(bv.reshape(1, 3))  # (1, 8)
    return rotor[0]  # (8,)


class HolographicMemoryBank:
    """
    Holographic Memory Bank â€” algebraic long-range memory.

    Stores distilled multivector summaries of processed chunks.
    Reads via geometric product relevance scoring.
    Writes via geometric fold + sandwich with context rotor.
    """

    def __init__(self, n_slots: int = 1_000_000, d_model: int = 216):
        self.n_slots = n_slots
        self.d_model = d_model

        # The bank: each slot is ONE summary multivector
        # Shape: [n_slots, 8], Storage: n_slots Ã— 8 Ã— 4 = 32 MB for 1M slots
        self.bank = xp.zeros((n_slots, 8), dtype=xp.float32)

        # Context rotor for each slot â€” encodes WHEN this was written
        # Shape: [n_slots, 8] (full MV, even sub-algebra active)
        self.context_rotors = xp.zeros((n_slots, 8), dtype=xp.float32)

        # Grade energy of each slot â€” for decay/eviction
        # Shape: [n_slots, 4]
        self.grade_energy = xp.zeros((n_slots, 4), dtype=xp.float32)

        # Write pointer â€” circular buffer
        self.write_head = 0

        # Number of valid slots
        self.n_valid = 0

        # Evictable mask
        self.evictable = xp.zeros(n_slots, dtype=bool)

        # Decay config
        self.decay_rates = list(DEFAULT_DECAY_RATES)
        self.decay_interval = 100
        self._chunks_since_decay = 0

    def write(self, chunk_output: np.ndarray, chunk_position: int):
        """
        Distill chunk output into a single summary MV and write to bank.

        Args:
            chunk_output: (seq_len, d_model, 8) â€” final layer output
            chunk_position: integer â€” which chunk in the stream
        """
        # Step 1: Mean pool over sequence
        pooled = xp.mean(chunk_output, axis=0)  # (d_model, 8)

        # Step 2: Geometric fold â€” pairwise GP reduction to single MV
        summary = geometric_fold(pooled)  # (8,)

        # Step 3: Context rotor
        context_rotor = _position_to_rotor(chunk_position)  # (8,)

        # Step 4: Sandwich product â€” bake context into content
        # memory = R * summary * ~R
        contextualized = batched_sandwich(
            context_rotor.reshape(1, 8),
            summary.reshape(1, 8),
        )[0]

        # Step 5: Find write slot â€” prefer evictable, else circular
        if self.n_valid < self.n_slots:
            slot = self.n_valid
        elif xp.any(self.evictable):
            # Evict lowest-energy slot
            evictable_indices = xp.where(self.evictable)[0]
            energies = self.grade_energy[evictable_indices].sum(axis=1)
            slot = evictable_indices[xp.argmin(energies)]
        else:
            slot = self.write_head % self.n_slots

        # Step 6: Write
        self.bank[slot] = contextualized
        self.context_rotors[slot] = context_rotor
        self.grade_energy[slot] = compute_grade_energy(
            contextualized.reshape(1, 8)
        )[0]
        self.write_head += 1
        self.n_valid = min(self.n_valid + 1, self.n_slots)

        # Periodic decay
        self._chunks_since_decay += 1
        if self._chunks_since_decay >= self.decay_interval:
            self._apply_decay()
            self._chunks_since_decay = 0

    def read(self, query: np.ndarray, top_k: int = 64) -> np.ndarray:
        """
        Read from memory bank using geometric product relevance.

        Args:
            query: (8,) â€” summary multivector of current chunk state
            top_k: number of memories to retrieve
        Returns:
            (8,) â€” weighted algebraic summary of top-k memories
        """
        return self.read_many(query, top_k=top_k)

    def read_many(self, queries: np.ndarray, top_k: int = 64) -> np.ndarray:
        """
        Batched memory read.

        Args:
            queries: (..., 8) â€” one or more summary multivectors
            top_k: number of memories to retrieve per query
        Returns:
            (..., 8) â€” weighted algebraic summaries
        """
        queries = xp.asarray(queries, dtype=xp.float32)
        single_query = queries.ndim == 1
        if single_query:
            queries = queries[xp.newaxis, :]

        leading_shape = queries.shape[:-1]
        flat_queries = queries.reshape(-1, 8)

        if self.n_valid == 0:
            empty = xp.zeros((flat_queries.shape[0], 8), dtype=xp.float32)
            return empty[0] if single_query else empty.reshape(*leading_shape, 8)

        n = self.n_valid
        top_k = min(top_k, n)
        n_queries = flat_queries.shape[0]

        # Step 1: Scalar product for relevance (grade-0 of query * memory)
        signature = xp.asarray(_SCALAR_SIGNATURE)
        relevance = xp.matmul(flat_queries * signature, self.bank[:n].T)  # (n_queries, n)

        # Step 2: Top-k selection
        if top_k == n:
            top_k_indices = xp.broadcast_to(
                xp.arange(n, dtype=xp.int32),
                (n_queries, n),
            ).copy()
        else:
            top_k_indices = xp.argpartition(relevance, n - top_k, axis=-1)[..., -top_k:]
        top_k_scores = xp.take_along_axis(relevance, top_k_indices, axis=-1)
        sort_order = xp.argsort(top_k_scores, axis=-1)
        top_k_indices = xp.take_along_axis(top_k_indices, sort_order, axis=-1)
        top_k_scores = xp.take_along_axis(top_k_scores, sort_order, axis=-1)
        top_k_memories = self.bank[top_k_indices]  # (n_queries, top_k, 8)

        # Step 3: Full GP with top-k memories
        query_top = xp.broadcast_to(
            flat_queries[:, xp.newaxis, :],
            (n_queries, top_k, 8),
        ).copy()
        full_interactions = batched_geom_prod(query_top, top_k_memories)

        # Step 4: Softmax-weighted sum
        weights = _softmax(top_k_scores, axis=-1)
        retrieved = xp.sum(weights[..., xp.newaxis] * full_interactions, axis=1)

        if single_query:
            return retrieved[0]
        return retrieved.reshape(*leading_shape, 8)

    def _apply_decay(self):
        """Apply grade-specific decay to all valid memories."""
        if self.n_valid > 0:
            self.evictable[:self.n_valid] = apply_grade_decay(
                self.bank, self.grade_energy,
                decay_rates=self.decay_rates,
                n_valid=self.n_valid
            )

    def memory_stats(self) -> dict:
        """Return diagnostic statistics about the memory bank."""
        if self.n_valid == 0:
            return {'n_valid': 0, 'n_evictable': 0}

        energy = self.grade_energy[:self.n_valid]
        return {
            'n_valid': self.n_valid,
            'n_evictable': int(xp.sum(self.evictable[:self.n_valid])),
            'mean_energy': float(energy.sum(axis=1).mean()),
            'grade_energy_mean': [float(energy[:, g].mean()) for g in range(4)],
            'write_head': self.write_head,
        }
