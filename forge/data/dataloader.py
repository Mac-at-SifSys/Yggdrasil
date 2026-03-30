"""
dataloader.py -- Data loader for Clifford HLM language model training.

Yields (tokens, targets) batches for next-token prediction.
Targets are simply the input shifted by one position.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from typing import Optional
from forge.data.tokenizer import BasicTokenizer


class CliffordDataLoader:
    """Streaming data loader for character-level language modelling.

    Reads a text file (or a string), tokenizes it with ``BasicTokenizer``,
    and yields fixed-length ``(input, target)`` windows.

    Parameters
    ----------
    text : str, optional
        Raw text data.  Provide either ``text`` or ``filepath``.
    filepath : str, optional
        Path to a text file to load.
    tokenizer : BasicTokenizer, optional
        Pre-built tokenizer.  If None, one is created from the data.
    batch_size : int
        Number of sequences per batch.
    seq_len : int
        Length of each input sequence (target is the same length, shifted by 1).
    shuffle : bool
        If True, shuffle the order of windows each epoch.
    """

    def __init__(
        self,
        text: Optional[str] = None,
        filepath: Optional[str] = None,
        tokenizer: Optional[BasicTokenizer] = None,
        batch_size: int = 32,
        seq_len: int = 64,
        shuffle: bool = True,
    ):
        # Load text
        if text is not None:
            self._raw = text
        elif filepath is not None:
            with open(filepath, "r", encoding="utf-8") as f:
                self._raw = f.read()
        else:
            raise ValueError("Provide either `text` or `filepath`")

        # Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BasicTokenizer(self._raw)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle

        # Tokenize the full corpus once
        self._tokens = self.tokenizer.encode(self._raw)

        # Build window start indices: need seq_len + 1 tokens per window
        n_windows = max(1, (len(self._tokens) - 1) // seq_len)
        self._starts = np.arange(n_windows) * seq_len

        self._epoch = 0

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def num_batches(self) -> int:
        return max(1, len(self._starts) // self.batch_size)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        """Yield (input_ids, target_ids) batches.

        input_ids  : np.ndarray, shape (batch_size, seq_len), dtype int64
        target_ids : np.ndarray, shape (batch_size, seq_len), dtype int64
                     (input shifted right by 1 -- next token prediction)
        """
        indices = self._starts.copy()
        if self.shuffle:
            np.random.shuffle(indices)

        for b in range(self.num_batches):
            batch_starts = indices[b * self.batch_size : (b + 1) * self.batch_size]
            inputs = np.zeros((len(batch_starts), self.seq_len), dtype=np.int64)
            targets = np.zeros((len(batch_starts), self.seq_len), dtype=np.int64)

            for i, start in enumerate(batch_starts):
                end = start + self.seq_len + 1
                # Clamp to corpus length
                chunk = self._tokens[start : min(end, len(self._tokens))]
                actual_len = min(len(chunk) - 1, self.seq_len)
                if actual_len <= 0:
                    continue
                inputs[i, :actual_len] = chunk[:actual_len]
                targets[i, :actual_len] = chunk[1:actual_len + 1]

            yield inputs, targets

        self._epoch += 1
