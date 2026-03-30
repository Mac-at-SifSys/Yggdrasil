"""
tokenizer.py -- Simple character-level tokenizer.

For the Clifford HLM stack we keep tokenization minimal so the interesting
work happens in the geometric algebra layers, not in a BPE vocabulary.
"""

import numpy as np
from typing import List, Optional, Dict


class BasicTokenizer:
    """Character-level tokenizer with special tokens.

    Maintains a bijective mapping between characters and integer IDs.
    Unknown characters at decode time are replaced with the <unk> token.

    Special tokens
    --------------
    0 : <pad>
    1 : <unk>
    2 : <bos>
    3 : <eos>
    """

    SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, text: Optional[str] = None):
        """Build vocabulary from optional training text.

        Parameters
        ----------
        text : str, optional
            If provided, the vocabulary is built from the unique characters
            in this text.  Otherwise call ``build_vocab`` later.
        """
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self._vocab_built = False

        if text is not None:
            self.build_vocab(text)

    def build_vocab(self, text: str):
        """Build character vocabulary from text."""
        # Start with special tokens
        self.char_to_id = {}
        self.id_to_char = {}
        for i, tok in enumerate(self.SPECIAL_TOKENS):
            self.char_to_id[tok] = i
            self.id_to_char[i] = tok

        # Add unique characters in sorted order for reproducibility
        chars = sorted(set(text))
        offset = len(self.SPECIAL_TOKENS)
        for i, ch in enumerate(chars):
            idx = offset + i
            self.char_to_id[ch] = idx
            self.id_to_char[idx] = ch

        self._vocab_built = True

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

    def encode(self, text: str, add_special: bool = False) -> np.ndarray:
        """Encode text to integer token IDs.

        Parameters
        ----------
        text : str
            Input string.
        add_special : bool
            If True, prepend <bos> and append <eos>.

        Returns
        -------
        np.ndarray, shape (seq_len,), dtype int64
        """
        ids = []
        if add_special:
            ids.append(self.bos_id)

        for ch in text:
            ids.append(self.char_to_id.get(ch, self.unk_id))

        if add_special:
            ids.append(self.eos_id)

        return np.array(ids, dtype=np.int64)

    def decode(self, token_ids, skip_special: bool = True) -> str:
        """Decode token IDs back to text.

        Parameters
        ----------
        token_ids : array-like of int
            Token indices.
        skip_special : bool
            If True, omit special tokens from the output.

        Returns
        -------
        str
        """
        special_ids = set(range(len(self.SPECIAL_TOKENS)))
        chars = []
        for tid in token_ids:
            tid = int(tid)
            if skip_special and tid in special_ids:
                continue
            chars.append(self.id_to_char.get(tid, "?"))
        return "".join(chars)
