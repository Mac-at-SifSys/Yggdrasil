"""
cross_entropy.py -- Cross-entropy loss on the scalar (grade-0) projection.

The HLM outputs multivector logits.  We project to grade-0 (scalar) and then
apply standard softmax cross-entropy, giving us a smooth bridge between the
Clifford representation and conventional classification targets.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
from rune.backend import xp
from rune.types.multivector import Multivector


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    shifted = logits - xp.max(logits, axis=-1, keepdims=True)
    exp_vals = xp.exp(shifted)
    return exp_vals / xp.sum(exp_vals, axis=-1, keepdims=True)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax along the last axis."""
    shifted = logits - xp.max(logits, axis=-1, keepdims=True)
    return shifted - xp.log(xp.sum(xp.exp(shifted), axis=-1, keepdims=True))


def clifford_cross_entropy(
    logits_mv: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Cross-entropy loss on the scalar projection of multivector logits.

    Parameters
    ----------
    logits_mv : np.ndarray, shape (batch, vocab_size, 8)
        Multivector logits -- one MV per vocabulary entry per sample.
        We extract the grade-0 (index 0) component as the logit.
    targets : np.ndarray, shape (batch,), dtype int
        Ground-truth token indices.

    Returns
    -------
    float
        Mean cross-entropy loss over the batch.
    """
    # Handle Multivector objects
    if isinstance(logits_mv, Multivector):
        logits_mv = logits_mv._data

    # Extract scalar (grade-0) component -> shape (batch, vocab_size)
    if logits_mv.ndim == 3 and logits_mv.shape[-1] == 8:
        scalar_logits = logits_mv[..., 0]  # (batch, vocab_size)
    elif logits_mv.ndim == 2:
        # Already scalar logits
        scalar_logits = logits_mv
    else:
        raise ValueError(
            f"Expected logits_mv shape (batch, vocab, 8) or (batch, vocab), "
            f"got {logits_mv.shape}"
        )

    batch_size = scalar_logits.shape[0]
    log_probs = _log_softmax(scalar_logits)

    # Gather the log-prob at the target index for each sample
    targets = targets.astype(xp.int64)
    nll = -log_probs[xp.arange(batch_size), targets]

    return float(xp.mean(nll))


def clifford_cross_entropy_with_grad(
    logits_mv: np.ndarray,
    targets: np.ndarray,
) -> tuple:
    """Cross-entropy loss with gradient w.r.t. the full MV logits array.

    Returns
    -------
    (loss: float, grad: np.ndarray of same shape as logits_mv)
    """
    if isinstance(logits_mv, Multivector):
        logits_mv = logits_mv._data

    if logits_mv.ndim == 3 and logits_mv.shape[-1] == 8:
        scalar_logits = logits_mv[..., 0]
        is_multivector = True
    elif logits_mv.ndim == 2:
        scalar_logits = logits_mv
        is_multivector = False
    else:
        raise ValueError(
            f"Expected (batch, vocab, 8) or (batch, vocab), got {logits_mv.shape}"
        )

    batch_size = scalar_logits.shape[0]
    probs = _softmax(scalar_logits)
    log_probs = xp.log(probs + 1e-12)

    targets = targets.astype(xp.int64)
    nll = -log_probs[xp.arange(batch_size), targets]
    loss = float(xp.mean(nll))

    # Gradient w.r.t. scalar_logits: (probs - one_hot) / batch_size
    one_hot = xp.zeros_like(probs)
    one_hot[xp.arange(batch_size), targets] = 1.0
    d_scalar = (probs - one_hot) / batch_size  # (batch, vocab)

    # Map back to full MV grad: only the grade-0 slot is nonzero
    if is_multivector:
        grad = xp.zeros_like(logits_mv)
        grad[..., 0] = d_scalar
    else:
        grad = d_scalar

    return loss, grad
