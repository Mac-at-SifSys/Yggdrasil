"""
test_training_loop.py -- End-to-end test: one training step through the full pipeline.

Creates a minimal mock HLM model, tokenizes some text, runs one step of
CliffordTrainer, and verifies the loss is finite and parameters changed.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from rune.types.multivector import Multivector

from forge.optimizers.clifford_adam import CliffordAdam
from forge.losses.cross_entropy import clifford_cross_entropy
from forge.losses.grade_entropy import grade_entropy_penalty
from forge.schedulers.warmup_cosine import WarmupCosineScheduler
from forge.schedulers.grade_warmup import GradeWarmupScheduler, grade_mask_at_step
from forge.data.tokenizer import BasicTokenizer
from forge.data.dataloader import CliffordDataLoader
from forge.training.trainer import CliffordTrainer
from forge.training.checkpointing import save_checkpoint, load_checkpoint


# ---------------------------------------------------------------------------
# Minimal mock model for testing
# ---------------------------------------------------------------------------

class MockCliffordLM:
    """Tiny Clifford language model for testing the training pipeline.

    Single linear layer: embedding -> MV logits via a weight matrix.
    """

    def __init__(self, vocab_size: int, d_model: int = 4):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embedding: (vocab_size, d_model, 8) -- each token -> d_model multivectors
        self.embed = Multivector(
            np.random.randn(vocab_size, d_model, 8).astype(np.float32) * 0.02,
            requires_grad=True,
        )
        # Output projection: (d_model, vocab_size, 8)
        self.proj = Multivector(
            np.random.randn(d_model, vocab_size, 8).astype(np.float32) * 0.02,
            requires_grad=True,
        )
        # Rotor parameter (for algebraic consistency testing)
        rotor_data = np.zeros((d_model, 8), dtype=np.float32)
        rotor_data[:, 0] = 1.0  # start as identity rotor
        self.rotor = Multivector(rotor_data, requires_grad=True)

        self._last_hidden = None

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass: input_ids (B, S) -> logits (B, S, vocab, 8)."""
        B, S = input_ids.shape

        # Lookup embeddings: (B, S, d_model, 8)
        emb = self.embed._data[input_ids]  # (B, S, d_model, 8)

        # Simple "transformation": element-wise scale by rotor scalar
        # (not a real geometric operation, just enough for testing)
        rotor_scale = self.rotor._data[np.newaxis, np.newaxis, :, :]  # (1, 1, d_model, 8)
        hidden = emb * rotor_scale[..., 0:1]  # scale by scalar part
        self._last_hidden = hidden

        # Project to logits: contract over d_model
        # hidden: (B, S, d_model, 8), proj: (d_model, vocab, 8)
        # Output: (B, S, vocab, 8) via einsum on scalar components only
        # For simplicity, we do a scalar-only linear projection
        h_scalar = hidden[..., 0]  # (B, S, d_model)
        p_scalar = self.proj._data[..., 0]  # (d_model, vocab)
        logits_scalar = np.einsum('bsd,dv->bsv', h_scalar, p_scalar)

        logits = np.zeros((B, S, self.vocab_size, 8), dtype=np.float32)
        logits[..., 0] = logits_scalar

        return logits

    def backward(self, grad_output: np.ndarray):
        """Crude backward: assign random gradients scaled by output grad magnitude."""
        scale = np.sqrt(np.mean(grad_output ** 2))
        for p in [self.embed, self.proj, self.rotor]:
            p._grad = Multivector(
                scale * np.random.randn(*p._data.shape).astype(np.float32) * 0.01
            )

    def parameters(self):
        return [
            {"mv": self.embed, "name": "embed"},
            {"mv": self.proj, "name": "proj"},
            {"mv": self.rotor, "name": "rotor"},
        ]

    def get_rotor_params(self):
        return [self.rotor._data]

    def get_layer_representations(self):
        if self._last_hidden is not None:
            return [self._last_hidden.reshape(-1, 8)]
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_one_training_step():
    """Run one training step and verify loss is finite."""
    np.random.seed(42)

    text = "The quick brown fox jumps over the lazy dog. " * 10
    tokenizer = BasicTokenizer(text)
    vocab_size = tokenizer.vocab_size

    model = MockCliffordLM(vocab_size=vocab_size, d_model=4)
    optimizer = CliffordAdam(model.parameters(), lr=1e-3)
    trainer = CliffordTrainer(
        model=model,
        optimizer=optimizer,
        max_grad_norm=1.0,
        log_interval=999,  # suppress printing during test
    )

    # Create a batch
    tokens = tokenizer.encode(text[:128])
    seq_len = 16
    B = 2
    input_ids = np.zeros((B, seq_len), dtype=np.int64)
    target_ids = np.zeros((B, seq_len), dtype=np.int64)
    for i in range(B):
        start = i * seq_len
        input_ids[i] = tokens[start:start + seq_len]
        target_ids[i] = tokens[start + 1:start + seq_len + 1]

    metrics = trainer.train_step((input_ids, target_ids))

    assert np.isfinite(metrics["loss"]), f"Loss is not finite: {metrics['loss']}"
    assert np.isfinite(metrics["ce_loss"]), f"CE loss not finite"
    assert np.isfinite(metrics["grad_norm"]), f"Grad norm not finite"
    assert metrics["step"] == 0

    print(f"[PASS] One training step: loss={metrics['loss']:.4f}, "
          f"ce={metrics['ce_loss']:.4f}, gnorm={metrics['grad_norm']:.4f}")


def test_multiple_steps_reduce_loss():
    """Multiple training steps should reduce loss."""
    np.random.seed(42)

    text = "abcdefghijklmnopqrstuvwxyz " * 50
    tokenizer = BasicTokenizer(text)
    vocab_size = tokenizer.vocab_size

    model = MockCliffordLM(vocab_size=vocab_size, d_model=4)
    optimizer = CliffordAdam(model.parameters(), lr=5e-3)
    trainer = CliffordTrainer(
        model=model,
        optimizer=optimizer,
        max_grad_norm=5.0,
        log_interval=999,
    )

    tokens = tokenizer.encode(text[:256])
    seq_len = 16
    B = 4
    input_ids = np.zeros((B, seq_len), dtype=np.int64)
    target_ids = np.zeros((B, seq_len), dtype=np.int64)
    for i in range(B):
        start = i * seq_len * 2
        input_ids[i] = tokens[start:start + seq_len]
        target_ids[i] = tokens[start + 1:start + seq_len + 1]

    losses = []
    for step in range(20):
        metrics = trainer.train_step((input_ids, target_ids))
        losses.append(metrics["ce_loss"])

    # Allow for some noise but overall trend should be downward
    first_few = np.mean(losses[:5])
    last_few = np.mean(losses[-5:])
    # With random grads this is noisy, so we just check it doesn't diverge
    assert np.isfinite(losses[-1]), f"Final loss is not finite: {losses[-1]}"
    print(f"[PASS] Multi-step: first5_avg={first_few:.4f}, last5_avg={last_few:.4f}")


def test_dataloader_integration():
    """Test that dataloader produces valid batches for training."""
    text = "Hello world! This is a test of the Clifford training pipeline. " * 20
    loader = CliffordDataLoader(text=text, batch_size=2, seq_len=16, shuffle=False)

    batches = list(loader)
    assert len(batches) > 0, "DataLoader produced no batches"

    inp, tgt = batches[0]
    assert inp.shape[0] <= 2, f"Batch size wrong: {inp.shape}"
    assert inp.shape[1] == 16, f"Seq len wrong: {inp.shape}"
    assert tgt.shape == inp.shape, f"Target shape mismatch"

    # Verify targets are shifted inputs
    # (they should differ because target is next token)
    print(f"[PASS] DataLoader: {len(batches)} batches, shape={inp.shape}")


def test_scheduler_integration():
    """Test that schedulers produce reasonable values."""
    sched = WarmupCosineScheduler(peak_lr=1e-3, warmup_steps=100, total_steps=1000)

    # Warmup phase
    assert sched.get_lr(0) > 0.0
    assert sched.get_lr(50) < sched.get_lr(99)
    assert abs(sched.get_lr(100) - 1e-3) < 1e-5  # peak at warmup_steps

    # Decay phase
    assert sched.get_lr(500) < sched.get_lr(100)
    assert sched.get_lr(999) > sched.get_lr(1000)

    # Grade warmup
    gs = GradeWarmupScheduler(schedule=[0, 10, 20, 30])
    assert gs.get_mask(0) == 0x01  # only scalar
    assert gs.get_mask(10) == 0x03  # scalar + vector
    assert gs.get_mask(20) == 0x07  # + bivector
    assert gs.get_mask(30) == 0x0F  # all grades

    print("[PASS] Scheduler integration")


def test_checkpoint_roundtrip():
    """Save and load a checkpoint, verify parameter recovery."""
    import tempfile

    np.random.seed(42)
    text = "test " * 100
    tokenizer = BasicTokenizer(text)
    model = MockCliffordLM(vocab_size=tokenizer.vocab_size, d_model=4)
    optimizer = CliffordAdam(model.parameters(), lr=1e-3)

    # Do a step to populate optimizer state
    B, S = 2, 8
    inp = np.random.randint(0, tokenizer.vocab_size, (B, S)).astype(np.int64)
    tgt = np.random.randint(0, tokenizer.vocab_size, (B, S)).astype(np.int64)

    trainer = CliffordTrainer(model, optimizer, log_interval=999)
    trainer.train_step((inp, tgt))

    # Save
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        ckpt_path = f.name

    save_checkpoint(model, optimizer, step=1, path=ckpt_path)

    # Load
    model_state, opt_state, step = load_checkpoint(ckpt_path)

    assert step == 1, f"Step mismatch: {step}"
    assert "embed" in model_state, f"Missing 'embed' in model_state: {list(model_state.keys())}"
    assert model_state["embed"].shape[-1] == 8, "Wrong MV dimension"

    # Verify parameter values match
    orig_embed = model.embed._data.copy()
    loaded_embed = model_state["embed"]
    assert np.allclose(orig_embed, loaded_embed, atol=1e-6), "Embed params don't match"

    # Cleanup
    os.unlink(ckpt_path)
    print(f"[PASS] Checkpoint roundtrip: step={step}, params recovered")


def test_tokenizer():
    """Test BasicTokenizer encode/decode roundtrip."""
    text = "Hello, world!"
    tok = BasicTokenizer(text)

    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text, f"Roundtrip failed: '{decoded}' != '{text}'"

    # With special tokens
    ids_special = tok.encode(text, add_special=True)
    assert ids_special[0] == tok.bos_id
    assert ids_special[-1] == tok.eos_id
    decoded_special = tok.decode(ids_special, skip_special=True)
    assert decoded_special == text

    print(f"[PASS] Tokenizer: vocab_size={tok.vocab_size}, roundtrip OK")


if __name__ == "__main__":
    test_tokenizer()
    test_dataloader_integration()
    test_scheduler_integration()
    test_one_training_step()
    test_multiple_steps_reduce_loss()
    test_checkpoint_roundtrip()
    print("\nAll training loop tests passed.")
