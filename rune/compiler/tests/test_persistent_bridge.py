"""
Tests for the HLM <-> persistent-engine bridge.
"""

from __future__ import annotations

from hlm_experiment.models.hlm_125m import HLM125M, HLM125MConfig
from rune.compiler.compiled_model import compile_model
from rune.compiler.ir import OpCode
from rune.engine.hlm_adapter import (
    can_attach,
    can_train,
    pack_hlm_core_params,
    persistent_engine_support_report,
)


def test_memory_augmented_hlm_is_persistent_engine_compatible():
    config = HLM125MConfig()
    report = persistent_engine_support_report(config)

    assert report.supported
    assert any("exact bank scan" in warning for warning in report.warnings)


def test_core_transformer_subset_with_positional_encoding_is_trainable():
    config = HLM125MConfig(
        d_model=8,
        n_layers=2,
        n_heads=2,
        d_ff=16,
        max_seq_len=16,
        memory_enabled=False,
        use_rotor_bias=False,
        use_positional_encoding=True,
    )
    attach = can_attach(config)
    train = can_train(config)
    report = persistent_engine_support_report(config)

    assert attach.supported
    assert train.supported
    assert report.supported


def test_pack_hlm_core_params_matches_engine_layout():
    model = HLM125M(HLM125MConfig(
        vocab_size=32,
        d_model=8,
        n_layers=2,
        n_heads=2,
        d_ff=16,
        max_seq_len=16,
        memory_enabled=False,
        use_rotor_bias=False,
    ))

    params = pack_hlm_core_params(model, batch_size=2, seq_len=8)

    assert params["embed.weight"].shape == (32, 8, 8)
    assert params["layer0.attn.Wq"].shape == (8, 8, 8)
    assert params["layer0.attn.Wo"].shape == (8, 8, 8)
    assert params["layer0.ffn.W1"].shape == (16, 8, 8)
    assert params["layer0.ffn.W2"].shape == (8, 16, 8)
    assert params["final_ln.gamma"].shape == (8, 8)
    assert params["layer0.attn.bq"].shape == (8, 8)
    assert params["pos.rotors"].shape == (16, 8, 8)


def test_compiler_emits_memory_commands_for_memory_enabled_hlm():
    model = HLM125M(HLM125MConfig(
        vocab_size=32,
        d_model=8,
        n_layers=4,
        n_heads=2,
        d_ff=16,
        max_seq_len=16,
        memory_enabled=True,
        memory_n_slots=32,
        memory_top_k=4,
        memory_layers=[1],
        memory_write_layer=3,
        use_rotor_bias=False,
    ))

    compilation = compile_model(
        model,
        batch_size=2,
        seq_len=8,
        with_training=True,
        include_memory=True,
    )

    ops = {node.op for node in compilation.ir_after.live_nodes()}
    assert OpCode.MEAN_POOL_SEQ in ops
    assert OpCode.MEMORY_READ in ops
    assert OpCode.MEMORY_GATE in ops
    assert OpCode.MEMORY_WRITE in ops

    command_ops = {cmd.op for cmd in compilation.execution_plan.commands}
    assert 'kern_mean_pool_seq' in command_ops
    assert 'kern_memory_read' in command_ops
    assert 'kern_memory_gate' in command_ops
    assert 'kern_memory_write' in command_ops


def test_compiled_model_forwards_max_grad_norm_to_engine(monkeypatch):
    from rune.compiler.compiled_model import CompiledModel

    captured = {}

    class DummyEngine:
        def compile_execution_plan(self, execution_plan, params_by_id, *, max_grad_norm=None):
            captured["max_grad_norm"] = max_grad_norm

    monkeypatch.setattr("rune.engine.cuda_engine.CUDAEngine", DummyEngine)

    model = HLM125M(HLM125MConfig(
        vocab_size=32,
        d_model=8,
        n_layers=1,
        n_heads=1,
        d_ff=16,
        max_seq_len=16,
        memory_enabled=False,
        use_rotor_bias=False,
        use_positional_encoding=False,
    ))

    CompiledModel.from_model(
        model,
        batch_size=2,
        seq_len=8,
        use_persistent_engine=True,
        strict_engine=True,
        max_grad_norm=1.5,
    )

    assert captured["max_grad_norm"] == 1.5


def test_execution_plan_tracks_gradient_buffers_per_parameter():
    model = HLM125M(HLM125MConfig(
        vocab_size=32,
        d_model=8,
        n_layers=1,
        n_heads=1,
        d_ff=16,
        max_seq_len=16,
        memory_enabled=False,
        use_rotor_bias=False,
        use_positional_encoding=False,
    ))

    compilation = compile_model(
        model,
        batch_size=2,
        seq_len=8,
        with_training=True,
    )

    plan = compilation.execution_plan
    assert plan.param_grad_buffer_map

    named = dict(model.named_parameters())
    for name, param in named.items():
        pid = id(param)
        assert pid in plan.param_buffer_map
        assert pid in plan.param_grad_buffer_map


def test_dense_clifford_ops_keep_dense_storage_after_grade_pruning():
    model = HLM125M(HLM125MConfig(
        vocab_size=32,
        d_model=8,
        n_layers=1,
        n_heads=1,
        d_ff=16,
        max_seq_len=16,
        memory_enabled=False,
        use_rotor_bias=False,
        use_positional_encoding=False,
    ))

    compilation = compile_model(
        model,
        batch_size=2,
        seq_len=8,
        with_training=True,
    )

    final_norm_specs = [
        spec for spec in compilation.execution_plan.buffer_specs.values()
        if spec.get("name") == "final_norm"
    ]
    assert final_norm_specs
    assert final_norm_specs[0]["storage_components"] == 8
