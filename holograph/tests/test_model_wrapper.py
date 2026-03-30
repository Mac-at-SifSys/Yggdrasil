"""
Tests for the top-level HolographLM wrapper and agent loading path.
"""

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
for name in ["sif", "sif.agents"]:
    sys.modules.pop(name, None)

import tempfile

import numpy as np

from holograph.model import HolographLM
from hlm_experiment.models.hlm_125m import HLM125MConfig
from sif.agents.hlm_agent import HLMAgent, HLMConfig as AgentConfig


def _tiny_config():
    return HLM125MConfig(
        vocab_size=32,
        d_model=4,
        n_layers=1,
        n_heads=2,
        d_ff=8,
        max_seq_len=8,
        use_rotor_bias=False,
        memory_enabled=False,
    )


def test_holographlm_roundtrip_and_helpers():
    np.random.seed(7)

    model = HolographLM(config=_tiny_config())
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = model.save(tmpdir)
        loaded = HolographLM.load(save_path)

        encoding = loaded.encode("hello")
        embedding = loaded.embed("hello")
        generation = loaded.generate("hello", max_tokens=2, temperature=0.0)
        attn = loaded.attention_map("hello world")

        assert len(encoding.components) == 8
        assert embedding.shape == (8,)
        assert isinstance(generation.text, str)
        assert generation.token_count == 2
        assert len(attn) == len(attn[0])


def test_hlm_agent_loads_holographlm_checkpoint():
    np.random.seed(9)

    model = HolographLM(config=_tiny_config())
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir)
        agent = HLMAgent(AgentConfig(model_path=tmpdir))
        response = agent.generate("hello", max_tokens=2, temperature=0.0)

        assert agent.available
        assert response.model_id == "hlm-native"
        assert response.multivector_meta is not None
