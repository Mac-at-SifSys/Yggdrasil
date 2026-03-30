#!/usr/bin/env python3
"""GeoFormer / MoE-HLM Chat Interface.

Interactive chat window for testing checkpoints from either model.

Usage:
    python geoformer/chat.py --checkpoint D:/Downloads/step_005000.pt
    python geoformer/chat.py --checkpoint D:/Downloads/step_005000.pt --port 7860
"""

import sys
import json
import argparse
import logging
import pickle
import io
import threading
from pathlib import Path
from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, Response, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("chat")

# ── Globals ──────────────────────────────────────────────────────────────
MODEL = None
TOKENIZER = None
CONFIG = None
DEVICE = None
STEP = None
MODEL_TYPE = None  # "geoformer" or "moe_hlm"

BLADE_NAMES = ["narrative", "causation", "affect", "wisdom",
               "relations", "ecology", "epistemics", "temporal"]


# ── Pickle compatibility for self-contained training scripts ─────────────

class _UnpicklerWithFallback(pickle.Unpickler):
    """Handle configs pickled from __main__ in Colab/Kaggle notebooks."""

    def find_class(self, module, name):
        # Redirect __main__.MoEHLMConfig to our local version
        if name == "MoEHLMConfig":
            return _MoEHLMConfigCompat
        if name == "MoEHLM":
            from geoformer.moe_hlm.model import MoEHLM
            return MoEHLM
        return super().find_class(module, name)


@dataclass
class _MoEHLMConfigCompat:
    """Compatible config that can unpickle from any training script variant."""
    d_model: int = 1536
    n_layers: int = 16
    n_heads: int = 24
    d_head: int = 64
    n_experts: int = 8
    top_k: int = 2
    router_aux_loss_weight: float = 0.01
    n_blades: int = 8
    d_blade: int = 128
    n_geometric_rounds: int = 2
    expert_d_ffn: int = 640
    vocab_size: int = 151936
    max_seq_len: int = 1024
    tou_n_primitives: int = 1486
    tou_d_prim: int = 128
    tou_every_n_layers: int = 4
    attn_dropout: float = 0.0
    embed_dropout: float = 0.1
    rope_theta: float = 10000.0
    gradient_checkpointing: bool = False  # off for inference
    init_std: float = 0.02

    @property
    def tou_attn_layers(self):
        return list(range(self.tou_every_n_layers - 1,
                          self.n_layers, self.tou_every_n_layers))


def _load_checkpoint(path: str, device: str):
    """Load checkpoint with fallback unpickler for __main__ configs."""
    # First try normal load
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        return ckpt
    except (ModuleNotFoundError, AttributeError) as e:
        log.info(f"Standard load failed ({e}), trying compatibility unpickler...")

    # Fallback: custom unpickler
    with open(path, "rb") as f:
        ckpt = _UnpicklerWithFallback(f).load()
    # Move tensors to target device
    def _to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_to_device(x) for x in obj)
        return obj
    return _to_device(ckpt)


# ── Model loading ────────────────────────────────────────────────────────

def _detect_model_type(ckpt):
    """Detect whether checkpoint is GeoFormer or MoE-HLM."""
    # Check by config key
    if "model_config" in ckpt:
        return "geoformer"
    if "config" in ckpt:
        cfg = ckpt["config"]
        if hasattr(cfg, "n_experts"):
            return "moe_hlm"
    # Check by state dict keys
    sd = ckpt.get("model_state_dict", {})
    for key in sd:
        if "moe" in key or "router" in key or "experts" in key:
            return "moe_hlm"
        if "blade_projector" in key or "blade_collapse" in key:
            return "geoformer"
    return "moe_hlm"  # default for newer checkpoints


def _load_moe_hlm(ckpt, device):
    """Load MoE-HLM model from checkpoint."""
    from geoformer.moe_hlm.config import MoEHLMConfig
    from geoformer.moe_hlm.model import MoEHLM

    # Extract config — may be our compat class or the real one
    raw_config = ckpt["config"]

    # Build a proper MoEHLMConfig from whatever we got
    config = MoEHLMConfig(
        d_model=getattr(raw_config, "d_model", 1536),
        n_layers=getattr(raw_config, "n_layers", 16),
        n_heads=getattr(raw_config, "n_heads", 24),
        d_head=getattr(raw_config, "d_head", 64),
        n_experts=getattr(raw_config, "n_experts", 8),
        top_k=getattr(raw_config, "top_k", 2),
        n_blades=getattr(raw_config, "n_blades", 8),
        d_blade=getattr(raw_config, "d_blade", 128),
        n_geometric_rounds=getattr(raw_config, "n_geometric_rounds", 2),
        expert_d_ffn=getattr(raw_config, "expert_d_ffn", 640),
        vocab_size=getattr(raw_config, "vocab_size", 151936),
        max_seq_len=getattr(raw_config, "max_seq_len", 1024),
        tou_n_primitives=getattr(raw_config, "tou_n_primitives", 1486),
        tou_d_prim=getattr(raw_config, "tou_d_prim", 128),
        tou_every_n_layers=getattr(raw_config, "tou_every_n_layers", 4),
        embed_dropout=0.0,  # no dropout at inference
        gradient_checkpointing=False,
    )

    model = MoEHLM(config).to(device)

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        log.warning(f"  Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        log.warning(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    step = ckpt.get("step", ckpt.get("global_step", "unknown"))
    return model, config, step


def _load_geoformer(ckpt, device):
    """Load GeoFormer model from checkpoint."""
    from geoformer.config import GeoFormerConfig
    from geoformer.model import GeoFormer

    config = ckpt["model_config"]
    model = GeoFormer(config).to(device)

    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    step = ckpt.get("global_step", ckpt.get("step", "unknown"))
    return model, config, step


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint, auto-detecting type."""
    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = _load_checkpoint(checkpoint_path, device)

    model_type = _detect_model_type(ckpt)
    log.info(f"  Detected model type: {model_type}")

    if model_type == "moe_hlm":
        model, config, step = _load_moe_hlm(ckpt, device)
    else:
        model, config, step = _load_geoformer(ckpt, device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Loaded step {step}, {n_params:,} params on {device}")
    return model, config, step, model_type


def load_tokenizer(config):
    from transformers import AutoTokenizer
    # Qwen2.5 vocab = 151936, Qwen2 = 50304
    vocab_size = getattr(config, "vocab_size", 50304)
    if vocab_size > 100_000:
        tok_name = "Qwen/Qwen2.5-0.5B"
    else:
        tok_name = "Qwen/Qwen2.5-0.5B"  # same tokenizer, different vocab truncation
    log.info(f"  Loading tokenizer: {tok_name} (vocab_size={vocab_size})")
    return AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)


# ── Generation (streaming via generator) ─────────────────────────────────

@torch.no_grad()
def generate_stream(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
):
    """Yield tokens one at a time for streaming."""
    tokens = TOKENIZER.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    max_seq = getattr(CONFIG, "max_seq_len", 2048)
    generated_ids = []
    expert_usage = []  # Track which experts fire (MoE-HLM)

    for _ in range(max_new_tokens):
        # Crop to context window
        idx_cond = input_ids if input_ids.shape[1] <= max_seq else \
            input_ids[:, -max_seq:]

        if MODEL_TYPE == "geoformer":
            outputs = MODEL(idx_cond, return_blade_activations=True)
        else:
            outputs = MODEL(idx_cond)

        logits = outputs["logits"][:, -1, :]  # (1, vocab)

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            for prev_id in set(generated_ids[-64:]):
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= repetition_penalty
                else:
                    logits[0, prev_id] *= repetition_penalty

        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                indices_to_remove = remove.scatter(1, sorted_indices, remove)
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        generated_ids.append(token_id)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Get blade/expert activations for the UI
        blades = None
        if MODEL_TYPE == "geoformer" and outputs.get("blade_activations"):
            acts = outputs["blade_activations"][-1][:, -1, :]  # (1, 8)
            blades = acts[0].cpu().tolist()
        elif MODEL_TYPE == "moe_hlm":
            # For MoE-HLM, show router distribution from last layer
            blades = _get_expert_usage(idx_cond)

        # Decode just this token
        token_text = TOKENIZER.decode([token_id], skip_special_tokens=True)

        # Check for EOS
        if token_id == TOKENIZER.eos_token_id:
            break

        yield token_text, blades


@torch.no_grad()
def _get_expert_usage(input_ids):
    """Get expert routing weights for the last token from the last MoE layer.

    Returns a list of 8 floats (one per expert) showing routing probability.
    """
    try:
        last_block = MODEL.blocks[-1]
        x = MODEL.token_embed(input_ids)
        x = MODEL.embed_dropout(x)
        # We need the hidden state at the last layer's MoE input
        # Quick approach: just run the router on the final norm output
        # This is approximate but avoids a full re-forward
        normed = last_block.norm2(
            last_block.norm1(x)  # approximate — good enough for visualization
        )
        logits = last_block.moe.router.gate(normed[:, -1:, :])  # (1, 1, n_experts)
        probs = F.softmax(logits, dim=-1)[0, 0]  # (n_experts,)
        return probs.cpu().tolist()
    except Exception:
        return None


# ── Flask App ────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GeoFormer Chat</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
    --user-bg: #1c2333; --bot-bg: #0d1117;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    height: 100vh; display: flex; flex-direction: column;
  }

  /* Header */
  .header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 12px 20px; display: flex; align-items: center; gap: 16px;
    flex-shrink: 0;
  }
  .header h1 { font-size: 18px; font-weight: 600; }
  .header .meta { font-size: 12px; color: var(--text-dim); }
  .header .step-badge {
    background: #1f6feb33; color: var(--accent); padding: 2px 10px;
    border-radius: 12px; font-size: 12px; font-weight: 500;
  }
  .header .model-badge {
    background: #23863633; color: #3fb950; padding: 2px 10px;
    border-radius: 12px; font-size: 12px; font-weight: 500;
  }

  /* Chat area */
  .chat-container {
    flex: 1; overflow-y: auto; padding: 20px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .message {
    max-width: 85%; padding: 12px 16px; border-radius: 12px;
    line-height: 1.6; white-space: pre-wrap; word-break: break-word;
    font-size: 14px;
  }
  .message.user {
    align-self: flex-end; background: var(--user-bg);
    border: 1px solid var(--border);
  }
  .message.bot {
    align-self: flex-start; background: var(--bot-bg);
    border: 1px solid var(--border);
  }
  .message .label {
    font-size: 11px; color: var(--text-dim); margin-bottom: 4px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }

  /* Blade/Expert bar */
  .blade-bar {
    display: flex; gap: 3px; margin-top: 8px; align-items: flex-end;
    height: 32px;
  }
  .blade-bar .blade {
    flex: 1; border-radius: 2px 2px 0 0; min-width: 0;
    position: relative; transition: height 0.15s ease;
  }
  .blade-bar .blade-label {
    font-size: 8px; color: var(--text-dim); text-align: center;
    margin-top: 2px; overflow: hidden; text-overflow: ellipsis;
  }
  .blade-colors {
    --b0: #f97316; --b1: #ef4444; --b2: #ec4899; --b3: #a855f7;
    --b4: #6366f1; --b5: #06b6d4; --b6: #22c55e; --b7: #eab308;
  }

  /* Input area */
  .input-area {
    background: var(--surface); border-top: 1px solid var(--border);
    padding: 16px 20px; flex-shrink: 0;
  }
  .input-row { display: flex; gap: 10px; max-width: 900px; margin: 0 auto; }
  .input-row textarea {
    flex: 1; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 14px; font-size: 14px; font-family: inherit;
    resize: none; outline: none; min-height: 44px; max-height: 120px;
  }
  .input-row textarea:focus { border-color: var(--accent); }
  .input-row button {
    background: var(--accent); color: #fff; border: none;
    border-radius: 8px; padding: 0 20px; font-size: 14px;
    cursor: pointer; font-weight: 500; white-space: nowrap;
  }
  .input-row button:hover { background: #4393e6; }
  .input-row button:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Settings drawer */
  .settings-toggle {
    background: none; color: var(--text-dim); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; cursor: pointer; font-size: 12px;
  }
  .settings-panel {
    display: none; padding: 8px 20px; background: var(--surface);
    border-top: 1px solid var(--border);
  }
  .settings-panel.open { display: flex; gap: 20px; flex-wrap: wrap; }
  .setting { display: flex; flex-direction: column; gap: 2px; }
  .setting label { font-size: 11px; color: var(--text-dim); }
  .setting input {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 4px 8px; width: 80px; font-size: 13px;
  }

  /* Scrollbar */
  .chat-container::-webkit-scrollbar { width: 6px; }
  .chat-container::-webkit-scrollbar-track { background: transparent; }
  .chat-container::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .generating-indicator {
    display: inline-block; width: 6px; height: 14px;
    background: var(--accent); animation: blink 0.8s infinite;
    vertical-align: text-bottom; margin-left: 1px;
  }
  @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0.2; } }

  /* Token counter */
  .token-counter {
    font-size: 11px; color: var(--text-dim); margin-top: 4px;
    text-align: right;
  }
</style>
</head>
<body>

<div class="header">
  <h1 id="modelTitle">HLM Chat</h1>
  <span class="step-badge" id="stepBadge">loading...</span>
  <span class="model-badge" id="modelBadge"></span>
  <span class="meta" id="modelMeta"></span>
  <div style="flex:1"></div>
  <button class="settings-toggle" onclick="toggleSettings()">Settings</button>
</div>

<div class="settings-panel" id="settingsPanel">
  <div class="setting">
    <label>Temperature</label>
    <input type="number" id="temperature" value="0.8" step="0.05" min="0" max="2">
  </div>
  <div class="setting">
    <label>Max Tokens</label>
    <input type="number" id="maxTokens" value="256" step="32" min="1" max="2048">
  </div>
  <div class="setting">
    <label>Top-K</label>
    <input type="number" id="topK" value="50" step="5" min="0" max="200">
  </div>
  <div class="setting">
    <label>Top-P</label>
    <input type="number" id="topP" value="0.9" step="0.05" min="0" max="1">
  </div>
  <div class="setting">
    <label>Rep. Penalty</label>
    <input type="number" id="repPenalty" value="1.1" step="0.05" min="1" max="2">
  </div>
</div>

<div class="chat-container" id="chat"></div>

<div class="input-area">
  <div class="input-row">
    <textarea id="userInput" rows="1" placeholder="Type a message..."
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send();}"></textarea>
    <button id="sendBtn" onclick="send()">Send</button>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
let barLabels = [];
let barColors = ['#f97316','#ef4444','#ec4899','#a855f7','#6366f1','#06b6d4','#22c55e','#eab308'];

// Auto-resize textarea
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
});

// Load model info
fetch('/api/info').then(r=>r.json()).then(info => {
  document.getElementById('stepBadge').textContent = 'Step ' + info.step;
  document.getElementById('modelBadge').textContent = info.model_type;
  document.getElementById('modelMeta').textContent =
    info.params + ' params | ' + info.device + ' | ctx ' + info.max_seq_len;

  if (info.model_type === 'moe_hlm') {
    document.getElementById('modelTitle').textContent = 'MoE-HLM Chat';
    barLabels = info.blade_names.map((n, i) => 'E' + i);  // Expert 0..7
  } else {
    document.getElementById('modelTitle').textContent = 'GeoFormer Chat';
    barLabels = info.blade_names.map(n => n.slice(0, 4));
  }
});

function toggleSettings() {
  document.getElementById('settingsPanel').classList.toggle('open');
}

function makeBladeBar(acts, isExpert) {
  if (!acts) return '';
  const max = Math.max(...acts.map(Math.abs), 0.001);
  let html = '<div class="blade-bar blade-colors">';
  for (let i = 0; i < acts.length; i++) {
    const h = Math.max(2, (Math.abs(acts[i]) / max) * 28);
    const label = isExpert ? ('Expert ' + i) : (barLabels[i] || 'b' + i);
    html += `<div class="blade" style="height:${h}px;background:${barColors[i % barColors.length]}"
      title="${label}: ${acts[i].toFixed(4)}"></div>`;
  }
  html += '</div>';
  html += '<div style="display:flex;gap:3px">';
  for (let i = 0; i < acts.length; i++) {
    const lbl = isExpert ? ('E' + i) : (barLabels[i] || 'b' + i);
    html += `<div class="blade-label" style="flex:1;font-size:7px;color:#8b949e;text-align:center">${lbl}</div>`;
  }
  html += '</div>';
  return html;
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

let generating = false;
let modelType = 'moe_hlm';

async function send() {
  const text = input.value.trim();
  if (!text || generating) return;

  // User message
  const userDiv = document.createElement('div');
  userDiv.className = 'message user';
  userDiv.innerHTML = '<div class="label">You</div>' + escapeHtml(text);
  chat.appendChild(userDiv);
  input.value = '';
  input.style.height = 'auto';
  scrollToBottom();

  // Bot message placeholder
  const botDiv = document.createElement('div');
  botDiv.className = 'message bot';
  const modelLabel = modelType === 'moe_hlm' ? 'MoE-HLM' : 'GeoFormer';
  botDiv.innerHTML = '<div class="label">' + modelLabel + '</div><span id="botText"></span><span class="generating-indicator"></span>';
  chat.appendChild(botDiv);
  scrollToBottom();

  generating = true;
  sendBtn.disabled = true;

  const params = {
    prompt: text,
    max_new_tokens: parseInt(document.getElementById('maxTokens').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    top_k: parseInt(document.getElementById('topK').value),
    top_p: parseFloat(document.getElementById('topP').value),
    repetition_penalty: parseFloat(document.getElementById('repPenalty').value),
  };

  let tokenCount = 0;

  try {
    const resp = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    let lastBlades = null;
    let buffer = '';

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          try {
            const obj = JSON.parse(data);
            if (obj.token) {
              fullText += obj.token;
              tokenCount++;
              const textSpan = botDiv.querySelector('#botText') || botDiv.querySelector('.bot-text');
              if (textSpan) textSpan.textContent = fullText;
              scrollToBottom();
            }
            if (obj.blades) lastBlades = obj.blades;
          } catch(e) {}
        }
      }
    }

    // Remove cursor, add blade bar + token count
    const cursor = botDiv.querySelector('.generating-indicator');
    if (cursor) cursor.remove();
    const textSpan = botDiv.querySelector('#botText');
    if (textSpan) textSpan.removeAttribute('id');
    if (lastBlades) {
      const isExpert = modelType === 'moe_hlm';
      botDiv.insertAdjacentHTML('beforeend', makeBladeBar(lastBlades, isExpert));
    }
    botDiv.insertAdjacentHTML('beforeend',
      `<div class="token-counter">${tokenCount} tokens</div>`);

  } catch(err) {
    botDiv.innerHTML += '<br><span style="color:#f85149">Error: ' + escapeHtml(err.message) + '</span>';
  }

  generating = false;
  sendBtn.disabled = false;
  scrollToBottom();
  input.focus();
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// Get model type for UI labels
fetch('/api/info').then(r=>r.json()).then(info => { modelType = info.model_type; });

input.focus();
</script>
</body>
</html>"""


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/info")
def info():
    n_params = sum(p.numel() for p in MODEL.parameters())
    if n_params >= 1e9:
        params_str = f"{n_params / 1e9:.1f}B"
    else:
        params_str = f"{n_params / 1e6:.0f}M"
    return jsonify({
        "step": STEP,
        "params": params_str,
        "device": str(DEVICE),
        "max_seq_len": getattr(CONFIG, "max_seq_len", 2048),
        "n_layers": getattr(CONFIG, "n_layers", 0),
        "model_type": MODEL_TYPE,
        "blade_names": BLADE_NAMES,
        "n_experts": getattr(CONFIG, "n_experts", 0),
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "empty prompt"}), 400

    max_new_tokens = data.get("max_new_tokens", 256)
    temperature = data.get("temperature", 0.8)
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.9)
    rep_penalty = data.get("repetition_penalty", 1.1)

    def event_stream():
        for token_text, blades in generate_stream(
            prompt, max_new_tokens, temperature, top_k, top_p, rep_penalty
        ):
            payload = {"token": token_text}
            if blades:
                payload["blades"] = blades
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    global MODEL, TOKENIZER, CONFIG, DEVICE, STEP, MODEL_TYPE

    p = argparse.ArgumentParser(description="GeoFormer / MoE-HLM Chat Interface")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--device", default=None,
                   help="Device (default: cuda if available)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    if args.device is None:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device

    MODEL, CONFIG, STEP, MODEL_TYPE = load_model(args.checkpoint, DEVICE)
    TOKENIZER = load_tokenizer(CONFIG)

    log.info(f"Starting chat server on http://{args.host}:{args.port}")
    import webbrowser
    webbrowser.open(f"http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
