#!/usr/bin/env python3
"""Lightweight CPU chat for TokenizedHLMv9 checkpoints.

Loads only model weights (skips 29GB memory banks).
Usage: python chat_cpu.py --checkpoint path/to/step_012000.pt
"""

import sys
import os
import argparse

# Add parent dir so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

# Import model and config from the training script
from TokenizedHLMv9_Colab import TokenizedHLMv9, TokenizedHLMv9Config

def load_model_cpu(checkpoint_path):
    """Load model on CPU, skip memory bank buffers."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = ckpt.get('config', TokenizedHLMv9Config())

    # Override memory to tiny sizes so model init doesn't eat RAM
    config.mem_3_slots = 1000
    config.mem_4_slots = 1000
    if hasattr(config, 'mem_2_slots'):
        config.mem_2_slots = 1000
    if hasattr(config, 'mem_8_slots'):
        config.mem_8_slots = 1000
    if hasattr(config, 'mem_16_slots'):
        config.mem_16_slots = 1000
    if hasattr(config, 'blade_mem_2_slots'):
        config.blade_mem_2_slots = 100
    if hasattr(config, 'blade_mem_4_slots'):
        config.blade_mem_4_slots = 100
    if hasattr(config, 'blade_mem_8_slots'):
        config.blade_mem_8_slots = 100
    if hasattr(config, 'blade_mem_16_slots'):
        config.blade_mem_16_slots = 100

    print(f"Building model ({config.n_layers}L, d={config.d_model})...")
    model = TokenizedHLMv9(config)

    # Filter checkpoint — skip memory buffers and mismatched shapes
    state_dict = ckpt['model_state_dict']
    model_state = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in state_dict.items():
        if 'bank' in k or 'counts' in k:
            skipped += 1
            continue
        if k in model_state and model_state[k].shape != v.shape:
            skipped += 1
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} params, skipped {skipped} memory buffers")
    if missing:
        # Filter out expected missing keys (memory banks)
        real_missing = [k for k in missing if 'bank' not in k and 'counts' not in k]
        if real_missing:
            print(f"Warning: {len(real_missing)} unexpected missing keys:")
            for k in real_missing[:5]:
                print(f"  {k}")

    step = ckpt.get('step', 0)
    loss = ckpt.get('loss', 0)
    is_sft = ckpt.get('sft', False) or 'sft' in os.path.basename(checkpoint_path).lower()
    print(f"Checkpoint: step {step}, loss {loss:.4f}, SFT={'yes' if is_sft else 'no'}")

    model.eval()
    return model, config, is_sft


def setup_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    return tokenizer


@torch.no_grad()
def generate(model, input_ids, tokenizer, max_new_tokens=200, temperature=0.8,
             top_k=50, top_p=0.9, rep_penalty=1.2):
    """Simple autoregressive generation on CPU with repetition penalty."""
    # Precompute stop token IDs from end/user markers
    stop_strings = ["<|end|>", "<|user|>", "<|system|>"]
    stop_ids = set()
    for s in stop_strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            stop_ids.add(ids[0])

    generated = []
    for i in range(max_new_tokens):
        idx_cond = input_ids[:, -model.config.max_seq_len:]
        result = model.forward(idx_cond)
        logits = result["logits"][:, -1, :] / temperature

        # Repetition penalty
        if rep_penalty != 1.0:
            prev_tokens = input_ids[0].tolist()
            for token_id in set(prev_tokens):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= rep_penalty
                else:
                    logits[0, token_id] *= rep_penalty

        # Top-k
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token[0].item()

        # Stop at end/user tokens
        if token_id in stop_ids:
            break

        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated.append(token_id)

        # Print token as it's generated
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        print(token_str, end='', flush=True)

    return input_ids


CHAT_USER_PREFIX = "\n<|user|>\n"
CHAT_ASST_PREFIX = "\n<|assistant|>\n"
CHAT_END = "\n<|end|>\n"

def format_sft_prompt(tokenizer, user_text, history=None):
    """Format prompt using SFT chat template — matches training encoding exactly."""
    parts = []
    if history:
        for user_turn, assistant_turn in history:
            parts.append(f"{CHAT_USER_PREFIX}{user_turn}{CHAT_END}")
            parts.append(f"{CHAT_ASST_PREFIX}{assistant_turn}{CHAT_END}")
    parts.append(f"{CHAT_USER_PREFIX}{user_text}{CHAT_END}")
    parts.append(CHAT_ASST_PREFIX)
    text = "".join(parts)
    # add_special_tokens=False — matches how SFT training encoded text
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Clamp any out-of-range token IDs to vocab size
    ids = [min(i, 50303) for i in ids]
    return torch.tensor([ids])


def extract_response(tokenizer, output_ids, input_len):
    """Decode only the newly generated tokens."""
    new_tokens = output_ids[0][input_len:].tolist()
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    # Stop at chat delimiter tokens
    for stop in ['\n<|end|>', '<|end|>', '\n<|user|>', '<|user|>', '<|system|>']:
        if stop in text:
            text = text[:text.index(stop)]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description='CPU Chat for TokenizedHLMv9')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt file')
    parser.add_argument('--max-tokens', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--rep-penalty', type=float, default=1.3, help='Repetition penalty (1.0=off, 1.3=mild, 1.5=strong)')
    parser.add_argument('--pretrain', action='store_true', help='Force pretrain mode (raw completion)')
    args = parser.parse_args()

    global tokenizer
    print("Loading tokenizer...")
    tokenizer = setup_tokenizer()

    model, config, is_sft = load_model_cpu(args.checkpoint)
    use_sft_mode = is_sft and not args.pretrain

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count/1e6:.1f}M")
    print(f"\n{'='*60}")
    if use_sft_mode:
        print("TokenizedHLMv9 Chat (SFT mode — instruction following)")
        print("Type your message. Type 'quit' to exit, 'reset' to clear history.")
    else:
        print("TokenizedHLMv9 Chat (pretrain mode — text completion)")
        print("Type a prompt and the model will continue it.")
        print("Type 'quit' to exit.")
    print(f"{'='*60}\n")

    history = []

    while True:
        try:
            prompt = input("YOU: " if use_sft_mode else ">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if prompt.strip().lower() in ('quit', 'exit', 'q'):
            break
        if prompt.strip().lower() == 'reset':
            history = []
            print("[History cleared]\n")
            continue
        if not prompt.strip():
            continue

        if use_sft_mode:
            input_ids = format_sft_prompt(tokenizer, prompt, history)
            input_len = input_ids.shape[1]
            print(f"\n[{input_len} tokens, generating...]\n")
            print("HLM-v9: ", end='', flush=True)
            output_ids = generate(model, input_ids, tokenizer, max_new_tokens=args.max_tokens,
                                  temperature=args.temperature, rep_penalty=args.rep_penalty)
            response = extract_response(tokenizer, output_ids, input_len)
            print(f"\n")
            history.append((prompt, response))
        else:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            print(f"\n[{len(input_ids[0])} input tokens, generating up to {args.max_tokens}...]\n")
            generate(model, input_ids, tokenizer, max_new_tokens=args.max_tokens,
                     temperature=args.temperature, rep_penalty=args.rep_penalty)
            print("\n")


if __name__ == '__main__':
    main()
