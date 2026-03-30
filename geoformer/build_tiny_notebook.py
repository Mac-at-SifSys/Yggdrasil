#!/usr/bin/env python3
"""Build TinyHLMMoE_Colab.ipynb from TinyHLMMoE_Colab.py"""

import json, re, pathlib

SRC = pathlib.Path(r"C:\Sif v3\Sif-v4\geoformer\TinyHLMMoE_Colab.py")
DST = pathlib.Path(r"C:\Sif v3\Sif-v4\geoformer\TinyHLMMoE_Colab.ipynb")

lines = SRC.read_text(encoding="utf-8").splitlines()

# Find cell boundary lines (the "# ===...===" separator lines)
# Each cell starts at "# Cell N:" header, preceded by the separator
CELL_HEADER = re.compile(r"^# Cell \d+:")
MAIN_HEADER = re.compile(r"^# ={20,}\s*$")

# Split into sections by locating separator+header pairs
sections = []  # (title, [lines])
current_title = None
current_lines = []

i = 0
while i < len(lines):
    line = lines[i]
    # Look for "# =======" line
    if MAIN_HEADER.match(line):
        # Check if next two lines are "# Cell N: ..." and "# ======="
        if i + 2 < len(lines) and CELL_HEADER.match(lines[i+1]) and MAIN_HEADER.match(lines[i+2]):
            # Save current section
            if current_title is not None:
                sections.append((current_title, current_lines))
            current_title = lines[i+1].strip("# ").strip()
            current_lines = []
            i += 3  # skip separator, header, separator
            # skip blank line after separator if present
            if i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
        elif i + 2 < len(lines) and lines[i+1].strip() == "# Main" and MAIN_HEADER.match(lines[i+2]):
            # Main/argparse section — save current, stop collecting
            if current_title is not None:
                sections.append((current_title, current_lines))
            current_title = None
            current_lines = []
            i += 3
            continue
    current_lines.append(line)
    i += 1

# Save last section if not Main
if current_title is not None and current_title != "Main":
    sections.append((current_title, current_lines))

def make_code_cell(source_lines):
    # Join lines, strip trailing blanks
    src = "\n".join(source_lines).rstrip() + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src
    }

def make_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text
    }

cells = []

# Title markdown
cells.append(make_markdown_cell(
    "# TinyHLMMoE — Tiny Holographic Language Model with Mixture of Experts\n"
    "\n"
    "~76M total params, ~48M active per forward pass.  \n"
    "Blade-routed MoE: 8 blade groups × 8 experts, top-2 active per blade.  \n"
    "No memory banks. No ToU injection. Clean, fast, iterable.\n"
    "\n"
    "**Expected training time:** ~2.5h on A100 at 150K tok/s for 10K steps (~1.3B tokens)\n"
    "\n"
    "---\n"
    "**Options:**\n"
    "- **Option A (Full Pipeline):** Run all cells → scroll to the bottom and run Option A\n"
    "- **Option B (SFT only):** Run all cells → run Option B (loads best.pt from Drive)\n"
    "- **Option C (Chat only):** Run all cells → run Option C (loads lora_final.pt or best.pt)\n"
))

# Definition cells (Cells 1-11)
for title, src_lines in sections:
    # Skip empty sections or Main
    if not src_lines or title == "Main":
        continue
    # Add markdown divider
    cells.append(make_markdown_cell(f"## {title}"))
    cells.append(make_code_cell(src_lines))

# ── Option A: Full Pipeline ──────────────────────────────────────────────────
cells.append(make_markdown_cell(
    "---\n"
    "## ▶ Option A — Full Pipeline: Pretrain → LoRA SFT → Chat\n"
    "\n"
    "Runs all 10K pretraining steps (~2.5h), then 2K LoRA SFT steps, then opens chat.\n"
    "Checkpoints saved to `/MyDrive/tiny_hlm_moe/` every 1,000 steps."
))
cells.append(make_code_cell([
    "# Option A: Full Pipeline",
    "model, tok = train()",
    "model, tok = sft_finetune(model, tok)",
    "chat(model, tok)",
]))

# ── Option B: SFT Only ───────────────────────────────────────────────────────
cells.append(make_markdown_cell(
    "---\n"
    "## ▶ Option B — LoRA SFT Only\n"
    "\n"
    "Loads `best.pt` from Drive, runs 2K LoRA SFT steps, then opens chat.  \n"
    "Use this if pretraining already completed in a previous session."
))
cells.append(make_code_cell([
    "# Option B: SFT only (loads pretrained checkpoint from Drive)",
    "model, tok = sft_finetune()",
    "chat(model, tok)",
]))

# ── Option C: Chat Only ──────────────────────────────────────────────────────
cells.append(make_markdown_cell(
    "---\n"
    "## ▶ Option C — Chat Only\n"
    "\n"
    "Loads `lora_final.pt` → `best.pt` → `final.pt` (first found).  \n"
    "Use this to test a previously trained model."
))
cells.append(make_code_cell([
    "# Option C: Chat only",
    "chat()",
]))

# ── Option D: Pretrain Only ──────────────────────────────────────────────────
cells.append(make_markdown_cell(
    "---\n"
    "## ▶ Option D — Pretrain Only (no SFT)\n"
    "\n"
    "Just runs pretraining and opens a raw (non-chat) generation test at the end."
))
cells.append(make_code_cell([
    "# Option D: Pretrain only",
    "model, tok = train()",
    "chat(model, tok)  # will use pretrained weights (no LoRA)",
]))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU",
        "colab": {
            "gpuType": "A100",
            "provenance": []
        }
    },
    "cells": cells
}

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {DST}")
print(f"Cells: {len(cells)}")
for i, c in enumerate(cells):
    ctype = c["cell_type"]
    if ctype == "code":
        src_preview = c["source"][:60].replace("\n", " ")
        print(f"  [{i:02d}] code: {src_preview!r}")
    else:
        src_preview = c["source"][:60].replace("\n", " ")
        print(f"  [{i:02d}] md:   {src_preview!r}")
