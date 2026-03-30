#!/usr/bin/env python3
"""Convert TinyHLMMoE_Colab.ipynb -> TinyHLMMoE_VSCode.ipynb

Changes:
  1. Strip Colab metadata (accelerator, colab provenance)
  2. Add pip-install cell at top
  3. Add CKPT_DIR config cell at top (editable checkpoint path)
  4. Replace Colab drive-mount block in Cell 1 with clean local setup
  5. Update markdown title to mention VS Code
"""

import json, pathlib, copy

SRC = pathlib.Path(r"C:\Sif v3\Sif-v4\geoformer\TinyHLMMoE_Colab.ipynb")
DST = pathlib.Path(r"C:\Sif v3\Sif-v4\geoformer\TinyHLMMoE_VSCode.ipynb")

nb = json.loads(SRC.read_text(encoding="utf-8"))
nb = copy.deepcopy(nb)

# ── 1. Clean metadata ────────────────────────────────────────────────────────
nb["metadata"].pop("accelerator", None)
nb["metadata"].pop("colab", None)
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3"
}

# ── 2. Build new cells list ──────────────────────────────────────────────────

def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": src}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

new_cells = []

# ── Title markdown (updated) ─────────────────────────────────────────────────
new_cells.append(md_cell(
    "# TinyHLMMoE — VS Code / Local Edition\n"
    "\n"
    "~76M total params, ~48M active per forward pass.  \n"
    "Blade-routed MoE: 8 blade groups × 8 experts, top-2 active per blade.\n"
    "\n"
    "**Requirements:** Python 3.10+, PyTorch 2.x, CUDA GPU recommended (RTX 3090+ / A100).  \n"
    "Run the **Install** cell first, then **Run All Cells**, then choose an option at the bottom.\n"
    "\n"
    "Checkpoints are saved to the path you configure in **Cell 0b** below.\n"
))

# ── Install cell ─────────────────────────────────────────────────────────────
new_cells.append(md_cell("## Install Dependencies\n\nRun once. Safe to skip if already installed."))
new_cells.append(code_cell(
    "import subprocess, sys\n"
    "\n"
    "def pip(*pkgs):\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=True)\n"
    "\n"
    "pip('torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu121')\n"
    "pip('transformers>=4.40', 'datasets>=2.18', 'huggingface_hub', 'accelerate')\n"
    "print('All packages installed.')\n"
))

# ── Checkpoint path config cell ───────────────────────────────────────────────
new_cells.append(md_cell(
    "## Checkpoint Directory\n\n"
    "Edit `CKPT_DIR` below to point wherever you want checkpoints saved."
))
new_cells.append(code_cell(
    "import os\n"
    "\n"
    "# ── Edit this path ──────────────────────────────────────────────────────\n"
    "CKPT_DIR = r'./tiny_hlm_moe'          # relative to notebook, or use absolute path\n"
    "# CKPT_DIR = r'D:/models/tiny_hlm_moe'  # example: absolute path on another drive\n"
    "# ────────────────────────────────────────────────────────────────────────\n"
    "\n"
    "os.makedirs(CKPT_DIR, exist_ok=True)\n"
    "print(f'Checkpoints will be saved to: {os.path.abspath(CKPT_DIR)}')\n"
))

# ── Patch Cell 1 (imports/GPU/drive) — replace drive mount block ──────────────
# The original cell 2 in Colab nb is index 2 (0=title md, 1=cell1 md, 2=cell1 code)
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = cell["source"]
    if "from google.colab import drive" in src:
        # Replace entire try/except drive block with local CKPT_DIR reference
        old_block = (
            "try:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "    DRIVE_CKPT_DIR = '/content/drive/MyDrive/tiny_hlm_moe'\n"
            "    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)\n"
            "    print(f\"Checkpoints: {DRIVE_CKPT_DIR}\")\n"
            "except ImportError:\n"
            "    DRIVE_CKPT_DIR = './tiny_hlm_moe'\n"
            "    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)\n"
            "    print(f\"Checkpoints: {DRIVE_CKPT_DIR}\")"
        )
        new_block = (
            "# CKPT_DIR is set in the 'Checkpoint Directory' cell above.\n"
            "# If running this cell standalone, fall back to a default.\n"
            "if 'CKPT_DIR' not in dir():\n"
            "    CKPT_DIR = './tiny_hlm_moe'\n"
            "    os.makedirs(CKPT_DIR, exist_ok=True)\n"
            "DRIVE_CKPT_DIR = CKPT_DIR  # alias so rest of code works unchanged\n"
            "print(f'Checkpoints: {os.path.abspath(DRIVE_CKPT_DIR)}')"
        )
        cell["source"] = src.replace(old_block, new_block)
        print("Patched: drive mount block replaced.")
        break

# ── Add remaining Colab cells (skip original title cell at index 0) ───────────
for cell in nb["cells"][1:]:   # skip old title
    new_cells.append(cell)

nb["cells"] = new_cells

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {DST}")
print(f"Total cells: {len(new_cells)}")
for i, c in enumerate(new_cells):
    kind = c["cell_type"]
    preview = c["source"].splitlines()[0][:70] if c["source"] else "(empty)"
    print(f"  [{i:02d}] {kind:8s}  {preview}")
