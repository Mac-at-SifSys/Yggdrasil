"""Package GeoFormer for cloud upload.

Creates a tar.gz archive containing everything needed to train on Lambda A100.

Usage:
    python -m geoformer.package_for_cloud

Produces:
    geoformer_cloud.tar.gz

On Lambda:
    tar xzf geoformer_cloud.tar.gz
    cd geoformer_cloud
    pip install torch transformers datasets tqdm wandb
    python -m geoformer.training.cloud_train_geoformer --output /workspace/checkpoints
"""

import os
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEOFORMER_DIR = PROJECT_ROOT / "geoformer"
BANK_PATH = PROJECT_ROOT / "modal_app" / "src" / "bank_full.json"


def package():
    output_name = "geoformer_cloud.tar.gz"
    output_path = PROJECT_ROOT / output_name

    files_to_include = []

    # All geoformer package files
    for root, dirs, files in os.walk(GEOFORMER_DIR):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py") or f.endswith(".json"):
                full_path = Path(root) / f
                arc_name = f"geoformer_cloud/{full_path.relative_to(PROJECT_ROOT)}"
                files_to_include.append((str(full_path), arc_name))

    # Include bank_full.json if it exists
    if BANK_PATH.exists():
        files_to_include.append(
            (str(BANK_PATH), "geoformer_cloud/bank_full.json")
        )

    # Create archive
    print(f"Creating {output_name}...")
    with tarfile.open(str(output_path), "w:gz") as tar:
        for src, arc in files_to_include:
            print(f"  + {arc}")
            tar.add(src, arcname=arc)

    size_mb = output_path.stat().st_size / 1e6
    print(f"\nCreated: {output_path} ({size_mb:.1f} MB)")
    print(f"Files: {len(files_to_include)}")

    print(f"""
Upload and run:
    scp {output_name} user@lambda-host:/workspace/
    ssh user@lambda-host
    cd /workspace && tar xzf {output_name} && cd geoformer_cloud
    pip install torch transformers datasets tqdm wandb
    python -m geoformer.training.cloud_train_geoformer \\
        --bank bank_full.json \\
        --output /workspace/geoformer-250m \\
        --wandb-project geoformer
""")


if __name__ == "__main__":
    package()
