"""
VayuSwarm — Model Downloader

Downloads trained model artifacts from the GitHub repository
into the local models/ directory.

Usage:
    python scripts/download_models.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

# ── Config ──────────────────────────────────────────────────────
GITHUB_REPO = "ved354/swam"
GITHUB_BRANCH = "main"
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Files to download, grouped by subdirectory
MODEL_FILES = {
    "yolo": [
        "best.pt",
        "class_mapping.json",
    ],
    "thermal": [
        "best_thermal.pth",
        "thermal_classifier.onnx",
        "thermal_metadata.json",
    ],
    "behavior": [
        "best_behavior.pth",
        "behavior_transformer.onnx",
        "behavior_metadata.json",
        "norm_mean.npy",
        "norm_std.npy",
    ],
    "llm_data": [
        "llm_training_data.jsonl",
    ],
}


def download_file(url: str, dest: Path, overwrite: bool = False) -> bool:
    """Download a file from URL to dest. Returns True if downloaded."""
    if dest.exists() and not overwrite:
        print(f"  ✓ {dest.name} (already exists)")
        return False

    try:
        print(f"  ↓ Downloading {dest.name} ...", end=" ", flush=True)
        urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"done ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    """Download all model files from GitHub."""
    overwrite = "--force" in sys.argv

    print(f"📦 VayuSwarm Model Downloader")
    print(f"   Source: github.com/{GITHUB_REPO} ({GITHUB_BRANCH})")
    print(f"   Target: {MODELS_DIR}")
    if overwrite:
        print(f"   Mode:   FORCE (re-download all)")
    print()

    total_downloaded = 0
    total_skipped = 0

    for subdir, files in MODEL_FILES.items():
        dest_dir = MODELS_DIR / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 models/{subdir}/")

        for filename in files:
            url = f"{BASE_URL}/models/{subdir}/{filename}"
            dest = dest_dir / filename

            if download_file(url, dest, overwrite):
                total_downloaded += 1
            else:
                total_skipped += 1

        print()

    print(f"✅ Done! Downloaded: {total_downloaded}, Skipped: {total_skipped}")

    # Verify critical files
    critical = [
        MODELS_DIR / "yolo" / "best.pt",
        MODELS_DIR / "thermal" / "thermal_classifier.onnx",
        MODELS_DIR / "behavior" / "behavior_transformer.onnx",
    ]
    missing = [f for f in critical if not f.exists()]
    if missing:
        print(f"\n⚠ Missing critical files:")
        for f in missing:
            print(f"  - {f.relative_to(PROJECT_ROOT)}")
        sys.exit(1)
    else:
        print(f"\n✅ All critical model files present!")


if __name__ == "__main__":
    main()
