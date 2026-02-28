"""
VayuSwarm — YOLOv8 Fine-Tuning on VisDrone Real Data
═══════════════════════════════════════════════════════
Run this on Kaggle with GPU T4/P100 enabled.

Uses the REAL VisDrone drone-patrol dataset from:
  https://huggingface.co/kilanisainikhil/VayuSwarm

Dataset: 540 train + 111 val + 549 test real aerial images
Classes: person, car, truck, motorcycle, bicycle (5 classes)
Format:  YOLO (class cx cy w h) — already clean and remapped

Runtime: ~30-60 min on T4 GPU
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1: Install & Imports
# ═══════════════════════════════════════════════════════════════

import subprocess
subprocess.check_call(["pip", "install", "-q", "ultralytics", "huggingface_hub", "hf-transfer"])

# Enable hf_transfer for faster, rate-limit-resilient downloads
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import json, shutil, tempfile
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CELL 2: Config
# ═══════════════════════════════════════════════════════════════

# Hugging Face repo with VisDrone training data (for downloading dataset)
HF_REPO = "kilanisainikhil/VayuSwarm"

# GitHub repo for pushing trained model output
GIT_REPO  = "https://github.com/ved354/swam.git"
GIT_USER  = "ved354"
GIT_EMAIL = "ved354@users.noreply.github.com"

# Load secrets via Kaggle UserSecretsClient (not os.environ)
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    HF_TOKEN  = _secrets.get_secret("HF_TOKEN")
    GIT_TOKEN = _secrets.get_secret("GIT_TOKEN")
    print(f"✅ Secrets loaded — HF_TOKEN starts with: {HF_TOKEN[:8] if HF_TOKEN else 'EMPTY'}")
except Exception as e:
    print(f"⚠ kaggle_secrets failed: {e}")
    HF_TOKEN  = os.environ.get("HF_TOKEN", "")
    GIT_TOKEN = os.environ.get("GIT_TOKEN", "")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is empty! Go to Add-ons → Secrets, add HF_TOKEN and enable the toggle.")

MODEL_BASE = "yolov8n.pt"       # Pre-trained COCO backbone
EPOCHS     = 100
BATCH_SIZE = 32
IMG_SIZE   = 640
PATIENCE   = 20
DEVICE     = 0                   # GPU 0

# Paths (Kaggle working directory)
WORK_DIR    = Path("/kaggle/working")
DATASET_DIR = WORK_DIR / "datasets" / "VisDrone"
OUTPUT_DIR  = WORK_DIR / "runs"

# 5 patrol classes matching the VisDrone remapped dataset
CLASSES = ["person", "car", "truck", "motorcycle", "bicycle"]

print(f"🎯 {len(CLASSES)} classes: {CLASSES}")
print(f"🔧 Training: {EPOCHS} epochs, batch {BATCH_SIZE}, {IMG_SIZE}px")

# ═══════════════════════════════════════════════════════════════
# CELL 3: Download Dataset from Hugging Face
# ═══════════════════════════════════════════════════════════════

from huggingface_hub import snapshot_download, login

# Explicitly login — more reliable than passing token param
login(token=HF_TOKEN, add_to_git_credential=False)
print("✅ Logged in to HuggingFace")

if DATASET_DIR.exists() and len(list((DATASET_DIR / "images" / "train").glob("*.jpg"))) > 0:
    print(f"✅ Dataset already exists at {DATASET_DIR}")
else:
    print(f"📥 Downloading VisDrone dataset from {HF_REPO}...")

    # max_workers=2 throttles concurrency to avoid HF rate-limiting (HTTP 429)
    local_path = snapshot_download(
        repo_id=HF_REPO,
        repo_type="model",
        allow_patterns=["datasets/VisDrone/**"],
        local_dir=str(WORK_DIR),
        token=HF_TOKEN,
        max_workers=2,
    )
    print(f"✅ Downloaded to {local_path}")

# Verify dataset
for split in ["train", "val", "test"]:
    img_count = len(list((DATASET_DIR / "images" / split).glob("*")))
    lbl_count = len(list((DATASET_DIR / "labels" / split).glob("*")))
    print(f"   {split}: {img_count} images, {lbl_count} labels")

# ═══════════════════════════════════════════════════════════════
# CELL 4: Create data.yaml
# ═══════════════════════════════════════════════════════════════

import yaml

data_config = {
    "path": str(DATASET_DIR),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(CLASSES),
    "names": CLASSES,
}

yaml_path = DATASET_DIR / "data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"✅ data.yaml written to {yaml_path}")
print(f"   Classes: {CLASSES}")

# ═══════════════════════════════════════════════════════════════
# CELL 5: Train YOLOv8
# ═══════════════════════════════════════════════════════════════

from ultralytics import YOLO

print(f"\n🚀 Starting YOLOv8 training...")
print(f"   Base model:  {MODEL_BASE}")
print(f"   Dataset:     VisDrone (real aerial images)")
print(f"   Epochs:      {EPOCHS}")
print(f"   Batch size:  {BATCH_SIZE}")
print(f"   Image size:  {IMG_SIZE}")

model = YOLO(MODEL_BASE)
results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    patience=PATIENCE,
    device=DEVICE,
    project=str(OUTPUT_DIR),
    name="patrol_rgb",
    exist_ok=True,
    # Data augmentation — tuned for aerial/drone patrol
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.3,       # vertical flip useful for aerial
    fliplr=0.5,
    mosaic=1.0,
    erasing=0.4,
    # Optimizer (auto selects SGD for YOLO)
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    # Saving
    save=True,
    plots=True,
    verbose=True,
)

print("✅ Training complete!")
SAVE_DIR = Path(results.save_dir)
print(f"📁 Model saved to: {SAVE_DIR}")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Evaluate
# ═══════════════════════════════════════════════════════════════

best_pt = SAVE_DIR / "weights" / "best.pt"
best_model = YOLO(str(best_pt))
metrics = best_model.val(data=str(yaml_path), split="test")

print(f"\n📊 Test Results (on {len(list((DATASET_DIR / 'images' / 'test').glob('*')))} real images):")
print(f"   mAP50:      {metrics.box.map50:.4f}")
print(f"   mAP50-95:   {metrics.box.map:.4f}")
print(f"   Precision:   {metrics.box.mp:.4f}")
print(f"   Recall:      {metrics.box.mr:.4f}")

# Per-class results
if hasattr(metrics.box, "ap_class_index"):
    print(f"\n   Per-class mAP50:")
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        cls_name = CLASSES[int(cls_idx)] if int(cls_idx) < len(CLASSES) else f"cls_{cls_idx}"
        print(f"     {cls_name:15s}  mAP50={metrics.box.ap50[i]:.4f}")

# ═══════════════════════════════════════════════════════════════
# CELL 7: Export & Save Metadata
# ═══════════════════════════════════════════════════════════════

print("\n📦 Exporting to ONNX...")
onnx_path = best_model.export(format="onnx", imgsz=IMG_SIZE, simplify=True)

metadata = {
    "model_name": "vayuswarm_patrol_rgb",
    "base_model": MODEL_BASE,
    "dataset": "VisDrone (real aerial drone images)",
    "dataset_source": f"https://huggingface.co/{HF_REPO}",
    "num_classes": len(CLASSES),
    "classes": {i: name for i, name in enumerate(CLASSES)},
    "training": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "train_images": len(list((DATASET_DIR / "images" / "train").glob("*"))),
        "val_images": len(list((DATASET_DIR / "images" / "val").glob("*"))),
        "test_images": len(list((DATASET_DIR / "images" / "test").glob("*"))),
    },
    "metrics": {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    },
}

metadata_path = SAVE_DIR / "class_mapping.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Exported: best.pt + ONNX + class_mapping.json")

# ═══════════════════════════════════════════════════════════════
# CELL 8: Push Trained Model to GitHub
# ═══════════════════════════════════════════════════════════════

if GIT_TOKEN:
    try:
        # Build authenticated clone URL
        auth_url = GIT_REPO.replace("https://", f"https://{GIT_USER}:{GIT_TOKEN}@")
        clone_dir = Path(tempfile.mkdtemp()) / "swam"

        print(f"\n📤 Pushing trained model to GitHub: {GIT_REPO}")

        # Clone the repo
        subprocess.check_call(["git", "clone", "--depth", "1", auth_url, str(clone_dir)])

        # Create models/yolo directory in the repo
        model_dir = clone_dir / "models" / "yolo"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy trained model files into the repo
        files_to_push = []
        for src, dst_name in [
            (best_pt, "best.pt"),
            (SAVE_DIR / "weights" / "last.pt", "last.pt"),
            (metadata_path, "class_mapping.json"),
        ]:
            if src.exists():
                shutil.copy2(src, model_dir / dst_name)
                files_to_push.append(dst_name)
                print(f"   ✅ Copied {dst_name} ({src.stat().st_size / 1024 / 1024:.1f} MB)")

        # Copy ONNX export if exists
        for onnx_file in SAVE_DIR.glob("**/*.onnx"):
            shutil.copy2(onnx_file, model_dir / onnx_file.name)
            files_to_push.append(onnx_file.name)
            print(f"   ✅ Copied {onnx_file.name}")

        # Copy training plots
        plots_dest = clone_dir / "models" / "yolo" / "plots"
        plots_dest.mkdir(exist_ok=True)
        for img in SAVE_DIR.glob("*.png"):
            shutil.copy2(img, plots_dest / img.name)
        print(f"   ✅ Copied training plots")

        # Git add, commit, push
        env = os.environ.copy()
        git_cmds = [
            ["git", "config", "user.name", GIT_USER],
            ["git", "config", "user.email", GIT_EMAIL],
            ["git", "add", "models/yolo/"],
            ["git", "commit", "-m",
             f"Add trained YOLOv8 patrol model — mAP50={metrics.box.map50:.4f}, "
             f"{len(CLASSES)} classes, {EPOCHS} epochs"],
            ["git", "push", "origin", "main"],
        ]

        for cmd in git_cmds:
            subprocess.check_call(cmd, cwd=str(clone_dir), env=env)

        print(f"\n✅ Model pushed to: {GIT_REPO}")
        print(f"   Files: {', '.join(files_to_push)}")

    except Exception as e:
        print(f"⚠ GitHub push failed: {e}")
        print(f"  → Model saved locally at: {SAVE_DIR}")
        print(f"  → Download from Kaggle Output tab instead")
else:
    print(f"\nℹ No GIT_TOKEN — model saved locally at: {SAVE_DIR}")
    print(f"  Add GIT_TOKEN in Kaggle Settings → Secrets to auto-push to GitHub")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

print(f"""
{'='*60}
🎉 VayuSwarm YOLOv8 Training Complete!
{'='*60}
Model:     {best_pt}
Classes:   {CLASSES}
mAP50:     {metrics.box.map50:.4f}
Precision: {metrics.box.mp:.4f}
Recall:    {metrics.box.mr:.4f}

To use in VayuSwarm:
  1. Copy best.pt → models/yolo/best.pt
  2. Copy class_mapping.json → models/yolo/class_mapping.json
  3. Run: python3 simulate.py --model models/yolo/best.pt
{'='*60}
""")
