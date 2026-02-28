"""
VayuSwarm — YOLOv8 Fine-Tuning for Weapons, Uniforms & Drone Detection
═══════════════════════════════════════════════════════════════════════
Run this on Kaggle with GPU T4 enabled.

Generates synthetic + augmented training data, then fine-tunes YOLOv8n.
Auto-pushes trained model to GitHub.

Runtime: ~30-45 min on T4 GPU
"""

# ═══════════════════════════════════════════════════════════════
# CELL 1: Install
# ═══════════════════════════════════════════════════════════════

import subprocess
subprocess.check_call(["pip", "install", "-q", "ultralytics", "gitpython", "Pillow"])

import os, json, shutil, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# ═══════════════════════════════════════════════════════════════
# CELL 2: Config
# ═══════════════════════════════════════════════════════════════

GITHUB_TOKEN = "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"
GITHUB_REPO = "ved354/swam"
GITHUB_BRANCH = "main"

MODEL_BASE = "yolov8n.pt"
EPOCHS = 80
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 15
DEVICE = 0

OUTPUT_DIR = Path("/kaggle/working/vayuswarm_yolo")
DATASET_DIR = Path("/kaggle/working/dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 13 VayuSwarm target classes
CLASSES = [
    "person",             # 0
    "vehicle_car",        # 1
    "vehicle_truck",      # 2
    "vehicle_motorcycle", # 3
    "weapon_rifle",       # 4
    "weapon_pistol",      # 5
    "weapon_knife",       # 6
    "uniform_military",   # 7
    "uniform_police",     # 8
    "uniform_civilian",   # 9
    "drone",              # 10
    "suspicious_package", # 11
    "fire",               # 12
]

print(f"🎯 {len(CLASSES)} classes: {CLASSES}")

# ═══════════════════════════════════════════════════════════════
# CELL 3: Synthetic Data Generator
# ═══════════════════════════════════════════════════════════════

# Color palettes for each class (to make them visually distinct)
CLASS_COLORS = {
    0:  [(200,150,120), (180,130,100), (160,120,90)],    # person — skin tones
    1:  [(50,50,200), (200,50,50), (50,200,50)],          # car — bright colors
    2:  [(100,100,100), (80,80,80), (120,120,120)],       # truck — gray/dark
    3:  [(200,200,50), (200,100,50), (50,200,200)],       # motorcycle
    4:  [(40,40,40), (60,50,40), (30,30,30)],             # rifle — dark metallic
    5:  [(50,50,50), (70,60,50), (40,40,40)],             # pistol — dark
    6:  [(180,180,180), (200,200,200), (160,160,160)],    # knife — silver
    7:  [(80,100,60), (70,90,50), (90,110,70)],           # military — camo green
    8:  [(30,40,100), (20,30,90), (40,50,110)],           # police — dark blue
    9:  [(180,50,50), (50,50,180), (50,180,50)],          # civilian — varied
    10: [(200,200,200), (220,220,220), (180,180,180)],    # drone — white/gray
    11: [(60,40,20), (80,60,30), (50,30,10)],             # package — brown
    12: [(255,100,0), (255,50,0), (255,150,0)],           # fire — orange/red
}

# Typical aspect ratios for each class (w_ratio, h_ratio)
CLASS_SHAPES = {
    0:  (0.15, 0.40),   # person — tall narrow
    1:  (0.30, 0.20),   # car — wide low
    2:  (0.35, 0.25),   # truck — wider
    3:  (0.15, 0.15),   # motorcycle — small square
    4:  (0.25, 0.05),   # rifle — very wide thin
    5:  (0.10, 0.07),   # pistol — small wide
    6:  (0.15, 0.03),   # knife — thin line
    7:  (0.15, 0.40),   # military uniform — person shaped
    8:  (0.15, 0.40),   # police uniform — person shaped
    9:  (0.15, 0.38),   # civilian — person shaped
    10: (0.12, 0.08),   # drone — small wide
    11: (0.12, 0.10),   # package — small square
    12: (0.20, 0.25),   # fire — irregular tall
}


def draw_object(draw, cls_id, cx, cy, w, h, img_size):
    """Draw a synthetic object on the image."""
    colors = CLASS_COLORS[cls_id]
    color = colors[np.random.randint(len(colors))]
    
    x1 = int((cx - w/2) * img_size)
    y1 = int((cy - h/2) * img_size)
    x2 = int((cx + w/2) * img_size)
    y2 = int((cy + h/2) * img_size)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_size-1, x2), min(img_size-1, y2)
    
    if cls_id in [0, 7, 8, 9]:  # Person-shaped
        # Body (rectangle)
        body_y1 = y1 + (y2-y1)//5
        draw.rectangle([x1, body_y1, x2, y2], fill=color)
        # Head (circle)
        head_r = max(3, (x2-x1)//3)
        head_cx = (x1+x2)//2
        head_cy = y1 + head_r
        draw.ellipse([head_cx-head_r, head_cy-head_r, head_cx+head_r, head_cy+head_r],
                     fill=(200,150,120))
        
    elif cls_id in [1, 2]:  # Vehicle
        draw.rectangle([x1, y1, x2, y2], fill=color)
        # Windows
        wy1 = y1 + (y2-y1)//4
        wy2 = y1 + (y2-y1)//2
        draw.rectangle([x1+5, wy1, x2-5, wy2], fill=(150,200,230))
        # Wheels
        wh = max(3, (y2-y1)//6)
        draw.ellipse([x1+5, y2-wh*2, x1+5+wh*2, y2], fill=(30,30,30))
        draw.ellipse([x2-5-wh*2, y2-wh*2, x2-5, y2], fill=(30,30,30))
        
    elif cls_id in [4, 5, 6]:  # Weapons
        draw.rectangle([x1, y1, x2, y2], fill=color)
        # Add detail line
        mid_y = (y1+y2)//2
        draw.line([(x1, mid_y), (x2, mid_y)], fill=(20,20,20), width=2)
        
    elif cls_id == 10:  # Drone
        cx_px, cy_px = (x1+x2)//2, (y1+y2)//2
        # Body
        br = max(3, min(x2-x1, y2-y1)//4)
        draw.ellipse([cx_px-br, cy_px-br, cx_px+br, cy_px+br], fill=color)
        # Arms
        draw.line([(x1, y1), (x2, y2)], fill=(100,100,100), width=2)
        draw.line([(x1, y2), (x2, y1)], fill=(100,100,100), width=2)
        # Props
        for px, py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            draw.ellipse([px-4, py-4, px+4, py+4], fill=(50,50,50))
            
    elif cls_id == 11:  # Package
        draw.rectangle([x1, y1, x2, y2], fill=color)
        draw.line([(x1, (y1+y2)//2), (x2, (y1+y2)//2)], fill=(40,30,10), width=2)
        draw.line([((x1+x2)//2, y1), ((x1+x2)//2, y2)], fill=(40,30,10), width=2)
        
    elif cls_id == 12:  # Fire
        for _ in range(8):
            fx = np.random.randint(x1, max(x1+1, x2))
            fy = np.random.randint(y1, max(y1+1, y2))
            fr = np.random.randint(3, max(4, (x2-x1)//3))
            fc = (255, np.random.randint(50,200), 0)
            draw.ellipse([fx-fr, fy-fr, fx+fr, fy+fr], fill=fc)
    else:
        draw.rectangle([x1, y1, x2, y2], fill=color)


def generate_background(size=640):
    """Generate varied backgrounds: sky, ground, urban, rural."""
    bg_type = np.random.choice(["urban", "rural", "sky", "indoor"])
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    
    if bg_type == "urban":
        # Gray concrete + buildings
        base = np.random.randint(120, 180)
        img = Image.fromarray(np.full((size, size, 3), base, dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        for _ in range(np.random.randint(2, 6)):
            bx = np.random.randint(0, size-50)
            bw = np.random.randint(40, 150)
            bh = np.random.randint(100, size)
            bc = tuple(np.random.randint(80, 160, 3).tolist())
            draw.rectangle([bx, size-bh, bx+bw, size], fill=bc)
            
    elif bg_type == "rural":
        # Green + brown ground, blue sky
        sky = np.random.randint(150, 200)
        ground = np.random.randint(60, 120)
        horizon = np.random.randint(size//3, 2*size//3)
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr[:horizon] = [sky, sky+30, 230]  # Blue sky
        arr[horizon:] = [ground, ground+40, ground-10]  # Green ground
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        
    elif bg_type == "sky":
        # Aerial view — for drone detection
        g = np.random.randint(80, 140)
        arr = np.random.normal(g, 15, (size, size, 3)).clip(0, 255).astype(np.uint8)
        arr[:, :, 0] = (arr[:, :, 0] * 0.8).astype(np.uint8)  # Less red
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        
    else:  # indoor
        base = np.random.randint(160, 220)
        arr = np.random.normal(base, 8, (size, size, 3)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        
    return img, draw


def generate_image(classes_in_image=None, size=640):
    """Generate one training image with random objects."""
    img, draw = generate_background(size)
    
    if classes_in_image is None:
        n_objects = np.random.randint(1, 6)
        classes_in_image = [np.random.randint(0, len(CLASSES)) for _ in range(n_objects)]
    
    labels = []
    for cls_id in classes_in_image:
        base_w, base_h = CLASS_SHAPES[cls_id]
        # Randomize size
        w = base_w * np.random.uniform(0.5, 1.5)
        h = base_h * np.random.uniform(0.5, 1.5)
        # Random position (keep within bounds)
        cx = np.random.uniform(w/2 + 0.02, 1 - w/2 - 0.02)
        cy = np.random.uniform(h/2 + 0.02, 1 - h/2 - 0.02)
        
        draw_object(draw, cls_id, cx, cy, w, h, size)
        labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    # Add noise
    arr = np.array(img).astype(np.float32)
    arr += np.random.normal(0, np.random.uniform(3, 12), arr.shape)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    
    # Random blur sometimes
    if np.random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5)))
    
    return img, labels


print("🔧 Generating training dataset...")

N_TRAIN = 2000
N_VAL = 400

for split, n_images in [("train", N_TRAIN), ("val", N_VAL)]:
    img_dir = DATASET_DIR / "merged" / split / "images"
    lbl_dir = DATASET_DIR / "merged" / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_images):
        # Ensure every class appears frequently
        if i < len(CLASSES) * 5:
            # First passes: one class per image to ensure coverage
            target_cls = [i % len(CLASSES)]
            # Sometimes add more objects
            if np.random.random() < 0.5:
                target_cls.append(np.random.randint(0, len(CLASSES)))
        else:
            target_cls = None  # Random
        
        img, labels = generate_image(target_cls)
        img.save(str(img_dir / f"{split}_{i:05d}.jpg"), quality=90)
        with open(lbl_dir / f"{split}_{i:05d}.txt", "w") as f:
            f.write("\n".join(labels))
        
        if (i+1) % 500 == 0:
            print(f"   [{split}] {i+1}/{n_images}")

print(f"✅ Generated {N_TRAIN} train + {N_VAL} val images")

# Create data.yaml
import yaml
data_yaml = {
    "path": str(DATASET_DIR / "merged"),
    "train": "train/images",
    "val": "val/images",
    "nc": len(CLASSES),
    "names": CLASSES,
}
yaml_path = DATASET_DIR / "merged" / "data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

# Verify
train_count = len(list((DATASET_DIR / "merged" / "train" / "images").glob("*.jpg")))
val_count = len(list((DATASET_DIR / "merged" / "val" / "images").glob("*.jpg")))
print(f"✅ Verified: {train_count} train, {val_count} val images")

# ═══════════════════════════════════════════════════════════════
# CELL 4: Train YOLOv8
# ═══════════════════════════════════════════════════════════════

from ultralytics import YOLO

print(f"🚀 Training YOLOv8 | {EPOCHS} epochs | batch {BATCH_SIZE} | {IMG_SIZE}px")

model = YOLO(MODEL_BASE)
results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE,
    patience=PATIENCE, device=DEVICE,
    project=str(OUTPUT_DIR), name="vayuswarm_detector",
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10.0, translate=0.1, scale=0.5,
    flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1,
    optimizer="AdamW", lr0=0.001, lrf=0.01,
    weight_decay=0.0005, warmup_epochs=3.0,
    save=True, save_period=10, plots=True, verbose=True,
)
print("✅ Training complete!")

# Get the actual save directory (Kaggle may rename it to vayuswarm_detector2, etc.)
SAVE_DIR = Path(results.save_dir)
print(f"📁 Model saved to: {SAVE_DIR}")

# ═══════════════════════════════════════════════════════════════
# CELL 5: Evaluate
# ═══════════════════════════════════════════════════════════════

best_model = YOLO(str(SAVE_DIR / "weights" / "best.pt"))
metrics = best_model.val(data=str(yaml_path))

print(f"\n📊 Results:")
print(f"   mAP50:     {metrics.box.map50:.4f}")
print(f"   mAP50-95:  {metrics.box.map:.4f}")
print(f"   Precision:  {metrics.box.mp:.4f}")
print(f"   Recall:     {metrics.box.mr:.4f}")

# ═══════════════════════════════════════════════════════════════
# CELL 6: Export ONNX + Save Metadata
# ═══════════════════════════════════════════════════════════════

print("📦 Exporting to ONNX...")
onnx_path = best_model.export(format="onnx", imgsz=IMG_SIZE, simplify=True)

mapping = {
    "model_name": "vayuswarm_yolov8n",
    "base_model": MODEL_BASE,
    "num_classes": len(CLASSES),
    "classes": {i: n for i, n in enumerate(CLASSES)},
    "metrics": {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    },
}
mapping_path = SAVE_DIR / "class_mapping.json"
with open(mapping_path, "w") as f:
    json.dump(mapping, f, indent=2)
print(f"✅ Exported ONNX + class mapping")

# ═══════════════════════════════════════════════════════════════
# CELL 7: Auto-Push to GitHub
# ═══════════════════════════════════════════════════════════════

if GITHUB_TOKEN and GITHUB_REPO:
    try:
        import git
        clone_dir = Path("/kaggle/working/repo_clone")
        if clone_dir.exists(): shutil.rmtree(clone_dir)

        print(f"📥 Cloning {GITHUB_REPO}...")
        try:
            repo = git.Repo.clone_from(f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git",
                                        str(clone_dir), branch=GITHUB_BRANCH)
        except:
            repo = git.Repo.clone_from(f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git",
                                        str(clone_dir))
            repo.git.checkout("-b", GITHUB_BRANCH)

        target = clone_dir / "models" / "yolo"
        target.mkdir(parents=True, exist_ok=True)

        weights_dir = SAVE_DIR / "weights"
        for src, dst in [(weights_dir/"best.pt", "best.pt"), (mapping_path, "class_mapping.json")]:
            if src.exists():
                shutil.copy2(str(src), str(target/dst))
                print(f"   📁 models/yolo/{dst}")
        for f in SAVE_DIR.glob("*.onnx"):
            shutil.copy2(str(f), str(target/f.name))
            print(f"   📁 models/yolo/{f.name}")

        repo.config_writer().set_value("user","name","VayuSwarm-Bot").release()
        repo.config_writer().set_value("user","email","bot@vayuswarm.ai").release()
        repo.git.add(A=True)
        if repo.is_dirty() or repo.untracked_files:
            repo.index.commit(f"🤖 YOLOv8 model (mAP50: {metrics.box.map50:.4f})")
            repo.remote("origin").push(GITHUB_BRANCH)
            print(f"✅ Pushed to: https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/models/yolo")
        shutil.rmtree(clone_dir, ignore_errors=True)
    except Exception as e:
        print(f"⚠ GitHub push failed: {e}")
        print(f"  → Model saved locally at: {SAVE_DIR}")
else:
    print(f"⚠ No GitHub creds — saved locally: {SAVE_DIR}")

print("\n🎉 YOLOv8 fine-tuning complete!")
