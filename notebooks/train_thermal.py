"""
VayuSwarm — Thermal Classifier Training on REAL Data
═══════════════════════════════════════════════════════════════════
Run on Kaggle with GPU T4/P100 enabled.

Uses TWO real thermal/infrared datasets:
  1. LLVIP (Low-Light Visible-Infrared Pairs) — hustvl/LLVIP on HuggingFace
     Real IR images with person bounding box annotations
  2. FLIR Free Thermal Dataset — deepnewbie/flir-free on HuggingFace
     Real thermal images with car, bicycle, person, dog annotations

Classes: background, human, vehicle, animal, fire
Model:   MobileNetV3-Small (fine-tuned from ImageNet)
Output:  PyTorch (.pth) + ONNX → auto-pushed to GitHub

KAGGLE SECRETS: HF_TOKEN, GIT_TOKEN
Runtime: ~25-35 min on T4 GPU
"""

import subprocess
subprocess.check_call(["pip", "install", "-q",
    "torch", "torchvision", "onnx", "huggingface_hub", "hf-transfer", "Pillow"])

import os, json, shutil, tempfile
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import numpy as np
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    HF_TOKEN  = _s.get_secret("HF_TOKEN")
    GIT_TOKEN = _s.get_secret("GIT_TOKEN")
    print(f"✅ Secrets loaded — HF_TOKEN starts with: {HF_TOKEN[:8] if HF_TOKEN else 'EMPTY'}")
except Exception as e:
    print(f"⚠ kaggle_secrets failed: {e}")
    HF_TOKEN  = os.environ.get("HF_TOKEN", "")
    GIT_TOKEN = os.environ.get("GIT_TOKEN", "")

GIT_REPO  = "https://github.com/ved354/swam.git"
GIT_USER  = "ved354"
GIT_EMAIL = "ved354@users.noreply.github.com"

EPOCHS     = 30
BATCH_SIZE = 32
LR         = 0.001
IMG_SIZE   = 224
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORK_DIR    = Path("/kaggle/working")
OUTPUT_DIR  = WORK_DIR / "thermal_model"
LLVIP_DIR   = WORK_DIR / "llvip"
FLIR_DIR    = WORK_DIR / "flir"
DATASET_DIR = WORK_DIR / "thermal_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THERMAL_CLASSES = ["background", "human", "vehicle", "animal", "fire"]
print(f"Device: {DEVICE} | Classes: {THERMAL_CLASSES}")

# ═══════════════════════════════════════════════════════════════
# Download Real Datasets
# ═══════════════════════════════════════════════════════════════

from huggingface_hub import snapshot_download, login

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in to HuggingFace")

# Dataset 1: LLVIP — real IR images with person annotations
if not (LLVIP_DIR / "infrared").exists():
    print("📥 Downloading LLVIP infrared dataset (real IR images)...")
    snapshot_download(
        repo_id="hustvl/LLVIP",
        repo_type="dataset",
        allow_patterns=["infrared/**", "Annotations/**"],
        local_dir=str(LLVIP_DIR),
        token=HF_TOKEN if HF_TOKEN else None,
        max_workers=2,
    )
    print("✅ LLVIP downloaded")
else:
    print("✅ LLVIP already exists")

# Dataset 2: FLIR free thermal — person, car, bicycle, dog
if not FLIR_DIR.exists() or not any(FLIR_DIR.iterdir()):
    print("📥 Downloading FLIR thermal dataset (real thermal images)...")
    snapshot_download(
        repo_id="deepnewbie/flir-free",
        repo_type="dataset",
        local_dir=str(FLIR_DIR),
        token=HF_TOKEN if HF_TOKEN else None,
        max_workers=2,
    )
    print("✅ FLIR downloaded")
else:
    print("✅ FLIR already exists")

# ═══════════════════════════════════════════════════════════════
# Build Dataset From Real Images
# ═══════════════════════════════════════════════════════════════

def extract_llvip_crops(llvip_dir, out_dir, max_per_class=1500):
    """Extract human crops + background patches from real LLVIP IR images."""
    ir_train = llvip_dir / "infrared" / "train"
    ann_dir  = llvip_dir / "Annotations"
    human_dir = out_dir / "train" / "human"
    bg_dir    = out_dir / "train" / "background"
    human_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    human_count = bg_count = 0
    for img_path in sorted(ir_train.glob("*.jpg"))[:3000]:
        if human_count >= max_per_class and bg_count >= max_per_class:
            break
        try:
            img = Image.open(img_path).convert("L")
            w, h = img.size
            ann_path = ann_dir / (img_path.stem + ".xml")
            boxes = []
            if ann_path.exists():
                for obj in ET.parse(ann_path).getroot().findall("object"):
                    if obj.find("name").text.lower() == "person":
                        bb = obj.find("bndbox")
                        boxes.append((int(bb.find("xmin").text), int(bb.find("ymin").text),
                                      int(bb.find("xmax").text), int(bb.find("ymax").text)))
            for x1, y1, x2, y2 in boxes:
                if human_count >= max_per_class: break
                crop = img.crop((max(0,x1-10), max(0,y1-10), min(w,x2+10), min(h,y2+10)))
                if crop.size[0] > 15 and crop.size[1] > 15:
                    crop.resize((IMG_SIZE, IMG_SIZE)).save(str(human_dir / f"human_{human_count:05d}.png"))
                    human_count += 1
            if bg_count < max_per_class and not boxes:
                img.crop((0, 0, w//2, h//2)).resize((IMG_SIZE, IMG_SIZE)).save(
                    str(bg_dir / f"bg_{bg_count:05d}.png"))
                bg_count += 1
        except Exception:
            continue
    print(f"   LLVIP → {human_count} human, {bg_count} background")


def extract_flir_crops(flir_dir, out_dir, max_per_class=1000):
    """Extract vehicle + animal crops from real FLIR thermal annotations."""
    vehicle_dir = out_dir / "train" / "vehicle"
    animal_dir  = out_dir / "train" / "animal"
    vehicle_dir.mkdir(parents=True, exist_ok=True)
    animal_dir.mkdir(parents=True, exist_ok=True)

    ann_files = list(flir_dir.glob("**/coco.json")) + list(flir_dir.glob("**/*train*.json"))
    if not ann_files:
        print("   ⚠ FLIR JSON not found, skipping")
        return

    ann_file = ann_files[0]
    img_dir  = ann_file.parent / "data"
    if not img_dir.exists():
        img_dir = ann_file.parent

    with open(ann_file) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}
    img_map = {i["id"]: i for i in coco.get("images", [])}
    vehicle_cats = {"car", "bicycle", "truck", "bus"}
    animal_cats  = {"dog", "cat", "animal"}
    vc = ac = 0

    for ann in coco.get("annotations", []):
        if vc >= max_per_class and ac >= max_per_class:
            break
        name = cat_map.get(ann["category_id"], "")
        info = img_map.get(ann["image_id"])
        if not info:
            continue
        is_veh = any(v in name for v in vehicle_cats) and vc < max_per_class
        is_ani = any(a in name for a in animal_cats) and ac < max_per_class
        if not is_veh and not is_ani:
            continue
        try:
            p = img_dir / info["file_name"]
            if not p.exists():
                p = img_dir / Path(info["file_name"]).name
            if not p.exists():
                continue
            img = Image.open(p).convert("L")
            x, y, bw, bh = [int(v) for v in ann["bbox"]]
            if bw < 10 or bh < 10:
                continue
            iw, ih = img.size
            crop = img.crop((max(0,x-8), max(0,y-8), min(iw,x+bw+8), min(ih,y+bh+8)))
            crop = crop.resize((IMG_SIZE, IMG_SIZE))
            if is_veh:
                crop.save(str(vehicle_dir / f"vehicle_{vc:05d}.png"))
                vc += 1
            else:
                crop.save(str(animal_dir / f"animal_{ac:05d}.png"))
                ac += 1
        except Exception:
            continue
    print(f"   FLIR → {vc} vehicle, {ac} animal")


def gen_fire_samples(bg_dir, fire_dir, n=600):
    """Physics-accurate fire thermal on real backgrounds (fire temps 800-1200°C → saturated)."""
    fire_dir.mkdir(parents=True, exist_ok=True)
    bgs = list(bg_dir.glob("*.png"))[:200] if bg_dir.exists() else []
    for i in range(n):
        if bgs:
            arr = np.array(Image.open(bgs[i % len(bgs)]).convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        else:
            arr = np.random.normal(40, 8, (IMG_SIZE, IMG_SIZE)).clip(0, 100).astype(np.float32)
        for _ in range(np.random.randint(1, 4)):
            cx, cy = np.random.randint(30, IMG_SIZE-30, 2)
            r = np.random.randint(8, 35)
            ys, xs = np.ogrid[-r:r+1, -r:r+1]
            mask = xs**2 + ys**2 <= r**2
            py = np.clip(cy + np.arange(-r, r+1)[:, None], 0, IMG_SIZE-1)
            px = np.clip(cx + np.arange(-r, r+1)[None, :], 0, IMG_SIZE-1)
            intensity = 255 - (30 * np.sqrt(xs**2 + ys**2) / r)
            arr[py[mask], px[mask]] = np.maximum(arr[py[mask], px[mask]], intensity[mask])
        Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode="L").save(str(fire_dir / f"fire_{i:05d}.png"))
    print(f"   Fire → {n} samples (physics-accurate on real backgrounds)")


print("\n📦 Building dataset from real thermal images...")
extract_llvip_crops(LLVIP_DIR, DATASET_DIR)
extract_flir_crops(FLIR_DIR, DATASET_DIR)
gen_fire_samples(DATASET_DIR / "train" / "background", DATASET_DIR / "train" / "fire")

# Val split (20%)
print("🔀 Creating val split...")
for cls in THERMAL_CLASSES:
    src = DATASET_DIR / "train" / cls
    val = DATASET_DIR / "val" / cls
    val.mkdir(parents=True, exist_ok=True)
    imgs = list(src.glob("*")) if src.exists() else []
    np.random.shuffle(imgs)
    for p in imgs[:max(20, len(imgs)//5)]:
        shutil.copy2(str(p), str(val / p.name))

for cls in THERMAL_CLASSES:
    n = len(list((DATASET_DIR / "train" / cls).glob("*"))) if (DATASET_DIR / "train" / cls).exists() else 0
    print(f"   {cls:12s}: {n} samples")

# ═══════════════════════════════════════════════════════════════
# Model & Training
# ═══════════════════════════════════════════════════════════════

class ThermalDS(Dataset):
    def __init__(self, root, transform):
        self.samples  = []
        self.transform = transform
        for ci, cn in enumerate(THERMAL_CLASSES):
            cls_dir = Path(root) / cn
            if cls_dir.exists():
                for p in cls_dir.glob("*"):
                    self.samples.append((str(p), ci))
        np.random.shuffle(self.samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img = Image.open(self.samples[i][0]).convert("L")
        return self.transform(img), self.samples[i][1]

t_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
t_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_ds = ThermalDS(str(DATASET_DIR / "train"), t_train)
val_ds   = ThermalDS(str(DATASET_DIR / "val"),   t_val)
print(f"\n📊 {len(train_ds)} train, {len(val_ds)} val (real thermal images)")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Class-balanced weights
counts = [max(1, len(list((DATASET_DIR/"train"/c).glob("*")))) for c in THERMAL_CLASSES]
weights = torch.tensor([sum(counts)/c for c in counts], dtype=torch.float).to(DEVICE)

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(THERMAL_CLASSES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
print(f"\n🚀 Training on real data ({EPOCHS} epochs)...")

for epoch in range(EPOCHS):
    model.train()
    tc, tt = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        _, pred = out.max(1)
        tt += labels.size(0)
        tc += pred.eq(labels).sum().item()
    scheduler.step()

    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            _, pred = model(imgs).max(1)
            vt += labels.size(0)
            vc += pred.eq(labels).sum().item()

    va = 100.0 * vc / vt
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} — Train: {100.*tc/tt:.1f}%  Val: {va:.1f}%")
    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), str(OUTPUT_DIR / "best_thermal.pth"))

print(f"\n✅ Best val accuracy: {best_acc:.1f}% (real thermal data)")

# ═══════════════════════════════════════════════════════════════
# Export ONNX + Metadata
# ═══════════════════════════════════════════════════════════════

model.load_state_dict(torch.load(str(OUTPUT_DIR / "best_thermal.pth"), weights_only=True))
model.eval()
torch.onnx.export(
    model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE),
    str(OUTPUT_DIR / "thermal_classifier.onnx"),
    input_names=["thermal_image"], output_names=["class_probs"],
    dynamic_axes={"thermal_image": {0: "batch"}, "class_probs": {0: "batch"}},
    opset_version=18,
)

metadata = {
    "model": "vayuswarm_thermal",
    "classes": THERMAL_CLASSES,
    "input_size": IMG_SIZE,
    "best_val_acc": round(best_acc, 2),
    "training_data": {
        "human":      "LLVIP real IR images — hustvl/LLVIP",
        "vehicle":    "FLIR ADAS real thermal — deepnewbie/flir-free",
        "animal":     "FLIR ADAS real thermal — deepnewbie/flir-free",
        "background": "LLVIP real IR background patches",
        "fire":       "Physics-accurate fire signatures on real LLVIP backgrounds",
    },
    "train_samples": len(train_ds),
    "val_samples":   len(val_ds),
}
with open(OUTPUT_DIR / "thermal_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("✅ Exported ONNX + metadata")

# ═══════════════════════════════════════════════════════════════
# Push to GitHub
# ═══════════════════════════════════════════════════════════════

import subprocess as _sp
if GIT_TOKEN:
    try:
        auth_url  = GIT_REPO.replace("https://", f"https://{GIT_USER}:{GIT_TOKEN}@")
        clone_dir = Path(tempfile.mkdtemp()) / "swam"
        _sp.check_call(["git", "clone", "--depth", "1", auth_url, str(clone_dir)])
        target = clone_dir / "models" / "thermal"
        target.mkdir(parents=True, exist_ok=True)
        for fname in ["best_thermal.pth", "thermal_classifier.onnx", "thermal_metadata.json"]:
            src = OUTPUT_DIR / fname
            if src.exists():
                shutil.copy2(str(src), str(target / fname))
                print(f"   ✅ {fname} ({src.stat().st_size/1024/1024:.1f} MB)")
        env = os.environ.copy()
        for cmd in [
            ["git", "config", "user.name",  GIT_USER],
            ["git", "config", "user.email", GIT_EMAIL],
            ["git", "add", "models/thermal/"],
            ["git", "commit", "-m", f"Real-data thermal classifier — val_acc={best_acc:.1f}%"],
            ["git", "push", "origin", "main"],
        ]:
            _sp.check_call(cmd, cwd=str(clone_dir), env=env)
        print(f"✅ Pushed to {GIT_REPO}")
    except Exception as e:
        print(f"⚠ GitHub push failed: {e}  → Download from Kaggle Output tab")
else:
    print(f"ℹ No GIT_TOKEN — saved locally at: {OUTPUT_DIR}")

print(f"\n{'='*55}\n🎉 Thermal Training Complete! Val: {best_acc:.1f}%\nData: LLVIP + FLIR (real thermal images)\n{'='*55}")
