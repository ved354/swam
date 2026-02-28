"""
VayuSwarm — Thermal Classifier Training
═══════════════════════════════════════════════════════════════════
Run on Kaggle with GPU. Trains MobileNetV3 to classify thermal blobs.

Classes: background, human, vehicle, animal, fire
Output: PyTorch + ONNX → auto-pushed to GitHub

KAGGLE SECRETS: GITHUB_TOKEN, GITHUB_REPO
Runtime: ~20 min on T4 GPU
"""

import subprocess
subprocess.check_call(["pip", "install", "-q", "torch", "torchvision", "onnx", "onnxscript", "gitpython"])

import json, shutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

GITHUB_TOKEN = "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"
GITHUB_REPO = "ved354/swam"
GITHUB_BRANCH = "main"

EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("/kaggle/working/thermal_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = Path("/kaggle/working/thermal_dataset")

THERMAL_CLASSES = ["background", "human", "vehicle", "animal", "fire"]

try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    GITHUB_TOKEN = GITHUB_TOKEN or s.get_secret("GITHUB_TOKEN")
    GITHUB_REPO = GITHUB_REPO or s.get_secret("GITHUB_REPO")
except Exception:
    pass

print(f"Device: {DEVICE} | Classes: {THERMAL_CLASSES}")

# ═══════════════════════════════════════════════════════════════
# Generate Synthetic Thermal Dataset
# ═══════════════════════════════════════════════════════════════

def _gen_thermal(cls_name):
    img = np.random.normal(80, 10, (IMG_SIZE, IMG_SIZE)).clip(0, 255).astype(np.uint8)
    if cls_name == "background":
        g = np.linspace(60, 100, IMG_SIZE).reshape(1, -1)
        img = (img * 0.7 + g * 0.3).astype(np.uint8)
    elif cls_name == "human":
        cx, cy = np.random.randint(60, 164, 2)
        w, h = np.random.randint(20, 40), np.random.randint(40, 80)
        y1, y2 = max(0, cy-h//2), min(IMG_SIZE, cy+h//2)
        x1, x2 = max(0, cx-w//2), min(IMG_SIZE, cx+w//2)
        img[y1:y2, x1:x2] = np.random.normal(180, 15, (y2-y1, x2-x1)).clip(150, 220).astype(np.uint8)
    elif cls_name == "vehicle":
        cx, cy = np.random.randint(50, 174, 2)
        w, h = np.random.randint(50, 80), np.random.randint(30, 50)
        y1, y2 = max(0, cy-h//2), min(IMG_SIZE, cy+h//2)
        x1, x2 = max(0, cx-w//2), min(IMG_SIZE, cx+w//2)
        img[y1:y2, x1:x2] = np.random.normal(140, 10, (y2-y1, x2-x1)).clip(120, 160).astype(np.uint8)
        er = np.random.randint(8, 15)
        ex, ey = x1 + max(5, (x2-x1)//4), cy
        for dy in range(-er, er+1):
            for dx in range(-er, er+1):
                if dx**2+dy**2 <= er**2:
                    py, px = ey+dy, ex+dx
                    if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                        img[py, px] = np.random.randint(200, 240)
    elif cls_name == "animal":
        cx, cy = np.random.randint(40, 184, 2)
        w, h = np.random.randint(15, 30), np.random.randint(10, 25)
        y1, y2 = max(0, cy-h//2), min(IMG_SIZE, cy+h//2)
        x1, x2 = max(0, cx-w//2), min(IMG_SIZE, cx+w//2)
        img[y1:y2, x1:x2] = np.random.normal(170, 12, (y2-y1, x2-x1)).clip(140, 200).astype(np.uint8)
    elif cls_name == "fire":
        cx, cy = np.random.randint(50, 174, 2)
        for _ in range(np.random.randint(3, 8)):
            sx, sy, r = cx+np.random.randint(-20,20), cy+np.random.randint(-30,10), np.random.randint(5,20)
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx**2+dy**2 <= r**2:
                        py, px = sy+dy, sx+dx
                        if 0 <= py < IMG_SIZE and 0 <= px < IMG_SIZE:
                            img[py, px] = np.random.randint(230, 255)
    return Image.fromarray(img, mode="L")

print("🔧 Generating thermal dataset...")
for split, n in [("train", 2000), ("val", 400)]:
    for cls_idx, cls_name in enumerate(THERMAL_CLASSES):
        d = DATASET_DIR / split / cls_name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n // len(THERMAL_CLASSES)):
            _gen_thermal(cls_name).save(str(d / f"{cls_name}_{i:04d}.png"))
print(f"✅ Generated 2000 train + 400 val images")

# ═══════════════════════════════════════════════════════════════
# Dataset & Model
# ═══════════════════════════════════════════════════════════════

class ThermalDS(Dataset):
    def __init__(self, root, transform):
        self.samples = []
        self.transform = transform
        for ci, cn in enumerate(THERMAL_CLASSES):
            for p in (Path(root)/cn).glob("*.png"):
                self.samples.append((str(p), ci))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        img = Image.open(self.samples[i][0]).convert("L")
        return self.transform(img), self.samples[i][1]

t_train = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10), transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)), transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
t_val = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)), transforms.Normalize([.5,.5,.5],[.5,.5,.5])])

train_loader = DataLoader(ThermalDS(str(DATASET_DIR/"train"), t_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(ThermalDS(str(DATASET_DIR/"val"), t_val), batch_size=BATCH_SIZE, num_workers=2)

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(THERMAL_CLASSES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ═══════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward(); optimizer.step()
        _, pred = model(imgs).max(1)
        total += labels.size(0); correct += pred.eq(labels).sum().item()
    scheduler.step()

    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            _, pred = model(imgs).max(1)
            vt += labels.size(0); vc += pred.eq(labels).sum().item()

    va = 100.*vc/vt
    if (epoch+1) % 5 == 0: print(f"Epoch {epoch+1}/{EPOCHS} — Val: {va:.1f}%")
    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), str(OUTPUT_DIR/"best_thermal.pth"))

print(f"\n✅ Best accuracy: {best_acc:.1f}%")

# ═══════════════════════════════════════════════════════════════
# Export ONNX
# ═══════════════════════════════════════════════════════════════

model.load_state_dict(torch.load(str(OUTPUT_DIR/"best_thermal.pth"), weights_only=True))
model.eval()
torch.onnx.export(model, torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(DEVICE),
    str(OUTPUT_DIR/"thermal_classifier.onnx"), input_names=["thermal_image"],
    output_names=["class_probs"], dynamic_axes={"thermal_image":{0:"batch"},"class_probs":{0:"batch"}}, opset_version=18)

meta = {"model":"vayuswarm_thermal","classes":THERMAL_CLASSES,"input_size":IMG_SIZE,"best_val_acc":best_acc}
with open(OUTPUT_DIR/"thermal_metadata.json","w") as f: json.dump(meta, f, indent=2)
print(f"✅ Exported ONNX")

# ═══════════════════════════════════════════════════════════════
# Auto-Push to GitHub
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

        target = clone_dir / "models" / "thermal"
        target.mkdir(parents=True, exist_ok=True)

        for f in [OUTPUT_DIR/"best_thermal.pth", OUTPUT_DIR/"thermal_classifier.onnx", OUTPUT_DIR/"thermal_metadata.json"]:
            if f.exists(): shutil.copy2(str(f), str(target/f.name))

        repo.config_writer().set_value("user","name","VayuSwarm-Bot").release()
        repo.config_writer().set_value("user","email","bot@vayuswarm.ai").release()
        repo.git.add(A=True)
        if repo.is_dirty() or repo.untracked_files:
            repo.index.commit(f"🤖 Add thermal classifier (val_acc: {best_acc:.1f}%)")
            repo.remote("origin").push(GITHUB_BRANCH)
            print(f"✅ Pushed to: https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/models/thermal")
        shutil.rmtree(clone_dir, ignore_errors=True)
    except Exception as e:
        print(f"⚠ GitHub push failed: {e}")
        print(f"  → Model is saved locally at: {OUTPUT_DIR}")
        print(f"  → Fix: Go to github.com/settings/tokens → Edit token → Enable 'Contents: Read and write'")
else:
    print(f"⚠ No GitHub creds — saved locally: {OUTPUT_DIR}")

print("\n🎉 Thermal classifier training complete!")
