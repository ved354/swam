"""
VayuSwarm — Behavior Transformer Training
═══════════════════════════════════════════════════════════════════
Run on Kaggle with GPU. Trains lightweight Transformer for trajectory classification.

Input:  30-frame window of [x, y, speed, heading, acceleration]
Output: behavior class (patrol, evasive, formation, stationary, approaching)
Auto-pushes trained model to GitHub.

KAGGLE SECRETS: GITHUB_TOKEN, GITHUB_REPO
Runtime: ~15 min on T4 GPU
"""

import subprocess
subprocess.check_call(["pip", "install", "-q", "torch", "onnx", "onnxscript", "gitpython", "scikit-learn"])

import json, math, shutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

GITHUB_TOKEN = "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"
GITHUB_REPO = "ved354/swam"
GITHUB_BRANCH = "main"

WINDOW = 30; FEAT = 5; DIM = 64; HEADS = 4; LAYERS = 2
EPOCHS = 50; BATCH = 128; LR = 0.001
N_TRAIN = 10000; N_VAL = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("/kaggle/working/behavior_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["patrol", "evasive", "formation", "stationary", "approaching"]

try:
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    GITHUB_TOKEN = GITHUB_TOKEN or s.get_secret("GITHUB_TOKEN")
    GITHUB_REPO = GITHUB_REPO or s.get_secret("GITHUB_REPO")
except Exception: pass

# ═══════════════════════════════════════════════════════════════
# Generate Synthetic Trajectories
# ═══════════════════════════════════════════════════════════════

def gen_traj(behavior, n=WINDOW):
    seq = np.zeros((n, FEAT))
    x, y = np.random.uniform(-100,100), np.random.uniform(-100,100)
    heading = np.random.uniform(0,360); speed = 0.0
    for t in range(n):
        if behavior == "patrol":
            speed = np.random.normal(5,.3); accel = np.random.normal(0,.2)
            if t % np.random.choice([10,15,20]) < 2: heading += np.random.normal(90,10)
        elif behavior == "evasive":
            speed = np.random.uniform(3,15); heading += np.random.normal(0,40)
            if np.random.random()<.3: heading += np.random.choice([-90,90,180])
            accel = np.random.normal(0,3)
        elif behavior == "formation":
            speed = np.random.normal(8,.2); heading += np.random.normal(0,1.5); accel = np.random.normal(0,.1)
        elif behavior == "stationary":
            speed = np.random.exponential(.2); heading = np.random.uniform(0,360); accel = np.random.normal(0,.05)
        elif behavior == "approaching":
            th = math.degrees(math.atan2(-y,-x)); heading = heading*.8+th*.2
            speed = 3+t*.3+np.random.normal(0,.5); accel = .3+np.random.normal(0,.1)
        heading %= 360; rad = math.radians(heading)
        x += speed*math.cos(rad)*.1; y += speed*math.sin(rad)*.1
        seq[t] = [x, y, speed, heading/360, accel]
    return seq

def gen_dataset(n):
    X = np.zeros((n, WINDOW, FEAT), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    per = n // len(CLASSES)
    for ci, cn in enumerate(CLASSES):
        for i in range(per):
            X[ci*per+i] = gen_traj(cn); y[ci*per+i] = ci
    p = np.random.permutation(len(y))
    return X[p], y[p]

print("🔧 Generating trajectories...")
X_tr, y_tr = gen_dataset(N_TRAIN)
X_va, y_va = gen_dataset(N_VAL)
mu, std = X_tr.mean((0,1)), X_tr.std((0,1))+1e-8
X_tr = (X_tr-mu)/std; X_va = (X_va-mu)/std
np.save(str(OUTPUT_DIR/"norm_mean.npy"), mu)
np.save(str(OUTPUT_DIR/"norm_std.npy"), std)
print(f"✅ Train: {X_tr.shape}, Val: {X_va.shape}")

# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════

class TrajDS(Dataset):
    def __init__(self,X,y): self.X=torch.FloatTensor(X); self.y=torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

class BehaviorTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(FEAT, DIM)
        self.pos = nn.Parameter(torch.randn(1, WINDOW, DIM)*.1)
        enc = nn.TransformerEncoderLayer(DIM, HEADS, DIM*2, .1, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, LAYERS)
        self.head = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM,DIM), nn.GELU(), nn.Dropout(.1), nn.Linear(DIM, len(CLASSES)))
    def forward(self, x):
        x = self.proj(x) + self.pos
        return self.head(self.tf(x).mean(1))

tr_loader = DataLoader(TrajDS(X_tr, y_tr), batch_size=BATCH, shuffle=True)
va_loader = DataLoader(TrajDS(X_va, y_va), batch_size=BATCH)

model = BehaviorTransformer().to(DEVICE)
crit = nn.CrossEntropyLoss()
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params:,} params")

# ═══════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════

best_acc = 0.0
for ep in range(EPOCHS):
    model.train()
    for X, y in tr_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(); crit(model(X), y).backward(); opt.step()
    sched.step()

    model.eval(); vc, vt = 0, 0; preds, labs = [], []
    with torch.no_grad():
        for X, y in va_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            p = model(X).argmax(1); vt += y.size(0); vc += p.eq(y).sum().item()
            preds.extend(p.cpu().numpy()); labs.extend(y.cpu().numpy())
    va = 100.*vc/vt
    if (ep+1)%10==0: print(f"Epoch {ep+1}/{EPOCHS} — Val: {va:.1f}%")
    if va > best_acc:
        best_acc = va; torch.save(model.state_dict(), str(OUTPUT_DIR/"best_behavior.pth"))

print(f"\n✅ Best: {best_acc:.1f}%")
print(classification_report(labs, preds, target_names=CLASSES))

# ═══════════════════════════════════════════════════════════════
# Export ONNX
# ═══════════════════════════════════════════════════════════════

model.load_state_dict(torch.load(str(OUTPUT_DIR/"best_behavior.pth"), weights_only=True)); model.eval()
torch.onnx.export(model, torch.randn(1,WINDOW,FEAT).to(DEVICE),
    str(OUTPUT_DIR/"behavior_transformer.onnx"), input_names=["trajectory"],
    output_names=["behavior"], dynamic_axes={"trajectory":{0:"batch"},"behavior":{0:"batch"}}, opset_version=18)

meta = {"model":"vayuswarm_behavior","classes":CLASSES,"window":WINDOW,"features":FEAT,"params":params,"best_acc":best_acc}
with open(OUTPUT_DIR/"behavior_metadata.json","w") as f: json.dump(meta, f, indent=2)
print("✅ Exported ONNX")

# ═══════════════════════════════════════════════════════════════
# Auto-Push to GitHub
# ═══════════════════════════════════════════════════════════════

if GITHUB_TOKEN and GITHUB_REPO:
    try:
        import git
        clone_dir = Path("/kaggle/working/repo_clone")
        if clone_dir.exists(): shutil.rmtree(clone_dir)
        try:
            repo = git.Repo.clone_from(f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git", str(clone_dir), branch=GITHUB_BRANCH)
        except:
            repo = git.Repo.clone_from(f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git", str(clone_dir))
            repo.git.checkout("-b", GITHUB_BRANCH)

        target = clone_dir/"models"/"behavior"
        target.mkdir(parents=True, exist_ok=True)
        for f in [OUTPUT_DIR/"best_behavior.pth", OUTPUT_DIR/"behavior_transformer.onnx",
                  OUTPUT_DIR/"behavior_metadata.json", OUTPUT_DIR/"norm_mean.npy", OUTPUT_DIR/"norm_std.npy"]:
            if f.exists(): shutil.copy2(str(f), str(target/f.name))

        repo.config_writer().set_value("user","name","VayuSwarm-Bot").release()
        repo.config_writer().set_value("user","email","bot@vayuswarm.ai").release()
        repo.git.add(A=True)
        if repo.is_dirty() or repo.untracked_files:
            repo.index.commit(f"🤖 Add behavior Transformer (val_acc: {best_acc:.1f}%)")
            repo.remote("origin").push(GITHUB_BRANCH)
            print(f"✅ Pushed to: https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/models/behavior")
        shutil.rmtree(clone_dir, ignore_errors=True)
    except Exception as e:
        print(f"⚠ GitHub push failed: {e}")
        print(f"  → Model saved locally at: {OUTPUT_DIR}")
        print(f"  → Fix: Go to github.com/settings/tokens → Edit token → Enable 'Contents: Read and write'")
else:
    print(f"⚠ No GitHub creds — saved locally: {OUTPUT_DIR}")

print("\n🎉 Behavior Transformer training complete!")
