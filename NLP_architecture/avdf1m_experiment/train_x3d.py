"""
train_x3d.py
AVDF1M train set → X3D_m fine-tune
레이블: visual_modified, both_modified=1 / real, audio_modified=0
"""

import os, sys, json, random, functools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import av

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as _Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = _Ftv

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

import functools
torch.load = functools.partial(torch.load, weights_only=False)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(X3D_CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 전처리 ────────────────────────────────────────────────────────
def rescale(x): return x / 255.0
def to_tc(x):   return x.permute(1,0,2,3)
def to_ct(x):   return x.permute(1,0,2,3)

transform = Compose([
    UniformTemporalSubsample(16), Lambda(rescale),
    Lambda(to_tc), Normalize([0.45,0.45,0.45],[0.225,0.225,0.225]),
    Lambda(to_ct), ShortSideScale(256), Resize((224,224))
])

def load_video(path):
    try:
        container = av.open(path)
        frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(frames) < 16: return None
        if len(frames) > 128:
            idx = np.linspace(0, len(frames)-1, 128, dtype=int)
            frames = [frames[i] for i in idx]
        return torch.from_numpy(np.stack(frames)).permute(3,0,1,2).float()
    except: return None

# ── Dataset ───────────────────────────────────────────────────────
class VideoDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        for offset in range(len(self.samples)):
            s = self.samples[(idx+offset) % len(self.samples)]
            v = load_video(s["path"])
            if v is not None:
                return transform(v), torch.tensor(float(s["label"]))
        raise RuntimeError("로드 가능한 샘플 없음")

# ── 샘플 구성 ─────────────────────────────────────────────────────
print("메타데이터 로드 중...")
with open(AVDF1M_TRAIN_META) as f:
    meta = json.load(f)

all_s = []
for m in meta:
    path = os.path.join(AVDF1M_TRAIN_ROOT, m["file"])
    if not os.path.exists(path): continue
    label = 1 if m["modify_type"] in ("visual_modified","both_modified") else 0
    all_s.append({"path": path, "label": label, "type": m["modify_type"]})

fake_s = [s for s in all_s if s["label"]==1]
real_s = [s for s in all_s if s["label"]==0]
n = min(len(fake_s), len(real_s), X3D_CFG["n_per_class"])
random.shuffle(fake_s); random.shuffle(real_s)
samples = fake_s[:n] + real_s[:n]
random.shuffle(samples)

n_val   = int(len(samples) * X3D_CFG["val_ratio"])
val_s   = samples[:n_val]
train_s = samples[n_val:]
print(f"train: {len(train_s):,}개  val: {len(val_s):,}개")
print(f"  fake: {sum(1 for s in train_s if s['label']==1):,} / "
      f"real: {sum(1 for s in train_s if s['label']==0):,}")

train_loader = DataLoader(VideoDataset(train_s),
    batch_size=X3D_CFG["batch_size"], shuffle=True,
    num_workers=2, pin_memory=True)
val_loader   = DataLoader(VideoDataset(val_s),
    batch_size=X3D_CFG["batch_size"], shuffle=False,
    num_workers=2, pin_memory=True)

# ── 모델 ─────────────────────────────────────────────────────────
print("X3D_m 로드 중...")
model = x3d_m(pretrained=False)
model.blocks[5].proj       = nn.Linear(2048, 1)
model.blocks[5].activation = nn.Identity()
if os.path.exists(FAKEAV_X3D_CKPT):
    model.load_state_dict(torch.load(FAKEAV_X3D_CKPT, map_location="cpu"))
    print("  FakeAV 가중치로 초기화 (transfer learning)")
model = model.to(DEVICE)

criterion    = nn.BCEWithLogitsLoss()
optimizer    = torch.optim.AdamW(model.parameters(),
                lr=X3D_CFG["lr"], weight_decay=0.01)
total_steps  = len(train_loader) * X3D_CFG["epochs"]
warmup_steps = int(total_steps * X3D_CFG["warmup_ratio"])
scheduler    = get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps)

# ── 평가 ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for videos, labels in loader:
        logits = model(videos.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    labels = np.array(all_labels); probs = np.array(all_probs)
    preds  = (probs>0.5).astype(int)
    return {"auc": roc_auc_score(labels,probs),
            "f1":  f1_score(labels,preds,zero_division=0),
            "acc": accuracy_score(labels,preds)}

# ── 학습 루프 ─────────────────────────────────────────────────────
best_auc = 0.0
log      = []
for epoch in range(1, X3D_CFG["epochs"]+1):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{X3D_CFG['epochs']} [X3D]")
    for videos, labels in pbar:
        videos = videos.to(DEVICE); labels = labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(videos).squeeze(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    train_loss /= len(train_loader)
    vm = evaluate(model, val_loader)
    print(f"\nEpoch {epoch} | train_loss={train_loss:.4f} | "
          f"AUC={vm['auc']*100:.2f}%  F1={vm['f1']*100:.2f}%  "
          f"ACC={vm['acc']*100:.2f}%")
    log.append({"epoch": epoch, "train_loss": train_loss, **vm})
    if vm["auc"] > best_auc:
        best_auc = vm["auc"]
        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                    "val_auc": best_auc, "cfg": X3D_CFG}, X3D_SAVE_PATH)
        print(f"  ✅ Best 저장 (AUC={best_auc*100:.2f}%)")

with open(os.path.join(RESULTS_DIR,"x3d_train_log.json"),"w") as f:
    json.dump(log, f, indent=2)
print(f"\n✅ X3D 학습 완료 | Best AUC: {best_auc*100:.2f}%")
print(f"   저장: {X3D_SAVE_PATH}")
