"""
train_aasist.py
AVDF1M train set → AASIST fine-tune
레이블: audio_modified, both_modified=1 / real, visual_modified=0
"""

import os, sys, json, random, functools
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import av
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

torch.load = functools.partial(torch.load, weights_only=False)
BASE_APP = os.path.expanduser("~/hsh/AIApplication")
sys.path.insert(0, BASE_APP)
from aasist.models.AASIST import Model as AASISTModel

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(AASIST_CFG_TRAIN["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 오디오 로드 ───────────────────────────────────────────────────
def load_audio(path, sr=16000, max_len=64000):
    try:
        container = av.open(path)
        if not container.streams.audio:
            container.close(); return None
        orig_sr = container.streams.audio[0].rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.ndim > 1: arr = arr.mean(axis=0)
            frames.append(arr)
        container.close()
        if not frames: return None
        wav = torch.from_numpy(np.concatenate(frames)).unsqueeze(0)
        if orig_sr != sr:
            wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
        if wav.shape[1] > max_len: wav = wav[:, :max_len]
        else: wav = torch.nn.functional.pad(wav,(0,max_len-wav.shape[1]))
        return wav.squeeze()
    except: return None

# ── Dataset ───────────────────────────────────────────────────────
class AudioDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        for offset in range(len(self.samples)):
            s = self.samples[(idx+offset) % len(self.samples)]
            a = load_audio(s["path"])
            if a is not None:
                return a, torch.tensor(float(s["label"]))
        raise RuntimeError("로드 가능한 샘플 없음")

# ── 샘플 구성 ─────────────────────────────────────────────────────
print("메타데이터 로드 중...")
with open(AVDF1M_TRAIN_META) as f:
    meta = json.load(f)

all_s = []
for m in meta:
    path = os.path.join(AVDF1M_TRAIN_ROOT, m["file"])
    if not os.path.exists(path): continue
    label = 1 if m["modify_type"] in ("audio_modified","both_modified") else 0
    all_s.append({"path": path, "label": label})

fake_s = [s for s in all_s if s["label"]==1]
real_s = [s for s in all_s if s["label"]==0]
n = min(len(fake_s), len(real_s), AASIST_CFG_TRAIN["n_per_class"])
random.shuffle(fake_s); random.shuffle(real_s)
samples = fake_s[:n] + real_s[:n]
random.shuffle(samples)

n_val   = int(len(samples) * AASIST_CFG_TRAIN["val_ratio"])
val_s   = samples[:n_val]
train_s = samples[n_val:]
print(f"train: {len(train_s):,}개  val: {len(val_s):,}개")

train_loader = DataLoader(AudioDataset(train_s),
    batch_size=AASIST_CFG_TRAIN["batch_size"], shuffle=True,
    num_workers=2, pin_memory=True)
val_loader   = DataLoader(AudioDataset(val_s),
    batch_size=AASIST_CFG_TRAIN["batch_size"], shuffle=False,
    num_workers=2, pin_memory=True)

# ── 모델 ─────────────────────────────────────────────────────────
print("AASIST 로드 중...")
with open(AASIST_CFG) as f:
    aasist_json = json.load(f)
model = AASISTModel(aasist_json["model_config"])
if os.path.exists(FAKEAV_AASIST_CKPT):
    model.load_state_dict(torch.load(FAKEAV_AASIST_CKPT, map_location="cpu"))
    print("  FakeAV 가중치로 초기화 (transfer learning)")
model = model.to(DEVICE)

criterion    = nn.CrossEntropyLoss()
optimizer    = torch.optim.AdamW(model.parameters(),
                lr=AASIST_CFG_TRAIN["lr"], weight_decay=0.01)
total_steps  = len(train_loader) * AASIST_CFG_TRAIN["epochs"]
warmup_steps = int(total_steps * AASIST_CFG_TRAIN["warmup_ratio"])
scheduler    = get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps)

# ── 평가 ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for audios, labels in loader:
        _, out = model(audios.to(DEVICE))
        probs  = torch.softmax(out,dim=1)[:,1].cpu().numpy()
        all_probs.extend(probs); all_labels.extend(labels.numpy())
    labels = np.array(all_labels); probs = np.array(all_probs)
    preds  = (probs>0.5).astype(int)
    return {"auc": roc_auc_score(labels,probs),
            "f1":  f1_score(labels,preds,zero_division=0),
            "acc": accuracy_score(labels,preds)}

# ── 학습 루프 ─────────────────────────────────────────────────────
best_auc = 0.0
log      = []
for epoch in range(1, AASIST_CFG_TRAIN["epochs"]+1):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader,
                desc=f"Epoch {epoch}/{AASIST_CFG_TRAIN['epochs']} [AASIST]")
    for audios, labels in pbar:
        audios = audios.to(DEVICE); labels = labels.long().to(DEVICE)
        optimizer.zero_grad()
        _, out = model(audios)
        loss   = criterion(out, labels)
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
                    "val_auc": best_auc, "cfg": AASIST_CFG_TRAIN},
                   AASIST_SAVE_PATH)
        print(f"  ✅ Best 저장 (AUC={best_auc*100:.2f}%)")

with open(os.path.join(RESULTS_DIR,"aasist_train_log.json"),"w") as f:
    json.dump(log, f, indent=2)
print(f"\n✅ AASIST 학습 완료 | Best AUC: {best_auc*100:.2f}%")
print(f"   저장: {AASIST_SAVE_PATH}")
