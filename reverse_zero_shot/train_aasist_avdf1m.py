"""
==============================================================================
[train_aasist_avdf1m.py] AASIST 학습 on AV-Deepfake1M

[설정]
  학습: Real 2000 + Fake 8000 (1:4)
  Epoch: 20, Batch: 8
  Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
  Loss: BCE with pos_weight=0.25 (1:4 비율 보정)
  Pretrained: PGF 학습 가중치에서 fine-tune (aasist_model_pgf_best.pth)

[출력]
  ./aasist_model_avdf1m_best.pth

[예상 시간] ~30분 (RTX 3060)

[사용법]
  cd ~/hsh/AIApplication/reverse_zero_shot
  python train_aasist_avdf1m.py
==============================================================================
"""
import sys
import os
import json
import time
import functools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path: sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)

torch.load = functools.partial(torch.load, weights_only=False)

from aasist.models.AASIST import Model as AASISTModel
from avdf1m_train_data import (
    build_avdf1m_train_val_split, load_audio_for_aasist
)


# ═══════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    'EPOCHS': 20,
    'BATCH_SIZE': 8,
    'LR': 5e-4,
    'WEIGHT_DECAY': 1e-4,
    'POS_WEIGHT': 0.25,
    'SEED': 42,
    'PATIENCE': 5,
    'PRETRAINED': 'aasist_model_pgf_best.pth',  # 더 가까운 도메인에서 시작
    'OUTPUT': 'aasist_model_avdf1m_best.pth',
    'AASIST_CONFIG': os.path.join(PARENT_DIR, "aasist/config/AASIST.conf"),
}


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class AASISTDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _try_load(self, idx):
        row = self.df.iloc[idx]
        if not os.path.exists(row['video_path']):
            return None
        audio = load_audio_for_aasist(row['video_path'])
        if audio is None: return None
        return audio, torch.tensor(float(row['video_label'])), idx

    def __getitem__(self, idx):
        for offset in range(len(self)):
            r = self._try_load((idx + offset) % len(self))
            if r is not None: return r
        raise RuntimeError("로드 가능한 샘플 없음")


# ═══════════════════════════════════════════════════════════════════
# 학습/평가
# ═══════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0; n = 0
    t0 = time.time()
    
    for batch_idx, (audio, label, _) in enumerate(loader):
        audio = audio.to(device); label = label.to(device)
        
        # AASIST: (hidden, logits 2-class)
        _, logits = model(audio)
        # BCE를 위해 fake(idx=1) logit 추출
        fake_logit = logits[:, 1] - logits[:, 0]
        
        loss = criterion(fake_logit, label)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(label); n += len(label)
        
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (batch_idx + 1) / elapsed
            print(f"   [E{epoch:02d} {batch_idx+1:4d}/{len(loader)}] "
                  f"loss={total_loss/n:.4f} speed={speed:.1f}/s")
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []; all_labels = []
    for audio, label, _ in loader:
        audio = audio.to(device)
        _, logits = model(audio)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(label.numpy().tolist())
    
    probs = np.array(all_probs); labels = np.array(all_labels)
    auc = roc_auc_score(labels, probs) * 100 if len(np.unique(labels)) > 1 else 0.0
    preds = (probs > 0.5).astype(int)
    f1 = f1_score(labels.astype(int), preds, zero_division=0) * 100
    return auc, f1


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("🎯 AASIST 학습 on AV-Deepfake1M")
    print("=" * 70)
    print(f"설정: {json.dumps(CONFIG, indent=2)}")
    
    torch.manual_seed(CONFIG['SEED'])
    np.random.seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Device: {device}")
    
    # 데이터
    train_df, val_df = build_avdf1m_train_val_split(seed=CONFIG['SEED'])
    train_ds = AASISTDataset(train_df)
    val_ds = AASISTDataset(val_df)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'],
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'],
                            shuffle=False, num_workers=2)
    
    # 모델
    with open(CONFIG['AASIST_CONFIG'], 'r') as f:
        aasist_cfg = json.load(f)
    model = AASISTModel(aasist_cfg['model_config'])
    
    # Pretrained load
    if os.path.exists(CONFIG['PRETRAINED']):
        print(f"\n📥 Pretrained 로드: {CONFIG['PRETRAINED']}")
        model.load_state_dict(torch.load(CONFIG['PRETRAINED'], map_location=device))
    else:
        print(f"\n⚠️ Pretrained 없음: {CONFIG['PRETRAINED']} — 처음부터 학습")
    
    model = model.to(device)
    
    # Loss/Optimizer
    pos_weight = torch.tensor([CONFIG['POS_WEIGHT']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG['LR'],
                                  weight_decay=CONFIG['WEIGHT_DECAY'])
    
    # 학습
    best_auc = 0; patience_counter = 0
    history = []
    
    for epoch in range(1, CONFIG['EPOCHS'] + 1):
        print(f"\n{'='*70}\n[Epoch {epoch}/{CONFIG['EPOCHS']}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_auc, val_f1 = evaluate(model, val_loader, device)
        
        print(f"\n   📊 train_loss={train_loss:.4f}  val_AUC={val_auc:.2f}%  val_F1={val_f1:.2f}%")
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_auc': val_auc, 'val_f1': val_f1})
        
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG['OUTPUT'])
            print(f"   💾 최고 성능 갱신 → {CONFIG['OUTPUT']}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['PATIENCE']:
                print(f"\n   ⏹️ Early stopping (patience={CONFIG['PATIENCE']})")
                break
    
    # 학습 history 저장
    pd.DataFrame(history).to_csv('aasist_avdf1m_history.csv', index=False)
    print(f"\n✅ 완료. 최고 AUC: {best_auc:.2f}%")
    print(f"   가중치: {CONFIG['OUTPUT']}")
    print(f"   history: aasist_avdf1m_history.csv")


if __name__ == "__main__":
    main()