"""
==============================================================================
[train_x3d_avdf1m.py] X3D_m 학습 on AV-Deepfake1M

[설정]
  학습: Real 2000 + Fake 8000 (1:4)
  Epoch: 20, Batch: 8
  Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
  Loss: BCE with pos_weight=0.25
  Pretrained: PGF 가중치 (x3d_model_pgf_best.pth)
  헤드 교체: blocks[5].proj = Linear(2048, 1)

[출력]
  ./x3d_model_avdf1m_best.pth

[예상 시간] ~2시간 (RTX 3060)

[사용법]
  cd ~/hsh/AIApplication/reverse_zero_shot
  python train_x3d_avdf1m.py
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path: sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)

torch.load = functools.partial(torch.load, weights_only=False)

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = Ftv

from pytorchvideo.models.hub import x3d_m

from avdf1m_train_data import (
    build_avdf1m_train_val_split,
    load_video_for_x3d,
    x3d_video_transform,
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
    'PRETRAINED_FROM_PGF': 'x3d_model_pgf_best.pth',
    'OUTPUT': 'x3d_model_avdf1m_best.pth',
}


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class X3DDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _try_load(self, idx):
        row = self.df.iloc[idx]
        if not os.path.exists(row['video_path']):
            return None
        video = load_video_for_x3d(row['video_path'])
        if video is None: return None
        video = x3d_video_transform(video)
        return video, torch.tensor(float(row['video_label'])), idx

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
    
    for batch_idx, (video, label, _) in enumerate(loader):
        video = video.to(device); label = label.to(device)
        logit = model(video).squeeze(-1)  # (B, 1) → (B,)
        loss = criterion(logit, label)
        
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
    for video, label, _ in loader:
        video = video.to(device)
        logit = model(video).squeeze(-1)
        probs = torch.sigmoid(logit).cpu().numpy()
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
    print("🎯 X3D_m 학습 on AV-Deepfake1M")
    print("=" * 70)
    print(f"설정: {json.dumps({k: v for k, v in CONFIG.items() if not k.startswith('PRETRAINED')}, indent=2)}")
    
    torch.manual_seed(CONFIG['SEED']); np.random.seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Device: {device}")
    
    train_df, val_df = build_avdf1m_train_val_split(seed=CONFIG['SEED'])
    train_ds = X3DDataset(train_df)
    val_ds = X3DDataset(val_df)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)
    
    # 모델 (헤드 교체)
    model = x3d_m(pretrained=False)
    model.blocks[5].proj = nn.Linear(2048, 1)
    model.blocks[5].activation = nn.Identity()
    
    # PGF 가중치에서 fine-tune
    if os.path.exists(CONFIG['PRETRAINED_FROM_PGF']):
        print(f"\n📥 PGF 가중치 로드: {CONFIG['PRETRAINED_FROM_PGF']}")
        state = torch.load(CONFIG['PRETRAINED_FROM_PGF'], map_location=device)
        # PGF 가중치는 보통 순수 state_dict 형태
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        try:
            model.load_state_dict(state)
            print("   ✅ 전체 가중치 로드 성공")
        except RuntimeError as e:
            print(f"   ⚠️ 일부 가중치 불일치, strict=False로 재시도")
            model.load_state_dict(state, strict=False)
    
    model = model.to(device)
    
    pos_weight = torch.tensor([CONFIG['POS_WEIGHT']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CONFIG['LR'],
                                  weight_decay=CONFIG['WEIGHT_DECAY'])
    
    best_auc = 0; patience_counter = 0; history = []
    
    for epoch in range(1, CONFIG['EPOCHS'] + 1):
        print(f"\n{'='*70}\n[Epoch {epoch}/{CONFIG['EPOCHS']}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_auc, val_f1 = evaluate(model, val_loader, device)
        
        print(f"\n   📊 train_loss={train_loss:.4f}  val_AUC={val_auc:.2f}%  val_F1={val_f1:.2f}%")
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_auc': val_auc, 'val_f1': val_f1})
        
        if val_auc > best_auc:
            best_auc = val_auc; patience_counter = 0
            # X3D는 기존 관습대로 순수 state_dict로 저장 (load 시 호환성)
            torch.save(model.state_dict(), CONFIG['OUTPUT'])
            # 추가로 메타데이터도 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'cfg': CONFIG,
                'val_auc': val_auc,
                'val_f1': val_f1,
                'epoch': epoch,
            }, CONFIG['OUTPUT'].replace('.pth', '_full.pth'))
            print(f"   💾 최고 성능 갱신 → {CONFIG['OUTPUT']}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['PATIENCE']:
                print(f"\n   ⏹️ Early stopping (patience={CONFIG['PATIENCE']})")
                break
    
    pd.DataFrame(history).to_csv('x3d_avdf1m_history.csv', index=False)
    print(f"\n✅ 완료. 최고 AUC: {best_auc:.2f}%")
    print(f"   가중치: {CONFIG['OUTPUT']}")
    print(f"   메타포함: {CONFIG['OUTPUT'].replace('.pth', '_full.pth')}")
    print(f"   history: x3d_avdf1m_history.csv")


if __name__ == "__main__":
    main()