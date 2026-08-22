"""
==============================================================================
[train_HSEmotion_avdf1m.py] HSEmotion+GRU 학습 on AV-Deepfake1M

[설정]
  학습: Real 2000 + Fake 8000 (1:4)
  Epoch: 20, Batch: 8
  Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
  Loss: BCE with pos_weight=0.25
  Backbone: enet_b0_8_best_afew (HSEmotion lib, 자동 다운로드)
  Unfreeze: 마지막 2 블록 (timm EfficientNet)
  Pretrained: PGF v2 가중치 (emotion_flow_lite_pgf_v2_best.pth)

[출력]
  ./emotion_flow_lite_avdf1m_best.pth

[예상 시간] ~2시간 (RTX 3060)
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

try:
    from emotion_deepfake_detector_lite import EmotionFlowDetectorLite
except ImportError:
    from train_HSEmotion import EmotionFlowDetectorLite

from avdf1m_train_data import build_avdf1m_train_val_split, load_frames_for_hsemo


# ═══════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    'EPOCHS': 20,
    'BATCH_SIZE': 8,
    'LR': 5e-4,
    'WEIGHT_DECAY': 1e-4,
    'POS_WEIGHT': 0.25,
    'NUM_FRAMES': 16,
    'GRU_HIDDEN': 64,
    'DROPOUT': 0.3,
    'SEED': 42,
    'PATIENCE': 5,
    'UNFREEZE_LAST_BLOCKS': 2,
    'MODEL_NAME': 'enet_b0_8_best_afew',
    'PRETRAINED_FROM_PGF': 'emotion_flow_lite_pgf_v2_best.pth',
    'OUTPUT': 'emotion_flow_lite_avdf1m_best.pth',
}


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class HSEmoDataset(Dataset):
    def __init__(self, df, num_frames=16):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def _try_load(self, idx):
        row = self.df.iloc[idx]
        if not os.path.exists(row['video_path']):
            return None
        frames = load_frames_for_hsemo(row['video_path'], self.num_frames)
        if frames is None: return None
        return frames, torch.tensor(float(row['video_label'])), idx

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
    
    for batch_idx, (frames, label, _) in enumerate(loader):
        frames = frames.to(device); label = label.to(device)
        logit, _ = model(frames)
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
    for frames, label, _ in loader:
        frames = frames.to(device)
        logit, _ = model(frames)
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
    print("🎯 HSEmotion+GRU 학습 on AV-Deepfake1M")
    print("=" * 70)
    print(f"설정: {json.dumps({k: v for k, v in CONFIG.items() if not k.startswith('PRETRAINED')}, indent=2)}")
    
    torch.manual_seed(CONFIG['SEED']); np.random.seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Device: {device}")
    
    train_df, val_df = build_avdf1m_train_val_split(seed=CONFIG['SEED'])
    train_ds = HSEmoDataset(train_df, CONFIG['NUM_FRAMES'])
    val_ds = HSEmoDataset(val_df, CONFIG['NUM_FRAMES'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)
    
    # 모델
    model = EmotionFlowDetectorLite(
        model_name=CONFIG['MODEL_NAME'],
        num_frames=CONFIG['NUM_FRAMES'],
        gru_hidden=CONFIG['GRU_HIDDEN'],
        dropout=CONFIG['DROPOUT'],
        device='cpu',
        unfreeze_last_blocks=CONFIG['UNFREEZE_LAST_BLOCKS'],
    ).to(device)
    
    # PGF v2 가중치에서 fine-tune
    if os.path.exists(CONFIG['PRETRAINED_FROM_PGF']):
        print(f"\n📥 PGF 가중치 로드: {CONFIG['PRETRAINED_FROM_PGF']}")
        ckpt = torch.load(CONFIG['PRETRAINED_FROM_PGF'], map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        try:
            model.load_state_dict(state)
            print("   ✅ 전체 가중치 로드 성공")
        except RuntimeError as e:
            print(f"   ⚠️ 일부 가중치 불일치, strict=False로 재시도")
            model.load_state_dict(state, strict=False)
    
    pos_weight = torch.tensor([CONFIG['POS_WEIGHT']]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['LR'],
        weight_decay=CONFIG['WEIGHT_DECAY'],
    )
    
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'cfg': CONFIG,
                'val_auc': val_auc,
                'val_f1': val_f1,
                'epoch': epoch,
            }, CONFIG['OUTPUT'])
            print(f"   💾 최고 성능 갱신 → {CONFIG['OUTPUT']}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['PATIENCE']:
                print(f"\n   ⏹️ Early stopping (patience={CONFIG['PATIENCE']})")
                break
    
    pd.DataFrame(history).to_csv('hsemo_avdf1m_history.csv', index=False)
    print(f"\n✅ 완료. 최고 AUC: {best_auc:.2f}%")
    print(f"   가중치: {CONFIG['OUTPUT']}")
    print(f"   history: hsemo_avdf1m_history.csv")


if __name__ == "__main__":
    main()