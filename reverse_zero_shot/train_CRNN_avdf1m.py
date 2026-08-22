"""
==============================================================================
[train_CRNN_avdf1m.py] CRNN+GRU 학습 on AV-Deepfake1M

[설정]
  학습: Real 2000 + Fake 8000 (1:4)
  Epoch: 20, Batch: 8
  Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
  Loss: BCE with pos_weight=0.25
  Pretrained: PGF v2 가중치 (audio_flow_deepfake_pgf_v2_best.pth)
  Backbone: audio_emotion_crnn_best.pth (RAVDESS 사전학습)

[출력]
  ./audio_flow_deepfake_avdf1m_best.pth

[예상 시간] ~1시간 (RTX 3060)

[사용법]
  cd ~/hsh/AIApplication/reverse_zero_shot
  python train_CRNN_avdf1m.py
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
    from audio_emotion_deepfake_detector import (
        AudioEmotionFlowDetector, extract_audio_segments
    )
except ImportError:
    from train_CRNN import AudioEmotionFlowDetector, extract_audio_segments

from avdf1m_train_data import build_avdf1m_train_val_split


# ═══════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    'EPOCHS': 20,
    'BATCH_SIZE': 8,
    'LR': 1e-4,                                            # 5e-4 → 1e-4 (낮춤)
    'WEIGHT_DECAY': 1e-4,
    'POS_WEIGHT': 0.25,
    'NUM_SEGMENTS': 16,
    'SEGMENT_DURATION': 3.0,
    'GRU_HIDDEN': 128,
    'DROPOUT': 0.4,
    'SEED': 42,
    'PATIENCE': 7,                                         # 5 → 7 (낮은 lr 보정)
    'UNFREEZE_PRETRAINED_GRU': True,
    'PRETRAINED_FROM_PGF': None,                           # PGF 로드 제거 (RAVDESS만 사용)
    'PRETRAINED_BACKBONE': os.path.join(PARENT_DIR, "audio_emotion_crnn_best.pth"),
    'OUTPUT': 'audio_flow_deepfake_avdf1m_v2_best.pth',    # v2로 구분
    'HISTORY': 'crnn_avdf1m_v2_history.csv',
}


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
class CRNNDataset(Dataset):
    def __init__(self, df, num_segments=16, segment_duration=3.0):
        self.df = df.reset_index(drop=True)
        self.num_segments = num_segments
        self.segment_duration = segment_duration

    def __len__(self):
        return len(self.df)

    def _try_load(self, idx):
        row = self.df.iloc[idx]
        if not os.path.exists(row['video_path']):
            return None
        segs = extract_audio_segments(
            row['video_path'],
            num_segments=self.num_segments,
            target_sr=16000,
            segment_duration=self.segment_duration,
        )
        if segs is None: return None
        return segs, torch.tensor(float(row['video_label'])), idx

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
    
    for batch_idx, (segs, label, _) in enumerate(loader):
        segs = segs.to(device); label = label.to(device)
        logit, _ = model(segs)
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
    for segs, label, _ in loader:
        segs = segs.to(device)
        logit, _ = model(segs)
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
    print("🎯 CRNN+GRU 학습 on AV-Deepfake1M")
    print("=" * 70)
    print(f"설정: {json.dumps({k: v for k, v in CONFIG.items() if not k.startswith('PRETRAINED')}, indent=2)}")
    
    torch.manual_seed(CONFIG['SEED']); np.random.seed(CONFIG['SEED'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Device: {device}")
    
    train_df, val_df = build_avdf1m_train_val_split(seed=CONFIG['SEED'])
    train_ds = CRNNDataset(train_df, CONFIG['NUM_SEGMENTS'], CONFIG['SEGMENT_DURATION'])
    val_ds = CRNNDataset(val_df, CONFIG['NUM_SEGMENTS'], CONFIG['SEGMENT_DURATION'])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)
    
    # 모델
    model = AudioEmotionFlowDetector(
        pretrained_path=CONFIG['PRETRAINED_BACKBONE'],
        num_segments=CONFIG['NUM_SEGMENTS'],
        gru_hidden=CONFIG['GRU_HIDDEN'],
        dropout=CONFIG['DROPOUT'],
        unfreeze_pretrained_gru=CONFIG['UNFREEZE_PRETRAINED_GRU'],
    ).to(device)
    
    # PGF v2 가중치에서 fine-tune (None이면 스킵)
    if CONFIG['PRETRAINED_FROM_PGF'] and os.path.exists(CONFIG['PRETRAINED_FROM_PGF']):
        print(f"\n📥 PGF 가중치 로드: {CONFIG['PRETRAINED_FROM_PGF']}")
        ckpt = torch.load(CONFIG['PRETRAINED_FROM_PGF'], map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        try:
            model.load_state_dict(state)
            print("   ✅ 전체 가중치 로드 성공")
        except RuntimeError as e:
            print(f"   ⚠️ 일부 가중치 불일치, strict=False로 재시도")
            model.load_state_dict(state, strict=False)
    else:
        print(f"\n🆕 PGF pretrain 없이 RAVDESS backbone만 사용 (from-scratch fine-tune)")
    
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
    
    pd.DataFrame(history).to_csv(CONFIG['HISTORY'], index=False)
    print(f"\n✅ 완료. 최고 AUC: {best_auc:.2f}%")
    print(f"   가중치: {CONFIG['OUTPUT']}")
    print(f"   history: {CONFIG['HISTORY']}")


if __name__ == "__main__":
    main()