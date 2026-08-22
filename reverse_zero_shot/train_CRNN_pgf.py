"""
==============================================================================
[PolyGlotFake 학습] CRNN+GRU+Attention (오디오 감정 모델)

[실행 위치] ~/hsh/AIApplication/reverse_zero_shot/
[필요 파일]
- 같은 폴더: polyglotfake_data.py
- 상위 폴더: audio_emotion_deepfake_detector.py, audio_emotion_crnn_best.pth

[출력]
- audio_flow_deepfake_pgf_best.pth
- audio_flow_logs_pgf/history.csv
==============================================================================
"""

import sys
import os
import functools
import time

# 경로 설정
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

torch.load = functools.partial(torch.load, weights_only=False)

from polyglotfake_data import build_polyglotfake_train_val

try:
    from audio_emotion_deepfake_detector import (
        AudioEmotionFlowDetector, extract_audio_segments
    )
except ImportError:
    try:
        from train_CRNN import (
            AudioEmotionFlowDetector, extract_audio_segments
        )
    except ImportError:
        print("❌ AudioEmotionFlowDetector import 실패")
        print(f"   상위 폴더({PARENT_DIR})에서 다음 파일 중 하나가 필요:")
        print("   - audio_emotion_deepfake_detector.py")
        print("   - train_CRNN.py")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class PolyGlotFakeAudioDataset(Dataset):
    def __init__(self, df, num_segments=16, segment_duration=3.0, target_sr=16000):
        self.df               = df.reset_index(drop=True)
        self.num_segments     = num_segments
        self.segment_duration = segment_duration
        self.target_sr        = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
        segments = extract_audio_segments(
            video_path, num_segments=self.num_segments,
            target_sr=self.target_sr, segment_duration=self.segment_duration
        )
        if segments is None:
            return self.__getitem__((idx + 1) % len(self))
        return segments, torch.tensor(row['video_label'], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 학습/검증
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (segments, labels) in enumerate(loader):
        segments, labels = segments.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(segments)
        loss = criterion(logits, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠️  E{epoch:02d} [{i:3d}] NaN/Inf, skip")
            optimizer.zero_grad()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"  E{epoch:02d} [{i:3d}/{len(loader)}] Loss={loss.item():.4f}")
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for segments, labels in loader:
        segments, labels = segments.to(device), labels.to(device)
        logits, _ = model(segments)
        total_loss += criterion(logits, labels).item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    preds = (np.array(all_probs) > 0.5).astype(int)
    acc = (preds == np.array(all_labels)).mean() * 100
    try: auc = roc_auc_score(all_labels, all_probs) * 100
    except: auc = 0.0
    return avg_loss, acc, auc


# ══════════════════════════════════════════════════════════════════════════════
# 3. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        CKPT_PATH        = "audio_flow_deepfake_pgf_best.pth",
        LOG_DIR          = "audio_flow_logs_pgf",
        # 사전학습 가중치는 상위 폴더에 있음
        PRETRAINED_PATH  = os.path.join(PARENT_DIR, "audio_emotion_crnn_best.pth"),

        BATCH_SIZE       = 8,
        EPOCHS           = 20,
        LR               = 5e-4,
        WEIGHT_DECAY     = 1e-4,
        VAL_RATIO        = 0.1,
        NUM_WORKERS      = 3,
        NUM_FAKE_SAMPLE  = 2000,
        NUM_SEGMENTS     = 16,
        SEGMENT_DURATION = 3.0,
        TARGET_SR        = 16000,
        GRU_HIDDEN       = 128,
        DROPOUT          = 0.4,
        PATIENCE         = 5,
        SEED             = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)

    if not os.path.exists(CFG['PRETRAINED_PATH']):
        print(f"❌ 사전학습 가중치 없음: {CFG['PRETRAINED_PATH']}")
        sys.exit(1)

    train_df, val_df = build_polyglotfake_train_val(
        num_fake_sample=CFG['NUM_FAKE_SAMPLE'],
        val_ratio=CFG['VAL_RATIO'], seed=CFG['SEED']
    )

    train_loader = DataLoader(
        PolyGlotFakeAudioDataset(train_df, CFG['NUM_SEGMENTS'],
                                 CFG['SEGMENT_DURATION'], CFG['TARGET_SR']),
        batch_size=CFG['BATCH_SIZE'], shuffle=True,
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        PolyGlotFakeAudioDataset(val_df, CFG['NUM_SEGMENTS'],
                                 CFG['SEGMENT_DURATION'], CFG['TARGET_SR']),
        batch_size=CFG['BATCH_SIZE'],
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )

    model = AudioEmotionFlowDetector(
        pretrained_path=CFG['PRETRAINED_PATH'],
        num_segments=CFG['NUM_SEGMENTS'],
        gru_hidden=CFG['GRU_HIDDEN'],
        dropout=CFG['DROPOUT']
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"🔧 학습 파라미터: {sum(p.numel() for p in trainable):,}개")

    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_weight_val = torch.tensor(
        [num_neg / num_pos], dtype=torch.float32
    ).to(device)
    print(f"⚖️  BCE pos_weight: {pos_weight_val.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = optim.AdamW(trainable, lr=CFG['LR'],
                            weight_decay=CFG['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['EPOCHS']
    )

    print("\n" + "="*60)
    print("🚀 PolyGlotFake 학습 시작 (CRNN+GRU+Attention)")
    print("="*60)

    best_auc, history = 0.0, []
    patience_counter, best_val_loss = 0, float('inf')

    for epoch in range(CFG['EPOCHS']):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc, val_auc = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n🏁 E{epoch:02d} | Train={train_loss:.4f} | "
              f"Val Loss={val_loss:.4f} | Acc={val_acc:.1f}% | "
              f"AUC={val_auc:.1f}% | LR={current_lr:.5f} | {elapsed:.0f}s\n")

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc,
        })
        pd.DataFrame(history).to_csv(
            os.path.join(CFG['LOG_DIR'], "history.csv"), index=False
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'cfg': CFG, 'epoch': epoch, 'val_auc': val_auc,
            }, CFG['CKPT_PATH'])
            print(f"  ⭐ Best 저장 (AUC={best_auc:.1f}%)")

        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
            if patience_counter >= CFG['PATIENCE']:
                print(f"  ⏹ Early Stopping")
                break

    print(f"\n✅ 완료! Best AUC: {best_auc:.1f}%")
    print(f"   체크포인트: {os.path.abspath(CFG['CKPT_PATH'])}")


if __name__ == "__main__":
    main()