"""
==============================================================================
[v2 강화 버전] PolyGlotFake CRNN 학습 (Pretrained GRU Unfreeze)
==============================================================================

[v1 대비 변경점]
1. unfreeze_pretrained_gru=True: 사전학습 GRU도 fine-tuning
2. Layer-wise LR: pretrained 1e-5, head 5e-4
3. SpecAugment 추가 (Frequency/Time Masking) — 단, mel_spec 단계에 적용 필요
4. Epoch 증가 (20 → 25)
5. 별도 가중치 파일명 (충돌 방지)

[참고]
CRNN은 v1에서도 P→P AUC 97.25%로 이미 우수함.
v2는 F→F (81.46%) 개선이 주 목적.
PolyGlotFake 학습본도 같이 v2로 만들어서 일관성 유지.

[출력]
- audio_flow_deepfake_pgf_v2_best.pth
- audio_flow_logs_pgf_v2/history.csv
==============================================================================
"""

import sys
import os
import functools
import time

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
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dataset (오디오 증강 추가는 모델 내부 mel_spec 단계에서 적용 필요)
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
        # 출력 (v2)
        CKPT_PATH        = "audio_flow_deepfake_pgf_v2_best.pth",
        LOG_DIR          = "audio_flow_logs_pgf_v2",
        PRETRAINED_PATH  = os.path.join(PARENT_DIR, "audio_emotion_crnn_best.pth"),

        # 학습 설정
        BATCH_SIZE       = 8,
        EPOCHS           = 25,           # 20 → 25
        WEIGHT_DECAY     = 1e-4,

        # NEW: Unfreeze 설정
        UNFREEZE_PRETRAINED_GRU = True,  # 사전학습 GRU도 fine-tune

        # NEW: Layer-wise LR
        LR_PRETRAINED    = 1e-5,         # 사전학습 GRU에는 작은 LR
        LR_HEAD          = 5e-4,         # head는 기존 LR

        # 데이터
        VAL_RATIO        = 0.1,
        NUM_WORKERS      = 3,
        NUM_FAKE_SAMPLE  = 2000,
        NUM_SEGMENTS     = 16,
        SEGMENT_DURATION = 3.0,
        TARGET_SR        = 16000,
        GRU_HIDDEN       = 128,
        DROPOUT          = 0.4,
        PATIENCE         = 7,            # 5 → 7
        SEED             = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)

    if not os.path.exists(CFG['PRETRAINED_PATH']):
        print(f"❌ 사전학습 가중치 없음: {CFG['PRETRAINED_PATH']}")
        sys.exit(1)

    # 데이터
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

    # 모델 (NEW: unfreeze_pretrained_gru=True)
    print(f"\n🔧 모델 생성 (unfreeze_pretrained_gru={CFG['UNFREEZE_PRETRAINED_GRU']})")
    model = AudioEmotionFlowDetector(
        pretrained_path=CFG['PRETRAINED_PATH'],
        num_segments=CFG['NUM_SEGMENTS'],
        gru_hidden=CFG['GRU_HIDDEN'],
        dropout=CFG['DROPOUT'],
        unfreeze_pretrained_gru=CFG['UNFREEZE_PRETRAINED_GRU']  # NEW
    ).to(device)

    # 학습 가능 파라미터 분류
    pretrained_params = []
    head_params = []
    pretrained_count = 0
    head_count = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # audio_backbone 안의 파라미터는 사전학습 (작은 LR)
        if 'audio_backbone' in name:
            pretrained_params.append(param)
            pretrained_count += param.numel()
        else:
            head_params.append(param)
            head_count += param.numel()
    
    total_count = pretrained_count + head_count
    print(f"📊 학습 가능 파라미터:")
    print(f"   Pretrained: {pretrained_count:,} ({pretrained_count/total_count*100:.1f}%)")
    print(f"   Head      : {head_count:,} ({head_count/total_count*100:.1f}%)")
    print(f"   총합      : {total_count:,}")

    # pos_weight
    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_weight_val = torch.tensor(
        [num_neg / num_pos], dtype=torch.float32
    ).to(device)
    print(f"⚖️  BCE pos_weight: {pos_weight_val.item():.4f}")

    # Layer-wise LR
    print(f"📐 Layer-wise LR:")
    print(f"   Pretrained LR: {CFG['LR_PRETRAINED']:.0e}")
    print(f"   Head LR      : {CFG['LR_HEAD']:.0e}")
    
    param_groups = []
    if pretrained_params:
        param_groups.append({
            'params': pretrained_params,
            'lr': CFG['LR_PRETRAINED'],
            'weight_decay': CFG['WEIGHT_DECAY']
        })
    param_groups.append({
        'params': head_params,
        'lr': CFG['LR_HEAD'],
        'weight_decay': CFG['WEIGHT_DECAY']
    })
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['EPOCHS']
    )

    # 학습
    print("\n" + "="*60)
    print("🚀 PolyGlotFake 학습 시작 (CRNN v2 — GRU Unfreeze)")
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
        lrs = [g['lr'] for g in optimizer.param_groups]
        print(f"\n🏁 E{epoch:02d} | Train={train_loss:.4f} | "
              f"Val Loss={val_loss:.4f} | Acc={val_acc:.1f}% | "
              f"AUC={val_auc:.1f}% | LR={lrs} | {elapsed:.0f}s\n")

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc,
            'lr_pretrained': lrs[0] if len(lrs) > 1 else 0,
            'lr_head': lrs[-1],
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
                print(f"  ⏹ Early Stopping ({CFG['PATIENCE']} epoch 미개선)")
                break

    print(f"\n✅ 완료! Best AUC: {best_auc:.1f}%")
    print(f"   체크포인트: {os.path.abspath(CFG['CKPT_PATH'])}")


if __name__ == "__main__":
    main()