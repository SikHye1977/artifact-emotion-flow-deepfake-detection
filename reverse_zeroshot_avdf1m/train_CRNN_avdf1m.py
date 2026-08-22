"""
==============================================================================
[AV-Deepfake1M 학습] CRNN+GRU+Attention (오디오 감정 모델)

[목적]
기존 train_CRNN_pgf.py를 AV-Deepfake1M용으로 변환.
모델 구조 완전히 동일.

[변경점]
  1. 데이터 로더: PolyGlotFakeAudioDataset → AVDF1MAudioDataset
  2. 데이터 분할: build_polyglotfake_train_val → build_avdf1m_train_val
  3. 가중치 저장: audio_flow_deepfake_avdf1m_best.pth
  4. 로그 폴더: audio_flow_logs_avdf1m/

[실행 위치] ~/hsh/AIApplication/reverse_zeroshot_avdf1m/

[필요 파일]
- 같은 폴더: avdf1m_data.py
- 상위 폴더 (~/hsh/AIApplication/):
    * audio_emotion_deepfake_detector.py (또는 train_CRNN.py)
    * audio_emotion_crnn_best.pth (RAVDESS 사전학습 가중치)
    * AV-Deepfake1M_RootFiles/ 데이터셋

[출력]
- audio_flow_deepfake_avdf1m_best.pth
- audio_flow_logs_avdf1m/history.csv
==============================================================================
"""

import sys
import os
import functools
import time

# ─ 경로 설정 ───────────────────────────────────────────────────────────
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

# 같은 폴더의 데이터 유틸
from avdf1m_data import build_avdf1m_train_val

# 상위 폴더의 모델 + 함수
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
class AVDF1MAudioDataset(Dataset):
    """
    AV-Deepfake1M 영상에서 16구간 오디오 추출.
    PolyGlotFakeAudioDataset과 입력/출력 형식 동일.
    """
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
# 2. 학습/검증 함수 (PGF 학습 코드 재사용)
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (segments, labels) in enumerate(loader):
        segments, labels = segments.to(device), labels.to(device)
        optimizer.zero_grad()
        # FP16 underflow 방지: FP32 강제
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
        # 출력 경로 (현재 폴더 기준)
        CKPT_PATH        = "audio_flow_deepfake_avdf1m_best.pth",
        LOG_DIR          = "audio_flow_logs_avdf1m",

        # 사전학습 가중치는 상위 폴더에 있음
        PRETRAINED_PATH  = os.path.join(PARENT_DIR, "audio_emotion_crnn_best.pth"),

        # 학습 설정 (FakeAVCeleb/PolyGlotFake와 동일)
        BATCH_SIZE       = 8,
        EPOCHS           = 20,
        LR               = 5e-4,
        WEIGHT_DECAY     = 1e-4,
        VAL_RATIO        = 0.1,
        NUM_WORKERS      = 3,
        NUM_SEGMENTS     = 16,
        SEGMENT_DURATION = 3.0,
        TARGET_SR        = 16000,
        GRU_HIDDEN       = 128,
        DROPOUT          = 0.4,
        PATIENCE         = 5,

        # AVDF1M 데이터 (Real 1000 + Fake 4000)
        N_REAL           = 1000,
        N_FAKE           = 4000,
        SEED             = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")
    print(f"📁 Parent dir : {PARENT_DIR}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)

    if not os.path.exists(CFG['PRETRAINED_PATH']):
        print(f"❌ 사전학습 가중치 없음: {CFG['PRETRAINED_PATH']}")
        sys.exit(1)

    # ── 데이터 분할 ─────────────────────────────────────────────────────
    train_df, val_df = build_avdf1m_train_val(
        n_real    = CFG['N_REAL'],
        n_fake    = CFG['N_FAKE'],
        val_ratio = CFG['VAL_RATIO'],
        seed      = CFG['SEED']
    )

    train_loader = DataLoader(
        AVDF1MAudioDataset(train_df, CFG['NUM_SEGMENTS'],
                            CFG['SEGMENT_DURATION'], CFG['TARGET_SR']),
        batch_size=CFG['BATCH_SIZE'], shuffle=True,
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        AVDF1MAudioDataset(val_df, CFG['NUM_SEGMENTS'],
                            CFG['SEGMENT_DURATION'], CFG['TARGET_SR']),
        batch_size=CFG['BATCH_SIZE'],
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )

    # ── 모델 ────────────────────────────────────────────────────────────
    model = AudioEmotionFlowDetector(
        pretrained_path=CFG['PRETRAINED_PATH'],
        num_segments=CFG['NUM_SEGMENTS'],
        gru_hidden=CFG['GRU_HIDDEN'],
        dropout=CFG['DROPOUT']
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"🔧 학습 파라미터: {sum(p.numel() for p in trainable):,}개")

    # pos_weight 자동 계산 (PGF/FAV 방식 그대로)
    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_weight_val = torch.tensor(
        [num_neg / num_pos], dtype=torch.float32
    ).to(device)
    print(f"⚖️  BCE pos_weight: {pos_weight_val.item():.4f} "
          f"(neg/pos = {num_neg}/{num_pos})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = optim.AdamW(trainable, lr=CFG['LR'],
                            weight_decay=CFG['WEIGHT_DECAY'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['EPOCHS']
    )

    # ── 학습 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🚀 AV-Deepfake1M 학습 시작 (CRNN+GRU+Attention)")
    print("=" * 60)

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

        # Early Stopping (PGF와 동일)
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