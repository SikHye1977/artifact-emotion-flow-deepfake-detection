"""
==============================================================================
[v2 강화 버전] FakeAVCeleb CRNN 학습 (Pretrained GRU Unfreeze)
==============================================================================

[목적]
양방향 일관성을 위해 FakeAVCeleb CRNN도 v2로 재학습.
주 목적: F→F AUC 81.46% 개선

[출력]
- audio_flow_deepfake_fav_v2_best.pth
- audio_flow_logs_fav_v2/history.csv
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
from sklearn.model_selection import train_test_split

torch.load = functools.partial(torch.load, weights_only=False)

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
# 1. FakeAVCeleb 데이터 로더
# ══════════════════════════════════════════════════════════════════════════════
def find_fakeavceleb_root():
    candidates = [
        os.path.abspath("FakeAVCeleb_v1.2"),
        os.path.abspath("../FakeAVCeleb_v1.2"),
        os.path.abspath("../../FakeAVCeleb_v1.2"),
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'meta_data.csv')):
            return path
    raise FileNotFoundError("FakeAVCeleb_v1.2 폴더를 찾을 수 없습니다.")


def build_fakeavceleb_train_val(num_fake_sample=2000, val_ratio=0.1, seed=42):
    base_dir = find_fakeavceleb_root()
    csv_path = os.path.join(base_dir, 'meta_data.csv')
    
    df = pd.read_csv(csv_path)
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)
    
    real_df = df[df['video_label'] == 0.0].reset_index(drop=True)
    fake_df = df[df['video_label'] == 1.0].reset_index(drop=True)
    
    print(f"📊 FakeAVCeleb 원본: Real={len(real_df)}, Fake={len(fake_df)}")
    
    n_fake = min(num_fake_sample, len(fake_df))
    sampled_fake = fake_df.sample(n=n_fake, random_state=seed)
    
    balanced_df = pd.concat([real_df, sampled_fake]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    
    def make_path(row):
        rel_dir = row.iloc[-2].replace("FakeAVCeleb", os.path.basename(base_dir))
        return os.path.join(os.path.dirname(base_dir), rel_dir, row['path'])
    
    balanced_df['video_path'] = balanced_df.apply(make_path, axis=1)
    exists_mask = balanced_df['video_path'].apply(os.path.exists)
    balanced_df = balanced_df[exists_mask].reset_index(drop=True)
    
    print(f"✂️  학습 데이터: 총 {len(balanced_df)}개")
    
    train_df, val_df = train_test_split(
        balanced_df, test_size=val_ratio,
        stratify=balanced_df['video_label'], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class FakeAVCelebAudioDataset(Dataset):
    def __init__(self, df, num_segments=16, segment_duration=3.0, target_sr=16000):
        self.df = df.reset_index(drop=True)
        self.num_segments = num_segments
        self.segment_duration = segment_duration
        self.target_sr = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        segments = extract_audio_segments(
            row['video_path'], num_segments=self.num_segments,
            target_sr=self.target_sr, segment_duration=self.segment_duration
        )
        if segments is None:
            return self.__getitem__((idx + 1) % len(self))
        return segments, torch.tensor(row['video_label'], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 학습/검증 (PGF v2와 동일)
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
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        CKPT_PATH        = "audio_flow_deepfake_fav_v2_best.pth",
        LOG_DIR          = "audio_flow_logs_fav_v2",
        PRETRAINED_PATH  = os.path.join(PARENT_DIR, "audio_emotion_crnn_best.pth"),

        BATCH_SIZE       = 8,
        EPOCHS           = 25,
        WEIGHT_DECAY     = 1e-4,

        # NEW: Unfreeze
        UNFREEZE_PRETRAINED_GRU = True,
        LR_PRETRAINED    = 1e-5,
        LR_HEAD          = 5e-4,

        VAL_RATIO        = 0.1,
        NUM_WORKERS      = 3,
        NUM_FAKE_SAMPLE  = 2000,
        NUM_SEGMENTS     = 16,
        SEGMENT_DURATION = 3.0,
        TARGET_SR        = 16000,
        GRU_HIDDEN       = 128,
        DROPOUT          = 0.4,
        PATIENCE         = 7,
        SEED             = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)

    if not os.path.exists(CFG['PRETRAINED_PATH']):
        print(f"❌ 사전학습 가중치 없음: {CFG['PRETRAINED_PATH']}")
        sys.exit(1)

    train_df, val_df = build_fakeavceleb_train_val(
        num_fake_sample=CFG['NUM_FAKE_SAMPLE'],
        val_ratio=CFG['VAL_RATIO'], seed=CFG['SEED']
    )

    train_loader = DataLoader(
        FakeAVCelebAudioDataset(train_df, CFG['NUM_SEGMENTS'],
                                CFG['SEGMENT_DURATION'], CFG['TARGET_SR']),
        batch_size=CFG['BATCH_SIZE'], shuffle=True,
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        FakeAVCelebAudioDataset(val_df, CFG['NUM_SEGMENTS'],
                                CFG['SEGMENT_DURATION'], CFG['TARGET_SR']),
        batch_size=CFG['BATCH_SIZE'],
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )

    print(f"\n🔧 모델 생성 (unfreeze_pretrained_gru={CFG['UNFREEZE_PRETRAINED_GRU']})")
    model = AudioEmotionFlowDetector(
        pretrained_path=CFG['PRETRAINED_PATH'],
        num_segments=CFG['NUM_SEGMENTS'],
        gru_hidden=CFG['GRU_HIDDEN'],
        dropout=CFG['DROPOUT'],
        unfreeze_pretrained_gru=CFG['UNFREEZE_PRETRAINED_GRU']
    ).to(device)

    pretrained_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'audio_backbone' in name:
            pretrained_params.append(param)
        else:
            head_params.append(param)
    
    print(f"📊 학습 가능 파라미터:")
    print(f"   Pretrained: {sum(p.numel() for p in pretrained_params):,}")
    print(f"   Head      : {sum(p.numel() for p in head_params):,}")

    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_weight_val = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    param_groups = []
    if pretrained_params:
        param_groups.append({
            'params': pretrained_params, 'lr': CFG['LR_PRETRAINED'],
            'weight_decay': CFG['WEIGHT_DECAY']
        })
    param_groups.append({
        'params': head_params, 'lr': CFG['LR_HEAD'],
        'weight_decay': CFG['WEIGHT_DECAY']
    })

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])

    print("\n" + "="*60)
    print("🚀 FakeAVCeleb 학습 시작 (CRNN v2)")
    print("="*60)

    best_auc, history = 0.0, []
    patience_counter, best_val_loss = 0, float('inf')

    for epoch in range(CFG['EPOCHS']):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lrs = [g['lr'] for g in optimizer.param_groups]
        print(f"\n🏁 E{epoch:02d} | Train={train_loss:.4f} | "
              f"Val Loss={val_loss:.4f} | Acc={val_acc:.1f}% | "
              f"AUC={val_auc:.1f}% | LR={lrs} | {elapsed:.0f}s\n")

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc,
        })
        pd.DataFrame(history).to_csv(os.path.join(CFG['LOG_DIR'], "history.csv"), index=False)

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


if __name__ == "__main__":
    main()