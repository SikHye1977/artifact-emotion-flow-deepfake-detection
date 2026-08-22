"""
==============================================================================
[PolyGlotFake 학습] MAE-DFER 백본 + GRU + Attention
Train Emotion Flow Detector with MAE-DFER on PolyGlotFake
==============================================================================

[주요 특징]
- HSEmotion v1이 PolyGlotFake에서 학습 실패 (P→P AUC 51.67%)
- MAE-DFER 백본은 VoxCeleb2 자기지도 학습 → 다국어 화자에 더 강건할 것
- 같은 학습 설정으로 비교 (4가지 시나리오 평가용)

[출력]
- emotion_flow_mae_pgf_best.pth
- emotion_mae_logs_pgf/history.csv
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
import av
from torchvision import transforms
from sklearn.metrics import roc_auc_score

torch.load = functools.partial(torch.load, weights_only=False)

from polyglotfake_data import build_polyglotfake_train_val
from emotion_deepfake_detector_mae import EmotionFlowDetectorMAE


# ══════════════════════════════════════════════════════════════════════════════
# 1. 영상 전처리 (FAV와 동일, 160x160)
# ══════════════════════════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def extract_uniform_frames(video_path, num_frames=16, transform=None):
    try:
        container = av.open(video_path)
        all_frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
    except Exception:
        return None
    if len(all_frames) < num_frames:
        return None
    if transform is None:
        transform = val_transform
    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    return torch.stack([transform(all_frames[i]) for i in indices])


# ══════════════════════════════════════════════════════════════════════════════
# 2. PolyGlotFake Dataset
# ══════════════════════════════════════════════════════════════════════════════
class PolyGlotFakeVideoDataset(Dataset):
    def __init__(self, df, num_frames=16, mode='train'):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.transform = train_transform if mode == 'train' else val_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
        if not os.path.exists(video_path):
            return self.__getitem__((idx + 1) % len(self))
        
        frames = extract_uniform_frames(video_path, self.num_frames, self.transform)
        if frames is None:
            return self.__getitem__((idx + 1) % len(self))
        
        return frames, torch.tensor(row['video_label'], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 학습/검증 (FAV와 동일)
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (frames, labels) in enumerate(loader):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(frames)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"  E{epoch:02d} [{i:3d}/{len(loader)}] Loss={loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(frames)
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
        MAE_DFER_PATH    = os.path.expanduser("~/hsh/AIApplication/mae_dfer"),
        PRETRAINED_CKPT  = os.path.expanduser(
            "~/hsh/AIApplication/mae_dfer/saved/pretrained/checkpoint-49.pth"
        ),
        
        CKPT_PATH        = "emotion_flow_mae_pgf_best.pth",
        LOG_DIR          = "emotion_mae_logs_pgf",
        
        NUM_FRAMES       = 16,
        BATCH_SIZE       = 4,
        EPOCHS           = 15,
        LR               = 1e-3,
        LR_BACKBONE      = 1e-5,
        WEIGHT_DECAY     = 1e-4,
        GRU_HIDDEN       = 64,
        DROPOUT          = 0.3,
        
        UNFREEZE_LAST_BLOCKS = 0,
        
        NUM_FAKE_SAMPLE  = 2000,
        VAL_RATIO        = 0.1,
        NUM_WORKERS      = 3,
        SEED             = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)
    
    if not os.path.exists(CFG['PRETRAINED_CKPT']):
        print(f"❌ MAE-DFER 사전학습 가중치 없음")
        sys.exit(1)
    
    # ── 데이터 로드 (PolyGlotFake) ─────────────────────
    train_df, val_df = build_polyglotfake_train_val(
        num_fake_sample=CFG['NUM_FAKE_SAMPLE'],
        val_ratio=CFG['VAL_RATIO'], seed=CFG['SEED']
    )
    
    train_loader = DataLoader(
        PolyGlotFakeVideoDataset(train_df, CFG['NUM_FRAMES'], mode='train'),
        batch_size=CFG['BATCH_SIZE'], shuffle=True,
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        PolyGlotFakeVideoDataset(val_df, CFG['NUM_FRAMES'], mode='val'),
        batch_size=CFG['BATCH_SIZE'],
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    
    # ── 모델 ─────────────────────────────────────────
    print(f"\n🔧 모델 생성")
    model = EmotionFlowDetectorMAE(
        mae_dfer_path=CFG['MAE_DFER_PATH'],
        pretrained_ckpt=CFG['PRETRAINED_CKPT'],
        gru_hidden=CFG['GRU_HIDDEN'],
        dropout=CFG['DROPOUT'],
        unfreeze_last_blocks=CFG['UNFREEZE_LAST_BLOCKS'],
    ).to(device)
    
    # pos_weight (PGF는 Real 766, Fake 2000)
    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_weight_val = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    print(f"⚖️  BCE pos_weight: {pos_weight_val.item():.4f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    
    # Optimizer
    if CFG['UNFREEZE_LAST_BLOCKS'] > 0:
        backbone_params = [p for n, p in model.named_parameters()
                          if 'backbone' in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters()
                      if 'backbone' not in n and p.requires_grad]
        
        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': CFG['LR_BACKBONE'],
                'weight_decay': CFG['WEIGHT_DECAY']
            })
        param_groups.append({
            'params': head_params,
            'lr': CFG['LR'],
            'weight_decay': CFG['WEIGHT_DECAY']
        })
        optimizer = optim.AdamW(param_groups)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params, lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY']
        )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # ── 학습 ───────────────────────────────────────
    print("\n" + "="*60)
    print("🚀 PolyGlotFake 학습 시작 (MAE-DFER 백본)")
    print("="*60)
    
    best_auc = 0.0
    history = []
    
    for epoch in range(CFG['EPOCHS']):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
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
    
    print(f"\n✅ 완료! Best AUC: {best_auc:.1f}%")


if __name__ == "__main__":
    main()