"""
==============================================================================
[v2 강화 버전] PolyGlotFake HSEmotion 학습 (Backbone Unfreeze)
==============================================================================

[v1 대비 변경점]
1. unfreeze_last_blocks=2: EfficientNet 마지막 2 블록 fine-tuning
2. Layer-wise LR: backbone 1e-5, head 1e-3 (catastrophic forgetting 방지)
3. 강화된 데이터 증강 (ColorJitter, RandomAffine, RandomErasing)
4. 학습 epoch 증가 (15 → 20)
5. Cosine LR scheduler 추가
6. 별도 가중치 파일명 (충돌 방지)

[필요 사항]
상위 폴더 train_HSEmotion.py에 unfreeze_last_blocks 파라미터 추가됨
(PATCH_GUIDE_HSEmotion.py 참고)

[출력]
- emotion_flow_lite_pgf_v2_best.pth (v2 가중치, v1과 분리)
- emotion_lite_logs_pgf_v2/history.csv
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

try:
    from emotion_deepfake_detector_lite import EmotionFlowDetectorLite
except ImportError:
    try:
        from train_HSEmotion import EmotionFlowDetectorLite
    except ImportError:
        print("❌ EmotionFlowDetectorLite import 실패")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 강화된 데이터 증강 (학습용)
# ══════════════════════════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    # 강화: 더 강한 색 변화
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    # NEW: 약간의 회전과 평행이동
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
    # NEW: 일부 영역 무작위 삭제 (occlusion robustness)
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

# 평가용 (증강 없음)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def extract_uniform_frames(video_path: str, num_frames: int = 16,
                            transform=None):
    try:
        container  = av.open(video_path)
        all_frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
    except Exception:
        return None
    if len(all_frames) < num_frames:
        return None
    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    if transform is None:
        transform = val_transform
    return torch.stack([transform(all_frames[i]) for i in indices])


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset (학습/평가 모드 분리)
# ══════════════════════════════════════════════════════════════════════════════
class PolyGlotFakeVideoDataset(Dataset):
    def __init__(self, df, num_frames=16, mode='train'):
        self.df         = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.transform  = train_transform if mode == 'train' else val_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
        frames = extract_uniform_frames(video_path, self.num_frames, self.transform)
        if frames is None:
            return self.__getitem__((idx + 1) % len(self))
        return frames, torch.tensor(row['video_label'], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 학습/검증
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (frames, labels) in enumerate(loader):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(frames)
            loss      = criterion(logits, labels)
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
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = labels.cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(targets.tolist())
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
        # 출력 (v2 별도 경로)
        CKPT_PATH       = "emotion_flow_lite_pgf_v2_best.pth",
        LOG_DIR         = "emotion_lite_logs_pgf_v2",

        # 모델 설정
        NUM_FRAMES      = 16,
        BATCH_SIZE      = 8,
        EPOCHS          = 20,           # 15 → 20
        GRU_HIDDEN      = 64,
        DROPOUT         = 0.3,
        MODEL_NAME      = 'enet_b0_8_best_afew',

        # NEW: Unfreeze 설정
        UNFREEZE_LAST_BLOCKS = 2,       # EfficientNet 마지막 2 블록

        # NEW: Layer-wise LR
        LR_BACKBONE     = 1e-5,         # 매우 작게 (catastrophic forgetting 방지)
        LR_HEAD         = 1e-3,         # 기존 LR 유지
        WEIGHT_DECAY    = 1e-4,

        # 데이터
        NUM_FAKE_SAMPLE = 2000,
        VAL_RATIO       = 0.1,
        NUM_WORKERS     = 3,
        SEED            = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)

    # 데이터
    train_df, val_df = build_polyglotfake_train_val(
        num_fake_sample = CFG['NUM_FAKE_SAMPLE'],
        val_ratio       = CFG['VAL_RATIO'],
        seed            = CFG['SEED']
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

    # 모델 (Unfreeze 옵션 포함)
    print(f"\n🔧 모델 생성 (unfreeze_last_blocks={CFG['UNFREEZE_LAST_BLOCKS']})")
    model = EmotionFlowDetectorLite(
        model_name=CFG['MODEL_NAME'],
        num_frames=CFG['NUM_FRAMES'],
        gru_hidden=CFG['GRU_HIDDEN'],
        dropout=CFG['DROPOUT'],
        device='cpu',
        unfreeze_last_blocks=CFG['UNFREEZE_LAST_BLOCKS']  # NEW
    ).to(device)

    # 학습 가능 파라미터 분류
    backbone_params = []
    head_params = []
    backbone_count = 0
    head_count = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
            backbone_count += param.numel()
        else:
            head_params.append(param)
            head_count += param.numel()
    
    total_count = backbone_count + head_count
    print(f"📊 학습 가능 파라미터:")
    print(f"   Backbone: {backbone_count:,} ({backbone_count/total_count*100:.1f}%)")
    print(f"   Head    : {head_count:,} ({head_count/total_count*100:.1f}%)")
    print(f"   총합    : {total_count:,}")

    # pos_weight
    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_weight_val = torch.tensor(
        [num_neg / num_pos], dtype=torch.float32
    ).to(device)
    print(f"⚖️  BCE pos_weight: {pos_weight_val.item():.4f}")

    # Layer-wise LR Optimizer
    print(f"📐 Layer-wise LR:")
    print(f"   Backbone LR: {CFG['LR_BACKBONE']:.0e}")
    print(f"   Head LR    : {CFG['LR_HEAD']:.0e}")
    
    param_groups = []
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': CFG['LR_BACKBONE'],
            'weight_decay': CFG['WEIGHT_DECAY']
        })
    param_groups.append({
        'params': head_params,
        'lr': CFG['LR_HEAD'],
        'weight_decay': CFG['WEIGHT_DECAY']
    })
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = optim.AdamW(param_groups)
    
    # NEW: Cosine LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['EPOCHS']
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 학습
    print("\n" + "="*60)
    print("🚀 PolyGlotFake 학습 시작 (HSEmotion v2 — Backbone Unfreeze)")
    print("="*60)

    best_auc = 0.0
    history  = []
    for epoch in range(CFG['EPOCHS']):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
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
            'lr_backbone': lrs[0] if len(lrs) > 1 else 0,
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

    print(f"\n✅ 완료! Best AUC: {best_auc:.1f}%")
    print(f"   체크포인트: {os.path.abspath(CFG['CKPT_PATH'])}")


if __name__ == "__main__":
    main()