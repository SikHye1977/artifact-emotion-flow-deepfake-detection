"""
==============================================================================
[v2 강화 버전] FakeAVCeleb HSEmotion 학습 (Backbone Unfreeze)
==============================================================================

[목적]
양방향 4가지 시나리오 일관성을 위해 FakeAVCeleb HSEmotion도 v2로 재학습.
PolyGlotFake v2와 동일한 설정 (unfreeze, layer-wise LR, augmentation).

[입력]
- 상위 폴더의 train_HSEmotion.py (수정됨, unfreeze_last_blocks 파라미터 추가)
- 상위 폴더의 FakeAVCeleb_v1.2/

[출력 (현재 폴더)]
- emotion_flow_lite_fav_v2_best.pth
- emotion_lite_logs_fav_v2/history.csv
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
from sklearn.model_selection import train_test_split

torch.load = functools.partial(torch.load, weights_only=False)

try:
    from emotion_deepfake_detector_lite import EmotionFlowDetectorLite
except ImportError:
    try:
        from train_HSEmotion import EmotionFlowDetectorLite
    except ImportError:
        print("❌ EmotionFlowDetectorLite import 실패")
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
    """기존 학습과 동일한 분할 (재현성)."""
    base_dir = find_fakeavceleb_root()
    csv_path = os.path.join(base_dir, 'meta_data.csv')
    
    df = pd.read_csv(csv_path)
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)
    
    real_df = df[df['video_label'] == 0.0].reset_index(drop=True)
    fake_df = df[df['video_label'] == 1.0].reset_index(drop=True)
    
    print(f"📊 FakeAVCeleb 원본: Real={len(real_df)}, Fake={len(fake_df)}")
    
    # Real 전량 + Fake 2000 샘플링
    n_fake = min(num_fake_sample, len(fake_df))
    sampled_fake = fake_df.sample(n=n_fake, random_state=seed)
    
    balanced_df = pd.concat([real_df, sampled_fake]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    
    # 영상 절대 경로 추가
    def make_path(row):
        rel_dir = row.iloc[-2].replace("FakeAVCeleb", os.path.basename(base_dir))
        return os.path.join(os.path.dirname(base_dir), rel_dir, row['path'])
    
    balanced_df['video_path'] = balanced_df.apply(make_path, axis=1)
    
    # 존재하는 파일만 필터
    exists_mask = balanced_df['video_path'].apply(os.path.exists)
    balanced_df = balanced_df[exists_mask].reset_index(drop=True)
    
    print(f"✂️  학습 데이터: 총 {len(balanced_df)}개 (파일 존재 필터링)")
    
    train_df, val_df = train_test_split(
        balanced_df, test_size=val_ratio,
        stratify=balanced_df['video_label'], random_state=seed
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    print(f"📂 Train: {len(train_df)}  |  Val: {len(val_df)}")
    return train_df, val_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. 강화된 데이터 증강 (v2와 동일)
# ══════════════════════════════════════════════════════════════════════════════
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
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
    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    if transform is None:
        transform = val_transform
    return torch.stack([transform(all_frames[i]) for i in indices])


class FakeAVCelebVideoDataset(Dataset):
    def __init__(self, df, num_frames=16, mode='train'):
        self.df = df.reset_index(drop=True)
        self.num_frames = num_frames
        self.transform = train_transform if mode == 'train' else val_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frames = extract_uniform_frames(row['video_path'], self.num_frames, self.transform)
        if frames is None:
            return self.__getitem__((idx + 1) % len(self))
        return frames, torch.tensor(row['video_label'], dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 학습/검증 (PGF v2와 동일)
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
        CKPT_PATH       = "emotion_flow_lite_fav_v2_best.pth",
        LOG_DIR         = "emotion_lite_logs_fav_v2",

        NUM_FRAMES      = 16,
        BATCH_SIZE      = 8,
        EPOCHS          = 20,
        GRU_HIDDEN      = 64,
        DROPOUT         = 0.3,
        MODEL_NAME      = 'enet_b0_8_best_afew',

        # NEW: Unfreeze
        UNFREEZE_LAST_BLOCKS = 2,
        LR_BACKBONE     = 1e-5,
        LR_HEAD         = 1e-3,
        WEIGHT_DECAY    = 1e-4,

        NUM_FAKE_SAMPLE = 2000,
        VAL_RATIO       = 0.1,
        NUM_WORKERS     = 3,
        SEED            = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)

    train_df, val_df = build_fakeavceleb_train_val(
        num_fake_sample=CFG['NUM_FAKE_SAMPLE'],
        val_ratio=CFG['VAL_RATIO'], seed=CFG['SEED']
    )

    train_loader = DataLoader(
        FakeAVCelebVideoDataset(train_df, CFG['NUM_FRAMES'], mode='train'),
        batch_size=CFG['BATCH_SIZE'], shuffle=True,
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        FakeAVCelebVideoDataset(val_df, CFG['NUM_FRAMES'], mode='val'),
        batch_size=CFG['BATCH_SIZE'],
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )

    print(f"\n🔧 모델 생성 (unfreeze_last_blocks={CFG['UNFREEZE_LAST_BLOCKS']})")
    model = EmotionFlowDetectorLite(
        model_name=CFG['MODEL_NAME'],
        num_frames=CFG['NUM_FRAMES'],
        gru_hidden=CFG['GRU_HIDDEN'],
        dropout=CFG['DROPOUT'],
        device='cpu',
        unfreeze_last_blocks=CFG['UNFREEZE_LAST_BLOCKS']
    ).to(device)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    print(f"📊 학습 가능 파라미터:")
    print(f"   Backbone: {sum(p.numel() for p in backbone_params):,}")
    print(f"   Head    : {sum(p.numel() for p in head_params):,}")

    num_neg = (train_df['video_label']==0).sum()
    num_pos = (train_df['video_label']==1).sum()
    pos_weight_val = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    print(f"⚖️  BCE pos_weight: {pos_weight_val.item():.4f}")

    param_groups = []
    if backbone_params:
        param_groups.append({
            'params': backbone_params, 'lr': CFG['LR_BACKBONE'],
            'weight_decay': CFG['WEIGHT_DECAY']
        })
    param_groups.append({
        'params': head_params, 'lr': CFG['LR_HEAD'],
        'weight_decay': CFG['WEIGHT_DECAY']
    })

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print("\n" + "="*60)
    print("🚀 FakeAVCeleb 학습 시작 (HSEmotion v2)")
    print("="*60)

    best_auc, history = 0.0, []
    for epoch in range(CFG['EPOCHS']):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
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

    print(f"\n✅ 완료! Best AUC: {best_auc:.1f}%")


if __name__ == "__main__":
    main()