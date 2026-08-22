"""
==============================================================================
[FakeAVCeleb 학습] MAE-DFER 백본 + GRU + Attention (Corrected Version)
Train Emotion Flow Detector with MAE-DFER on FakeAVCeleb
==============================================================================

[수정 및 개선 사항]
- 시공간 일관성(Temporal Consistency)을 유지하는 Video Transform 적용
- BCEWithLogitsLoss 차원 충돌(Broadcasting Bug) 해결
- Validation AUC/ACC 계산 시 shape 불일치 해결
==============================================================================
"""

import sys
import os
import functools
import time
import random

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
import torchvision.transforms.functional as F_tv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

torch.load = functools.partial(torch.load, weights_only=False)

from emotion_deepfake_detector_mae import EmotionFlowDetectorMAE

# ══════════════════════════════════════════════════════════════════════════════
# 1. 영상 전처리 (Temporal Consistency 유지)
# ══════════════════════════════════════════════════════════════════════════════
def apply_video_transform(frames_list, mode='train'):
    """
    frames_list: 16장의 PIL Image 리스트
    16장 모두에 '동일한' Augmentation을 적용하여 감정 흐름의 왜곡을 방지합니다.
    """
    # 1. 공통 Resize (MAE-DFER 사양: 160x160)
    frames = [F_tv.resize(img, (160, 160)) for img in frames_list]
    
    if mode == 'train':
        # 2. 비디오 전체에 대해 50% 확률로 동일하게 좌우 반전
        if random.random() > 0.5:
            frames = [F_tv.hflip(img) for img in frames]
            
        # 3. 비디오 전체에 대해 동일한 밝기/대비/채도 조절 적용
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        
        frames = [F_tv.adjust_brightness(img, brightness_factor) for img in frames]
        frames = [F_tv.adjust_contrast(img, contrast_factor) for img in frames]
        frames = [F_tv.adjust_saturation(img, saturation_factor) for img in frames]

    # 4. 공통 ToTensor & Normalize
    # (참고: 일반적인 ImageNet 기반 ViT 모델들은 아래의 정규화 수치를 주로 사용합니다. 
    # 만약 기존의 0.5를 유지해야 한다면 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]로 변경하세요.)
    frames = [F_tv.to_tensor(img) for img in frames]
    frames = [F_tv.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in frames]
    
    # [16, 3, 160, 160] 형태로 반환
    return torch.stack(frames)


def extract_uniform_frames(video_path, num_frames=16, mode='val'):
    try:
        container = av.open(video_path)
        all_frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
    except Exception:
        return None
        
    if len(all_frames) < num_frames:
        return None
        
    # 균등한 간격으로 프레임 인덱스 추출
    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    
    # 추출된 프레임을 일괄 PIL Image로 변환
    selected_pil = [transforms.ToPILImage()(all_frames[i]) for i in indices]
    
    # 비디오 전체 변환 적용
    return apply_video_transform(selected_pil, mode)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FakeAVCeleb Dataset
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

class FakeAVCelebDataset(Dataset):
    def __init__(self, df, base_dir, num_frames=16, mode='train'):
        self.df = df.reset_index(drop=True)
        self.base_dir = base_dir
        self.num_frames = num_frames
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row.iloc[-2].replace("FakeAVCeleb", os.path.basename(self.base_dir))
        video_path = os.path.join(os.path.dirname(self.base_dir), rel_path, row['path'])
        
        if not os.path.exists(video_path):
            return self.__getitem__((idx + 1) % len(self))
        
        frames = extract_uniform_frames(video_path, self.num_frames, self.mode)
        if frames is None:
            return self.__getitem__((idx + 1) % len(self))
        
        label = 1.0 if row['method'] != 'real' else 0.0
        return frames, torch.tensor(label, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 학습/검증 로직
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    for i, (frames, labels) in enumerate(loader):
        # non_blocking=True를 사용하여 데이터 전송 속도 최적화
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) 
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(frames)
            # [수정] 차원 불일치 방지: logits를 [Batch] 형태로 평탄화
            logits = logits.view(-1)
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
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(frames)
            # [수정] 차원 불일치 방지
            logits = logits.view(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
        # [수정] Batch 단위의 예측값을 안전하게 1차원 리스트로 변환
        probs = torch.sigmoid(logits).cpu().numpy()
        probs = np.atleast_1d(probs).tolist()
        
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy().tolist())
        
    avg_loss = total_loss / len(loader)
    preds = (np.array(all_probs) > 0.5).astype(int)
    acc = (preds == np.array(all_labels)).mean() * 100
    
    try: 
        auc = roc_auc_score(all_labels, all_probs) * 100
    except: 
        auc = 0.0
        
    return avg_loss, acc, auc


# ══════════════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        # MAE-DFER
        MAE_DFER_PATH    = os.path.expanduser("~/hsh/AIApplication/mae_dfer"),
        PRETRAINED_CKPT  = os.path.expanduser(
            "~/hsh/AIApplication/mae_dfer/saved/pretrained/checkpoint-49.pth"
        ),
        
        # 출력
        CKPT_PATH        = "emotion_flow_mae_fav_best.pth",
        LOG_DIR          = "emotion_mae_logs_fav",
        
        # 학습
        NUM_FRAMES       = 16,
        BATCH_SIZE       = 4,                # 160x160이지만 백본 큼 → 작게 시작
        EPOCHS           = 15,
        LR               = 1e-3,             # head LR
        LR_BACKBONE      = 1e-5,             # 만약 unfreeze 시 사용
        WEIGHT_DECAY     = 1e-4,
        GRU_HIDDEN       = 64,               # HSEmotion과 동일
        DROPOUT          = 0.3,
        
        # Backbone freeze 설정
        UNFREEZE_LAST_BLOCKS = 0,            # 일단 완전 동결 (안전)
        
        # 데이터
        NUM_FAKE_SAMPLE  = 2000,
        VAL_RATIO        = 0.1,
        NUM_WORKERS      = 3,
        SEED             = 42,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"📁 Working dir: {os.getcwd()}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)

    # 가중치 확인
    if not os.path.exists(CFG['PRETRAINED_CKPT']):
        print(f"❌ MAE-DFER 사전학습 가중치 없음: {CFG['PRETRAINED_CKPT']}")
        sys.exit(1)

    # ── 데이터 로드 ─────────────────────────────────
    base_dir = find_fakeavceleb_root()
    csv_path = os.path.join(base_dir, 'meta_data.csv')
    
    df = pd.read_csv(csv_path)
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)
    
    real_df = df[df['video_label'] == 0.0]
    fake_df = df[df['video_label'] == 1.0]
    print(f"📊 원본 데이터: Real={len(real_df)}, Fake={len(fake_df)}")
    
    sampled_fake = fake_df.sample(
        n=min(CFG['NUM_FAKE_SAMPLE'], len(fake_df)),
        random_state=CFG['SEED']
    )
    balanced_df = pd.concat([real_df, sampled_fake]).sample(
        frac=1, random_state=CFG['SEED']
    ).reset_index(drop=True)
    print(f"✂️  학습용 데이터: 총 {len(balanced_df)}개")
    
    train_df, val_df = train_test_split(
        balanced_df, test_size=CFG['VAL_RATIO'],
        stratify=balanced_df['video_label'], random_state=CFG['SEED']
    )
    print(f"📂 Train: {len(train_df)} | Val: {len(val_df)}")
    
    # DataLoader
    train_loader = DataLoader(
        FakeAVCelebDataset(train_df, base_dir, CFG['NUM_FRAMES'], mode='train'),
        batch_size=CFG['BATCH_SIZE'], shuffle=True,
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        FakeAVCelebDataset(val_df, base_dir, CFG['NUM_FRAMES'], mode='val'),
        batch_size=CFG['BATCH_SIZE'], shuffle=False,
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
    
    # ── 손실 함수 ────────────────────────────────────
    # [수정] 밸런싱을 이미 진행했으므로, 극단적인 pos_weight로 인한 불안정을 방지합니다.
    num_neg = len(train_df[train_df['video_label'] == 0.0])
    num_pos = len(train_df[train_df['video_label'] == 1.0])
    # 비율에 맞춰 weight 재계산
    weight_val = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight_val = torch.tensor([weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    print(f"⚖️  BCE pos_weight: {pos_weight_val.item():.4f}")
    
    # ── Optimizer ──────────────────────────────────
    if CFG['UNFREEZE_LAST_BLOCKS'] > 0:
        # Layer-wise LR
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
        # 단일 LR (head만 학습)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params, lr=CFG['LR'], weight_decay=CFG['WEIGHT_DECAY']
        )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # ── 학습 루프 ──────────────────────────────────
    print("\n" + "="*60)
    print("🚀 FakeAVCeleb 학습 시작 (MAE-DFER 백본)")
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
    print(f"   체크포인트: {os.path.abspath(CFG['CKPT_PATH'])}")

if __name__ == "__main__":
    main()