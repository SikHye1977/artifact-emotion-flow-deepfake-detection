"""
==============================================================================
[빠른 실험용] 감정 흐름 기반 딥페이크 탐지 모델 (Lite) - Bottleneck + Unfreeze
Emotion Flow Deepfake Detector — with Optional Backbone Fine-tuning
==============================================================================

[v2 변경사항]
1. EmotionFlowDetectorLite에 unfreeze_last_blocks 파라미터 추가
   - 기본값 0: 완전 동결 (기존 동작 유지)
   - >0: EfficientNet-B0의 마지막 N개 블록 학습 가능
2. Layer-wise Learning Rate (Backbone과 Head에 다른 LR 적용)
3. 강화된 데이터 증강 옵션
4. Cosine LR Scheduler 추가

[기존 호환성]
- unfreeze_last_blocks 파라미터를 안 넘기면 기존과 동일 동작
- 기존 체크포인트(emotion_flow_lite_best.pth) 그대로 로드 가능

[아키텍처는 기존과 동일]
"""

import sys
import os
import functools
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import av
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

torch.load = functools.partial(torch.load, weights_only=False)

try:
    from hsemotion.facial_emotions import HSEmotionRecognizer
except ImportError:
    print("❌ hsemotion 미설치: pip install hsemotion")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Temporal Attention (변경 없음)
# ══════════════════════════════════════════════════════════════════════════════
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.w = nn.Linear(hidden_size, 1)

    def forward(self, gru_out: torch.Tensor):
        scores  = self.w(gru_out)
        weights = F.softmax(scores, dim=1)
        context = (weights * gru_out).sum(dim=1)
        return context, weights.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 메인 모델 (Unfreeze 옵션 추가)
# ══════════════════════════════════════════════════════════════════════════════
class EmotionFlowDetectorLite(nn.Module):
    def __init__(
        self,
        model_name : str = 'enet_b0_8_best_afew',
        num_frames : int = 16,
        gru_hidden : int = 64,
        dropout    : float = 0.3,
        device     : str = 'cpu',
        unfreeze_last_blocks : int = 0,   # 🆕 추가: 0이면 기존 동작 (완전 동결)
    ):
        super().__init__()
        self.num_frames = num_frames
        self.unfreeze_last_blocks = unfreeze_last_blocks

        # ── HSEmotion 로드 ─────────────────────────────────────────
        print("🧠 HSEmotion 로드 중...")
        fer = HSEmotionRecognizer(model_name=model_name, device=device)
        self.backbone = fer.model
        self.fer = fer

        self.register_buffer('cls_weight',
            torch.tensor(fer.classifier_weights, dtype=torch.float32))
        self.register_buffer('cls_bias',
            torch.tensor(fer.classifier_bias, dtype=torch.float32))

        # ── 기본: 모든 backbone 파라미터 동결 ────────────────────────
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # ── 🆕 Unfreeze 옵션 적용 ───────────────────────────────────
        if unfreeze_last_blocks > 0:
            self._unfreeze_last_blocks(unfreeze_last_blocks)
        else:
            print("✅ Backbone 동결 완료 (unfreeze_last_blocks=0)")

        # ── Bottleneck (변경 없음) ─────────────────────────────────
        self.feat_reduce = nn.Sequential(
            nn.Linear(1280, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.input_norm = nn.LayerNorm(136)

        self.gru = nn.GRU(
            input_size  = 136,
            hidden_size = gru_hidden,
            num_layers  = 1,
            batch_first = True,
            dropout     = 0.0
        )

        self.attention = TemporalAttention(gru_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def _unfreeze_last_blocks(self, n_blocks: int):
        """
        🆕 EfficientNet의 마지막 n_blocks 개를 학습 가능하게 만듦.
        Catastrophic forgetting 방지를 위해 Head보다 작은 LR 사용 권장.
        """
        unfrozen_count = 0
        method_used = "none"

        # Method 1: torchvision 스타일 EfficientNet (features 속성)
        if hasattr(self.backbone, 'features'):
            features = self.backbone.features
            num_blocks = len(features)
            unfreeze_from = max(0, num_blocks - n_blocks)
            
            for i in range(unfreeze_from, num_blocks):
                for param in features[i].parameters():
                    param.requires_grad = True
                    unfrozen_count += param.numel()
            
            if hasattr(self.backbone, 'classifier'):
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True
                    unfrozen_count += param.numel()
            method_used = "torchvision"
        
        # Method 2: timm 스타일 EfficientNet (blocks 속성)
        elif hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            num_blocks = len(blocks)
            unfreeze_from = max(0, num_blocks - n_blocks)
            
            for i in range(unfreeze_from, num_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
                    unfrozen_count += param.numel()
            
            for attr_name in ['conv_head', 'bn2', 'global_pool', 'classifier']:
                if hasattr(self.backbone, attr_name):
                    module = getattr(self.backbone, attr_name)
                    if isinstance(module, nn.Module):
                        for param in module.parameters():
                            param.requires_grad = True
                            unfrozen_count += param.numel()
            method_used = "timm"
        
        # Method 3: Generic fallback (마지막 20% 파라미터)
        else:
            print('  ⚠️  표준 구조 미발견 — 마지막 20% 파라미터 unfreeze')
            all_params = list(self.backbone.parameters())
            unfreeze_n = max(1, len(all_params) // 5)
            for param in all_params[-unfreeze_n:]:
                param.requires_grad = True
                unfrozen_count += param.numel()
            method_used = "generic"
        
        print(f"🔓 Backbone 마지막 {n_blocks} 블록 Unfreeze ({method_used})")
        print(f"   학습 가능 backbone 파라미터: {unfrozen_count:,}")

    @torch.no_grad()
    def _extract_feat_only_no_grad(self, x: torch.Tensor):
        """기존 동작: backbone이 완전 동결일 때 (no_grad 컨텍스트)"""
        feat   = self.backbone(x)
        logit  = F.linear(feat, self.cls_weight, self.cls_bias)
        emotion = F.softmax(logit, dim=-1)
        return feat, emotion

    def _extract_feat_with_grad(self, x: torch.Tensor):
        """🆕 unfreeze 모드: backbone에 gradient 흐름 (no_grad 없음)"""
        feat   = self.backbone(x)
        # 분류기 weight/bias는 buffer라서 gradient 안 흐름 (감정 logit은 고정 분류기 사용)
        logit  = F.linear(feat, self.cls_weight, self.cls_bias)
        emotion = F.softmax(logit, dim=-1)
        return feat, emotion

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)

        # 🆕 Unfreeze 여부에 따라 다른 경로 사용
        if self.unfreeze_last_blocks > 0:
            # backbone에 gradient 흐름 (학습 가능)
            feat, emotion = self._extract_feat_with_grad(x_flat)
        else:
            # 기존 동작: backbone 완전 동결, no_grad
            feat, emotion = self._extract_feat_only_no_grad(x_flat)
        
        # Bottleneck
        reduced_feat = self.feat_reduce(feat)
        
        # Concat (128 + 8 = 136)
        combined = torch.cat([reduced_feat, emotion], dim=-1)
        combined = combined.view(B, T, 136)
        combined = self.input_norm(combined)

        # GRU & Attention
        gru_out, _ = self.gru(combined)
        context, attn_w = self.attention(gru_out)

        logit = self.classifier(context)
        return logit.squeeze(1), attn_w


# ══════════════════════════════════════════════════════════════════════════════
# 3. 영상 전처리 (강화된 증강 옵션 추가)
# ══════════════════════════════════════════════════════════════════════════════
def get_train_transform(strong_aug: bool = False):
    """🆕 strong_aug=True 시 더 강한 증강 적용"""
    if strong_aug:
        return transforms.Compose([
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
    else:
        # 기존 기본 증강
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])


val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


# 기본 transform (기존 호환성)
frame_transform = get_train_transform(strong_aug=False)


def extract_uniform_frames(video_path: str, num_frames: int = 16, transform=None):
    try:
        container  = av.open(video_path)
        all_frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
    except Exception:
        return None

    if len(all_frames) < num_frames:
        return None

    if transform is None:
        transform = frame_transform

    indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    return torch.stack([transform(all_frames[i]) for i in indices])


class FakeAVCelebDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_dir: str,
                 num_frames: int = 16, transform=None):
        self.df         = df.reset_index(drop=True)
        self.base_dir   = base_dir
        self.num_frames = num_frames
        self.transform  = transform if transform is not None else frame_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        rel_path   = row.iloc[-2].replace("FakeAVCeleb", self.base_dir)
        video_path = os.path.join(rel_path, row['path'])

        frames = extract_uniform_frames(video_path, self.num_frames, self.transform)
        if frames is None:
            return self.__getitem__((idx + 1) % len(self))

        label = 1.0 if row['method'] != 'real' else 0.0
        return frames, torch.tensor(label, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 학습 / 검증 루프
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    
    for i, (frames, labels) in enumerate(loader):
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
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

    return total_loss / len(loader), pd.DataFrame()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits, _ = model(frames)
            total_loss += criterion(logits, labels).item()

        probs = torch.sigmoid(logits).cpu().numpy()
        targets = labels.cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(targets.tolist())

    avg_loss = total_loss / len(loader)
    preds = (np.array(all_probs) > 0.5).astype(int)
    acc = (preds == np.array(all_labels)).mean() * 100
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except Exception:
        auc = 0.0

    return avg_loss, acc, auc, pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main (Layer-wise LR + Cosine Scheduler 추가)
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        BASE_DIR    = "FakeAVCeleb_v1.2",
        CSV_PATH    = "FakeAVCeleb_v1.2/meta_data.csv",
        CKPT_PATH   = "emotion_flow_lite_v2_best.pth",  # 🆕 v2 별도 저장
        NUM_FRAMES  = 16,
        BATCH_SIZE  = 8,
        EPOCHS      = 20,                # 🆕 15 → 20
        WEIGHT_DECAY= 1e-4,
        GRU_HIDDEN  = 64,
        DROPOUT     = 0.3,
        MODEL_NAME  = 'enet_b0_8_best_afew',
        NUM_FAKE_SAMPLE = 2000,

        # 🆕 새 옵션들
        UNFREEZE_LAST_BLOCKS = 2,        # EfficientNet 마지막 2 블록 unfreeze
        LR_BACKBONE = 1e-5,              # backbone은 작은 LR
        LR_HEAD     = 1e-3,              # head는 기존 LR
        STRONG_AUG  = True,              # 강화된 데이터 증강
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── 데이터 로드 ─────────────────────────────────────────────
    df = pd.read_csv(CFG['CSV_PATH'])
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)

    real_df = df[df['video_label'] == 0.0]
    fake_df = df[df['video_label'] == 1.0]

    print(f"📊 원본 데이터: Real={len(real_df)}, Fake={len(fake_df)}")
    
    sampled_fake = fake_df.sample(n=min(CFG['NUM_FAKE_SAMPLE'], len(fake_df)), random_state=42)
    balanced_df = pd.concat([real_df, sampled_fake]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✂️  학습용 데이터: 총 {len(balanced_df)}개 (Real {len(real_df)} : Fake {len(sampled_fake)})")

    train_df, val_df = train_test_split(
        balanced_df, test_size=0.1, stratify=balanced_df['video_label'], random_state=42
    )

    # 🆕 학습/검증용 transform 분리
    train_tf = get_train_transform(strong_aug=CFG['STRONG_AUG'])
    
    train_loader = DataLoader(
        FakeAVCelebDataset(train_df, CFG['BASE_DIR'], CFG['NUM_FRAMES'], transform=train_tf),
        batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        FakeAVCelebDataset(val_df, CFG['BASE_DIR'], CFG['NUM_FRAMES'], transform=val_transform),
        batch_size=CFG['BATCH_SIZE'], num_workers=4, pin_memory=True
    )

    # ── 모델 (🆕 unfreeze 옵션) ────────────────────────────────
    model = EmotionFlowDetectorLite(
        model_name = CFG['MODEL_NAME'],
        num_frames = CFG['NUM_FRAMES'],
        gru_hidden = CFG['GRU_HIDDEN'],
        dropout    = CFG['DROPOUT'],
        device     = 'cpu',
        unfreeze_last_blocks = CFG['UNFREEZE_LAST_BLOCKS']  # 🆕
    ).to(device)

    # 🆕 학습 가능 파라미터를 backbone과 head로 분리
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    bb_count = sum(p.numel() for p in backbone_params)
    head_count = sum(p.numel() for p in head_params)
    print(f"📊 학습 가능 파라미터:")
    print(f"   Backbone: {bb_count:,}")
    print(f"   Head    : {head_count:,}")
    print(f"   총합    : {bb_count + head_count:,}")

    # pos_weight
    num_neg = len(real_df)
    num_pos = len(sampled_fake)
    pos_weight_val = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    print(f"⚖️  BCE Loss pos_weight: {pos_weight_val.item():.4f}")

    # 🆕 Layer-wise LR Optimizer
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

    optimizer = optim.AdamW(param_groups)
    
    # 🆕 Cosine LR Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['EPOCHS'])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # 학습 루프
    print("\n" + "="*60)
    print(f"🚀 학습 시작 (HSEmotion v2 — Unfreeze + Layer-wise LR)")
    print("="*60)
    
    best_auc = 0.0
    history = []
    for epoch in range(CFG['EPOCHS']):
        t0 = time.time()
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        val_loss, val_acc, val_auc, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        lrs = [g['lr'] for g in optimizer.param_groups]
        print(f"\n🏁 E{epoch:02d} | Train={train_loss:.4f} | Val Loss={val_loss:.4f} | "
              f"Acc={val_acc:.1f}% | AUC={val_auc:.1f}% | LR={lrs} | {elapsed:.0f}s\n")

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': val_auc,
        })
        pd.DataFrame(history).to_csv("emotion_lite_history_v2.csv", index=False)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'cfg': CFG,
                'epoch': epoch,
                'val_auc': val_auc,
            }, CFG['CKPT_PATH'])
            print(f"  ⭐ Best 모델 저장 (AUC={best_auc:.1f}%)")

    print(f"\n✅ 완료! Best AUC: {best_auc:.1f}%")


if __name__ == "__main__":
    main()