"""
==============================================================================
[오디오 기반] 감정 흐름 딥페이크 탐지 모델 - GRU Unfreeze 옵션 추가
Audio Emotion Flow Deepfake Detector — with Optional GRU Fine-tuning
==============================================================================

[v2 변경사항]
1. AudioEmotionFlowDetector에 unfreeze_pretrained_gru 파라미터 추가
   - 기본값 False: 완전 동결 (기존 동작 유지)
   - True: 사전학습된 GRU + classifier 학습 가능
2. Layer-wise Learning Rate (사전학습 부분과 head에 다른 LR)

[기존 호환성]
- unfreeze_pretrained_gru를 안 넘기면 기존과 동일 동작
- 기존 체크포인트 그대로 로드 가능

[핵심 아이디어는 기존과 동일]
"""

import sys
import os
import functools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio
import pandas as pd
import numpy as np
import av
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

torch.load = functools.partial(torch.load, weights_only=False)


# ══════════════════════════════════════════════════════════════════════════════
# 1. AudioEmotionCRNN (변경 없음)
# ══════════════════════════════════════════════════════════════════════════════
class AudioEmotionCRNN(nn.Module):
    """RAVDESS 학습된 사전학습 모델 (구조 변경 없음)"""
    def __init__(self, num_classes: int = 8):
        super().__init__()

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.gru = nn.GRU(input_size=512, hidden_size=128,
                          num_layers=1, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor):
        return self.forward_with_feat(x)[1]

    def forward_with_feat(self, x: torch.Tensor):
        if self.training:
            x = x + torch.randn_like(x) * 1e-6

        x = self.mel_spec(x)
        x = torch.clamp(x, min=1e-5)
        x = self.amplitude_to_db(x)

        x = self.cnn(x)
        B, C, F_dim, T_dim = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, T_dim, C * F_dim)

        gru_out, hn = self.gru(x)
        feat = hn.squeeze(0)
        logit = self.classifier(feat)
        return feat, logit


# ══════════════════════════════════════════════════════════════════════════════
# 2. Temporal Attention (변경 없음)
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
# 3. 메인 모델 (Unfreeze 옵션 추가)
# ══════════════════════════════════════════════════════════════════════════════
class AudioEmotionFlowDetector(nn.Module):
    def __init__(
        self,
        pretrained_path : str,
        num_segments    : int = 16,
        gru_hidden      : int = 128,
        dropout         : float = 0.4,
        unfreeze_pretrained_gru : bool = False,  # 🆕 추가
    ):
        super().__init__()
        self.num_segments = num_segments
        self.unfreeze_pretrained_gru = unfreeze_pretrained_gru

        # ── 사전 학습 모델 로드 ──────────────────────────────────────
        print(f"🎵 AudioEmotionCRNN 가중치 로드: {pretrained_path}")
        self.audio_backbone = AudioEmotionCRNN(num_classes=8)
        state_dict = torch.load(pretrained_path, map_location='cpu')
        self.audio_backbone.load_state_dict(state_dict)

        # ── 기본: 완전 동결 ───────────────────────────────────────
        for param in self.audio_backbone.parameters():
            param.requires_grad = False
        
        # ── 🆕 Unfreeze 옵션 적용 ────────────────────────────────
        if unfreeze_pretrained_gru:
            self._unfreeze_pretrained_gru()
        else:
            print("✅ Audio backbone 동결 완료 (기본)")

        # ── 입력 정규화 ───────────────────────────────────────────
        self.input_dim  = 128 + 8
        self.input_norm = nn.LayerNorm(self.input_dim)

        # ── 시계열 GRU (변경 없음) ──────────────────────────────────
        self.gru = nn.GRU(
            input_size  = self.input_dim,
            hidden_size = gru_hidden,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.3
        )

        self.attention = TemporalAttention(gru_hidden)

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_hidden),
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def _unfreeze_pretrained_gru(self):
        """
        🆕 사전학습된 audio backbone에서 GRU와 classifier만 unfreeze.
        CNN은 음향 특징 추출에 일반적이므로 동결 유지.
        GRU는 시계열 적응이 필요하므로 unfreeze.
        """
        unfrozen_count = 0
        unfrozen_layers = []
        
        for name, param in self.audio_backbone.named_parameters():
            # GRU 또는 classifier만 unfreeze
            if 'gru' in name.lower() or 'classifier' in name.lower():
                param.requires_grad = True
                unfrozen_count += param.numel()
                unfrozen_layers.append(name)
        
        print(f"🔓 Audio backbone GRU + classifier Unfreeze")
        print(f"   학습 가능 파라미터: {unfrozen_count:,}")
        print(f"   대상 레이어: {len(unfrozen_layers)}개")

    def forward(self, x: torch.Tensor):
        B, N, C, S = x.shape

        # 🆕 unfreeze 모드에 따라 다른 경로
        if self.unfreeze_pretrained_gru:
            # gradient 흐름 (학습 가능)
            # 단, mel_spec → log 부분은 여전히 FP32로 처리
            x_flat = x.view(B * N, C, S).float()
            with torch.amp.autocast('cuda', enabled=False):
                feat, emo_logit = self.audio_backbone.forward_with_feat(x_flat)
                emo_prob = F.softmax(emo_logit, dim=-1)
        else:
            # 기존 동작: no_grad 컨텍스트
            x_flat = x.view(B * N, C, S).float()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                feat, emo_logit = self.audio_backbone.forward_with_feat(x_flat)
                emo_prob = F.softmax(emo_logit, dim=-1)

        feat     = feat.to(x.dtype)
        emo_prob = emo_prob.to(x.dtype)

        combined = torch.cat([feat, emo_prob], dim=-1)
        combined = combined.view(B, N, self.input_dim)
        combined = self.input_norm(combined)

        gru_out, _ = self.gru(combined)
        context, attn_w = self.attention(gru_out)

        logit = self.classifier(context)
        return logit.squeeze(1), attn_w


# ══════════════════════════════════════════════════════════════════════════════
# 4. 영상에서 오디오 추출 (변경 없음)
# ══════════════════════════════════════════════════════════════════════════════
def extract_audio_segments(
    video_path: str,
    num_segments: int = 16,
    target_sr: int = 16000,
    segment_duration: float = 3.0
):
    try:
        container = av.open(video_path)
        audio_stream = container.streams.audio[0]
        orig_sr = audio_stream.sample_rate

        chunks = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            chunks.append(arr)
        container.close()

        if not chunks:
            return None

        waveform = np.concatenate(chunks, axis=1)
        waveform = torch.from_numpy(waveform).float()

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.shape[0] == 1:
            pass
        else:
            waveform = waveform.unsqueeze(0)

        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=target_sr
            )
            waveform = resampler(waveform)

        total_samples = waveform.shape[1]
        segment_samples = int(target_sr * segment_duration)

        if total_samples < segment_samples:
            pad = segment_samples - total_samples
            waveform = F.pad(waveform, (0, pad))
            total_samples = segment_samples

        centers = np.linspace(
            segment_samples // 2,
            total_samples - segment_samples // 2,
            num_segments,
            dtype=int
        )

        segments = []
        for c in centers:
            start = max(0, c - segment_samples // 2)
            end   = start + segment_samples
            if end > total_samples:
                end   = total_samples
                start = end - segment_samples
            seg = waveform[:, start:end]
            if seg.shape[1] < segment_samples:
                seg = F.pad(seg, (0, segment_samples - seg.shape[1]))
            segments.append(seg)

        return torch.stack(segments)

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 5. Dataset (변경 없음)
# ══════════════════════════════════════════════════════════════════════════════
class FakeAVCelebAudioDataset(Dataset):
    def __init__(self, df, base_dir, num_segments=16,
                 segment_duration=3.0, target_sr=16000):
        self.df               = df.reset_index(drop=True)
        self.base_dir         = base_dir
        self.num_segments     = num_segments
        self.segment_duration = segment_duration
        self.target_sr        = target_sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        rel_path   = row.iloc[-2].replace("FakeAVCeleb", self.base_dir)
        video_path = os.path.join(rel_path, row['path'])

        segments = extract_audio_segments(
            video_path,
            num_segments     = self.num_segments,
            target_sr        = self.target_sr,
            segment_duration = self.segment_duration
        )

        if segments is None:
            return self.__getitem__((idx + 1) % len(self))

        label = 1.0 if row['method'] != 'real' else 0.0
        return segments, torch.tensor(label, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 6. 학습 / 검증 루프 (변경 없음)
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    # 🆕 unfreeze 시에도 BatchNorm은 eval 유지 (running stats 안정화)
    model.audio_backbone.eval()

    total_loss = 0.0
    logs = []

    for i, (segments, labels) in enumerate(loader):
        step_t = time.time()
        segments, labels = segments.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(segments)
            loss      = criterion(logits, labels)

        if not torch.isfinite(loss):
            print(f"  ⚠️  NaN/Inf loss 감지, 배치 스킵")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            probs   = torch.sigmoid(logits).cpu().numpy()
            preds   = (probs > 0.5).astype(int)
            targets = labels.cpu().numpy()

        for prob, pred, tgt in zip(probs, preds, targets):
            logs.append({
                '레이블 (0:진짜 1:가짜)': int(tgt),
                'Fake 확률 (%)':         round(float(prob) * 100, 2),
                '예측':                  int(pred),
                '정답 여부':             int(pred == int(tgt))
            })

        total_loss += loss.item()
        if i % 10 == 0:
            print(f"  E{epoch:02d} [{i:3d}/{len(loader)}] "
                  f"Loss={loss.item():.4f}  ({time.time()-step_t:.2f}s)")

    return total_loss / len(loader), pd.DataFrame(logs)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    logs = []

    for segments, labels in loader:
        segments, labels = segments.to(device), labels.to(device)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = model(segments)
            total_loss += criterion(logits, labels).item()

        probs   = torch.sigmoid(logits).cpu().numpy()
        preds   = (probs > 0.5).astype(int)
        targets = labels.cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(targets.tolist())

        for prob, pred, tgt in zip(probs, preds, targets):
            logs.append({
                '레이블 (0:진짜 1:가짜)': int(tgt),
                'Fake 확률 (%)':         round(float(prob) * 100, 2),
                '예측':                  int(pred),
                '정답 여부':             int(pred == int(tgt))
            })

    avg_loss = total_loss / len(loader)
    acc      = sum(l['정답 여부'] for l in logs) / len(logs) * 100

    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except Exception:
        auc = 0.0

    return avg_loss, acc, auc, pd.DataFrame(logs)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Main (Layer-wise LR 추가)
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        BASE_DIR         = "FakeAVCeleb_v1.2",
        CSV_PATH         = "FakeAVCeleb_v1.2/meta_data.csv",
        PRETRAINED_PATH  = "audio_emotion_crnn_best.pth",
        CKPT_PATH        = "audio_flow_deepfake_v2_best.pth",  # 🆕 v2 별도 저장
        LOG_DIR          = "audio_flow_logs_v2",

        NUM_SEGMENTS     = 16,
        SEGMENT_DURATION = 3.0,
        TARGET_SR        = 16000,

        BATCH_SIZE       = 8,
        EPOCHS           = 25,           # 🆕 20 → 25
        WEIGHT_DECAY     = 1e-4,
        VAL_RATIO        = 0.1,
        NUM_WORKERS      = 4,
        NUM_FAKE_SAMPLE  = 2000,
        GRU_HIDDEN       = 128,
        DROPOUT          = 0.4,
        PATIENCE         = 7,            # 🆕 5 → 7

        # 🆕 새 옵션들
        UNFREEZE_PRETRAINED_GRU = True,  # 사전학습 GRU 학습 가능
        LR_PRETRAINED    = 1e-5,         # 사전학습 GRU에는 작은 LR
        LR_HEAD          = 5e-4,         # head는 기존 LR
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    os.makedirs(CFG['LOG_DIR'], exist_ok=True)

    if not os.path.exists(CFG['PRETRAINED_PATH']):
        print(f"❌ 사전 학습 가중치 없음: {CFG['PRETRAINED_PATH']}")
        sys.exit(1)

    # ── 데이터 로드 ──────────────────────────────────────────
    df = pd.read_csv(CFG['CSV_PATH'])
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)

    real_df = df[df['video_label'] == 0.0]
    fake_df = df[df['video_label'] == 1.0]
    print(f"📊 원본 데이터: Real={len(real_df)}, Fake={len(fake_df)}")

    n_fake = min(CFG['NUM_FAKE_SAMPLE'], len(fake_df))
    sampled_fake = fake_df.sample(n=n_fake, random_state=42)
    balanced_df = pd.concat([real_df, sampled_fake]).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    print(f"✂️  학습 데이터: 총 {len(balanced_df)}개 "
          f"(Real {len(real_df)} : Fake {len(sampled_fake)})")

    train_df, val_df = train_test_split(
        balanced_df,
        test_size    = CFG['VAL_RATIO'],
        stratify     = balanced_df['video_label'],
        random_state = 42
    )
    print(f"📂 Train: {len(train_df)}  |  Val: {len(val_df)}")

    train_loader = DataLoader(
        FakeAVCelebAudioDataset(
            train_df, CFG['BASE_DIR'],
            CFG['NUM_SEGMENTS'], CFG['SEGMENT_DURATION'], CFG['TARGET_SR']
        ),
        batch_size=CFG['BATCH_SIZE'], shuffle=True,
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        FakeAVCelebAudioDataset(
            val_df, CFG['BASE_DIR'],
            CFG['NUM_SEGMENTS'], CFG['SEGMENT_DURATION'], CFG['TARGET_SR']
        ),
        batch_size=CFG['BATCH_SIZE'],
        num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )

    # ── 모델 (🆕 unfreeze 옵션) ──────────────────────────────
    model = AudioEmotionFlowDetector(
        pretrained_path = CFG['PRETRAINED_PATH'],
        num_segments    = CFG['NUM_SEGMENTS'],
        gru_hidden      = CFG['GRU_HIDDEN'],
        dropout         = CFG['DROPOUT'],
        unfreeze_pretrained_gru = CFG['UNFREEZE_PRETRAINED_GRU']  # 🆕
    ).to(device)

    # 🆕 학습 가능 파라미터 분류
    pretrained_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'audio_backbone' in name:
            pretrained_params.append(param)
        else:
            head_params.append(param)
    
    pp_count = sum(p.numel() for p in pretrained_params)
    hp_count = sum(p.numel() for p in head_params)
    print(f"📊 학습 가능 파라미터:")
    print(f"   Pretrained: {pp_count:,}")
    print(f"   Head      : {hp_count:,}")
    print(f"   총합      : {pp_count + hp_count:,}")

    # pos_weight
    pos_weight = torch.tensor(
        [len(real_df) / len(sampled_fake)], dtype=torch.float32
    ).to(device)
    print(f"⚖️  BCE pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 🆕 Layer-wise LR
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

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['EPOCHS'], eta_min=CFG['LR_HEAD'] / 50
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── 학습 루프 ────────────────────────────────────────────
    best_auc      = 0.0
    best_val_loss = float('inf')
    no_improve    = 0
    history       = []

    print("=" * 60)
    print("🚀 오디오 감정 흐름 학습 시작 (CRNN v2 — GRU Unfreeze)")
    print("=" * 60)

    for epoch in range(CFG['EPOCHS']):
        t0 = time.time()

        train_loss, train_log = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )
        val_loss, val_acc, val_auc, val_log = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        lrs = [g['lr'] for g in optimizer.param_groups]

        print(f"\n🏁 E{epoch:02d} | "
              f"Train={train_loss:.4f} | "
              f"Val Loss={val_loss:.4f} | "
              f"Acc={val_acc:.1f}% | "
              f"AUC={val_auc:.1f}% | "
              f"LR={lrs} | {elapsed:.0f}s\n")

        train_log.to_csv(
            os.path.join(CFG['LOG_DIR'], f"e{epoch:02d}_train.csv"),
            index=False, encoding='utf-8-sig'
        )
        val_log.to_csv(
            os.path.join(CFG['LOG_DIR'], f"e{epoch:02d}_val.csv"),
            index=False, encoding='utf-8-sig'
        )

        history.append({
            'epoch': epoch, 'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'val_acc':  round(val_acc,  2),
            'val_auc':  round(val_auc,  2),
        })
        pd.DataFrame(history).to_csv(
            os.path.join(CFG['LOG_DIR'], "history.csv"), index=False
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc, 'val_auc': val_auc, 'cfg': CFG,
            }, CFG['CKPT_PATH'])
            print(f"  ⭐ Best 모델 저장 (AUC={best_auc:.1f}%)\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CFG['PATIENCE']:
                print(f"  ⏹ Early Stopping ({CFG['PATIENCE']} epoch 미개선)")
                break

    print("=" * 60)
    print(f"✅ 완료! Best AUC: {best_auc:.1f}%")
    print(f"   체크포인트: {CFG['CKPT_PATH']}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 8. 단일 영상 추론 유틸 (변경 없음)
# ══════════════════════════════════════════════════════════════════════════════
EMOTION_LABELS = ['Neutral', 'Calm', 'Happy', 'Sad',
                  'Angry', 'Fearful', 'Disgust', 'Surprised']


def predict_video(video_path: str, ckpt_path: str,
                  pretrained_path: str = 'audio_emotion_crnn_best.pth',
                  device: str = 'cpu'):
    device = torch.device(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    cfg    = ckpt.get('cfg', {})

    model = AudioEmotionFlowDetector(
        pretrained_path = pretrained_path,
        num_segments    = cfg.get('NUM_SEGMENTS', 16),
        gru_hidden      = cfg.get('GRU_HIDDEN', 128),
        dropout         = cfg.get('DROPOUT', 0.4),
        unfreeze_pretrained_gru = cfg.get('UNFREEZE_PRETRAINED_GRU', False),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    segments = extract_audio_segments(
        video_path,
        num_segments     = cfg.get('NUM_SEGMENTS', 16),
        target_sr        = cfg.get('TARGET_SR', 16000),
        segment_duration = cfg.get('SEGMENT_DURATION', 3.0)
    )
    if segments is None:
        raise ValueError(f"오디오 추출 실패: {video_path}")

    seg_t = segments.unsqueeze(0).to(device)

    with torch.no_grad():
        N, C, S = segments.shape
        flat    = segments.view(N, C, S).to(device)
        _, emo_logit = model.audio_backbone.forward_with_feat(flat)
        emo_prob = F.softmax(emo_logit, dim=-1).cpu().numpy()

        logit, attn = model(seg_t)
        fake_prob   = torch.sigmoid(logit).item()

    verdict      = "FAKE" if fake_prob > 0.5 else "REAL"
    attn_weights = attn.squeeze(0).cpu().numpy()

    print(f"\n🎬 {os.path.basename(video_path)}")
    print(f"   판정  : {verdict}  (Fake {fake_prob*100:.1f}%)")
    print(f"\n   구간별 감정 흐름:")
    for t in range(len(emo_prob)):
        dom = EMOTION_LABELS[np.argmax(emo_prob[t])]
        bar = "█" * int(attn_weights[t] * 200)
        print(f"   [{t:2d}] {dom:10s} attn={attn_weights[t]:.3f} {bar}")

    return fake_prob, attn_weights, emo_prob, verdict


if __name__ == "__main__":
    main()