"""
==============================================================================
[v2 통합 평가] 4가지 시나리오 모두 평가
Unified Evaluation: All 4 Scenarios with Emotion v2 (Unfreezed)
==============================================================================

[목적]
v2 emotion 모델 학습 완료 후, 4가지 시나리오를 모두 평가하여
'Emotion 도입이 도움된다'는 주장을 데이터로 강화한다.

[가중치 매핑]
                  X3D / AASIST          HSEmotion / CRNN
  Scenario 1 F→F: v1 (상위 폴더)         v2 (FAV 학습)
  Scenario 2 F→P: v1 (상위 폴더)         v2 (FAV 학습)
  Scenario 3 P→F: PGF (현재 폴더)        v2 (PGF 학습)
  Scenario 4 P→P: PGF (현재 폴더)        v2 (PGF 학습)

[사용법]
  python evaluate_v2.py --scenario FF   # 단일 시나리오
  python evaluate_v2.py --scenario all  # 전체 4가지
  
  # 평가셋 크기 조정
  python evaluate_v2.py --scenario all --n_real 500 --n_fake 500

[출력 (현재 폴더)]
  evaluation_v2_report/
    ├── FF_predictions.csv
    ├── FF_metrics.csv
    ├── FP_predictions.csv
    ├── FP_metrics.csv
    ├── PF_predictions.csv
    ├── PF_metrics.csv
    ├── PP_predictions.csv
    ├── PP_metrics.csv
    └── all_scenarios_comparison.csv  ← 최종 비교표
==============================================================================
"""

import sys
import os
import json
import time
import argparse
import functools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import av
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = Ftv

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torchvision import transforms as T

torch.load = functools.partial(torch.load, weights_only=False)

# 모델 클래스 import
try:
    from emotion_deepfake_detector_lite import EmotionFlowDetectorLite
except ImportError:
    try:
        from train_HSEmotion import EmotionFlowDetectorLite
    except ImportError:
        print("❌ EmotionFlowDetectorLite import 실패")
        sys.exit(1)

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

try:
    from aasist.models.AASIST import Model as AASISTModel
except ImportError:
    print("❌ aasist 모듈 import 실패")
    sys.exit(1)

from polyglotfake_data import build_polyglotfake_dataframe


# ══════════════════════════════════════════════════════════════════════════════
# 가중치 매핑 정의 (시나리오별)
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS_MAP = {
    'FF': {  # FakeAVCeleb 학습 가중치 사용
        'x3d':       os.path.join(PARENT_DIR, 'x3d_model_best_final.pth'),
        'aasist':    os.path.join(PARENT_DIR, 'aasist_model_best_final.pth'),
        'hsemo':     'emotion_flow_lite_fav_v2_best.pth',  # 현재 폴더
        'crnn':      'audio_flow_deepfake_fav_v2_best.pth',
        'eval_set':  'fakeavceleb',
        'name':      'F→F (FakeAVCeleb In-Domain)',
    },
    'FP': {  # FakeAVCeleb 학습 가중치 사용 (평가 데이터만 PGF)
        'x3d':       os.path.join(PARENT_DIR, 'x3d_model_best_final.pth'),
        'aasist':    os.path.join(PARENT_DIR, 'aasist_model_best_final.pth'),
        'hsemo':     'emotion_flow_lite_fav_v2_best.pth',
        'crnn':      'audio_flow_deepfake_fav_v2_best.pth',
        'eval_set':  'polyglotfake',
        'name':      'F→P (Cross: FakeAVCeleb→PolyGlotFake)',
    },
    'PF': {  # PolyGlotFake 학습 가중치 사용
        'x3d':       'x3d_model_pgf_best.pth',
        'aasist':    'aasist_model_pgf_best.pth',
        'hsemo':     'emotion_flow_lite_pgf_v2_best.pth',
        'crnn':      'audio_flow_deepfake_pgf_v2_best.pth',
        'eval_set':  'fakeavceleb',
        'name':      'P→F (Cross: PolyGlotFake→FakeAVCeleb)',
    },
    'PP': {  # PolyGlotFake 학습 가중치 사용
        'x3d':       'x3d_model_pgf_best.pth',
        'aasist':    'aasist_model_pgf_best.pth',
        'hsemo':     'emotion_flow_lite_pgf_v2_best.pth',
        'crnn':      'audio_flow_deepfake_pgf_v2_best.pth',
        'eval_set':  'polyglotfake',
        'name':      'P→P (PolyGlotFake In-Domain)',
    },
}

AASIST_CONFIG     = os.path.join(PARENT_DIR, "aasist/config/AASIST.conf")
CRNN_PRETRAINED   = os.path.join(PARENT_DIR, "audio_emotion_crnn_best.pth")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전처리 (영상/오디오 로드)
# ══════════════════════════════════════════════════════════════════════════════
def rescale_video(x): return x / 255.0
def permute_to_tc(x): return x.permute(1, 0, 2, 3)
def permute_to_ct(x): return x.permute(1, 0, 2, 3)

x3d_video_transform = Compose([
    UniformTemporalSubsample(16),
    Lambda(rescale_video),
    Lambda(permute_to_tc),
    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    Lambda(permute_to_ct),
    ShortSideScale(size=256),
    Resize((224, 224))
])

hsemo_frame_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_video_for_x3d(path, max_frames=128):
    try:
        container = av.open(path)
        all_frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(all_frames) < 16: return None
        if len(all_frames) > max_frames:
            indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
            all_frames = [all_frames[i] for i in indices]
        video = np.stack(all_frames)
        return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
    except Exception:
        return None


def load_frames_for_hsemo(path, num_frames=16):
    try:
        container = av.open(path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames < num_frames:
            container.close()
            return None
        target_indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())
        sampled = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in target_indices:
                sampled[i] = frame.to_rgb().to_ndarray()
            if len(sampled) >= num_frames: break
        container.close()
        if len(sampled) < num_frames: return None
        sorted_keys = sorted(sampled.keys())
        return torch.stack([hsemo_frame_transform(sampled[k]) for k in sorted_keys])
    except Exception:
        return None


def load_audio_for_aasist(video_path, target_sr=16000, max_length=64000):
    try:
        container = av.open(video_path)
        if not container.streams.audio:
            container.close()
            return None
        sample_rate = container.streams.audio[0].rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            else:
                arr = arr.astype(np.float32)
            if len(arr.shape) > 1 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            elif len(arr.shape) == 1:
                arr = arr[np.newaxis, :]
            frames.append(arr)
        container.close()
        if not frames: return None
        waveform = np.concatenate(frames, axis=-1)
        waveform = torch.from_numpy(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.shape[1]))
        return waveform.squeeze()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. 평가셋 구성
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


def build_fakeavceleb_eval_set(n_real=500, n_fake=500, train_fake_n=2000, seed=42):
    """
    FakeAVCeleb 평가셋: 학습에 사용 안 한 Fake + Real 일부.
    """
    base_dir = find_fakeavceleb_root()
    csv_path = os.path.join(base_dir, 'meta_data.csv')
    df = pd.read_csv(csv_path)
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)

    real_df = df[df['video_label'] == 0.0].reset_index(drop=True)
    fake_df = df[df['video_label'] == 1.0].reset_index(drop=True)

    sampled_real = real_df.sample(n=min(n_real, len(real_df)), random_state=seed)
    train_fake_idx = fake_df.sample(n=train_fake_n, random_state=seed).index
    unseen_fake_df = fake_df.drop(train_fake_idx).reset_index(drop=True)
    sampled_fake = unseen_fake_df.sample(
        n=min(n_fake, len(unseen_fake_df)), random_state=seed + 1
    )

    val_df = pd.concat([sampled_real, sampled_fake]).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    val_df.attrs['base_dir'] = base_dir
    val_df.attrs['type'] = 'fakeavceleb'
    return val_df


def build_polyglotfake_eval_set(n_real=300, n_fake=1000, train_fake_n=2000, seed=42):
    """
    PolyGlotFake 평가셋 구성.
    F→P: 학습 도메인이 FakeAVCeleb이므로 PolyGlotFake 전체 사용 가능
    P→P: 학습에 사용된 2000 Fake 제외
    """
    df = build_polyglotfake_dataframe()
    real_df = df[df['video_label'] == 0.0].reset_index(drop=True)
    fake_df = df[df['video_label'] == 1.0].reset_index(drop=True)

    sampled_real = real_df.sample(n=min(n_real, len(real_df)), random_state=seed + 100)

    # PGF 학습에 사용된 Fake 제외
    train_fake_idx = fake_df.sample(n=train_fake_n, random_state=seed).index
    unseen_fake_df = fake_df.drop(train_fake_idx).reset_index(drop=True)

    # TTS×Sync 균등 샘플링
    sampled_fake_list = []
    n_per_combo = max(1, n_fake // 10)
    for tts in unseen_fake_df['tts'].unique():
        for sync in unseen_fake_df['sync'].unique():
            sub = unseen_fake_df[
                (unseen_fake_df['tts']==tts) & (unseen_fake_df['sync']==sync)
            ]
            if len(sub) == 0: continue
            sample_size = min(n_per_combo, len(sub))
            sampled_fake_list.append(sub.sample(n=sample_size, random_state=seed + 200))

    sampled_fake = pd.concat(sampled_fake_list)
    if len(sampled_fake) > n_fake:
        sampled_fake = sampled_fake.sample(n=n_fake, random_state=seed + 300)

    val_df = pd.concat([sampled_real, sampled_fake]).sample(
        frac=1, random_state=seed + 400
    ).reset_index(drop=True)
    val_df.attrs['type'] = 'polyglotfake'
    return val_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset (FakeAVCeleb / PolyGlotFake 통합)
# ══════════════════════════════════════════════════════════════════════════════
class MultiInputDataset(Dataset):
    def __init__(self, df, dataset_type, num_segments=16, segment_duration=3.0):
        self.df = df.reset_index(drop=True)
        self.dataset_type = dataset_type
        self.num_segments = num_segments
        self.segment_duration = segment_duration
        
        if dataset_type == 'fakeavceleb':
            self.base_dir = df.attrs.get('base_dir')

    def __len__(self):
        return len(self.df)

    def _get_video_path(self, idx):
        row = self.df.iloc[idx]
        if self.dataset_type == 'polyglotfake':
            return row['video_path']
        else:  # fakeavceleb
            rel_path = row.iloc[-2].replace("FakeAVCeleb",
                                             os.path.basename(self.base_dir))
            video_path = os.path.join(os.path.dirname(self.base_dir), rel_path,
                                       row['path'])
            if not os.path.exists(video_path):
                video_path = os.path.join(self.base_dir,
                                           row.iloc[-2].replace("FakeAVCeleb/", ""),
                                           row['path'])
            return video_path

    def _try_load(self, idx):
        video_path = self._get_video_path(idx)
        if not os.path.exists(video_path):
            return None

        x3d_v = load_video_for_x3d(video_path)
        if x3d_v is None: return None
        x3d_v = x3d_video_transform(x3d_v)

        aasist_a = load_audio_for_aasist(video_path)
        if aasist_a is None: return None

        hsemo_f = load_frames_for_hsemo(video_path)
        if hsemo_f is None: return None

        crnn_s = extract_audio_segments(
            video_path, num_segments=self.num_segments,
            target_sr=16000, segment_duration=self.segment_duration
        )
        if crnn_s is None: return None

        row = self.df.iloc[idx]
        return (x3d_v, aasist_a, hsemo_f, crnn_s,
                torch.tensor(float(row['video_label'])), idx)

    def __getitem__(self, idx):
        for offset in range(len(self)):
            r = self._try_load((idx + offset) % len(self))
            if r is not None:
                return r
        raise RuntimeError("로드 가능한 샘플 없음")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_models(scenario, device):
    """시나리오에 맞는 가중치를 로드."""
    weights = WEIGHTS_MAP[scenario]
    
    print(f"\n🧠 모델 로드 중 (Scenario: {scenario})...")
    print(f"   X3D    : {weights['x3d']}")
    print(f"   AASIST : {weights['aasist']}")
    print(f"   HSEmotion: {weights['hsemo']}")
    print(f"   CRNN   : {weights['crnn']}")

    # X3D
    x3d = x3d_m(pretrained=False)
    x3d.blocks[5].proj       = nn.Linear(2048, 1)
    x3d.blocks[5].activation = nn.Identity()
    x3d.load_state_dict(torch.load(weights['x3d'], map_location=device))
    x3d = x3d.to(device).eval()

    # AASIST
    with open(AASIST_CONFIG, 'r') as f:
        config = json.load(f)
    aasist = AASISTModel(config['model_config'])
    aasist.load_state_dict(torch.load(weights['aasist'], map_location=device))
    aasist = aasist.to(device).eval()

    # HSEmotion v2 (unfreeze 옵션 포함)
    hs_ckpt = torch.load(weights['hsemo'], map_location=device)
    hs_cfg  = hs_ckpt.get('cfg', {})
    hs_state = hs_ckpt.get('model_state_dict', hs_ckpt)
    hs_hidden = (hs_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in hs_state else 64)
    
    # v2 가중치는 unfreeze=2로 학습된 모델
    unfreeze_blocks = hs_cfg.get('UNFREEZE_LAST_BLOCKS', 2)
    
    hsemo = EmotionFlowDetectorLite(
        model_name=hs_cfg.get('MODEL_NAME', 'enet_b0_8_best_afew'),
        num_frames=hs_cfg.get('NUM_FRAMES', 16),
        gru_hidden=hs_hidden,
        dropout=hs_cfg.get('DROPOUT', 0.3),
        device='cpu',
        unfreeze_last_blocks=unfreeze_blocks  # v2 학습과 일치
    ).to(device)
    hsemo.load_state_dict(hs_state)
    hsemo.eval()

    # CRNN v2 (unfreeze 옵션 포함)
    cr_ckpt = torch.load(weights['crnn'], map_location=device)
    cr_cfg  = cr_ckpt.get('cfg', {})
    cr_state = cr_ckpt.get('model_state_dict', cr_ckpt)
    cr_hidden = (cr_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in cr_state else 128)
    
    unfreeze_gru = cr_cfg.get('UNFREEZE_PRETRAINED_GRU', True)
    
    crnn = AudioEmotionFlowDetector(
        pretrained_path=CRNN_PRETRAINED,
        num_segments=cr_cfg.get('NUM_SEGMENTS', 16),
        gru_hidden=cr_hidden,
        dropout=cr_cfg.get('DROPOUT', 0.4),
        unfreeze_pretrained_gru=unfreeze_gru
    ).to(device)
    crnn.load_state_dict(cr_state)
    crnn.eval()

    for m in [x3d, aasist, hsemo, crnn]:
        for p in m.parameters():
            p.requires_grad = False

    print("  ✅ 모든 모델 로드 완료")
    return x3d, aasist, hsemo, crnn


# ══════════════════════════════════════════════════════════════════════════════
# 5. 추론 + 분석
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_one_sample(x3d_v, aasist_a, hsemo_f, crnn_s,
                     x3d, aasist, hsemo, crnn, device):
    v_in = x3d_v.unsqueeze(0).to(device)
    p_v_art = torch.sigmoid(x3d(v_in)).item()
    del v_in

    a_in = aasist_a.unsqueeze(0).to(device)
    _, a_out = aasist(a_in)
    p_a_art = torch.softmax(a_out, dim=1)[0, 1].item()
    del a_in, a_out

    hf_in = hsemo_f.unsqueeze(0).to(device)
    e_logit, _ = hsemo(hf_in)
    p_v_emo = torch.sigmoid(e_logit).item()
    del hf_in, e_logit

    cs_in = crnn_s.unsqueeze(0).to(device)
    c_logit, _ = crnn(cs_in)
    p_a_emo = torch.sigmoid(c_logit).item()
    del cs_in, c_logit

    return p_v_art, p_a_art, p_v_emo, p_a_emo


def prob_or(*ps):
    r = np.ones_like(ps[0])
    for p in ps: r *= (1.0 - p)
    return 1.0 - r


def compute_metrics(probs, labels, t=0.5):
    preds = (probs > t).astype(int)
    li = labels.astype(int)
    cm = confusion_matrix(li, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    try: auc = roc_auc_score(labels, probs) * 100
    except: auc = 0.0
    return dict(
        auc=auc,
        acc=accuracy_score(li, preds) * 100,
        precision=precision_score(li, preds, zero_division=0) * 100,
        recall=recall_score(li, preds, zero_division=0) * 100,
        f1=f1_score(li, preds, zero_division=0) * 100,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. 단일 시나리오 평가
# ══════════════════════════════════════════════════════════════════════════════
def run_scenario(scenario, args, report_dir):
    weights = WEIGHTS_MAP[scenario]
    print("\n" + "=" * 70)
    print(f"🎯 Scenario: {weights['name']}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 가중치 파일 존재 확인
    paths_to_check = [weights['x3d'], weights['aasist'],
                      weights['hsemo'], weights['crnn']]
    missing = [p for p in paths_to_check if not os.path.exists(p)]
    if missing:
        print(f"❌ 누락된 가중치 파일:")
        for p in missing:
            print(f"   - {p}")
        return None

    # 평가셋 구성
    if weights['eval_set'] == 'fakeavceleb':
        val_df = build_fakeavceleb_eval_set(
            n_real=args.n_real, n_fake=args.n_fake
        )
    else:
        val_df = build_polyglotfake_eval_set(
            n_real=args.n_real_pgf, n_fake=args.n_fake_pgf
        )

    print(f"📂 평가셋: {len(val_df)}개")
    print(f"   Real: {(val_df['video_label']==0).sum()}")
    print(f"   Fake: {(val_df['video_label']==1).sum()}")

    # 모델 로드
    x3d, aasist, hsemo, crnn = load_models(scenario, device)

    # 추론
    dataset = MultiInputDataset(val_df, weights['eval_set'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"\n🚀 추론 시작 ({len(dataset)}개)\n")
    results = []
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        x3d_v, aasist_a, hsemo_f, crnn_s, label, sample_idx = batch
        x3d_v    = x3d_v.squeeze(0)
        aasist_a = aasist_a.squeeze(0)
        hsemo_f  = hsemo_f.squeeze(0)
        crnn_s   = crnn_s.squeeze(0)

        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            p_v_art, p_a_art, p_v_emo, p_a_emo = infer_one_sample(
                x3d_v, aasist_a, hsemo_f, crnn_s,
                x3d, aasist, hsemo, crnn, device
            )

        idx = sample_idx.item()
        meta = val_df.iloc[idx]
        result = {
            'video_label':  int(meta['video_label']),
            'p_v_artifact': round(p_v_art * 100, 2),
            'p_a_artifact': round(p_a_art * 100, 2),
            'p_v_emotion':  round(p_v_emo * 100, 2),
            'p_a_emotion':  round(p_a_emo * 100, 2),
        }
        # 메타데이터 추가
        if 'method' in meta.index:
            result['method'] = meta['method']
        if 'tts' in meta.index:
            result['tts'] = meta['tts']
        if 'sync' in meta.index:
            result['sync'] = meta['sync']
        results.append(result)

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (batch_idx + 1) / elapsed
            eta = (len(loader) - batch_idx - 1) / speed
            print(f"  [{batch_idx+1:4d}/{len(loader)}] "
                  f"속도: {speed:.1f}/s ETA: {eta/60:.1f}분")

    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(report_dir, f'{scenario}_predictions.csv'),
                  index=False, encoding='utf-8-sig')

    # ── 분석 ─────────────────────────────────────────────────
    labels = df_out['video_label'].values.astype(float)
    p_v_art = df_out['p_v_artifact'].values / 100
    p_a_art = df_out['p_a_artifact'].values / 100
    p_v_emo = df_out['p_v_emotion'].values / 100
    p_a_emo = df_out['p_a_emotion'].values / 100

    art_or = prob_or(p_v_art, p_a_art)
    emo_or = prob_or(p_v_emo, p_a_emo)
    final_or = prob_or(art_or, emo_or)

    print("\n" + "-" * 70)
    print(f"🎯 결과: {weights['name']}")
    print("-" * 70)

    strategies = {
        'X3D 단독':              p_v_art,
        'AASIST 단독':           p_a_art,
        'HSEmotion+GRU 단독':    p_v_emo,
        'CRNN+GRU 단독':         p_a_emo,
        'Score_artifact':        art_or,
        'Score_emotion':         emo_or,
        '🌟 Score_final':        final_or,
    }

    summary = []
    print(f"\n  {'전략':<30} {'AUC':>7} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, probs in strategies.items():
        m = compute_metrics(probs, labels)
        summary.append({'strategy': name, **m})
        print(f"  {name:<30} {m['auc']:>6.2f}% {m['acc']:>6.2f}% "
              f"{m['precision']:>6.2f}% {m['recall']:>6.2f}% {m['f1']:>6.2f}%")

    pd.DataFrame(summary).round(2).to_csv(
        os.path.join(report_dir, f'{scenario}_metrics.csv'),
        index=False, encoding='utf-8-sig'
    )

    return {
        'scenario': scenario,
        'name': weights['name'],
        'summary': summary,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. 종합 비교표 생성
# ══════════════════════════════════════════════════════════════════════════════
def make_comparison_table(all_results, report_dir):
    """4가지 시나리오 종합 비교표"""
    print("\n" + "=" * 70)
    print("📊 4가지 시나리오 종합 비교 (v2 — Emotion Unfreeze)")
    print("=" * 70)

    # AUC, F1 추출
    rows = []
    for result in all_results:
        if result is None: continue
        s = result['summary']
        # 인덱스: 0=X3D 1=AASIST 2=HSEmo 3=CRNN 4=art 5=emo 6=final
        rows.append({
            'scenario': result['scenario'],
            'name': result['name'],
            'X3D_auc':       s[0]['auc'], 'X3D_f1': s[0]['f1'],
            'AASIST_auc':    s[1]['auc'], 'AASIST_f1': s[1]['f1'],
            'HSEmo_auc':     s[2]['auc'], 'HSEmo_f1': s[2]['f1'],
            'CRNN_auc':      s[3]['auc'], 'CRNN_f1': s[3]['f1'],
            'artifact_auc':  s[4]['auc'], 'artifact_f1': s[4]['f1'],
            'emotion_auc':   s[5]['auc'], 'emotion_f1': s[5]['f1'],
            'final_auc':     s[6]['auc'], 'final_f1': s[6]['f1'],
        })

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(os.path.join(report_dir, 'all_scenarios_comparison.csv'),
                  index=False, encoding='utf-8-sig')

    # 화면 출력 (AUC)
    print(f"\n[AUC 비교]")
    print(f"  {'시나리오':<22} {'Artifact':>10} {'Emotion':>10} {'Final':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for r in rows:
        print(f"  {r['scenario']:<22} {r['artifact_auc']:>9.2f}% "
              f"{r['emotion_auc']:>9.2f}% {r['final_auc']:>9.2f}%")

    # 화면 출력 (F1)
    print(f"\n[F1 비교]")
    print(f"  {'시나리오':<22} {'Artifact':>10} {'Emotion':>10} {'Final':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for r in rows:
        print(f"  {r['scenario']:<22} {r['artifact_f1']:>9.2f}% "
              f"{r['emotion_f1']:>9.2f}% {r['final_f1']:>9.2f}%")

    # 단일 모델 비교 (Emotion 모델 개선 효과 확인)
    print(f"\n[Emotion 단독 모델 v2 성능]")
    print(f"  {'시나리오':<22} {'HSEmotion':>10} {'CRNN':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    for r in rows:
        print(f"  {r['scenario']:<22} AUC {r['HSEmo_auc']:>5.2f}%/F1 {r['HSEmo_f1']:>5.2f}%   "
              f"AUC {r['CRNN_auc']:>5.2f}%/F1 {r['CRNN_f1']:>5.2f}%")

    # v1 vs v2 비교 (참고용)
    print(f"\n[v1 → v2 비교 — 단일 Emotion 모델 AUC]")
    v1_baselines = {
        'FF': {'HSEmo': 92.58, 'CRNN': 81.46},
        'FP': {'HSEmo': 52.08, 'CRNN': 70.02},
        'PF': {'HSEmo': 52.20, 'CRNN': 65.33},
        'PP': {'HSEmo': 51.67, 'CRNN': 97.25},
    }
    print(f"  {'시나리오':<10} {'HSEmotion':>20} {'CRNN':>20}")
    print(f"  {'-'*10} {'-'*20} {'-'*20}")
    for r in rows:
        s = r['scenario']
        if s in v1_baselines:
            v1 = v1_baselines[s]
            hse_diff = r['HSEmo_auc'] - v1['HSEmo']
            cr_diff = r['CRNN_auc'] - v1['CRNN']
            hse_sign = '+' if hse_diff >= 0 else ''
            cr_sign = '+' if cr_diff >= 0 else ''
            print(f"  {s:<10} {v1['HSEmo']:.2f}% → {r['HSEmo_auc']:.2f}% ({hse_sign}{hse_diff:+5.2f}p)   "
                  f"{v1['CRNN']:.2f}% → {r['CRNN_auc']:.2f}% ({cr_sign}{cr_diff:+5.2f}p)")

    print(f"\n💾 종합 비교표: {os.path.join(report_dir, 'all_scenarios_comparison.csv')}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="v2 통합 평가 (4가지 시나리오)")
    parser.add_argument('--scenario', type=str, default='all',
                        choices=['FF', 'FP', 'PF', 'PP', 'all'],
                        help="평가 시나리오 (기본: all)")
    parser.add_argument('--n_real', type=int, default=500,
                        help="FakeAVCeleb Real 개수")
    parser.add_argument('--n_fake', type=int, default=500,
                        help="FakeAVCeleb Fake 개수")
    parser.add_argument('--n_real_pgf', type=int, default=300,
                        help="PolyGlotFake Real 개수")
    parser.add_argument('--n_fake_pgf', type=int, default=1000,
                        help="PolyGlotFake Fake 개수")
    parser.add_argument('--report_dir', type=str, default='evaluation_v2_report',
                        help="결과 저장 폴더")
    args = parser.parse_args()

    print(f"📁 Working dir: {os.getcwd()}")
    print(f"💾 Report dir : {os.path.abspath(args.report_dir)}")
    os.makedirs(args.report_dir, exist_ok=True)

    # 시나리오 선택
    if args.scenario == 'all':
        scenarios = ['FF', 'FP', 'PF', 'PP']
    else:
        scenarios = [args.scenario]

    # 각 시나리오 실행
    all_results = []
    for s in scenarios:
        result = run_scenario(s, args, args.report_dir)
        all_results.append(result)

    # 종합 비교표
    if len(scenarios) > 1:
        make_comparison_table(all_results, args.report_dir)

    print("\n" + "=" * 70)
    print("✅ v2 평가 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()