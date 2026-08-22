"""
==============================================================================
[v3 평가] 4가지 시나리오 통합 평가
HSEmotion → MAE-DFER 교체, CRNN은 v1 유지
==============================================================================

[가중치 매핑]
                  X3D / AASIST          MAE-DFER          CRNN
  Scenario 1 F→F: v1 (상위 폴더)        FAV v3 (현폴더)   v1 (상위 폴더)
  Scenario 2 F→P: v1 (상위 폴더)        FAV v3 (현폴더)   v1 (상위 폴더)
  Scenario 3 P→F: PGF (현폴더)          PGF v3 (현폴더)   PGF v1 (현폴더)
  Scenario 4 P→P: PGF (현폴더)          PGF v3 (현폴더)   PGF v1 (현폴더)

[v3 = MAE-DFER 기반]
- 백본: MAE-DFER (85M, LGI-Former)
- 입력 해상도: 160x160 (Temporal Consistency 적용)
- 정규화: ImageNet 표준 (0.485, 0.456, 0.406) - 학습 코드와 동기화됨!
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

import torchvision.transforms.functional as F_tv
from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize
from torchvision import transforms as T

torch.load = functools.partial(torch.load, weights_only=False)

# ── 모델 클래스 import ─────────────────────────────────────────
try:
    from emotion_deepfake_detector_mae import EmotionFlowDetectorMAE
except ImportError:
    print("❌ EmotionFlowDetectorMAE import 실패")
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
# 가중치 매핑
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS_MAP = {
    'FF': {
        'x3d':       os.path.join(PARENT_DIR, 'x3d_model_best_final.pth'),
        'aasist':    os.path.join(PARENT_DIR, 'aasist_model_best_final.pth'),
        'maedfer':   'emotion_flow_mae_fav_best.pth',            
        'crnn':      os.path.join(PARENT_DIR, 'audio_flow_deepfake_best.pth'),  
        'eval_set':  'fakeavceleb',
        'name':      'F→F (FakeAVCeleb In-Domain)',
    },
    'FP': {
        'x3d':       os.path.join(PARENT_DIR, 'x3d_model_best_final.pth'),
        'aasist':    os.path.join(PARENT_DIR, 'aasist_model_best_final.pth'),
        'maedfer':   'emotion_flow_mae_fav_best.pth',            
        'crnn':      os.path.join(PARENT_DIR, 'audio_flow_deepfake_best.pth'),  
        'eval_set':  'polyglotfake',
        'name':      'F→P (Cross: FakeAVCeleb→PolyGlotFake)',
    },
    'PF': {
        'x3d':       'x3d_model_pgf_best.pth',
        'aasist':    'aasist_model_pgf_best.pth',
        'maedfer':   'emotion_flow_mae_pgf_best.pth',            
        'crnn':      'audio_flow_deepfake_pgf_best.pth',         
        'eval_set':  'fakeavceleb',
        'name':      'P→F (Cross: PolyGlotFake→FakeAVCeleb)',
    },
    'PP': {
        'x3d':       'x3d_model_pgf_best.pth',
        'aasist':    'aasist_model_pgf_best.pth',
        'maedfer':   'emotion_flow_mae_pgf_best.pth',            
        'crnn':      'audio_flow_deepfake_pgf_best.pth',         
        'eval_set':  'polyglotfake',
        'name':      'P→P (PolyGlotFake In-Domain)',
    },
}

AASIST_CONFIG    = os.path.join(PARENT_DIR, "aasist/config/AASIST.conf")
CRNN_PRETRAINED  = os.path.join(PARENT_DIR, "audio_emotion_crnn_best.pth")
MAE_DFER_PATH    = os.path.expanduser("~/hsh/AIApplication/mae_dfer")
MAE_DFER_PRETRAIN = os.path.join(MAE_DFER_PATH, "saved/pretrained/checkpoint-49.pth")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전처리
# ══════════════════════════════════════════════════════════════════════════════
def rescale_video(x): return x / 255.0
def permute_to_tc(x): return x.permute(1, 0, 2, 3)
def permute_to_ct(x): return x.permute(1, 0, 2, 3)

# X3D용 (224x224)
x3d_video_transform = Compose([
    UniformTemporalSubsample(16),
    Lambda(rescale_video),
    Lambda(permute_to_tc),
    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    Lambda(permute_to_ct),
    ShortSideScale(size=256),
    Resize((224, 224))
])

# [수정] MAE-DFER용 평가 Transform (학습 코드와 동일하게 적용)
def apply_video_transform_val(frames_list):
    # 공통 Resize (160x160)
    frames = [F_tv.resize(img, (160, 160)) for img in frames_list]
    # 공통 ToTensor & Normalize (ImageNet 표준)
    frames = [F_tv.to_tensor(img) for img in frames]
    frames = [F_tv.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for img in frames]
    return torch.stack(frames)


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


def load_frames_for_maedfer(path, num_frames=16):
    """MAE-DFER용: 160x160, 16 frames, Temporal Consistency 유지"""
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
        
        # [수정] 추출된 프레임들을 PIL로 변환 후 일괄 Transform 적용
        selected_pil = [T.ToPILImage()(sampled[k]) for k in sorted_keys]
        return apply_video_transform_val(selected_pil)
    except Exception:
        return None

# (오디오 전처리 등 나머지 로직은 변경점 없으므로 그대로 사용합니다)
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
    base_dir = find_fakeavceleb_root()
    csv_path = os.path.join(base_dir, 'meta_data.csv')
    df = pd.read_csv(csv_path)
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)
    real_df = df[df['video_label'] == 0.0].reset_index(drop=True)
    fake_df = df[df['video_label'] == 1.0].reset_index(drop=True)
    sampled_real = real_df.sample(n=min(n_real, len(real_df)), random_state=seed)
    train_fake_idx = fake_df.sample(n=train_fake_n, random_state=seed).index
    unseen_fake_df = fake_df.drop(train_fake_idx).reset_index(drop=True)
    sampled_fake = unseen_fake_df.sample(n=min(n_fake, len(unseen_fake_df)), random_state=seed + 1)
    val_df = pd.concat([sampled_real, sampled_fake]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df.attrs['base_dir'] = base_dir
    val_df.attrs['type'] = 'fakeavceleb'
    return val_df

def build_polyglotfake_eval_set(n_real=300, n_fake=1000, train_fake_n=2000, seed=42):
    df = build_polyglotfake_dataframe()
    real_df = df[df['video_label'] == 0.0].reset_index(drop=True)
    fake_df = df[df['video_label'] == 1.0].reset_index(drop=True)
    sampled_real = real_df.sample(n=min(n_real, len(real_df)), random_state=seed + 100)
    train_fake_idx = fake_df.sample(n=train_fake_n, random_state=seed).index
    unseen_fake_df = fake_df.drop(train_fake_idx).reset_index(drop=True)
    sampled_fake_list = []
    n_per_combo = max(1, n_fake // 10)
    for tts in unseen_fake_df['tts'].unique():
        for sync in unseen_fake_df['sync'].unique():
            sub = unseen_fake_df[(unseen_fake_df['tts']==tts) & (unseen_fake_df['sync']==sync)]
            if len(sub) == 0: continue
            sample_size = min(n_per_combo, len(sub))
            sampled_fake_list.append(sub.sample(n=sample_size, random_state=seed + 200))
    sampled_fake = pd.concat(sampled_fake_list)
    if len(sampled_fake) > n_fake:
        sampled_fake = sampled_fake.sample(n=n_fake, random_state=seed + 300)
    val_df = pd.concat([sampled_real, sampled_fake]).sample(frac=1, random_state=seed + 400).reset_index(drop=True)
    val_df.attrs['type'] = 'polyglotfake'
    return val_df

# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset
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
        else:
            rel_path = row.iloc[-2].replace("FakeAVCeleb", os.path.basename(self.base_dir))
            video_path = os.path.join(os.path.dirname(self.base_dir), rel_path, row['path'])
            if not os.path.exists(video_path):
                video_path = os.path.join(self.base_dir, row.iloc[-2].replace("FakeAVCeleb/", ""), row['path'])
            return video_path

    def _try_load(self, idx):
        video_path = self._get_video_path(idx)
        if not os.path.exists(video_path): return None

        x3d_v = load_video_for_x3d(video_path)
        if x3d_v is None: return None
        x3d_v = x3d_video_transform(x3d_v)

        aasist_a = load_audio_for_aasist(video_path)
        if aasist_a is None: return None

        maedfer_f = load_frames_for_maedfer(video_path)
        if maedfer_f is None: return None

        crnn_s = extract_audio_segments(video_path, num_segments=self.num_segments, target_sr=16000, segment_duration=self.segment_duration)
        if crnn_s is None: return None

        row = self.df.iloc[idx]
        return (x3d_v, aasist_a, maedfer_f, crnn_s, torch.tensor(float(row['video_label'])), idx)

    def __getitem__(self, idx):
        for offset in range(len(self)):
            r = self._try_load((idx + offset) % len(self))
            if r is not None: return r
        raise RuntimeError("로드 가능한 샘플 없음")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_models(scenario, device):
    weights = WEIGHTS_MAP[scenario]
    print(f"\n🧠 모델 로드 중 (Scenario: {scenario})...")
    print(f"   X3D      : {weights['x3d']}")
    print(f"   AASIST   : {weights['aasist']}")
    print(f"   MAE-DFER : {weights['maedfer']}")
    print(f"   CRNN v1  : {weights['crnn']}")

    x3d = x3d_m(pretrained=False)
    x3d.blocks[5].proj = nn.Linear(2048, 1)
    x3d.blocks[5].activation = nn.Identity()
    x3d.load_state_dict(torch.load(weights['x3d'], map_location=device))
    x3d = x3d.to(device).eval()

    with open(AASIST_CONFIG, 'r') as f: config = json.load(f)
    aasist = AASISTModel(config['model_config'])
    aasist.load_state_dict(torch.load(weights['aasist'], map_location=device))
    aasist = aasist.to(device).eval()

    mae_ckpt = torch.load(weights['maedfer'], map_location=device)
    mae_cfg = mae_ckpt.get('cfg', {})
    mae_state = mae_ckpt.get('model_state_dict', mae_ckpt)
    maedfer = EmotionFlowDetectorMAE(
        mae_dfer_path=MAE_DFER_PATH,
        pretrained_ckpt=MAE_DFER_PRETRAIN,
        gru_hidden=mae_cfg.get('GRU_HIDDEN', 64),
        dropout=mae_cfg.get('DROPOUT', 0.3),
        unfreeze_last_blocks=mae_cfg.get('UNFREEZE_LAST_BLOCKS', 0),
    ).to(device)
    maedfer.load_state_dict(mae_state)
    maedfer.eval()

    cr_ckpt = torch.load(weights['crnn'], map_location=device)
    cr_cfg = cr_ckpt.get('cfg', {})
    cr_state = cr_ckpt.get('model_state_dict', cr_ckpt)
    cr_hidden = cr_state['gru.weight_hh_l0'].shape[1] if 'gru.weight_hh_l0' in cr_state else 128
    crnn = AudioEmotionFlowDetector(
        pretrained_path=CRNN_PRETRAINED,
        num_segments=cr_cfg.get('NUM_SEGMENTS', 16),
        gru_hidden=cr_hidden,
        dropout=cr_cfg.get('DROPOUT', 0.4),
        unfreeze_pretrained_gru=False,
    ).to(device)
    crnn.load_state_dict(cr_state)
    crnn.eval()

    for m in [x3d, aasist, maedfer, crnn]:
        for p in m.parameters(): p.requires_grad = False
    print("  ✅ 모든 모델 로드 완료")
    return x3d, aasist, maedfer, crnn

# ══════════════════════════════════════════════════════════════════════════════
# 5. 추론 및 메트릭 계산
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_one_sample(x3d_v, aasist_a, maedfer_f, crnn_s, x3d, aasist, maedfer, crnn, device):
    v_in = x3d_v.unsqueeze(0).to(device)
    p_v_art = torch.sigmoid(x3d(v_in)).item()
    del v_in

    a_in = aasist_a.unsqueeze(0).to(device)
    _, a_out = aasist(a_in)
    p_a_art = torch.softmax(a_out, dim=1)[0, 1].item()
    del a_in, a_out

    mf_in = maedfer_f.unsqueeze(0).to(device)
    e_logit, _ = maedfer(mf_in)
    p_v_emo = torch.sigmoid(e_logit).item()
    del mf_in, e_logit

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
    return dict(auc=auc, acc=accuracy_score(li, preds) * 100, precision=precision_score(li, preds, zero_division=0) * 100, recall=recall_score(li, preds, zero_division=0) * 100, f1=f1_score(li, preds, zero_division=0) * 100, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

# ══════════════════════════════════════════════════════════════════════════════
# 6. 단일 시나리오 평가
# ══════════════════════════════════════════════════════════════════════════════
def run_scenario(scenario, args, report_dir):
    weights = WEIGHTS_MAP[scenario]
    print("\n" + "=" * 70)
    print(f"🎯 Scenario: {weights['name']}")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    missing = [p for p in [weights['x3d'], weights['aasist'], weights['maedfer'], weights['crnn']] if not os.path.exists(p)]
    if missing:
        print(f"❌ 누락된 가중치 파일:")
        for p in missing: print(f"   - {p}")
        return None

    if weights['eval_set'] == 'fakeavceleb': val_df = build_fakeavceleb_eval_set(n_real=args.n_real, n_fake=args.n_fake)
    else: val_df = build_polyglotfake_eval_set(n_real=args.n_real_pgf, n_fake=args.n_fake_pgf)

    print(f"📂 평가셋: {len(val_df)}개 (Real: {(val_df['video_label']==0).sum()}, Fake: {(val_df['video_label']==1).sum()})")
    x3d, aasist, maedfer, crnn = load_models(scenario, device)

    dataset = MultiInputDataset(val_df, weights['eval_set'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"\n🚀 추론 시작 ({len(dataset)}개)\n")
    results, t0 = [], time.time()
    for batch_idx, batch in enumerate(loader):
        x3d_v, aasist_a, maedfer_f, crnn_s, label, sample_idx = batch
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            p_v_art, p_a_art, p_v_emo, p_a_emo = infer_one_sample(x3d_v.squeeze(0), aasist_a.squeeze(0), maedfer_f.squeeze(0), crnn_s.squeeze(0), x3d, aasist, maedfer, crnn, device)
        
        meta = val_df.iloc[sample_idx.item()]
        result = {'video_label': int(meta['video_label']), 'p_v_artifact': round(p_v_art * 100, 2), 'p_a_artifact': round(p_a_art * 100, 2), 'p_v_emotion': round(p_v_emo * 100, 2), 'p_a_emotion': round(p_a_emo * 100, 2)}
        if 'method' in meta.index: result['method'] = meta['method']
        if 'tts' in meta.index: result['tts'] = meta['tts']
        if 'sync' in meta.index: result['sync'] = meta['sync']
        results.append(result)

        if (batch_idx + 1) % 50 == 0:
            speed = (batch_idx + 1) / (time.time() - t0)
            print(f"  [{batch_idx+1:4d}/{len(loader)}] 속도: {speed:.1f}/s ETA: {(len(loader) - batch_idx - 1) / speed / 60:.1f}분")

    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(report_dir, f'{scenario}_predictions.csv'), index=False, encoding='utf-8-sig')

    labels, p_v_art, p_a_art, p_v_emo, p_a_emo = df_out['video_label'].values, df_out['p_v_artifact'].values/100, df_out['p_a_artifact'].values/100, df_out['p_v_emotion'].values/100, df_out['p_a_emotion'].values/100
    art_or, emo_or = prob_or(p_v_art, p_a_art), prob_or(p_v_emo, p_a_emo)
    final_or = prob_or(art_or, emo_or)

    print("\n" + "-" * 70 + f"\n🎯 결과: {weights['name']}\n" + "-" * 70)
    strategies = {'X3D 단독': p_v_art, 'AASIST 단독': p_a_art, 'MAE-DFER 단독': p_v_emo, 'CRNN v1 단독': p_a_emo, 'Score_artifact': art_or, 'Score_emotion': emo_or, '🌟 Score_final': final_or}
    summary = []
    print(f"\n  {'전략':<30} {'AUC':>7} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7}\n  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, probs in strategies.items():
        m = compute_metrics(probs, labels)
        summary.append({'strategy': name, **m})
        print(f"  {name:<30} {m['auc']:>6.2f}% {m['acc']:>6.2f}% {m['precision']:>6.2f}% {m['recall']:>6.2f}% {m['f1']:>6.2f}%")

    pd.DataFrame(summary).round(2).to_csv(os.path.join(report_dir, f'{scenario}_metrics.csv'), index=False, encoding='utf-8-sig')
    return {'scenario': scenario, 'name': weights['name'], 'summary': summary}

# ══════════════════════════════════════════════════════════════════════════════
# 7. 종합 비교표
# ══════════════════════════════════════════════════════════════════════════════
def make_comparison_table(all_results, report_dir):
    print("\n" + "=" * 70 + "\n📊 4가지 시나리오 종합 비교 (v3 — MAE-DFER + CRNN v1)\n" + "=" * 70)
    rows = [{'scenario': r['scenario'], 'name': r['name'], **{f"{k}_{metric}": r['summary'][i][metric] for i, k in enumerate(['X3D', 'AASIST', 'MAEDFER', 'CRNN', 'artifact', 'emotion', 'final']) for metric in ['auc', 'f1']}} for r in all_results if r]
    pd.DataFrame(rows).to_csv(os.path.join(report_dir, 'all_scenarios_comparison.csv'), index=False, encoding='utf-8-sig')

    for metric in ['AUC', 'F1']:
        metric_key = metric.lower()
        print(f"\n[{metric} 비교]\n  {'시나리오':<22} {'Artifact':>10} {'Emotion':>10} {'Final':>10}\n  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
        for r in rows: print(f"  {r['scenario']:<22} {r[f'artifact_{metric_key}']:>9.2f}% {r[f'emotion_{metric_key}']:>9.2f}% {r[f'final_{metric_key}']:>9.2f}%")
    print(f"\n💾 종합 비교표: {os.path.join(report_dir, 'all_scenarios_comparison.csv')}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="v3 평가 (MAE-DFER + CRNN v1)")
    parser.add_argument('--scenario', type=str, default='all', choices=['FF', 'FP', 'PF', 'PP', 'all'])
    parser.add_argument('--n_real', type=int, default=500)
    parser.add_argument('--n_fake', type=int, default=500)
    parser.add_argument('--n_real_pgf', type=int, default=300)
    parser.add_argument('--n_fake_pgf', type=int, default=1000)
    parser.add_argument('--report_dir', type=str, default='evaluation_v4_report')
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    scenarios = ['FF', 'FP', 'PF', 'PP'] if args.scenario == 'all' else [args.scenario]
    all_results = [run_scenario(s, args, args.report_dir) for s in scenarios]
    if len(scenarios) > 1: make_comparison_table(all_results, args.report_dir)
    print("\n" + "=" * 70 + "\n✅ v3 평가 완료\n" + "=" * 70)

if __name__ == "__main__": main()