"""
==============================================================================
[AV-Deepfake1M Zero-Shot 평가] v2 - 폴더명 자동 탐색
==============================================================================

[v2 변경점]
- 폴더명 자동 탐색 (AV-Deepfake1M_RootFiles 등 여러 후보)
- extracted_val/val 또는 val 직접 구조 자동 인식

[데이터셋 구조]
  AV-Deepfake1M_RootFiles/
  └── extracted_val/val/{speaker}/{youtube_id}/{seq}/
      ├── real.mp4
      ├── fake_video_real_audio.mp4
      ├── real_video_fake_audio.mp4
      └── fake_video_fake_audio.mp4

[사용법]
  cd ~/hsh/AIApplication
  python evaluate_avdf1m_zeroshot.py
==============================================================================
"""

import sys
import os
import json
import time
import argparse
import functools
import glob
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


# ══════════════════════════════════════════════════════════════════════════════
# 가중치 경로
# ══════════════════════════════════════════════════════════════════════════════
CKPT_PATHS = {
    'x3d':              'x3d_model_best_final.pth',
    'aasist':           'aasist_model_best_final.pth',
    'aasist_config':    'aasist/config/AASIST.conf',
    'hsemo':            'emotion_flow_lite_best.pth',
    'crnn':             'audio_flow_deepfake_best.pth',
    'crnn_pretrained':  'audio_emotion_crnn_best.pth',
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전처리
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
# 2. AV-Deepfake1M 폴더 자동 탐색
# ══════════════════════════════════════════════════════════════════════════════
LABEL_TYPES = {
    'real.mp4':                       ('real',              0),
    'fake_video_real_audio.mp4':      ('fake_video_only',   1),
    'real_video_fake_audio.mp4':      ('fake_audio_only',   1),
    'fake_video_fake_audio.mp4':      ('fake_both',         1),
}


def find_dataset_root(data_root: str):
    """데이터 루트 폴더 자동 탐색."""
    if os.path.isdir(data_root):
        return data_root
    
    candidates = [
        'AV-Deepfake1M_RootFiles',
        'AV_Deepfake1M_RootFiles',
        'AV-Deepfake1M',
        'AV_Deepfake1M',
        'AVDeepfake1M',
        'av_deepfake_1m',
        'av-deepfake1m',
    ]
    
    for c in candidates:
        if os.path.isdir(c):
            print(f"   ℹ️  '{data_root}' 못 찾음 → '{c}' 사용")
            return c
    
    # 현재 폴더에서 'deepfake' 포함된 모든 디렉토리 검색
    found_dirs = []
    try:
        for entry in os.listdir('.'):
            if os.path.isdir(entry) and ('deepfake' in entry.lower() or entry.lower().startswith('av')):
                found_dirs.append(entry)
    except Exception:
        pass
    
    if found_dirs:
        print(f"   ℹ️  자동 탐색 후보:")
        for d in found_dirs:
            print(f"        - {d}")
        for d in found_dirs:
            if 'deepfake' in d.lower():
                print(f"   ✅ '{d}' 자동 선택")
                return d
    
    raise FileNotFoundError(
        f"데이터 루트 폴더 못 찾음. 시도: {[data_root] + candidates}\n"
        f"--data_root 옵션으로 정확한 경로를 지정하세요."
    )


def find_split_path(data_root: str, split: str = 'val'):
    """split 데이터 경로 자동 탐색."""
    base_candidates = [
        os.path.join(data_root, f'extracted_{split}', split),
        os.path.join(data_root, f'extracted_{split}'),
        os.path.join(data_root, split),
    ]
    
    for c in base_candidates:
        if not os.path.isdir(c):
            continue
        try:
            entries = os.listdir(c)
        except PermissionError:
            continue
        if not entries: continue
        
        # id로 시작하는 폴더 있으면 확정
        if any(e.startswith('id') for e in entries):
            return c
        # 또는 첫 항목이 디렉토리 + 그 안에 또 디렉토리/mp4
        first = os.path.join(c, entries[0])
        if os.path.isdir(first):
            try:
                sub = os.listdir(first)
                if any(s.endswith('.mp4') or os.path.isdir(os.path.join(first, s)) for s in sub):
                    return c
            except Exception:
                continue
    
    raise FileNotFoundError(
        f"{split} 데이터 폴더 못 찾음. 시도한 경로:\n" +
        "\n".join(f"  - {c}" for c in base_candidates)
    )


def scan_avdf1m_dataset(data_root: str, split: str = 'val'):
    """AV-Deepfake1M 폴더 스캔."""
    data_root = find_dataset_root(data_root)
    base_path = find_split_path(data_root, split)
    
    print(f"📂 데이터 루트: {os.path.abspath(data_root)}")
    print(f"📂 스캔 경로 : {os.path.abspath(base_path)}")
    
    all_files = []
    for label_fname, (type_name, label) in LABEL_TYPES.items():
        pattern = os.path.join(base_path, '**', label_fname)
        matches = glob.glob(pattern, recursive=True)
        for path in matches:
            rel = os.path.relpath(path, base_path)
            parts = rel.split(os.sep)
            if len(parts) >= 4:
                speaker = parts[0]
                youtube_id = parts[1]
                seq_id = parts[2]
            else:
                speaker = youtube_id = seq_id = 'unknown'
            
            all_files.append({
                'video_path':  path,
                'video_label': float(label),
                'fake_type':   type_name,
                'speaker':     speaker,
                'youtube_id':  youtube_id,
                'seq_id':      seq_id,
                'label_file':  label_fname,
            })
    
    df = pd.DataFrame(all_files)
    return df


def build_balanced_eval_set(df: pd.DataFrame, n_per_type: int = 250, seed: int = 42):
    """각 변조 유형별 균등 샘플링."""
    print(f"\n📊 전체 파일 분포:")
    if len(df) == 0:
        print("  ⚠️ 발견된 파일 없음!")
        return df
    print(df['fake_type'].value_counts().to_string())
    
    sampled_list = []
    for type_name in ['real', 'fake_video_only', 'fake_audio_only', 'fake_both']:
        sub = df[df['fake_type'] == type_name]
        n = min(n_per_type, len(sub))
        if n == 0:
            print(f"  ⚠️  {type_name}: 0개 — skip")
            continue
        sampled = sub.sample(n=n, random_state=seed)
        sampled_list.append(sampled)
        print(f"  {type_name}: {n}개 샘플링")
    
    if not sampled_list:
        return pd.DataFrame()
    
    val_df = pd.concat(sampled_list).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    print(f"\n총 평가셋: {len(val_df)}개")
    print(f"  Real: {(val_df['video_label']==0).sum()}")
    print(f"  Fake: {(val_df['video_label']==1).sum()}")
    
    return val_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class AVDeepfake1MDataset(Dataset):
    def __init__(self, df, num_segments=16, segment_duration=3.0):
        self.df = df.reset_index(drop=True)
        self.num_segments = num_segments
        self.segment_duration = segment_duration

    def __len__(self):
        return len(self.df)

    def _try_load(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path']
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
def load_fav_models(device):
    print(f"\n🧠 FakeAVCeleb v1 가중치 로드...")

    x3d = x3d_m(pretrained=False)
    x3d.blocks[5].proj       = nn.Linear(2048, 1)
    x3d.blocks[5].activation = nn.Identity()
    x3d.load_state_dict(torch.load(CKPT_PATHS['x3d'], map_location=device))
    x3d = x3d.to(device).eval()
    print(f"  ✅ X3D")

    with open(CKPT_PATHS['aasist_config'], 'r') as f:
        config = json.load(f)
    aasist = AASISTModel(config['model_config'])
    aasist.load_state_dict(torch.load(CKPT_PATHS['aasist'], map_location=device))
    aasist = aasist.to(device).eval()
    print(f"  ✅ AASIST")

    hs_ckpt = torch.load(CKPT_PATHS['hsemo'], map_location=device)
    hs_cfg  = hs_ckpt.get('cfg', {})
    hs_state = hs_ckpt.get('model_state_dict', hs_ckpt)
    hs_hidden = (hs_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in hs_state else 64)
    
    hsemo = EmotionFlowDetectorLite(
        model_name=hs_cfg.get('MODEL_NAME', 'enet_b0_8_best_afew'),
        num_frames=hs_cfg.get('NUM_FRAMES', 16),
        gru_hidden=hs_hidden,
        dropout=hs_cfg.get('DROPOUT', 0.3),
        device='cpu',
    ).to(device)
    hsemo.load_state_dict(hs_state)
    hsemo.eval()
    print(f"  ✅ HSEmotion (hidden={hs_hidden})")

    cr_ckpt = torch.load(CKPT_PATHS['crnn'], map_location=device)
    cr_cfg  = cr_ckpt.get('cfg', {})
    cr_state = cr_ckpt.get('model_state_dict', cr_ckpt)
    cr_hidden = (cr_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in cr_state else 128)
    
    crnn = AudioEmotionFlowDetector(
        pretrained_path=CKPT_PATHS['crnn_pretrained'],
        num_segments=cr_cfg.get('NUM_SEGMENTS', 16),
        gru_hidden=cr_hidden,
        dropout=cr_cfg.get('DROPOUT', 0.4),
    ).to(device)
    crnn.load_state_dict(cr_state)
    crnn.eval()
    print(f"  ✅ CRNN (hidden={cr_hidden})")

    for m in [x3d, aasist, hsemo, crnn]:
        for p in m.parameters():
            p.requires_grad = False

    return x3d, aasist, hsemo, crnn


# ══════════════════════════════════════════════════════════════════════════════
# 5. 추론
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
# 6. 변조 유형별 분석
# ══════════════════════════════════════════════════════════════════════════════
def per_type_analysis(df_out, report_dir):
    labels = df_out['video_label'].values.astype(float)
    p_v_art = df_out['p_v_artifact'].values / 100
    p_a_art = df_out['p_a_artifact'].values / 100
    p_v_emo = df_out['p_v_emotion'].values / 100
    p_a_emo = df_out['p_a_emotion'].values / 100
    
    art_or = prob_or(p_v_art, p_a_art)
    emo_or = prob_or(p_v_emo, p_a_emo)
    final_or = prob_or(art_or, emo_or)
    
    print("\n" + "=" * 70)
    print("📊 변조 유형별 분석 (Real + 각 변조 유형)")
    print("=" * 70)
    
    real_idx = df_out[df_out['fake_type'] == 'real'].index.tolist()
    
    type_results = []
    for fake_type in ['fake_video_only', 'fake_audio_only', 'fake_both']:
        fake_idx = df_out[df_out['fake_type'] == fake_type].index.tolist()
        if not fake_idx: continue
        
        sub_idx = np.array(real_idx + fake_idx)
        sub_labels = labels[sub_idx]
        
        print(f"\n[{fake_type}]  N_fake={len(fake_idx)} + N_real={len(real_idx)}")
        print(f"  {'전략':<25} {'AUC':>7} {'F1':>7} {'Recall':>8}")
        print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*8}")
        
        strategies = {
            'X3D (영상 artifact)':    p_v_art[sub_idx],
            'AASIST (음성 artifact)': p_a_art[sub_idx],
            'HSEmotion (영상 emo)':   p_v_emo[sub_idx],
            'CRNN (음성 emo)':        p_a_emo[sub_idx],
            'Score_artifact':         art_or[sub_idx],
            'Score_emotion':          emo_or[sub_idx],
            'Score_final':            final_or[sub_idx],
        }
        
        row = {'fake_type': fake_type, 'n_fake': len(fake_idx)}
        for name, probs in strategies.items():
            m = compute_metrics(probs, sub_labels)
            row[f'{name}_AUC'] = round(m['auc'], 2)
            row[f'{name}_F1']  = round(m['f1'], 2)
            row[f'{name}_R']   = round(m['recall'], 2)
            print(f"  {name:<25} {m['auc']:>6.2f}% {m['f1']:>6.2f}% {m['recall']:>7.2f}%")
        
        type_results.append(row)
    
    pd.DataFrame(type_results).to_csv(
        os.path.join(report_dir, 'per_type_analysis.csv'),
        index=False, encoding='utf-8-sig'
    )
    
    return type_results


# ══════════════════════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AV-Deepfake1M Zero-Shot 평가")
    parser.add_argument('--data_root', type=str, default='AV-Deepfake1M_RootFiles',
                        help="데이터 루트 폴더 (자동 탐색 가능)")
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'train'])
    parser.add_argument('--n_per_type', type=int, default=250,
                        help="변조 유형별 샘플 (총 4*N)")
    parser.add_argument('--report_dir', type=str, default='avdf1m_zeroshot_report')
    args = parser.parse_args()

    print("=" * 70)
    print("🎯 AV-Deepfake1M Zero-Shot 평가 (FakeAVCeleb 학습 모델)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")
    os.makedirs(args.report_dir, exist_ok=True)
    
    print(f"\n📦 가중치 확인:")
    missing = []
    for name, path in CKPT_PATHS.items():
        exists = os.path.exists(path)
        marker = "✅" if exists else "❌"
        print(f"  {marker} {name:<18} {path}")
        if not exists:
            missing.append(name)
    if missing:
        print(f"\n❌ 누락 파일: {missing}")
        sys.exit(1)
    
    print(f"\n📂 AV-Deepfake1M 스캔 (split={args.split})...")
    all_df = scan_avdf1m_dataset(args.data_root, args.split)
    print(f"   전체 발견: {len(all_df)}개")
    
    if len(all_df) == 0:
        print("❌ 발견된 mp4 파일 없음. 데이터셋 구조 확인 필요.")
        sys.exit(1)
    
    val_df = build_balanced_eval_set(all_df, args.n_per_type)
    
    x3d, aasist, hsemo, crnn = load_fav_models(device)
    
    dataset = AVDeepfake1MDataset(val_df)
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
        results.append({
            'video_label':  int(meta['video_label']),
            'fake_type':    meta['fake_type'],
            'speaker':      meta['speaker'],
            'youtube_id':   meta['youtube_id'],
            'seq_id':       meta['seq_id'],
            'p_v_artifact': round(p_v_art * 100, 2),
            'p_a_artifact': round(p_a_art * 100, 2),
            'p_v_emotion':  round(p_v_emo * 100, 2),
            'p_a_emotion':  round(p_a_emo * 100, 2),
        })
        
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (batch_idx + 1) / elapsed
            eta = (len(loader) - batch_idx - 1) / speed
            print(f"  [{batch_idx+1:4d}/{len(loader)}] "
                  f"속도: {speed:.1f}/s ETA: {eta/60:.1f}분")
    
    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(args.report_dir, 'predictions.csv'),
                  index=False, encoding='utf-8-sig')
    
    # 전체 분석
    labels = df_out['video_label'].values.astype(float)
    p_v_art = df_out['p_v_artifact'].values / 100
    p_a_art = df_out['p_a_artifact'].values / 100
    p_v_emo = df_out['p_v_emotion'].values / 100
    p_a_emo = df_out['p_a_emotion'].values / 100
    
    art_or = prob_or(p_v_art, p_a_art)
    emo_or = prob_or(p_v_emo, p_a_emo)
    final_or = prob_or(art_or, emo_or)
    
    print("\n" + "=" * 70)
    print("🎯 전체 결과: AV-Deepfake1M Zero-Shot")
    print("=" * 70)
    
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
        os.path.join(args.report_dir, 'metrics_overall.csv'),
        index=False, encoding='utf-8-sig'
    )
    
    per_type_analysis(df_out, args.report_dir)
    
    # 시나리오 비교
    print("\n" + "=" * 70)
    print("📊 다른 시나리오와 비교 (참고)")
    print("=" * 70)
    
    overall_art_auc = summary[4]['auc']
    overall_emo_auc = summary[5]['auc']
    overall_final_auc = summary[6]['auc']
    overall_art_f1 = summary[4]['f1']
    overall_emo_f1 = summary[5]['f1']
    overall_final_f1 = summary[6]['f1']
    
    print(f"\n[AUC 비교]")
    print(f"  {'시나리오':<32} {'Artifact':>10} {'Emotion':>10} {'Final':>10}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'F→F (FAV In-Domain)':<32} {'99.99%':>10} {'94.52%':>10} {'99.86%':>10}")
    print(f"  {'F→P (Cross FAV→PGF)':<32} {'56.78%':>10} {'64.02%':>10} {'64.73%':>10}")
    print(f"  {'F→AVDF1M (NEW Cross)':<32} {overall_art_auc:>9.2f}% {overall_emo_auc:>9.2f}% {overall_final_auc:>9.2f}%")
    
    print(f"\n[F1 비교]")
    print(f"  {'시나리오':<32} {'Artifact':>10} {'Emotion':>10} {'Final':>10}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'F→F (FAV In-Domain)':<32} {'99.70%':>10} {'85.51%':>10} {'87.85%':>10}")
    print(f"  {'F→P (Cross FAV→PGF)':<32} {'7.97%':>10} {'86.07%':>10} {'91.10%':>10}")
    print(f"  {'F→AVDF1M (NEW Cross)':<32} {overall_art_f1:>9.2f}% {overall_emo_f1:>9.2f}% {overall_final_f1:>9.2f}%")
    
    avdf1m_summary = pd.DataFrame([{
        'scenario': 'F→AVDF1M',
        'artifact_auc': overall_art_auc, 'artifact_f1': overall_art_f1,
        'emotion_auc':  overall_emo_auc, 'emotion_f1': overall_emo_f1,
        'final_auc':    overall_final_auc, 'final_f1': overall_final_f1,
    }])
    avdf1m_summary.to_csv(
        os.path.join(args.report_dir, 'scenario_summary.csv'),
        index=False, encoding='utf-8-sig'
    )
    
    print(f"\n💾 결과 저장: {os.path.abspath(args.report_dir)}/")
    print("=" * 70)
    print("✅ AV-Deepfake1M Zero-Shot 평가 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()