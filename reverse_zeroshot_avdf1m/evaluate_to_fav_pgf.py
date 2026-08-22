"""
==============================================================================
[AV-Deepfake1M 학습 → FAV/PGF Zero-Shot 평가]
A→F (AVDF1M → FakeAVCeleb) 및 A→P (AVDF1M → PolyGlotFake)
한 스크립트에서 둘 다 지원.

[실행 위치] ~/hsh/AIApplication/reverse_zeroshot_avdf1m/

[필요 파일]
- 같은 폴더 (학습 산출물):
    * x3d_model_avdf1m_best.pth
    * aasist_model_avdf1m_best.pth
    * emotion_flow_lite_avdf1m_best.pth
    * audio_flow_deepfake_avdf1m_best.pth
- 상위 폴더 (~/hsh/AIApplication/):
    * emotion_deepfake_detector_lite.py, audio_emotion_deepfake_detector.py
    * audio_emotion_crnn_best.pth (RAVDESS 사전학습)
    * aasist/config/AASIST.conf
    * FakeAVCeleb_v1.2/ 또는 PolyGlotFake/

[사용법]
  python evaluate_to_fav_pgf.py --target fav      # A→F만
  python evaluate_to_fav_pgf.py --target pgf      # A→P만
  python evaluate_to_fav_pgf.py --target both     # 양쪽

[출력]
- avdf1m_to_fav_report/predictions.csv, metrics_by_strategy.csv
- avdf1m_to_pgf_report/predictions.csv, metrics_by_strategy.csv
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

# 경로 설정
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# torchvision 하위 호환
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

# 상위 폴더의 모델 클래스
try:
    from emotion_deepfake_detector_lite import EmotionFlowDetectorLite
except ImportError:
    from train_HSEmotion import EmotionFlowDetectorLite

try:
    from audio_emotion_deepfake_detector import (
        AudioEmotionFlowDetector, extract_audio_segments
    )
except ImportError:
    from train_CRNN import (
        AudioEmotionFlowDetector, extract_audio_segments
    )

from aasist.models.AASIST import Model as AASISTModel


# ══════════════════════════════════════════════════════════════════════════════
# 가중치 경로 (AVDF1M 학습본 + 상위 폴더 의존성)
# ══════════════════════════════════════════════════════════════════════════════
AVDF1M_CKPTS = {
    'x3d':              'x3d_model_avdf1m_best.pth',
    'aasist':           'aasist_model_avdf1m_best.pth',
    'aasist_config':    os.path.join(PARENT_DIR, 'aasist/config/AASIST.conf'),
    'hsemo':            'emotion_flow_lite_avdf1m_best.pth',
    'crnn':             'audio_flow_deepfake_avdf1m_best.pth',
    'crnn_pretrained':  os.path.join(PARENT_DIR, 'audio_emotion_crnn_best.pth'),
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전처리 (FAV/AVDF1M 평가 코드와 동일)
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
# 2. FakeAVCeleb 평가셋 빌더
# ══════════════════════════════════════════════════════════════════════════════
def build_fav_eval_set(n_real=500, n_fake=500, seed=42):
    """FakeAVCeleb 평가셋 (Real n_real + Fake n_fake)."""
    fav_candidates = [
        'FakeAVCeleb_v1.2', 'FakeAVCeleb',
        os.path.join(PARENT_DIR, 'FakeAVCeleb_v1.2'),
        os.path.join(PARENT_DIR, 'FakeAVCeleb'),
    ]
    fav_root = None
    for c in fav_candidates:
        if os.path.isdir(c):
            fav_root = os.path.abspath(c)
            break
    if fav_root is None:
        raise FileNotFoundError(
            "FakeAVCeleb 폴더 못 찾음. 시도: " + ", ".join(fav_candidates)
        )
    print(f"📂 FakeAVCeleb 루트: {fav_root}")

    all_mp4 = glob.glob(os.path.join(fav_root, '**', '*.mp4'), recursive=True)

    real_files, fake_files = [], []
    for p in all_mp4:
        lower = p.lower()
        # FakeAVCeleb 폴더 구조: RealVideo-RealAudio = Real, 나머지 fake 패턴
        if 'realvideo-realaudio' in lower or 'real_video_real_audio' in lower:
            real_files.append(p)
        elif any(k in lower for k in [
            'fakevideo', 'fakeaudio', 'faceswap', 'fsgan',
            'wav2lip', 'rtvc', 'sv2tts'
        ]):
            fake_files.append(p)

    print(f"   Real 후보: {len(real_files)}개, Fake 후보: {len(fake_files)}개")

    if len(real_files) == 0 or len(fake_files) == 0:
        raise RuntimeError(
            "FakeAVCeleb에서 Real/Fake 파일을 찾을 수 없음. "
            "폴더 구조 확인 필요."
        )

    rng = np.random.default_rng(seed)
    n_r = min(n_real, len(real_files))
    n_f = min(n_fake, len(fake_files))
    real_idx = rng.choice(len(real_files), n_r, replace=False)
    fake_idx = rng.choice(len(fake_files), n_f, replace=False)

    rows = []
    for i in real_idx:
        rows.append({'video_path': real_files[i], 'video_label': 0.0})
    for i in fake_idx:
        rows.append({'video_path': fake_files[i], 'video_label': 1.0})

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"   평가셋: {len(df)}개 "
          f"(Real {(df['video_label']==0).sum()}, "
          f"Fake {(df['video_label']==1).sum()})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. PolyGlotFake 평가셋 빌더
# ══════════════════════════════════════════════════════════════════════════════
def build_pgf_eval_set(n_real=500, n_fake=500, seed=42):
    """PolyGlotFake 평가셋."""
    pgf_candidates = [
        'PolyGlotFake',
        os.path.join(PARENT_DIR, 'PolyGlotFake'),
    ]
    pgf_root = None
    for c in pgf_candidates:
        if os.path.isdir(c):
            pgf_root = os.path.abspath(c)
            break
    if pgf_root is None:
        raise FileNotFoundError(
            "PolyGlotFake 폴더 못 찾음. 시도: " + ", ".join(pgf_candidates)
        )
    print(f"📂 PolyGlotFake 루트: {pgf_root}")

    # 1차 시도: json_file/*.json
    json_dir = os.path.join(pgf_root, 'json_file')
    all_rows = []
    if os.path.isdir(json_dir):
        for jf in glob.glob(os.path.join(json_dir, '*.json')):
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                entries = data if isinstance(data, list) else data.get('data', [])
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    vp = (e.get('video_path') or e.get('path') or
                          e.get('video') or e.get('file'))
                    if not vp:
                        continue
                    if not os.path.isabs(vp):
                        vp = os.path.join(pgf_root, vp)
                    lbl = e.get('label')
                    if lbl is None:
                        lbl = 1.0 if e.get('fake', False) else 0.0
                    else:
                        if isinstance(lbl, str):
                            lbl = 1.0 if 'fake' in lbl.lower() else 0.0
                        else:
                            lbl = float(lbl)
                    all_rows.append({'video_path': vp, 'video_label': lbl})
            except Exception:
                continue

    # 2차 시도: 폴더 구조로 추정
    if not all_rows:
        print("   ℹ️ JSON에서 로드 실패, 폴더 구조로 탐색")
        for p in glob.glob(os.path.join(pgf_root, '**', '*.mp4'), recursive=True):
            lower = p.lower()
            if '/real' in lower and '/fake' not in lower:
                all_rows.append({'video_path': p, 'video_label': 0.0})
            elif '/fake' in lower or 'fake' in os.path.basename(p).lower():
                all_rows.append({'video_path': p, 'video_label': 1.0})

    df = pd.DataFrame(all_rows)
    print(f"   전체 발견: {len(df)}개")
    if len(df) == 0:
        raise RuntimeError("PolyGlotFake에서 데이터를 찾을 수 없음")

    real_df = df[df['video_label']==0.0]
    fake_df = df[df['video_label']==1.0]
    print(f"   Real: {len(real_df)}, Fake: {len(fake_df)}")

    n_r = min(n_real, len(real_df))
    n_f = min(n_fake, len(fake_df))
    sampled = pd.concat([
        real_df.sample(n=n_r, random_state=seed),
        fake_df.sample(n=n_f, random_state=seed+1),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"   평가셋: {len(sampled)}개 "
          f"(Real {n_r}, Fake {n_f})")
    return sampled


# ══════════════════════════════════════════════════════════════════════════════
# 4. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class EvalDataset(Dataset):
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
# 5. AVDF1M 학습 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_avdf1m_models(device):
    print(f"\n🧠 AVDF1M 학습 가중치 로드...")

    # X3D
    x3d = x3d_m(pretrained=False)
    x3d.blocks[5].proj       = nn.Linear(2048, 1)
    x3d.blocks[5].activation = nn.Identity()
    x3d.load_state_dict(torch.load(AVDF1M_CKPTS['x3d'], map_location=device))
    x3d = x3d.to(device).eval()
    print(f"  ✅ X3D")

    # AASIST
    with open(AVDF1M_CKPTS['aasist_config'], 'r') as f:
        config = json.load(f)
    aasist = AASISTModel(config['model_config'])
    aasist.load_state_dict(torch.load(AVDF1M_CKPTS['aasist'], map_location=device))
    aasist = aasist.to(device).eval()
    print(f"  ✅ AASIST")

    # HSEmotion
    hs_ckpt = torch.load(AVDF1M_CKPTS['hsemo'], map_location=device)
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

    # CRNN
    cr_ckpt = torch.load(AVDF1M_CKPTS['crnn'], map_location=device)
    cr_cfg  = cr_ckpt.get('cfg', {})
    cr_state = cr_ckpt.get('model_state_dict', cr_ckpt)
    cr_hidden = (cr_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in cr_state else 128)
    crnn = AudioEmotionFlowDetector(
        pretrained_path=AVDF1M_CKPTS['crnn_pretrained'],
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
# 6. 추론 & 평가 헬퍼
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
# 7. 한 타깃에 대한 평가
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_on_target(target_name, val_df, models, device, report_dir):
    x3d, aasist, hsemo, crnn = models
    os.makedirs(report_dir, exist_ok=True)

    dataset = EvalDataset(val_df)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"\n🚀 추론 시작 ({len(dataset)}개) — {target_name.upper()}\n")
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
            'video_path':   meta['video_path'],
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
    df_out.to_csv(os.path.join(report_dir, 'predictions.csv'),
                  index=False, encoding='utf-8-sig')

    # 분석
    labels = df_out['video_label'].values.astype(float)
    p_v_art = df_out['p_v_artifact'].values / 100
    p_a_art = df_out['p_a_artifact'].values / 100
    p_v_emo = df_out['p_v_emotion'].values / 100
    p_a_emo = df_out['p_a_emotion'].values / 100

    art_or = prob_or(p_v_art, p_a_art)
    emo_or = prob_or(p_v_emo, p_a_emo)
    final_or = prob_or(art_or, emo_or)

    print("\n" + "=" * 70)
    print(f"🎯 결과: AVDF1M → {target_name.upper()}")
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
    print(f"\n  {'전략':<25} {'AUC':>7} {'Acc':>7} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, probs in strategies.items():
        m = compute_metrics(probs, labels)
        summary.append({'strategy': name, **m})
        print(f"  {name:<25} {m['auc']:>6.2f}% {m['acc']:>6.2f}% "
              f"{m['precision']:>6.2f}% {m['recall']:>6.2f}% {m['f1']:>6.2f}%")

    pd.DataFrame(summary).round(2).to_csv(
        os.path.join(report_dir, 'metrics_by_strategy.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"\n💾 결과 저장: {os.path.abspath(report_dir)}/")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='both',
                        choices=['fav', 'pgf', 'both'])
    parser.add_argument('--n_per_class', type=int, default=500,
                        help="Real 및 Fake 각각의 평가 샘플 수 (기본 500)")
    args = parser.parse_args()

    print("=" * 70)
    print("🎯 역방향 Zero-Shot 평가: AVDF1M 학습 → FAV/PGF")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")
    print(f"📁 Parent dir : {PARENT_DIR}")

    # 가중치 확인
    print(f"\n📦 가중치 확인:")
    missing = []
    for name, path in AVDF1M_CKPTS.items():
        exists = os.path.exists(path)
        marker = "✅" if exists else "❌"
        print(f"  {marker} {name:<18} {path}")
        if not exists:
            missing.append(name)
    if missing:
        print(f"\n❌ 누락 파일: {missing}")
        sys.exit(1)

    # 모델 1회 로드
    models = load_avdf1m_models(device)

    # FAV
    if args.target in ['fav', 'both']:
        print("\n" + "█" * 70)
        print("█  A→F : AVDF1M 학습 → FakeAVCeleb")
        print("█" * 70)
        try:
            fav_df = build_fav_eval_set(
                n_real=args.n_per_class, n_fake=args.n_per_class
            )
            evaluate_on_target('fav', fav_df, models, device,
                               'avdf1m_to_fav_report')
        except Exception as e:
            print(f"❌ FAV 평가 실패: {e}")

    # PGF
    if args.target in ['pgf', 'both']:
        print("\n" + "█" * 70)
        print("█  A→P : AVDF1M 학습 → PolyGlotFake")
        print("█" * 70)
        try:
            pgf_df = build_pgf_eval_set(
                n_real=args.n_per_class, n_fake=args.n_per_class
            )
            evaluate_on_target('pgf', pgf_df, models, device,
                               'avdf1m_to_pgf_report')
        except Exception as e:
            print(f"❌ PGF 평가 실패: {e}")

    print("\n" + "=" * 70)
    print("✅ 역방향 Zero-Shot 평가 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()