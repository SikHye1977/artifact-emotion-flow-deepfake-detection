"""
==============================================================================
[Phase 1] P→AVDF1M 평가
PolyGlotFake로 학습된 가중치 4개를 사용해 AV-Deepfake1M zero-shot 평가
==============================================================================

[목적]
9-시나리오 매트릭스의 P→AVDF1M 칸을 채우기 위한 스크립트.
모델은 PGF 학습 가중치 그대로 사용, 평가셋만 AVDF1M.

[가중치 매핑]
  X3D       : x3d_model_pgf_best.pth                    (현재 폴더)
  AASIST    : aasist_model_pgf_best.pth                 (현재 폴더)
  HSEmotion : emotion_flow_lite_pgf_v2_best.pth         (현재 폴더, v2)
  CRNN      : audio_flow_deepfake_pgf_v2_best.pth       (현재 폴더, v2)

[사용법]
  cd ~/hsh/AIApplication/reverse_zero_shot     # PGF 가중치가 있는 폴더
  python evaluate_pgf_to_avdf1m.py

[출력]
  pgf_to_avdf1m_report/
    ├── predictions.csv         (1000 샘플, 4 모델 확률 + fake_type)
    └── metrics.csv             (7가지 전략의 AUC/F1/Recall)
==============================================================================
"""
import sys
import os
import json
import time
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

# torchvision 호환성
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
    from train_HSEmotion import EmotionFlowDetectorLite

try:
    from audio_emotion_deepfake_detector import (
        AudioEmotionFlowDetector, extract_audio_segments
    )
except ImportError:
    from train_CRNN import AudioEmotionFlowDetector, extract_audio_segments

from aasist.models.AASIST import Model as AASISTModel


# ═══════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════
WEIGHTS = {
    'x3d':    'x3d_model_pgf_best.pth',
    'aasist': 'aasist_model_pgf_best.pth',
    'hsemo':  'emotion_flow_lite_pgf_v2_best.pth',
    'crnn':   'audio_flow_deepfake_pgf_v2_best.pth',
}

AASIST_CONFIG   = os.path.join(PARENT_DIR, "aasist/config/AASIST.conf")
CRNN_PRETRAINED = os.path.join(PARENT_DIR, "audio_emotion_crnn_best.pth")
AVDF1M_ROOT     = os.path.expanduser("~/hsh/AIApplication/AV-Deepfake1M_RootFiles")
AVDF1M_META     = os.path.join(AVDF1M_ROOT, "val_metadata.json")
AVDF1M_VIDEO    = os.path.join(AVDF1M_ROOT, "extracted_val/val")  # 압축 해제된 val 영상

N_PER_TYPE = 250  # real, fake_video_only, fake_audio_only, fake_both 각각
SEED = 42
REPORT_DIR = 'pgf_to_avdf1m_report'


# ═══════════════════════════════════════════════════════════════════
# 1. 전처리 (X3D / AASIST / HSEmotion 입력 준비)
# ═══════════════════════════════════════════════════════════════════
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
    Resize((224, 224)),
])

hsemo_frame_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_video_for_x3d(path, max_frames=128):
    try:
        container = av.open(path)
        frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(frames) < 16: return None
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames)-1, max_frames, dtype=int)
            frames = [frames[i] for i in idx]
        video = np.stack(frames)
        return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
    except Exception:
        return None


def load_frames_for_hsemo(path, num_frames=16):
    try:
        container = av.open(path)
        stream = container.streams.video[0]
        total = stream.frames
        if total < num_frames:
            container.close(); return None
        targets = set(np.linspace(0, total-1, num_frames, dtype=int).tolist())
        sampled = {}
        for i, frame in enumerate(container.decode(video=0)):
            if i in targets:
                sampled[i] = frame.to_rgb().to_ndarray()
            if len(sampled) >= num_frames: break
        container.close()
        if len(sampled) < num_frames: return None
        keys = sorted(sampled.keys())
        return torch.stack([hsemo_frame_transform(sampled[k]) for k in keys])
    except Exception:
        return None


def load_audio_for_aasist(video_path, target_sr=16000, max_length=64000):
    try:
        container = av.open(video_path)
        if not container.streams.audio:
            container.close(); return None
        sr = container.streams.audio[0].rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.dtype == np.int16:   arr = arr.astype(np.float32)/32768.0
            elif arr.dtype == np.int32: arr = arr.astype(np.float32)/2147483648.0
            else:                       arr = arr.astype(np.float32)
            if arr.ndim > 1 and arr.shape[0] > arr.shape[1]: arr = arr.T
            elif arr.ndim == 1: arr = arr[np.newaxis, :]
            frames.append(arr)
        container.close()
        if not frames: return None
        wav = torch.from_numpy(np.concatenate(frames, axis=-1))
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        if wav.shape[1] > max_length:
            wav = wav[:, :max_length]
        else:
            wav = torch.nn.functional.pad(wav, (0, max_length - wav.shape[1]))
        return wav.squeeze()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# 2. AVDF1M 평가셋 구성
# ═══════════════════════════════════════════════════════════════════
def build_avdf1m_eval_set(metadata_path, n_per_type=250, seed=42):
    """AVDF1M val에서 modify_type별로 균등 샘플링."""
    print(f"📂 메타데이터 로드: {metadata_path}")
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    print(f"   총 엔트리: {len(meta)}")
    
    # modify_type별 분류
    by_type = {'real': [], 'visual_modified': [], 'audio_modified': [], 'both_modified': []}
    for entry in meta:
        mt = entry.get('modify_type', '')
        if mt in by_type:
            by_type[mt].append(entry)
    
    print(f"\n각 modify_type 분포:")
    for k, v in by_type.items():
        print(f"   {k}: {len(v)}")
    
    # 균등 샘플링
    rng = np.random.RandomState(seed)
    selected = []
    type_map = {
        'real':            'real',
        'visual_modified': 'fake_video_only',
        'audio_modified':  'fake_audio_only',
        'both_modified':   'fake_both',
    }
    for mt, fake_type in type_map.items():
        pool = by_type[mt]
        if len(pool) < n_per_type:
            print(f"   ⚠️  {mt}: {len(pool)}개밖에 없어서 전부 사용")
            picks = pool
        else:
            idx = rng.choice(len(pool), size=n_per_type, replace=False)
            picks = [pool[i] for i in idx]
        for entry in picks:
            video_label = 0.0 if mt == 'real' else 1.0
            selected.append({
                'file_rel':    entry['file'],
                'video_label': video_label,
                'fake_type':   fake_type,
                'modify_type': mt,
                'audio_model': entry.get('audio_model'),
            })
    
    df = pd.DataFrame(selected)
    # 셔플
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"\n✅ 평가셋 구성: {len(df)}개")
    print(f"   Real: {(df['video_label']==0).sum()}")
    print(f"   Fake: {(df['video_label']==1).sum()}")
    return df


# ═══════════════════════════════════════════════════════════════════
# 3. Dataset
# ═══════════════════════════════════════════════════════════════════
class AVDF1MDataset(Dataset):
    def __init__(self, df, video_root, num_segments=16, segment_duration=3.0):
        self.df = df.reset_index(drop=True)
        self.video_root = video_root
        self.num_segments = num_segments
        self.segment_duration = segment_duration

    def __len__(self):
        return len(self.df)

    def _get_video_path(self, idx):
        row = self.df.iloc[idx]
        return os.path.join(self.video_root, row['file_rel'])

    def _try_load(self, idx):
        video_path = self._get_video_path(idx)
        if not os.path.exists(video_path):
            # AVDF1M 구조에 따라 경로 보정 시도
            alt = os.path.join(os.path.dirname(self.video_root), self.df.iloc[idx]['file_rel'])
            if os.path.exists(alt):
                video_path = alt
            else:
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


# ═══════════════════════════════════════════════════════════════════
# 4. 모델 로드 (PGF 가중치)
# ═══════════════════════════════════════════════════════════════════
def load_models(device):
    print("\n🧠 모델 로드 중 (PGF 가중치)...")
    for k, v in WEIGHTS.items():
        exists = "✅" if os.path.exists(v) else "❌"
        print(f"   {exists} {k:<10}: {v}")
    
    missing = [v for v in WEIGHTS.values() if not os.path.exists(v)]
    if missing:
        print(f"\n❌ 누락된 가중치: {missing}")
        sys.exit(1)
    
    # X3D
    x3d = x3d_m(pretrained=False)
    x3d.blocks[5].proj       = nn.Linear(2048, 1)
    x3d.blocks[5].activation = nn.Identity()
    x3d.load_state_dict(torch.load(WEIGHTS['x3d'], map_location=device))
    x3d = x3d.to(device).eval()
    
    # AASIST
    with open(AASIST_CONFIG, 'r') as f:
        config = json.load(f)
    aasist = AASISTModel(config['model_config'])
    aasist.load_state_dict(torch.load(WEIGHTS['aasist'], map_location=device))
    aasist = aasist.to(device).eval()
    
    # HSEmotion (PGF v2 — unfreeze 적용된 가중치)
    hs_ckpt = torch.load(WEIGHTS['hsemo'], map_location=device)
    hs_cfg  = hs_ckpt.get('cfg', {})
    hs_state = hs_ckpt.get('model_state_dict', hs_ckpt)
    hs_hidden = (hs_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in hs_state else 64)
    unfreeze_blocks = hs_cfg.get('UNFREEZE_LAST_BLOCKS', 2)
    
    hsemo = EmotionFlowDetectorLite(
        model_name=hs_cfg.get('MODEL_NAME', 'enet_b0_8_best_afew'),
        num_frames=hs_cfg.get('NUM_FRAMES', 16),
        gru_hidden=hs_hidden,
        dropout=hs_cfg.get('DROPOUT', 0.3),
        device='cpu',
        unfreeze_last_blocks=unfreeze_blocks,
    ).to(device)
    hsemo.load_state_dict(hs_state)
    hsemo.eval()
    
    # CRNN (PGF v2)
    cr_ckpt = torch.load(WEIGHTS['crnn'], map_location=device)
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
        unfreeze_pretrained_gru=unfreeze_gru,
    ).to(device)
    crnn.load_state_dict(cr_state)
    crnn.eval()
    
    for m in [x3d, aasist, hsemo, crnn]:
        for p in m.parameters():
            p.requires_grad = False
    
    print("  ✅ 모든 모델 로드 완료")
    return x3d, aasist, hsemo, crnn


# ═══════════════════════════════════════════════════════════════════
# 5. 추론
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_one(x3d_v, aasist_a, hsemo_f, crnn_s,
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
    preds = (probs > t).astype(int); li = labels.astype(int)
    cm = confusion_matrix(li, preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    try: auc = roc_auc_score(labels, probs)*100
    except: auc = 0.0
    return dict(
        auc=round(auc, 2),
        acc=round(accuracy_score(li,preds)*100, 2),
        precision=round(precision_score(li,preds,zero_division=0)*100, 2),
        recall=round(recall_score(li,preds,zero_division=0)*100, 2),
        f1=round(f1_score(li,preds,zero_division=0)*100, 2),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


# ═══════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("🎯 P→AVDF1M 평가 (PGF 학습 → AV-Deepfake1M 추론)")
    print("=" * 70)
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Device: {device}")
    
    # 평가셋 구성
    val_df = build_avdf1m_eval_set(AVDF1M_META, n_per_type=N_PER_TYPE, seed=SEED)
    
    # 모델 로드
    x3d, aasist, hsemo, crnn = load_models(device)
    
    # 영상 루트 자동 탐색 (확장됨)
    video_root = AVDF1M_VIDEO
    if not os.path.isdir(video_root):
        candidates = [
            os.path.join(AVDF1M_ROOT, "extracted_val/val"),
            os.path.join(AVDF1M_ROOT, "extracted_val"),
            os.path.join(AVDF1M_ROOT, "val_extracted"),
            AVDF1M_ROOT,
        ]
        for c in candidates:
            # 후보 안에 첫 메타 file이 실제로 있는지 검증
            test_path = os.path.join(c, val_df.iloc[0]['file_rel'])
            if os.path.exists(test_path):
                video_root = c
                print(f"   ✅ 영상 루트 자동 설정: {video_root}")
                break
        else:
            print(f"   ❌ 영상 루트를 찾을 수 없습니다.")
            print(f"      예상 파일: {val_df.iloc[0]['file_rel']}")
            print(f"      후보:")
            for c in candidates:
                print(f"        - {c}")
            sys.exit(1)
    else:
        # 기본 경로가 존재하더라도 실제 파일이 있는지 검증
        test_path = os.path.join(video_root, val_df.iloc[0]['file_rel'])
        if not os.path.exists(test_path):
            print(f"   ⚠️  {video_root} 존재하지만 영상이 없음. 검색 중...")
            for c in [os.path.join(AVDF1M_ROOT, "extracted_val/val"),
                      os.path.join(AVDF1M_ROOT, "extracted_val")]:
                tp = os.path.join(c, val_df.iloc[0]['file_rel'])
                if os.path.exists(tp):
                    video_root = c
                    print(f"   ✅ 영상 루트 변경: {video_root}")
                    break
        else:
            print(f"   ✅ 영상 루트 확인: {video_root}")
    
    # 추론
    dataset = AVDF1MDataset(val_df, video_root)
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
            p_v_art, p_a_art, p_v_emo, p_a_emo = infer_one(
                x3d_v, aasist_a, hsemo_f, crnn_s,
                x3d, aasist, hsemo, crnn, device
            )
        
        idx = sample_idx.item()
        meta = val_df.iloc[idx]
        results.append({
            'video_label':  int(meta['video_label']),
            'fake_type':    meta['fake_type'],
            'modify_type':  meta['modify_type'],
            'audio_model':  meta.get('audio_model'),
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
    
    # 저장
    df_out = pd.DataFrame(results)
    pred_path = os.path.join(REPORT_DIR, 'predictions.csv')
    df_out.to_csv(pred_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 예측 저장: {pred_path}")
    
    # ─── 분석 ─────────────────────────────────────────────
    labels = df_out['video_label'].values.astype(float)
    p_v_art = df_out['p_v_artifact'].values / 100
    p_a_art = df_out['p_a_artifact'].values / 100
    p_v_emo = df_out['p_v_emotion'].values / 100
    p_a_emo = df_out['p_a_emotion'].values / 100
    
    art_or = prob_or(p_v_art, p_a_art)
    emo_or = prob_or(p_v_emo, p_a_emo)
    final_or = prob_or(art_or, emo_or)
    
    print("\n" + "=" * 70)
    print("🎯 P→AVDF1M 결과")
    print("=" * 70)
    
    strategies = {
        'X3D 단독':           p_v_art,
        'AASIST 단독':        p_a_art,
        'HSEmotion+GRU 단독': p_v_emo,
        'CRNN+GRU 단독':      p_a_emo,
        'Score_artifact':     art_or,
        'Score_emotion':      emo_or,
        '🌟 Score_final':     final_or,
    }
    
    summary = []
    print(f"\n  {'전략':<28} {'AUC':>7} {'Acc':>7} {'F1':>7} {'Recall':>8}")
    print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for name, probs in strategies.items():
        m = compute_metrics(probs, labels)
        summary.append({'strategy': name, **m})
        print(f"  {name:<28} {m['auc']:>6.2f}% {m['acc']:>6.2f}% "
              f"{m['f1']:>6.2f}% {m['recall']:>7.2f}%")
    
    pd.DataFrame(summary).to_csv(
        os.path.join(REPORT_DIR, 'metrics.csv'),
        index=False, encoding='utf-8-sig'
    )
    
    # fake_type별 breakdown
    print("\n" + "=" * 70)
    print("📊 fake_type별 Score_final 성능")
    print("=" * 70)
    for ft in ['real', 'fake_video_only', 'fake_audio_only', 'fake_both']:
        sub = df_out[df_out['fake_type'] == ft]
        if len(sub) == 0: continue
        sub_idx = sub.index.values
        sub_final = final_or[sub_idx]
        sub_label = labels[sub_idx]
        if ft == 'real':
            fp_rate = (sub_final > 0.5).mean() * 100
            print(f"  {ft:<20} N={len(sub):>3}  FP rate={fp_rate:.2f}%")
        else:
            # real + 해당 fake_type으로 AUC 계산
            real_idx = df_out[df_out['fake_type']=='real'].index.values
            combined_idx = np.concatenate([real_idx, sub_idx])
            combined_final = final_or[combined_idx]
            combined_label = labels[combined_idx]
            m = compute_metrics(combined_final, combined_label)
            print(f"  {ft:<20} N={len(sub):>3}  AUC={m['auc']:>5.2f}%  F1={m['f1']:>5.2f}%  Recall={m['recall']:>5.2f}%")
    
    print(f"\n✅ 완료. 결과: {os.path.abspath(REPORT_DIR)}/")


if __name__ == "__main__":
    main()