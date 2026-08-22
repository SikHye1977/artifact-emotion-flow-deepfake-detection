"""
==============================================================================
[evaluate_fav_breakdown.py] FAV 학습 가중치 → FAV, PGF 평가 + 합성 기법별 분해

[목적]
F→F (FakeAVCeleb in-domain)와 F→P (PolyGlotFake zero-shot) 시나리오에서
각 합성 기법별로 4모델 + 3융합 성능을 분해 분석.

[분해 기준]
  FAV  : method 컬럼 (FaceSwap, FSGAN, Wav2Lip, RTVC, FaceSwap-Wav2Lip 등)
  PGF  : tts (5종) × sync (2종) 컬럼

[가중치]
  X3D       : x3d_model_best_final.pth
  AASIST    : aasist_model_best_final.pth
  HSEmotion : emotion_flow_lite_fav_v2_best.pth
  CRNN      : audio_flow_deepfake_fav_v2_best.pth

[출력]
  fav_breakdown_report/
    ├── fav_to_fav_predictions.csv      (method 컬럼 포함)
    ├── fav_to_pgf_predictions.csv      (tts, sync 컬럼 포함)
    ├── fav_to_fav_breakdown.csv        (method별 7전략 metric)
    └── fav_to_pgf_breakdown.csv        (tts, sync별 metric)

[사용법]
  cd ~/hsh/AIApplication
  python evaluate_fav_breakdown.py
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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score

# 호환성
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

# reverse_zero_shot 폴더도 sys.path에 추가 (polyglotfake_data.py 위치)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
for sub in ['reverse_zero_shot', '.', '..']:
    p = os.path.abspath(os.path.join(CURRENT_DIR, sub))
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# 모델 클래스
try:
    from emotion_deepfake_detector_lite import EmotionFlowDetectorLite
except ImportError:
    from train_HSEmotion import EmotionFlowDetectorLite
try:
    from audio_emotion_deepfake_detector import AudioEmotionFlowDetector, extract_audio_segments
except ImportError:
    from train_CRNN import AudioEmotionFlowDetector, extract_audio_segments
from aasist.models.AASIST import Model as AASISTModel
from polyglotfake_data import build_polyglotfake_dataframe


# ═══════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════
# 가중치 후보 (존재하는 첫 번째 사용)
WEIGHTS_CANDIDATES = {
    'x3d':    ['x3d_model_best_final.pth'],
    'aasist': ['aasist_model_best_final.pth'],
    'hsemo':  ['emotion_flow_lite_fav_v2_best.pth',
               'emotion_flow_lite_v2_best.pth',
               'emotion_flow_lite_best.pth'],
    'crnn':   ['audio_flow_deepfake_fav_v2_best.pth',
               'audio_flow_deepfake_v2_best.pth',
               'audio_flow_deepfake_best.pth'],
}


def resolve_weights():
    """존재하는 파일로 WEIGHTS 딕셔너리 구성."""
    resolved = {}
    for key, candidates in WEIGHTS_CANDIDATES.items():
        for c in candidates:
            if os.path.exists(c):
                resolved[key] = c
                break
        else:
            print(f"   ❌ {key}: 후보 모두 없음 → {candidates}")
            resolved[key] = candidates[0]  # 첫 번째를 디폴트로 (에러 발생용)
    return resolved


WEIGHTS = resolve_weights()
AASIST_CONFIG = "aasist/config/AASIST.conf"
CRNN_PRETRAINED = "audio_emotion_crnn_best.pth"

# FAV 평가 설정 (학습에 사용한 fake는 제외)
FAV_N_REAL = 500
FAV_N_FAKE = 500
FAV_TRAIN_FAKE_N = 2000
SEED = 42

# PGF 평가 설정
PGF_N_REAL = 500
PGF_N_FAKE = 500  # tts×sync 조합 균등 샘플링

REPORT_DIR = 'fav_breakdown_report'


# ═══════════════════════════════════════════════════════════════════
# 1. 전처리
# ═══════════════════════════════════════════════════════════════════
def rescale(x): return x / 255.0
def perm_tc(x): return x.permute(1, 0, 2, 3)
def perm_ct(x): return x.permute(1, 0, 2, 3)

x3d_transform = Compose([
    UniformTemporalSubsample(16), Lambda(rescale), Lambda(perm_tc),
    Normalize([0.45]*3, [0.225]*3), Lambda(perm_ct),
    ShortSideScale(size=256), Resize((224, 224)),
])
hsemo_transform = T.Compose([
    T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_video_for_x3d(path, max_frames=128):
    try:
        c = av.open(path)
        frames = [f.to_rgb().to_ndarray() for f in c.decode(video=0)]
        c.close()
        if len(frames) < 16: return None
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames)-1, max_frames, dtype=int)
            frames = [frames[i] for i in idx]
        return torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).to(torch.float32)
    except Exception:
        return None


def load_frames_for_hsemo(path, num_frames=16):
    try:
        c = av.open(path)
        total = c.streams.video[0].frames
        if total < num_frames: c.close(); return None
        targets = set(np.linspace(0, total-1, num_frames, dtype=int).tolist())
        sampled = {}
        for i, f in enumerate(c.decode(video=0)):
            if i in targets: sampled[i] = f.to_rgb().to_ndarray()
            if len(sampled) >= num_frames: break
        c.close()
        if len(sampled) < num_frames: return None
        keys = sorted(sampled.keys())
        return torch.stack([hsemo_transform(sampled[k]) for k in keys])
    except Exception:
        return None


def load_audio_for_aasist(video_path, target_sr=16000, max_length=64000):
    try:
        c = av.open(video_path)
        if not c.streams.audio: c.close(); return None
        sr = c.streams.audio[0].rate
        frames = []
        for frame in c.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.dtype == np.int16:   arr = arr.astype(np.float32)/32768.0
            elif arr.dtype == np.int32: arr = arr.astype(np.float32)/2147483648.0
            else:                       arr = arr.astype(np.float32)
            if arr.ndim > 1 and arr.shape[0] > arr.shape[1]: arr = arr.T
            elif arr.ndim == 1: arr = arr[np.newaxis, :]
            frames.append(arr)
        c.close()
        if not frames: return None
        wav = torch.from_numpy(np.concatenate(frames, axis=-1))
        if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr: wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        if wav.shape[1] > max_length: wav = wav[:, :max_length]
        else: wav = torch.nn.functional.pad(wav, (0, max_length - wav.shape[1]))
        return wav.squeeze()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# 2. 평가셋 구성
# ═══════════════════════════════════════════════════════════════════
def find_fakeavceleb_root():
    candidates = [
        os.path.expanduser("~/hsh/AIApplication/FakeAVCeleb_v1.2"),
        os.path.expanduser("~/hsh/FakeAVCeleb_v1.2"),
        os.path.expanduser("~/FakeAVCeleb_v1.2"),
        "FakeAVCeleb_v1.2",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "meta_data.csv")):
            return c
    raise FileNotFoundError("FakeAVCeleb_v1.2 폴더를 찾을 수 없음")


def build_fav_eval_df():
    base_dir = find_fakeavceleb_root()
    df = pd.read_csv(os.path.join(base_dir, 'meta_data.csv'))
    df['video_label'] = df['method'].apply(lambda x: 0.0 if x == 'real' else 1.0)
    
    real_df = df[df.video_label == 0]
    fake_df = df[df.video_label == 1]
    
    # 학습에 사용된 fake 인덱스 제외
    train_fake_idx = fake_df.sample(n=FAV_TRAIN_FAKE_N, random_state=SEED).index
    unseen_fake = fake_df.drop(train_fake_idx)
    
    sampled_real = real_df.sample(n=FAV_N_REAL, random_state=SEED)
    sampled_fake = unseen_fake.sample(n=FAV_N_FAKE, random_state=SEED+1)
    
    val_df = pd.concat([sampled_real, sampled_fake]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # video_path 컬럼 생성
    # 메타데이터 구조:
    #   'Unnamed: 9' = "FakeAVCeleb/RealVideo-RealAudio/African/men/id00076" (디렉토리)
    #   'path'       = "00109.mp4" (파일명)
    # 실제 경로 = base_dir의 부모 + Unnamed:9 (이름 교체) + path
    base_parent = os.path.dirname(base_dir)
    base_name = os.path.basename(base_dir)  # 'FakeAVCeleb_v1.2'
    
    def build_path(row):
        rel_dir = str(row['Unnamed: 9']).replace('FakeAVCeleb', base_name)
        # 예상 경로
        p = os.path.join(base_parent, rel_dir, row['path'])
        if os.path.exists(p):
            return p
        # 대안: base_dir 안에 직접 (FakeAVCeleb/ prefix 제거)
        rel_no_prefix = str(row['Unnamed: 9'])
        if rel_no_prefix.startswith('FakeAVCeleb/'):
            rel_no_prefix = rel_no_prefix[len('FakeAVCeleb/'):]
        p2 = os.path.join(base_dir, rel_no_prefix, row['path'])
        return p2
    
    val_df['video_path'] = val_df.apply(build_path, axis=1)
    
    print(f"\n📂 FAV 평가셋: Real {FAV_N_REAL} + Fake {FAV_N_FAKE} = {len(val_df)}")
    print(f"   method 분포:")
    for m, n in val_df['method'].value_counts().items():
        print(f"     {m}: {n}")
    
    # Sanity check (처음 10개)
    n_exists = sum(1 for p in val_df['video_path'].head(10) if os.path.exists(p))
    print(f"\n   [Sanity check] 처음 10개 중 {n_exists}/10개 존재")
    if n_exists < 10:
        missing = [p for p in val_df['video_path'].head(10) if not os.path.exists(p)][:3]
        print(f"   누락 예시:")
        for ex in missing:
            print(f"     ❌ {ex}")
    
    return val_df


def build_pgf_eval_df():
    df = build_polyglotfake_dataframe()
    
    real_df = df[df.video_label == 0]
    fake_df = df[df.video_label == 1]
    
    # tts × sync 조합별 균등 샘플링
    sampled_real = real_df.sample(n=min(PGF_N_REAL, len(real_df)), random_state=SEED)
    
    fake_groups = fake_df.groupby(['tts', 'sync'])
    n_per_group = max(1, PGF_N_FAKE // len(fake_groups))
    sampled_fakes = []
    for (tts, sync), grp in fake_groups:
        n_take = min(n_per_group, len(grp))
        sampled_fakes.append(grp.sample(n=n_take, random_state=SEED))
    sampled_fake = pd.concat(sampled_fakes)
    
    val_df = pd.concat([sampled_real, sampled_fake]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"\n📂 PGF 평가셋: Real {len(sampled_real)} + Fake {len(sampled_fake)} = {len(val_df)}")
    print(f"   tts × sync 분포 (fake만):")
    for (tts, sync), n in val_df[val_df.video_label==1].groupby(['tts','sync']).size().items():
        print(f"     {tts:>15} × {sync:>15}: {n}")
    return val_df


# ═══════════════════════════════════════════════════════════════════
# 3. Dataset (FAV/PGF 공통)
# ═══════════════════════════════════════════════════════════════════
class MultiModalDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _try_load(self, idx):
        row = self.df.iloc[idx]
        path = row['video_path']
        if not os.path.exists(path):
            return None
        v = load_video_for_x3d(path)
        if v is None: return None
        v = x3d_transform(v)
        a = load_audio_for_aasist(path)
        if a is None: return None
        f = load_frames_for_hsemo(path)
        if f is None: return None
        s = extract_audio_segments(path, num_segments=16, target_sr=16000, segment_duration=3.0)
        if s is None: return None
        return v, a, f, s, torch.tensor(float(row['video_label'])), idx

    def __getitem__(self, idx):
        for offset in range(len(self)):
            r = self._try_load((idx + offset) % len(self))
            if r is not None: return r
        raise RuntimeError("로드 가능한 샘플 없음")


# ═══════════════════════════════════════════════════════════════════
# 4. 모델 로드
# ═══════════════════════════════════════════════════════════════════
def load_models(device):
    print("\n🧠 모델 로드 (FAV 가중치)...")
    for k, v in WEIGHTS.items():
        ok = "✅" if os.path.exists(v) else "❌"
        print(f"   {ok} {k:<8}: {v}")
    missing = [v for v in WEIGHTS.values() if not os.path.exists(v)]
    if missing:
        print(f"❌ 누락: {missing}"); sys.exit(1)
    
    # X3D
    x3d = x3d_m(pretrained=False)
    x3d.blocks[5].proj = nn.Linear(2048, 1)
    x3d.blocks[5].activation = nn.Identity()
    x3d.load_state_dict(torch.load(WEIGHTS['x3d'], map_location=device))
    x3d = x3d.to(device).eval()
    
    # AASIST
    with open(AASIST_CONFIG) as f: cfg = json.load(f)
    aasist = AASISTModel(cfg['model_config'])
    aasist.load_state_dict(torch.load(WEIGHTS['aasist'], map_location=device))
    aasist = aasist.to(device).eval()
    
    # HSEmotion
    hs_ckpt = torch.load(WEIGHTS['hsemo'], map_location=device)
    hs_cfg = hs_ckpt.get('cfg', {})
    hs_state = hs_ckpt.get('model_state_dict', hs_ckpt)
    hs_hidden = (hs_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in hs_state else 64)
    hsemo = EmotionFlowDetectorLite(
        model_name=hs_cfg.get('MODEL_NAME', 'enet_b0_8_best_afew'),
        num_frames=hs_cfg.get('NUM_FRAMES', 16),
        gru_hidden=hs_hidden,
        dropout=hs_cfg.get('DROPOUT', 0.3),
        device='cpu',
        unfreeze_last_blocks=hs_cfg.get('UNFREEZE_LAST_BLOCKS', 2),
    ).to(device)
    hsemo.load_state_dict(hs_state)
    hsemo.eval()
    
    # CRNN
    cr_ckpt = torch.load(WEIGHTS['crnn'], map_location=device)
    cr_cfg = cr_ckpt.get('cfg', {})
    cr_state = cr_ckpt.get('model_state_dict', cr_ckpt)
    cr_hidden = (cr_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in cr_state else 128)
    crnn = AudioEmotionFlowDetector(
        pretrained_path=CRNN_PRETRAINED,
        num_segments=cr_cfg.get('NUM_SEGMENTS', 16),
        gru_hidden=cr_hidden,
        dropout=cr_cfg.get('DROPOUT', 0.4),
        unfreeze_pretrained_gru=cr_cfg.get('UNFREEZE_PRETRAINED_GRU', True),
    ).to(device)
    crnn.load_state_dict(cr_state)
    crnn.eval()
    
    for m in [x3d, aasist, hsemo, crnn]:
        for p in m.parameters(): p.requires_grad = False
    print("  ✅ 4 모델 로드 완료\n")
    return x3d, aasist, hsemo, crnn


# ═══════════════════════════════════════════════════════════════════
# 5. 추론 & 분석
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer(v, a, f, s, x3d, aasist, hsemo, crnn, device):
    p_v_art = torch.sigmoid(x3d(v.unsqueeze(0).to(device))).item()
    _, a_out = aasist(a.unsqueeze(0).to(device))
    p_a_art = torch.softmax(a_out, dim=1)[0, 1].item()
    e_logit, _ = hsemo(f.unsqueeze(0).to(device))
    p_v_emo = torch.sigmoid(e_logit).item()
    c_logit, _ = crnn(s.unsqueeze(0).to(device))
    p_a_emo = torch.sigmoid(c_logit).item()
    return p_v_art, p_a_art, p_v_emo, p_a_emo


def run_inference(val_df, models, device, name):
    x3d, aasist, hsemo, crnn = models
    ds = MultiModalDataset(val_df)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"\n🚀 {name} 추론 ({len(ds)}개)\n")
    results = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        v, a, f, s, lbl, idx = batch
        with torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
            p_v_art, p_a_art, p_v_emo, p_a_emo = infer(
                v.squeeze(0), a.squeeze(0), f.squeeze(0), s.squeeze(0),
                x3d, aasist, hsemo, crnn, device
            )
        results.append({
            'idx': idx.item(), 'video_label': int(lbl.item()),
            'p_v_artifact': p_v_art, 'p_a_artifact': p_a_art,
            'p_v_emotion': p_v_emo,  'p_a_emotion': p_a_emo,
        })
        if (bi+1) % 50 == 0:
            sp = (bi+1) / (time.time() - t0)
            print(f"  [{bi+1:4d}/{len(loader)}] {sp:.1f}/s ETA {(len(loader)-bi-1)/sp/60:.1f}분")
    return pd.DataFrame(results)


def prob_or(*ps):
    r = np.ones_like(ps[0])
    for p in ps: r *= (1.0 - p)
    return 1.0 - r


def compute_metrics(probs, labels, t=0.5):
    preds = (probs > t).astype(int); li = labels.astype(int)
    auc = roc_auc_score(labels, probs) * 100 if len(np.unique(labels)) > 1 else 0.0
    return {
        'AUC': round(auc, 2),
        'Acc': round(accuracy_score(li, preds) * 100, 2),
        'Prec': round(precision_score(li, preds, zero_division=0) * 100, 2),
        'Recall': round(recall_score(li, preds, zero_division=0) * 100, 2),
        'F1': round(f1_score(li, preds, zero_division=0) * 100, 2),
        'N': len(labels),
    }


def analyze_breakdown(preds_df, val_df, group_cols, scenario_name):
    """합성 기법별 metric 계산. group_cols로 그룹화 (e.g. ['method'] 또는 ['tts','sync'])."""
    # 메타 컬럼 병합
    merged = preds_df.merge(val_df[['video_label'] + group_cols].reset_index(),
                             left_on='idx', right_on='index', suffixes=('', '_meta'))
    
    # 확률 계산
    p_v_art = merged['p_v_artifact'].values
    p_a_art = merged['p_a_artifact'].values
    p_v_emo = merged['p_v_emotion'].values
    p_a_emo = merged['p_a_emotion'].values
    s_art = prob_or(p_v_art, p_a_art)
    s_emo = prob_or(p_v_emo, p_a_emo)
    s_final = prob_or(s_art, s_emo)
    
    merged['Score_artifact'] = s_art
    merged['Score_emotion']  = s_emo
    merged['Score_final']    = s_final
    
    # 전체 metric
    print(f"\n{'='*70}\n📊 {scenario_name} — 전체 결과\n{'='*70}")
    labels = merged['video_label'].values.astype(float)
    overall = []
    print(f"  {'전략':<22} {'AUC':>7} {'Acc':>7} {'F1':>7} {'Recall':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    strategies = {
        'X3D': p_v_art, 'AASIST': p_a_art,
        'HSEmotion': p_v_emo, 'CRNN': p_a_emo,
        'Score_artifact': s_art, 'Score_emotion': s_emo,
        '🌟 Score_final': s_final,
    }
    for sname, probs in strategies.items():
        m = compute_metrics(probs, labels)
        overall.append({'strategy': sname, 'group': 'ALL', **m})
        print(f"  {sname:<22} {m['AUC']:>6.2f}% {m['Acc']:>6.2f}% {m['F1']:>6.2f}% {m['Recall']:>6.2f}%")
    
    # 그룹별 metric
    print(f"\n{'='*70}\n📊 {scenario_name} — {'/'.join(group_cols)}별 분해\n{'='*70}")
    
    # Real만 따로 처리 (FP rate)
    real_mask = (merged['video_label'] == 0)
    real_subset = merged[real_mask]
    if len(real_subset) > 0:
        print(f"\n  [Real 전체] N={len(real_subset)}")
        for sname, col in [('X3D', p_v_art), ('AASIST', p_a_art),
                           ('HSEmo', p_v_emo), ('CRNN', p_a_emo),
                           ('Score_final', s_final)]:
            fp = (col[real_mask] > 0.5).mean() * 100
            print(f"     {sname:<14} FP rate = {fp:5.2f}%")
    
    # 각 fake 그룹
    breakdown = []
    fake_subset = merged[~real_mask].copy()
    
    for group_vals, grp in fake_subset.groupby(group_cols):
        if isinstance(group_vals, tuple):
            label = " × ".join(str(v) for v in group_vals)
        else:
            label = str(group_vals)
        
        if len(grp) < 3: continue  # 너무 작으면 스킵
        
        # Real(반)와 합쳐서 평가하면 confusing. 대신 fake recall만 보고하는 게 낫다
        # 또는 real 전부 + 이 그룹 fake로 sub-evaluation
        grp_idx = grp.index.values
        eval_indices = np.concatenate([real_subset.index.values, grp_idx])
        eval_labels = merged.loc[eval_indices, 'video_label'].values.astype(float)
        
        print(f"\n  [{label}] N_fake={len(grp)}, N_real={len(real_subset)} (혼합 평가)")
        print(f"     {'전략':<16} {'AUC':>7} {'F1':>7} {'Recall':>8}")
        for sname, col in [
            ('X3D', merged.loc[eval_indices, 'p_v_artifact'].values),
            ('AASIST', merged.loc[eval_indices, 'p_a_artifact'].values),
            ('HSEmo', merged.loc[eval_indices, 'p_v_emotion'].values),
            ('CRNN', merged.loc[eval_indices, 'p_a_emotion'].values),
            ('Score_art', merged.loc[eval_indices, 'Score_artifact'].values),
            ('Score_emo', merged.loc[eval_indices, 'Score_emotion'].values),
            ('Score_final', merged.loc[eval_indices, 'Score_final'].values),
        ]:
            m = compute_metrics(col, eval_labels)
            print(f"     {sname:<16} {m['AUC']:>6.2f}% {m['F1']:>6.2f}% {m['Recall']:>7.2f}%")
            row = {'scenario': scenario_name, 'strategy': sname, 'group': label,
                   'N_fake': len(grp), 'N_real': len(real_subset), **m}
            breakdown.append(row)
    
    return pd.DataFrame(overall), pd.DataFrame(breakdown), merged


# ═══════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("🎯 F→F, F→P 평가 + 합성 기법별 분해")
    print("=" * 70)
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")
    
    # 평가셋 구성
    fav_df = build_fav_eval_df()
    pgf_df = build_pgf_eval_df()
    
    # 모델 로드 (한 번)
    models = load_models(device)
    
    # F→F 추론
    fav_preds = run_inference(fav_df, models, device, "F→F (FakeAVCeleb)")
    fav_overall, fav_breakdown, fav_merged = analyze_breakdown(
        fav_preds, fav_df, ['method'], "F→F"
    )
    fav_merged.to_csv(os.path.join(REPORT_DIR, 'fav_to_fav_predictions.csv'),
                      index=False, encoding='utf-8-sig')
    fav_breakdown.to_csv(os.path.join(REPORT_DIR, 'fav_to_fav_breakdown.csv'),
                         index=False, encoding='utf-8-sig')
    
    # F→P 추론
    pgf_preds = run_inference(pgf_df, models, device, "F→P (PolyGlotFake)")
    # PGF: tts와 sync 각각 따로 + 조합
    pgf_overall_t, pgf_break_tts, pgf_merged_t = analyze_breakdown(
        pgf_preds, pgf_df, ['tts'], "F→P (by TTS)"
    )
    pgf_overall_s, pgf_break_sync, pgf_merged_s = analyze_breakdown(
        pgf_preds, pgf_df, ['sync'], "F→P (by Sync)"
    )
    pgf_overall_c, pgf_break_combo, pgf_merged_c = analyze_breakdown(
        pgf_preds, pgf_df, ['tts', 'sync'], "F→P (by TTS×Sync)"
    )
    
    pgf_merged_c.to_csv(os.path.join(REPORT_DIR, 'fav_to_pgf_predictions.csv'),
                        index=False, encoding='utf-8-sig')
    pd.concat([pgf_break_tts, pgf_break_sync, pgf_break_combo]).to_csv(
        os.path.join(REPORT_DIR, 'fav_to_pgf_breakdown.csv'),
        index=False, encoding='utf-8-sig'
    )
    
    print(f"\n✅ 완료. 결과: {os.path.abspath(REPORT_DIR)}/")
    print(f"   - fav_to_fav_predictions.csv  (FAV method별)")
    print(f"   - fav_to_fav_breakdown.csv")
    print(f"   - fav_to_pgf_predictions.csv  (PGF tts/sync별)")
    print(f"   - fav_to_pgf_breakdown.csv")


if __name__ == "__main__":
    main()