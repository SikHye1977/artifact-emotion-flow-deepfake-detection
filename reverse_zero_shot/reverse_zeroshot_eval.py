"""
==============================================================================
[역방향 Zero-shot] PolyGlotFake 학습 모델 → FakeAVCeleb 평가

[import 방식]
train_HSEmotion_pgf.py / train_CRNN_pgf.py 와 동일한 패턴 사용.
학습이 성공했으므로 동일한 import도 정상 작동함.
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
    recall_score, f1_score
)

# 경로 설정 (학습 스크립트와 동일)
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

# ══════════════════════════════════════════════════════════════════════════════
# 모델 클래스 import (train_*_pgf.py와 동일한 패턴)
# ══════════════════════════════════════════════════════════════════════════════

# HSEmotion 모델 클래스
try:
    from emotion_deepfake_detector_lite import EmotionFlowDetectorLite
except ImportError:
    try:
        from train_HSEmotion import EmotionFlowDetectorLite
    except ImportError:
        print("❌ EmotionFlowDetectorLite 클래스 import 실패")
        print(f"   상위 폴더({PARENT_DIR})에서 다음 파일 중 하나가 필요:")
        print("   - emotion_deepfake_detector_lite.py")
        print("   - train_HSEmotion.py")
        sys.exit(1)

# CRNN 모델 클래스 + 함수
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
        print(f"   상위 폴더({PARENT_DIR})에서 다음 파일 중 하나가 필요:")
        print("   - audio_emotion_deepfake_detector.py")
        print("   - train_CRNN.py")
        sys.exit(1)

# AASIST
try:
    from aasist.models.AASIST import Model as AASISTModel
except ImportError:
    print("❌ aasist 모듈 import 실패")
    print(f"   상위 폴더({PARENT_DIR})에 aasist/ 폴더가 있어야 합니다")
    sys.exit(1)


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
# 2. FakeAVCeleb 평가셋
# ══════════════════════════════════════════════════════════════════════════════
def find_fakeavceleb_root():
    candidates = [
        os.path.abspath("FakeAVCeleb_v1.2"),
        os.path.abspath("../FakeAVCeleb_v1.2"),
        os.path.abspath("../../FakeAVCeleb_v1.2"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            csv_path = os.path.join(path, 'meta_data.csv')
            if os.path.isfile(csv_path):
                return path
    raise FileNotFoundError("FakeAVCeleb_v1.2 폴더를 찾을 수 없습니다.")


def build_fakeavceleb_eval_set(base_dir=None, n_real=500, n_fake=500,
                                train_fake_n=2000, seed=42):
    if base_dir is None:
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
    return val_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class FakeAVCelebMultiInputDataset(Dataset):
    def __init__(self, df, base_dir, num_segments=16, segment_duration=3.0):
        self.df               = df.reset_index(drop=True)
        self.base_dir         = base_dir
        self.num_segments     = num_segments
        self.segment_duration = segment_duration

    def __len__(self):
        return len(self.df)

    def _try_load(self, idx):
        row = self.df.iloc[idx]
        rel_path = row.iloc[-2].replace("FakeAVCeleb",
                                         os.path.basename(self.base_dir))
        video_path = os.path.join(os.path.dirname(self.base_dir), rel_path,
                                   row['path'])
        if not os.path.exists(video_path):
            video_path = os.path.join(self.base_dir,
                                       row.iloc[-2].replace("FakeAVCeleb/", ""),
                                       row['path'])
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
def load_pgf_models(ckpt_paths, device):
    print("\n🧠 PolyGlotFake 학습 모델 로드 중...")

    x3d = x3d_m(pretrained=False)
    x3d.blocks[5].proj       = nn.Linear(2048, 1)
    x3d.blocks[5].activation = nn.Identity()
    x3d.load_state_dict(torch.load(ckpt_paths['x3d'], map_location=device))
    x3d = x3d.to(device).eval()
    print("  ✅ X3D_m (PGF 학습)")

    with open(ckpt_paths['aasist_config'], 'r') as f:
        config = json.load(f)
    aasist = AASISTModel(config['model_config'])
    aasist.load_state_dict(torch.load(ckpt_paths['aasist'], map_location=device))
    aasist = aasist.to(device).eval()
    print("  ✅ AASIST (PGF 학습)")

    hs_ckpt = torch.load(ckpt_paths['hsemo'], map_location=device)
    hs_cfg  = hs_ckpt.get('cfg', {})
    hs_state = hs_ckpt.get('model_state_dict', hs_ckpt)
    hs_hidden = (hs_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in hs_state else 64)
    hsemo = EmotionFlowDetectorLite(
        model_name=hs_cfg.get('MODEL_NAME', 'enet_b0_8_best_afew'),
        num_frames=hs_cfg.get('NUM_FRAMES', 16),
        gru_hidden=hs_hidden, dropout=hs_cfg.get('DROPOUT', 0.3), device='cpu'
    ).to(device)
    hsemo.load_state_dict(hs_state)
    hsemo.eval()
    print(f"  ✅ HSEmotion+GRU (PGF 학습, hidden={hs_hidden})")

    cr_ckpt = torch.load(ckpt_paths['crnn'], map_location=device)
    cr_cfg  = cr_ckpt.get('cfg', {})
    cr_state = cr_ckpt.get('model_state_dict', cr_ckpt)
    cr_hidden = (cr_state['gru.weight_hh_l0'].shape[1]
                 if 'gru.weight_hh_l0' in cr_state else 128)
    crnn = AudioEmotionFlowDetector(
        pretrained_path=ckpt_paths['crnn_pretrained'],
        num_segments=cr_cfg.get('NUM_SEGMENTS', 16),
        gru_hidden=cr_hidden, dropout=cr_cfg.get('DROPOUT', 0.4)
    ).to(device)
    crnn.load_state_dict(cr_state)
    crnn.eval()
    print(f"  ✅ CRNN+GRU (PGF 학습, hidden={cr_hidden})")

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
    try: auc = roc_auc_score(labels, probs) * 100
    except: auc = 0.0
    return dict(
        auc=auc, acc=accuracy_score(li, preds) * 100,
        precision=precision_score(li, preds, zero_division=0) * 100,
        recall=recall_score(li, preds, zero_division=0) * 100,
        f1=f1_score(li, preds, zero_division=0) * 100,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        REPORT_DIR      = "reverse_zeroshot_report",

        # PGF 학습 가중치 (현재 폴더)
        X3D_CKPT        = "x3d_model_pgf_best.pth",
        AASIST_CKPT     = "aasist_model_pgf_best.pth",
        HSEMO_CKPT      = "emotion_flow_lite_pgf_best.pth",
        CRNN_CKPT       = "audio_flow_deepfake_pgf_best.pth",

        # 상위 폴더 파일
        AASIST_CONFIG   = os.path.join(PARENT_DIR, "aasist/config/AASIST.conf"),
        CRNN_PRETRAINED = os.path.join(PARENT_DIR, "audio_emotion_crnn_best.pth"),

        BATCH_SIZE      = 1,
        NUM_WORKERS     = 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    print(f"📁 Working dir: {os.getcwd()}")
    os.makedirs(CFG['REPORT_DIR'], exist_ok=True)

    # 체크포인트 확인
    print(f"\n📦 체크포인트 확인:")
    paths = [
        ('X3D PGF',         CFG['X3D_CKPT']),
        ('AASIST PGF',      CFG['AASIST_CKPT']),
        ('HSEmotion PGF',   CFG['HSEMO_CKPT']),
        ('CRNN PGF',        CFG['CRNN_CKPT']),
        ('AASIST config',   CFG['AASIST_CONFIG']),
        ('CRNN 사전학습',    CFG['CRNN_PRETRAINED']),
    ]
    missing = []
    for name, p in paths:
        exists = os.path.exists(p)
        marker = "✅" if exists else "❌"
        print(f"   {marker} {name:<18} {p}")
        if not exists:
            missing.append(name)
    if missing:
        print(f"\n❌ 누락된 파일: {missing}")
        sys.exit(1)

    # FakeAVCeleb 평가셋
    print(f"\n📂 FakeAVCeleb 평가셋 구성 중...")
    val_df = build_fakeavceleb_eval_set()
    fakeav_base = val_df.attrs['base_dir']
    print(f"   기본 경로: {fakeav_base}")
    print(f"   총 {len(val_df)}개 (Real {(val_df['video_label']==0).sum()}, "
          f"Fake {(val_df['video_label']==1).sum()})")

    # 모델 로드
    ckpt_paths = {
        'x3d': CFG['X3D_CKPT'], 'aasist': CFG['AASIST_CKPT'],
        'aasist_config': CFG['AASIST_CONFIG'],
        'hsemo': CFG['HSEMO_CKPT'], 'crnn': CFG['CRNN_CKPT'],
        'crnn_pretrained': CFG['CRNN_PRETRAINED'],
    }
    x3d, aasist, hsemo, crnn = load_pgf_models(ckpt_paths, device)

    # 추론
    dataset = FakeAVCelebMultiInputDataset(val_df, fakeav_base)
    loader = DataLoader(dataset, batch_size=CFG['BATCH_SIZE'],
                        shuffle=False, num_workers=CFG['NUM_WORKERS'])

    print(f"\n🚀 역방향 Zero-shot 추론 시작 ({len(dataset)}개)\n")
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
            'method':       meta['method'],
            'p_v_artifact': round(p_v_art * 100, 2),
            'p_a_artifact': round(p_a_art * 100, 2),
            'p_v_emotion':  round(p_v_emo * 100, 2),
            'p_a_emotion':  round(p_a_emo * 100, 2),
        })

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (batch_idx + 1) / elapsed
            print(f"  [{batch_idx+1:4d}/{len(loader)}] 속도: {speed:.1f}/s")

    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(CFG['REPORT_DIR'], 'predictions.csv'),
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

    print("\n" + "="*70)
    print("🎯 역방향 Zero-shot 결과 (PolyGlotFake 학습 → FakeAVCeleb 평가)")
    print("="*70)

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
        os.path.join(CFG['REPORT_DIR'], 'metrics_by_strategy.csv'),
        index=False, encoding='utf-8-sig'
    )

    # 양방향 비교
    print("\n" + "="*70)
    print("📊 양방향 Cross-Dataset 비교")
    print("="*70)

    forward = {
        'Score_artifact': {'auc': 56.78, 'f1': 7.97},
        'Score_emotion':  {'auc': 64.02, 'f1': 86.07},
        'Score_final':    {'auc': 64.73, 'f1': 91.10},
    }

    reverse = {
        'Score_artifact': summary[4],
        'Score_emotion':  summary[5],
        'Score_final':    summary[6],
    }

    print(f"\n  {'전략':<20} {'정방향(F→P)':>15} {'역방향(P→F)':>15}")
    print(f"  {'─'*20} {'─'*15} {'─'*15}")
    print(f"\n  AUC 비교:")
    for k in forward:
        print(f"  {k:<20} {forward[k]['auc']:>13.2f}% {reverse[k]['auc']:>13.2f}%")
    print(f"\n  F1 비교:")
    for k in forward:
        print(f"  {k:<20} {forward[k]['f1']:>13.2f}% {reverse[k]['f1']:>13.2f}%")

    cmp_df = pd.DataFrame([
        {'strategy': k,
         'forward_auc': forward[k]['auc'], 'reverse_auc': reverse[k]['auc'],
         'forward_f1': forward[k]['f1'], 'reverse_f1': reverse[k]['f1']}
        for k in forward
    ])
    cmp_df.to_csv(os.path.join(CFG['REPORT_DIR'], 'comparison_table.csv'),
                  index=False, encoding='utf-8-sig')

    print(f"\n💾 결과 저장: {os.path.abspath(CFG['REPORT_DIR'])}/")
    print("\n" + "="*70)
    print("✅ 역방향 Zero-shot 평가 완료")
    print("="*70)


if __name__ == "__main__":
    main()