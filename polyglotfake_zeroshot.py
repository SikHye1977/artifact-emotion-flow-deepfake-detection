"""
==============================================================================
[PolyGlotFake Zero-shot 평가] 4-Way Probabilistic OR Architecture
Multilingual Cross-Dataset Generalization Test
==============================================================================

[목적]
FakeAVCeleb으로 학습한 4개 모델을 PolyGlotFake에서 추론만 수행:
  - X3D_m       (비디오 아티팩트)   ─ FakeAVCeleb fsgan/wav2lip 학습
  - AASIST      (오디오 아티팩트)   ─ FakeAVCeleb wav2lip 학습
  - HSEmotion+GRU (비디오 감정)     ─ AffectNet 사전학습 + FakeAVCeleb fine-tune
  - CRNN+GRU    (오디오 감정)       ─ RAVDESS 사전학습 + FakeAVCeleb fine-tune

학습 도메인과 다른 데이터에서 일반화 능력을 측정.

[평가 데이터]
  - Real: 766개 (7개 언어)
  - Fake: 14,472개 (5 TTS × 2 sync × 7×6 언어쌍)
  - 총: 15,238개

[종합 분석]
  1. 전체 지표 (AUC, Acc, P, R, F1)
  2. 모델별 단독 성능
  3. Fusion 전략 5종 비교 (4방향 OR / Mean / Weighted / 조건부)
  4. 립싱크 기법별 (VideoRetalking vs Wav2Lip)
  5. TTS 기법별 (Xtts/Bark/MicroTts/Tacotron/Vall-E)
  6. 언어별 (영어 vs 비영어)
  7. FakeAVCeleb 결과와 cross-dataset 비교

[전체 시간 추정]
  GPU에서 약 1초/샘플 → 전체 약 4시간 예상
  중간 저장(checkpoint)으로 끊겨도 재개 가능
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
import torch.nn.functional as F
import torchaudio
import av
from torch.utils.data import Dataset, DataLoader

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

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# PyTorch 2.6+ 보안 정책 우회
torch.load = functools.partial(torch.load, weights_only=False)

# 기존 모델 클래스 import
try:
    from train_HSEmotion import EmotionFlowDetectorLite
except ImportError:
    print("❌ train_HSEmotion.py 미발견")
    sys.exit(1)

try:
    from train_CRNN import AudioEmotionFlowDetector, extract_audio_segments
except ImportError:
    print("❌ train_CRNN.py 미발견")
    sys.exit(1)

try:
    from aasist.models.AASIST import Model as AASISTModel
except ImportError:
    print("❌ aasist 미발견")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 전처리 함수들
# ══════════════════════════════════════════════════════════════════════════════
def rescale_video(x): return x / 255.0
def permute_to_tc(x): return x.permute(1, 0, 2, 3)
def permute_to_ct(x): return x.permute(1, 0, 2, 3)

# X3D용 전처리
x3d_video_transform = Compose([
    UniformTemporalSubsample(16),
    Lambda(rescale_video),
    Lambda(permute_to_tc),
    Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
    Lambda(permute_to_ct),
    ShortSideScale(size=256),
    Resize((224, 224))
])

# HSEmotion용 프레임 전처리
hsemo_frame_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_video_for_x3d(path, max_frames=128):
    """
    X3D용 영상 로드: (3, T, H, W)
    긴 영상은 max_frames로 사전 다운샘플링하여 메모리 절약.
    이후 UniformTemporalSubsample(16)에서 다시 16프레임으로 줄어듦.
    """
    try:
        container = av.open(path)
        all_frames = []
        for f in container.decode(video=0):
            all_frames.append(f.to_rgb().to_ndarray())
        container.close()
        if len(all_frames) < 16:
            return None
        # 메모리 절약: max_frames개로 미리 다운샘플 (이후 transform이 16개로 줄임)
        if len(all_frames) > max_frames:
            indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
            all_frames = [all_frames[i] for i in indices]
        video = np.stack(all_frames)
        return torch.from_numpy(video).permute(3, 0, 1, 2).to(torch.float32)
    except Exception:
        return None


def load_frames_for_hsemo(path, num_frames=16):
    """
    HSEmotion용 균등 샘플 프레임: (T, 3, 224, 224)
    스트리밍 방식으로 필요한 프레임만 디코딩하여 메모리 절약.
    """
    try:
        container = av.open(path)
        # 먼저 총 프레임 수 추정 (대부분 빠름)
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
            if len(sampled) >= num_frames:
                break
        container.close()
        if len(sampled) < num_frames:
            return None
        # 인덱스 순서대로 정렬해서 transform 적용
        sorted_keys = sorted(sampled.keys())
        return torch.stack([hsemo_frame_transform(sampled[k]) for k in sorted_keys])
    except Exception:
        return None


def load_audio_for_aasist(video_path, target_sr=16000, max_length=64000):
    """AASIST용 오디오: (samples,) — 4초 16kHz"""
    try:
        container = av.open(video_path)
        if not container.streams.audio:
            container.close()
            return None
        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate
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
        if not frames:
            return None
        waveform = np.concatenate(frames, axis=-1)
        waveform = torch.from_numpy(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            )
            waveform = resampler(waveform)
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        else:
            pad_size = max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        return waveform.squeeze()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. 메타데이터 → DataFrame
# ══════════════════════════════════════════════════════════════════════════════
def build_polyglotfake_df(json_dir: str) -> pd.DataFrame:
    """
    PolyGlotFake JSON에서 평가용 DataFrame 구성.
    """
    real_path = os.path.join(json_dir, 'real_json_file/all_real_video.json')
    fake_path = os.path.join(json_dir, 'fake_Json_file/all_fake_video.json')

    with open(real_path, 'r', encoding='utf-8') as f:
        real_data = json.load(f)
    with open(fake_path, 'r', encoding='utf-8') as f:
        fake_data = json.load(f)

    rows = []

    # Real 영상
    for v in real_data['videos']:
        rows.append({
            'filename':    v['filename'],
            'video_label': 0,            # Real
            'lang':        v['lang'],
            'target_lang': v['lang'],    # Real은 동일
            'tts':         'real',
            'sync':        'real',
            'rel_dir':     f"real/{v['lang']}",
        })

    # Fake 영상
    for v in fake_data['video']:
        rows.append({
            'filename':    v['filename'],
            'video_label': 1,            # Fake
            'lang':        v['raw_lang'],
            'target_lang': v['target_lang'],
            'tts':         v['tts_technique'],
            'sync':        v['sync_tech'],
            'rel_dir':     f"fake/to_{v['target_lang']}",
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════════
class PolyGlotFakeDataset(Dataset):
    """
    한 영상에서 4개 모델 입력을 모두 생성:
      - x3d_video       (3, T, H, W)
      - aasist_audio    (samples,)
      - hsemo_frames    (T, 3, 224, 224)
      - crnn_segments   (N, 1, samples)
    """
    def __init__(self, df: pd.DataFrame, base_dir: str,
                 num_frames: int = 16,
                 num_segments: int = 16,
                 segment_duration: float = 3.0):
        self.df               = df.reset_index(drop=True)
        self.base_dir         = base_dir
        self.num_frames       = num_frames
        self.num_segments     = num_segments
        self.segment_duration = segment_duration

    def __len__(self):
        return len(self.df)

    def _try_load(self, idx):
        import gc
        row = self.df.iloc[idx]
        video_path = os.path.join(self.base_dir, row['rel_dir'], row['filename'])
        if not os.path.exists(video_path):
            return None

        x3d_v   = load_video_for_x3d(video_path)
        if x3d_v is None: return None

        x3d_v   = x3d_video_transform(x3d_v)

        aasist_a = load_audio_for_aasist(video_path)
        if aasist_a is None: return None

        hsemo_f = load_frames_for_hsemo(video_path, self.num_frames)
        if hsemo_f is None: return None

        crnn_s  = extract_audio_segments(
            video_path,
            num_segments     = self.num_segments,
            target_sr        = 16000,
            segment_duration = self.segment_duration
        )
        if crnn_s is None: return None

        return (x3d_v, aasist_a, hsemo_f, crnn_s,
                torch.tensor(float(row['video_label'])),
                idx)

    def __getitem__(self, idx):
        # 손상 영상 시 다음 샘플로 fallback
        for offset in range(len(self)):
            result = self._try_load((idx + offset) % len(self))
            if result is not None:
                return result
        raise RuntimeError("로드 가능한 샘플 없음")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_all_models(ckpt_paths: dict, device: torch.device):
    print("🧠 4개 모델 로드 중...")

    # X3D
    x3d = x3d_m(pretrained=False)
    x3d.blocks[5].proj       = nn.Linear(2048, 1)
    x3d.blocks[5].activation = nn.Identity()
    x3d.load_state_dict(torch.load(ckpt_paths['x3d'], map_location=device))
    x3d = x3d.to(device).eval()
    print("  ✅ X3D_m")

    # AASIST
    with open(ckpt_paths['aasist_config'], 'r') as f:
        config = json.load(f)
    aasist = AASISTModel(config['model_config'])
    aasist.load_state_dict(torch.load(ckpt_paths['aasist'], map_location=device))
    aasist = aasist.to(device).eval()
    print("  ✅ AASIST")

    # HSEmotion (체크포인트에서 hidden 자동 감지)
    hs_ckpt = torch.load(ckpt_paths['hsemo'], map_location=device)
    hs_cfg  = hs_ckpt.get('cfg', {})
    hs_state = hs_ckpt.get('model_state_dict', hs_ckpt)
    if 'gru.weight_hh_l0' in hs_state:
        hs_hidden = hs_state['gru.weight_hh_l0'].shape[1]
    else:
        hs_hidden = hs_cfg.get('GRU_HIDDEN', 64)
    hsemo = EmotionFlowDetectorLite(
        model_name = hs_cfg.get('MODEL_NAME', 'enet_b0_8_best_afew'),
        num_frames = hs_cfg.get('NUM_FRAMES', 16),
        gru_hidden = hs_hidden,
        dropout    = hs_cfg.get('DROPOUT', 0.3),
        device     = 'cpu'
    ).to(device)
    hsemo.load_state_dict(hs_state)
    hsemo.eval()
    print(f"  ✅ HSEmotion+GRU (hidden={hs_hidden})")

    # CRNN+GRU
    cr_ckpt = torch.load(ckpt_paths['crnn'], map_location=device)
    cr_cfg  = cr_ckpt.get('cfg', {})
    cr_state = cr_ckpt.get('model_state_dict', cr_ckpt)
    if 'gru.weight_hh_l0' in cr_state:
        cr_hidden = cr_state['gru.weight_hh_l0'].shape[1]
    else:
        cr_hidden = cr_cfg.get('GRU_HIDDEN', 128)
    crnn = AudioEmotionFlowDetector(
        pretrained_path = ckpt_paths['crnn_pretrained'],
        num_segments    = cr_cfg.get('NUM_SEGMENTS', 16),
        gru_hidden      = cr_hidden,
        dropout         = cr_cfg.get('DROPOUT', 0.4)
    ).to(device)
    crnn.load_state_dict(cr_state)
    crnn.eval()
    print(f"  ✅ CRNN+GRU (hidden={cr_hidden})")

    # 모두 동결
    for m in [x3d, aasist, hsemo, crnn]:
        for p in m.parameters():
            p.requires_grad = False

    return x3d, aasist, hsemo, crnn


# ══════════════════════════════════════════════════════════════════════════════
# 5. 추론 함수
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_one_sample(x3d_v, aasist_a, hsemo_f, crnn_s,
                     x3d, aasist, hsemo, crnn, device):
    """
    배치 1개에 대해 4개 모델 확률 동시 추출.
    각 모델 추론 후 즉시 텐서 삭제로 메모리 누수 방지.
    """
    # X3D
    v_in = x3d_v.unsqueeze(0).to(device, non_blocking=False)
    v_logit = x3d(v_in)
    p_v_art = torch.sigmoid(v_logit).item()
    del v_in, v_logit

    # AASIST
    a_in = aasist_a.unsqueeze(0).to(device, non_blocking=False)
    _, a_out = aasist(a_in)
    p_a_art = torch.softmax(a_out, dim=1)[0, 1].item()
    del a_in, a_out

    # HSEmotion
    hf_in = hsemo_f.unsqueeze(0).to(device, non_blocking=False)
    e_logit, _ = hsemo(hf_in)
    p_v_emo = torch.sigmoid(e_logit).item()
    del hf_in, e_logit

    # CRNN
    cs_in = crnn_s.unsqueeze(0).to(device, non_blocking=False)
    c_logit, _ = crnn(cs_in)
    p_a_emo = torch.sigmoid(c_logit).item()
    del cs_in, c_logit

    return p_v_art, p_a_art, p_v_emo, p_a_emo


# ══════════════════════════════════════════════════════════════════════════════
# 6. Fusion 전략
# ══════════════════════════════════════════════════════════════════════════════
def prob_or(*probs):
    result = np.ones_like(probs[0])
    for p in probs:
        result *= (1.0 - p)
    return 1.0 - result


def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs > threshold).astype(int)
    labels_int = labels.astype(int)
    try:
        auc = roc_auc_score(labels, probs) * 100
    except Exception:
        auc = 0.0
    acc  = accuracy_score(labels_int, preds) * 100
    prec = precision_score(labels_int, preds, zero_division=0) * 100
    rec  = recall_score(labels_int, preds, zero_division=0) * 100
    f1   = f1_score(labels_int, preds, zero_division=0) * 100
    cm = confusion_matrix(labels_int, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return dict(auc=auc, acc=acc, precision=prec, recall=rec, f1=f1,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


# ══════════════════════════════════════════════════════════════════════════════
# 7. Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    CFG = dict(
        BASE_DIR        = "PolyGlotFake",
        JSON_DIR        = "PolyGlotFake/json_file",
        REPORT_DIR      = "polyglotfake_report",

        # 체크포인트
        X3D_CKPT        = "x3d_model_best_final.pth",
        AASIST_CKPT     = "aasist_model_best_final.pth",
        AASIST_CONFIG   = "./aasist/config/AASIST.conf",
        HSEMO_CKPT      = "emotion_flow_lite_best.pth",
        CRNN_CKPT       = "audio_flow_deepfake_best.pth",
        CRNN_PRETRAINED = "audio_emotion_crnn_best.pth",

        # 추론 설정
        BATCH_SIZE      = 1,    # 4개 모델 동시 처리라 메모리 안전
        NUM_WORKERS     = 0,    # 워커 없이 메인 스레드 처리 (가장 안정적)
        SAVE_EVERY      = 500,  # 500개마다 중간 저장
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    os.makedirs(CFG['REPORT_DIR'], exist_ok=True)

    # ── 메타데이터 로드 ─────────────────────────────────────────────
    print("\n📂 PolyGlotFake 메타데이터 로드 중...")
    df = build_polyglotfake_df(CFG['JSON_DIR'])
    print(f"   총 {len(df)}개 영상 "
          f"(Real {(df['video_label']==0).sum()} / "
          f"Fake {(df['video_label']==1).sum()})")
    print(f"   언어: {sorted(df['lang'].unique())}")
    print(f"   TTS: {sorted(df['tts'].unique())}")
    print(f"   Sync: {sorted(df['sync'].unique())}")

    # ── 모델 로드 ────────────────────────────────────────────────────
    ckpt_paths = {
        'x3d':              CFG['X3D_CKPT'],
        'aasist':           CFG['AASIST_CKPT'],
        'aasist_config':    CFG['AASIST_CONFIG'],
        'hsemo':            CFG['HSEMO_CKPT'],
        'crnn':             CFG['CRNN_CKPT'],
        'crnn_pretrained':  CFG['CRNN_PRETRAINED'],
    }
    for name, path in ckpt_paths.items():
        if not os.path.exists(path):
            print(f"❌ 체크포인트 없음: {name} → {path}")
            sys.exit(1)

    x3d, aasist, hsemo, crnn = load_all_models(ckpt_paths, device)

    # ── 중간 저장 파일 확인 (재개 지원) ─────────────────────────────
    checkpoint_csv = os.path.join(CFG['REPORT_DIR'], "polyglotfake_predictions.csv")
    if os.path.exists(checkpoint_csv):
        print(f"\n♻️  기존 추론 결과 발견: {checkpoint_csv}")
        prev_df = pd.read_csv(checkpoint_csv)
        done_files = set(prev_df['filename'].tolist())
        print(f"   {len(done_files)}개 완료, 나머지부터 재개")
        results = prev_df.to_dict('records')
        df_remain = df[~df['filename'].isin(done_files)].reset_index(drop=True)
    else:
        results = []
        df_remain = df

    # ── DataLoader ──────────────────────────────────────────────────
    dataset = PolyGlotFakeDataset(df_remain, CFG['BASE_DIR'])
    # NUM_WORKERS=0이면 prefetch_factor, persistent_workers 사용 불가
    loader_kwargs = dict(
        batch_size  = CFG['BATCH_SIZE'],
        shuffle     = False,
        num_workers = CFG['NUM_WORKERS'],
        pin_memory  = False,
    )
    if CFG['NUM_WORKERS'] > 0:
        loader_kwargs['prefetch_factor']    = 2
        loader_kwargs['persistent_workers'] = False
        loader_kwargs['timeout']            = 300
    loader = DataLoader(dataset, **loader_kwargs)

    # ── 추론 시작 ────────────────────────────────────────────────────
    print(f"\n🚀 Zero-shot 추론 시작 (남은 {len(df_remain)}개)\n")
    t_start = time.time()

    for batch_idx, batch in enumerate(loader):
        x3d_v, aasist_a, hsemo_f, crnn_s, label, sample_idx = batch
        x3d_v    = x3d_v.squeeze(0)
        aasist_a = aasist_a.squeeze(0)
        hsemo_f  = hsemo_f.squeeze(0)
        crnn_s   = crnn_s.squeeze(0)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            p_v_art, p_a_art, p_v_emo, p_a_emo = infer_one_sample(
                x3d_v, aasist_a, hsemo_f, crnn_s,
                x3d, aasist, hsemo, crnn, device
            )

        idx = sample_idx.item()
        meta = df_remain.iloc[idx]
        results.append({
            'filename':         meta['filename'],
            'video_label':      int(meta['video_label']),
            'lang':             meta['lang'],
            'target_lang':      meta['target_lang'],
            'tts':              meta['tts'],
            'sync':             meta['sync'],
            'p_v_artifact':     round(p_v_art * 100, 2),
            'p_a_artifact':     round(p_a_art * 100, 2),
            'p_v_emotion':      round(p_v_emo * 100, 2),
            'p_a_emotion':      round(p_a_emo * 100, 2),
        })

        if (batch_idx + 1) % 50 == 0:
            import gc
            elapsed = time.time() - t_start
            speed = (batch_idx + 1) / elapsed
            eta = (len(df_remain) - batch_idx - 1) / speed
            # 가비지 컬렉션 + GPU 캐시 정리
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            # 시스템 메모리 사용량 확인
            try:
                import psutil
                ram_used = psutil.virtual_memory().used / 1024**3
                ram_total = psutil.virtual_memory().total / 1024**3
                ram_str = f"RAM: {ram_used:.1f}/{ram_total:.1f}GB"
            except ImportError:
                ram_str = ""
            print(f"  [{batch_idx+1:5d}/{len(df_remain)}] "
                  f"속도: {speed:.1f}/s  ETA: {eta/60:.1f}분  "
                  f"GPU: {torch.cuda.memory_reserved()/1024**3:.1f}GB  {ram_str}")

        # 중간 저장
        if (batch_idx + 1) % CFG['SAVE_EVERY'] == 0:
            pd.DataFrame(results).to_csv(
                checkpoint_csv, index=False, encoding='utf-8-sig'
            )

    # 최종 저장
    df_out = pd.DataFrame(results)
    df_out.to_csv(checkpoint_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 전체 추론 결과: {checkpoint_csv}")
    print(f"⏱  총 소요: {(time.time()-t_start)/60:.1f}분")

    # ══════════════════════════════════════════════════════════════════
    # 종합 분석
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📊 종합 분석 시작")
    print("=" * 60)

    labels  = df_out['video_label'].values.astype(float)
    p_v_art = df_out['p_v_artifact'].values / 100.0
    p_a_art = df_out['p_a_artifact'].values / 100.0
    p_v_emo = df_out['p_v_emotion'].values  / 100.0
    p_a_emo = df_out['p_a_emotion'].values  / 100.0

    art_or  = prob_or(p_v_art, p_a_art)
    emo_or  = prob_or(p_v_emo, p_a_emo)
    fusion_strategies = {
        '비디오 아티팩트 단독':   p_v_art,
        '오디오 아티팩트 단독':   p_a_art,
        '비디오 감정 단독':       p_v_emo,
        '오디오 감정 단독':       p_a_emo,
        '아티팩트 OR':            art_or,
        '감정 OR':                emo_or,
        '🌟 4방향 OR (제출 그림)': prob_or(art_or, emo_or),
        '4방향 Mean':             (p_v_art + p_a_art + p_v_emo + p_a_emo) / 4,
        'Weighted (art=0.7)':     0.7 * art_or + 0.3 * emo_or,
    }

    # ── 1. 전체 지표 ────────────────────────────────────────────────
    print("\n[1] 전체 데이터 평가")
    print("-" * 60)
    summary_rows = []
    for name, probs in fusion_strategies.items():
        m = compute_metrics(probs, labels)
        summary_rows.append({'strategy': name, **m})
        print(f"  {name:<30} AUC={m['auc']:5.2f}% "
              f"Acc={m['acc']:5.2f}%  P={m['precision']:5.2f}% "
              f"R={m['recall']:5.2f}%  F1={m['f1']:5.2f}%")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(CFG['REPORT_DIR'], "overall_metrics.csv"),
        index=False, encoding='utf-8-sig'
    )

    # ── 2. 립싱크 기법별 ─────────────────────────────────────────────
    print("\n[2] 립싱크 기법별 (Fake만 대상)")
    print("-" * 60)
    fake_df = df_out[df_out['video_label'] == 1]
    sync_rows = []
    for sync in fake_df['sync'].unique():
        sub = fake_df[fake_df['sync'] == sync]
        sub_idx = sub.index.values
        # 같은 인덱스의 Fake와 모든 Real을 결합해서 평가
        combined_idx = np.concatenate([
            np.where(df_out['video_label'] == 0)[0],
            sub_idx
        ])
        for name, probs in [('🌟 4방향 OR', fusion_strategies['🌟 4방향 OR (제출 그림)']),
                            ('4방향 Mean', fusion_strategies['4방향 Mean'])]:
            m = compute_metrics(probs[combined_idx], labels[combined_idx])
            sync_rows.append({
                'sync': sync, 'fake_count': len(sub),
                'strategy': name, **m
            })
            print(f"  [{sync:18s}] ({len(sub):5d}개) {name:<15} "
                  f"AUC={m['auc']:5.2f}%  F1={m['f1']:5.2f}%")

    pd.DataFrame(sync_rows).to_csv(
        os.path.join(CFG['REPORT_DIR'], "by_sync.csv"),
        index=False, encoding='utf-8-sig'
    )

    # ── 3. TTS 기법별 ────────────────────────────────────────────────
    print("\n[3] TTS 기법별 (Fake만 대상)")
    print("-" * 60)
    tts_rows = []
    for tts in fake_df['tts'].unique():
        if tts == 'real': continue
        sub = fake_df[fake_df['tts'] == tts]
        combined_idx = np.concatenate([
            np.where(df_out['video_label'] == 0)[0],
            sub.index.values
        ])
        for name, probs in [('🌟 4방향 OR', fusion_strategies['🌟 4방향 OR (제출 그림)']),
                            ('4방향 Mean', fusion_strategies['4방향 Mean'])]:
            m = compute_metrics(probs[combined_idx], labels[combined_idx])
            tts_rows.append({
                'tts': tts, 'fake_count': len(sub),
                'strategy': name, **m
            })
            print(f"  [{tts:12s}] ({len(sub):5d}개) {name:<15} "
                  f"AUC={m['auc']:5.2f}%  F1={m['f1']:5.2f}%")

    pd.DataFrame(tts_rows).to_csv(
        os.path.join(CFG['REPORT_DIR'], "by_tts.csv"),
        index=False, encoding='utf-8-sig'
    )

    # ── 4. 언어별 ────────────────────────────────────────────────────
    print("\n[4] 언어별 (target_lang 기준)")
    print("-" * 60)
    lang_rows = []
    for lang in sorted(df_out['target_lang'].unique()):
        sub = df_out[df_out['target_lang'] == lang]
        for name in ['🌟 4방향 OR (제출 그림)', '4방향 Mean']:
            probs = fusion_strategies[name]
            m = compute_metrics(probs[sub.index.values],
                                labels[sub.index.values])
            n_real = (sub['video_label'] == 0).sum()
            n_fake = (sub['video_label'] == 1).sum()
            lang_rows.append({
                'target_lang': lang,
                'real': int(n_real), 'fake': int(n_fake),
                'strategy': name, **m
            })
            print(f"  [{lang}] (R={n_real:3d} F={n_fake:5d}) {name:<25} "
                  f"AUC={m['auc']:5.2f}%  F1={m['f1']:5.2f}%")

    pd.DataFrame(lang_rows).to_csv(
        os.path.join(CFG['REPORT_DIR'], "by_language.csv"),
        index=False, encoding='utf-8-sig'
    )

    # ── 5. FakeAVCeleb 비교 ─────────────────────────────────────────
    print("\n[5] FakeAVCeleb vs PolyGlotFake 비교")
    print("-" * 60)
    print(f"  데이터셋          AUC      Acc      F1")
    print(f"  ────────────────  ───────  ───────  ───────")
    print(f"  FakeAVCeleb       99.99%   99.70%   99.70%   (학습 도메인)")
    pgf_4or = next(r for r in summary_rows if r['strategy'] == '🌟 4방향 OR (제출 그림)')
    print(f"  PolyGlotFake      {pgf_4or['auc']:5.2f}%   "
          f"{pgf_4or['acc']:5.2f}%   {pgf_4or['f1']:5.2f}%   (Zero-shot)")
    print(f"\n  도메인 갭(AUC): {99.99 - pgf_4or['auc']:+.2f}%p")

    print("\n" + "=" * 60)
    print(f"✅ Zero-shot 평가 완료")
    print(f"   결과 폴더: {CFG['REPORT_DIR']}")
    print("=" * 60)


if __name__ == "__main__":
    main()