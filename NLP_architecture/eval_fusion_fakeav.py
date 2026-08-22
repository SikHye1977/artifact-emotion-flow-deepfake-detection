"""
eval_fusion_fakeav.py  ── Option A: FakeAVCeleb fusion 평가
─────────────────────────────────────────────────────────────
[아키텍처]
  X3D_m   → score_v  (비디오 아티팩트)
  AASIST  → score_a  (오디오 아티팩트)
  NLP     → score_t  (transcript 분류)

[Fusion]
  Final = 1 − (1−score_v)(1−score_a)(1−score_t)   확률적 OR

[데이터]
  FakeAVCeleb val set (in-domain)
  → FakeAVCeleb/dataset/FakeAVCeleb_v1.2/ 구조 탐색
"""

import os, sys, json, time, functools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import av
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              f1_score, confusion_matrix)
from transformers import AutoTokenizer, AutoModel
from pytorchvideo.models.hub import x3d_m
# torchvision 하위 호환 패치 (pytorchvideo import 전에 반드시 위치)
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as _Ftv
    import sys
    sys.modules["torchvision.transforms.functional_tensor"] = _Ftv

from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize
import whisper, subprocess, tempfile
import scipy.io.wavfile as wavfile

torch.load = functools.partial(torch.load, weights_only=False)

# torchvision 하위 호환 패치
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as _Ftv
    import sys
    sys.modules["torchvision.transforms.functional_tensor"] = _Ftv

# ── 경로 설정 ─────────────────────────────────────────────────────
BASE     = os.path.expanduser("~/hsh/AIApplication")
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

X3D_CKPT    = os.path.join(BASE, "x3d_model_best_final.pth")
AASIST_CKPT = os.path.join(BASE, "aasist_model_best_final.pth")
AASIST_CFG  = os.path.join(BASE, "aasist/config/AASIST.conf")
NLP_CKPT    = os.path.join(OUT_DIR, "nlp_best.pth")
FAV_ROOT    = os.path.join(BASE, "FakeAVCeleb/dataset/FakeAVCeleb_v1.2")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════
# 1. FakeAVCeleb 데이터 구조 탐색
# ══════════════════════════════════════════════════════════════════
def build_fakeav_samples(root):
    """
    FakeAVCeleb_v1.2 폴더 구조:
      RealVideo-RealAudio/  → real
      FakeVideo-RealAudio/  → fake
      RealVideo-FakeAudio/  → fake
      FakeVideo-FakeAudio/  → fake
    """
    samples = []
    label_map = {
        "RealVideo-RealAudio": 0,
        "FakeVideo-RealAudio": 1,
        "RealVideo-FakeAudio": 1,
        "FakeVideo-FakeAudio": 1,
    }
    for folder, label in label_map.items():
        folder_path = os.path.join(root, folder)
        if not os.path.exists(folder_path):
            continue
        for dirpath, _, files in os.walk(folder_path):
            for f in files:
                if f.endswith(".mp4"):
                    samples.append({
                        "path":   os.path.join(dirpath, f),
                        "label":  label,
                        "folder": folder
                    })
    return samples

# ══════════════════════════════════════════════════════════════════
# 2. 전처리
# ══════════════════════════════════════════════════════════════════
def rescale(x): return x / 255.0
def to_tc(x):   return x.permute(1, 0, 2, 3)
def to_ct(x):   return x.permute(1, 0, 2, 3)

x3d_transform = Compose([
    UniformTemporalSubsample(16),
    Lambda(rescale),
    Lambda(to_tc),
    Normalize([0.45,0.45,0.45],[0.225,0.225,0.225]),
    Lambda(to_ct),
    ShortSideScale(256),
    Resize((224,224))
])

def load_video(path):
    try:
        container = av.open(path)
        frames = [f.to_rgb().to_ndarray() for f in container.decode(video=0)]
        container.close()
        if len(frames) < 16: return None
        if len(frames) > 128:
            idx = np.linspace(0, len(frames)-1, 128, dtype=int)
            frames = [frames[i] for i in idx]
        v = np.stack(frames)
        return torch.from_numpy(v).permute(3,0,1,2).float()
    except: return None

def load_audio_aasist(path, sr=16000, max_len=64000):
    try:
        container = av.open(path)
        if not container.streams.audio:
            container.close(); return None
        orig_sr = container.streams.audio[0].rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.ndim > 1: arr = arr.mean(axis=0)
            frames.append(arr)
        container.close()
        if not frames: return None
        wav = np.concatenate(frames)
        wav = torch.from_numpy(wav).unsqueeze(0)
        if orig_sr != sr:
            wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
        if wav.shape[1] > max_len:
            wav = wav[:, :max_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, max_len-wav.shape[1]))
        return wav.squeeze()
    except: return None

def extract_audio_16k(video_path, out_wav):
    try:
        container = av.open(video_path)
        if not container.streams.audio:
            container.close(); return False
        sr = container.streams.audio[0].rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.ndim > 1: arr = arr.mean(axis=0)
            frames.append(arr)
        container.close()
        if not frames: return False
        wav  = np.concatenate(frames)
        peak = np.abs(wav).max()
        if peak > 0: wav = wav / peak * 0.95
        wavfile.write(out_wav+".tmp.wav", sr, (wav*32767).astype(np.int16))
        subprocess.run(["ffmpeg","-y","-i",out_wav+".tmp.wav",
                        "-ar","16000","-ac","1",out_wav],
                       capture_output=True, timeout=30)
        os.remove(out_wav+".tmp.wav")
        return True
    except: return False

# ══════════════════════════════════════════════════════════════════
# 3. 모델 정의 및 로드
# ══════════════════════════════════════════════════════════════════

# ── X3D ──────────────────────────────────────────────────────────
def load_x3d(ckpt):
    model = x3d_m(pretrained=False)
    model.blocks[5].proj       = nn.Linear(2048, 1)
    model.blocks[5].activation = nn.Identity()
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model.to(DEVICE).eval()

# ── AASIST ───────────────────────────────────────────────────────
sys.path.insert(0, BASE)
from aasist.models.AASIST import Model as AASISTModel

def load_aasist(ckpt, cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = AASISTModel(cfg['model_config'])
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model.to(DEVICE).eval()

# ── NLP Classifier ───────────────────────────────────────────────
class NLPClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, input_ids, attention_mask):
        out   = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls   = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)

def load_nlp(ckpt):
    ckpt_data = torch.load(ckpt, map_location=DEVICE)
    model = NLPClassifier()
    model.load_state_dict(ckpt_data['state_dict'])
    return model.to(DEVICE).eval()

# ══════════════════════════════════════════════════════════════════
# 4. 추론
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_x3d(model, path):
    v = load_video(path)
    if v is None: return None
    v = x3d_transform(v).unsqueeze(0).to(DEVICE)
    logit = model(v)
    return torch.sigmoid(logit).item()

@torch.no_grad()
def infer_aasist(model, path):
    a = load_audio_aasist(path)
    if a is None: return None
    a = a.unsqueeze(0).to(DEVICE)
    _, out = model(a)
    return torch.softmax(out, dim=1)[0, 1].item()

@torch.no_grad()
def infer_nlp(model, tokenizer, whisper_model, path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    ok = extract_audio_16k(path, tmp_wav)
    if not ok:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)
        return None, None
    try:
        result = whisper_model.transcribe(tmp_wav, task="transcribe", verbose=False)
        transcript = result["text"].strip()
        lang       = result["language"]
    except:
        return None, None
    finally:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)

    # 언어 불일치 (PGF용 — FakeAV에선 항상 en이므로 작동 안 함)
    if lang != "en":
        return 0.92, lang

    if not transcript:
        return 0.5, lang

    enc = tokenizer(transcript, return_tensors="pt",
                    truncation=True, max_length=128,
                    padding="max_length")
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    logit = model(ids, mask)
    score = torch.sigmoid(logit).item()
    return score, lang

def prob_or(*scores):
    result = 1.0
    for s in scores:
        result *= (1.0 - s)
    return 1.0 - result

# ══════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════
def main():
    # 데이터 구조 탐색
    print("FakeAVCeleb 샘플 탐색 중...")
    samples = build_fakeav_samples(FAV_ROOT)
    if not samples:
        print(f"❌ 샘플 없음: {FAV_ROOT}")
        print("폴더 구조 확인:")
        os.system(f"ls {FAV_ROOT} 2>/dev/null || ls {os.path.dirname(FAV_ROOT)}")
        sys.exit(1)

    from collections import Counter
    dist = Counter(s['folder'] for s in samples)
    print(f"총 {len(samples)}개")
    for k,v in dist.items():
        print(f"  {k}: {v}개")

    # 모델 로드
    print("\n모델 로드 중...")
    x3d_model    = load_x3d(X3D_CKPT)
    aasist_model = load_aasist(AASIST_CKPT, AASIST_CFG)
    nlp_model    = load_nlp(NLP_CKPT)
    tokenizer    = AutoTokenizer.from_pretrained("xlm-roberta-base")
    whisper_model = whisper.load_model("medium")
    print("로드 완료\n")

    # 중간 저장 지원
    RESULT_PATH = os.path.join(OUT_DIR, "fusion_fakeav_results.json")
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            results = json.load(f)
        done = {r["path"] for r in results}
        print(f"기존 {len(results)}개 발견 → 이어서 진행")
    else:
        results = []
        done    = set()

    todo = [s for s in samples if s["path"] not in done]
    print(f"남은 작업: {len(todo)}개\n")

    SAVE_EVERY = 200
    t_start    = time.time()

    for idx, s in enumerate(tqdm(todo, desc="Fusion Eval (FakeAV)")):
        sv = infer_x3d(x3d_model, s["path"])
        sa = infer_aasist(aasist_model, s["path"])
        st, lang = infer_nlp(nlp_model, tokenizer, whisper_model, s["path"])

        if sv is None or sa is None or st is None:
            continue

        final = prob_or(sv, sa, st)

        results.append({
            "path":    s["path"],
            "label":   s["label"],
            "folder":  s["folder"],
            "score_v": round(sv, 4),
            "score_a": round(sa, 4),
            "score_t": round(st, 4),
            "score_final": round(final, 4),
            "lang":    lang,
        })

        if (idx+1) % SAVE_EVERY == 0:
            with open(RESULT_PATH, "w") as f:
                json.dump(results, f, ensure_ascii=False)
            elapsed = time.time() - t_start
            print(f"\n[{idx+1}/{len(todo)}] 저장 | "
                  f"경과 {elapsed/3600:.1f}h | "
                  f"남은 예상 {elapsed/(idx+1)*(len(todo)-idx-1)/3600:.1f}h")

    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False)

    # ── 성능 평가 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("성능 평가")
    print(f"{'='*60}")

    labels  = np.array([r["label"]      for r in results])
    sv_arr  = np.array([r["score_v"]    for r in results])
    sa_arr  = np.array([r["score_a"]    for r in results])
    st_arr  = np.array([r["score_t"]    for r in results])
    fin_arr = np.array([r["score_final"] for r in results])

    def report(name, probs):
        preds = (probs > 0.5).astype(int)
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, zero_division=0)
        cm  = confusion_matrix(labels, preds)
        print(f"\n  [{name}]")
        print(f"    AUC={auc*100:.2f}%  ACC={acc*100:.2f}%  F1={f1*100:.2f}%")
        print(f"    CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
        return {"name":name, "auc":auc, "acc":acc, "f1":f1}

    rows = []
    rows.append(report("X3D 단독",          sv_arr))
    rows.append(report("AASIST 단독",       sa_arr))
    rows.append(report("NLP 단독",          st_arr))
    rows.append(report("X3D+AASIST OR",     prob_or(sv_arr, sa_arr)))
    rows.append(report("X3D+NLP OR",        prob_or(sv_arr, st_arr)))
    rows.append(report("AASIST+NLP OR",     prob_or(sa_arr, st_arr)))
    rows.append(report("3방향 OR (최종)",    fin_arr))

    # 비교 요약
    print(f"\n{'='*60}")
    print(f"  {'전략':<20} {'AUC':>8} {'ACC':>8} {'F1':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for r in rows:
        print(f"  {r['name']:<20} {r['auc']*100:>7.2f}% {r['acc']*100:>7.2f}% {r['f1']*100:>7.2f}%")

    # CSV 저장
    report_path = os.path.join(OUT_DIR, "fusion_fakeav_report.csv")
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"\n✅ 완료 | 결과: {RESULT_PATH}")
    print(f"         리포트: {report_path}")

if __name__ == "__main__":
    main()
