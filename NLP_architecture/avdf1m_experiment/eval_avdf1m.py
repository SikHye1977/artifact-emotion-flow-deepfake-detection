"""
eval_avdf1m.py
─────────────────────────────────────────────────────────────────────
AVDF1M In-domain 평가
학습: AVDF1M train → x3d_avdf1m_best.pth, aasist_avdf1m_best.pth
평가: AVDF1M val  (57,340개 전체)

최종 fusion 레이블:
  fake (1): audio_modified, visual_modified, both_modified
  real (0): real

모달리티별 학습 레이블:
  X3D:    visual_modified, both_modified = fake
  AASIST: audio_modified,  both_modified = fake
  NLP:    audio_modified,  both_modified = fake (기존 token_nlp_best.pth)
"""

import os, sys, json, time, functools, tempfile, subprocess
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import av, whisper
import scipy.io.wavfile as wavfile
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              f1_score, confusion_matrix)
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

torch.load = functools.partial(torch.load, weights_only=False)

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as _Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = _Ftv

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

BASE_APP = os.path.expanduser("~/hsh/AIApplication")
sys.path.insert(0, BASE_APP)
from aasist.models.AASIST import Model as AASISTModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

RESULT_PATH = os.path.join(RESULTS_DIR, "avdf1m_indomain_results.json")
NLP_CKPT    = os.path.join(BASE_APP, "NLP_architecture/token_nlp_best.pth")

# ── 전처리 ────────────────────────────────────────────────────────
def rescale(x): return x / 255.0
def to_tc(x):   return x.permute(1,0,2,3)
def to_ct(x):   return x.permute(1,0,2,3)

x3d_transform = Compose([
    UniformTemporalSubsample(16), Lambda(rescale),
    Lambda(to_tc), Normalize([0.45,0.45,0.45],[0.225,0.225,0.225]),
    Lambda(to_ct), ShortSideScale(256), Resize((224,224))
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
        return torch.from_numpy(np.stack(frames)).permute(3,0,1,2).float()
    except: return None

def load_audio(path, sr=16000, max_len=64000):
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
        wav = torch.from_numpy(np.concatenate(frames)).unsqueeze(0)
        if orig_sr != sr:
            wav = torchaudio.transforms.Resample(orig_sr, sr)(wav)
        if wav.shape[1] > max_len: wav = wav[:,:max_len]
        else: wav = torch.nn.functional.pad(wav,(0,max_len-wav.shape[1]))
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
        wav = np.concatenate(frames)
        peak = np.abs(wav).max()
        if peak > 0: wav = wav / peak * 0.95
        wavfile.write(out_wav+".tmp.wav", sr, (wav*32767).astype(np.int16))
        subprocess.run(["ffmpeg","-y","-i",out_wav+".tmp.wav",
                        "-ar","16000","-ac","1",out_wav],
                       capture_output=True, timeout=30)
        os.remove(out_wav+".tmp.wav")
        return True
    except: return False

# ── 모델 정의 ─────────────────────────────────────────────────────
class TokenNLPClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", dropout=0.3):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.token_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.config.hidden_size, 1))
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        return self.token_head(out.last_hidden_state).squeeze(-1)

def load_x3d(ckpt_path):
    model = x3d_m(pretrained=False)
    model.blocks[5].proj       = nn.Linear(2048, 1)
    model.blocks[5].activation = nn.Identity()
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model.to(DEVICE).eval()

def load_aasist(ckpt_path):
    with open(AASIST_CFG) as f: cfg = json.load(f)
    model = AASISTModel(cfg["model_config"])
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model.to(DEVICE).eval()

def load_nlp(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = TokenNLPClassifier()
    model.load_state_dict(ckpt["state_dict"])
    tok   = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return model.to(DEVICE).eval(), tok

# ── 추론 ─────────────────────────────────────────────────────────
@torch.no_grad()
def infer_x3d(model, path):
    v = load_video(path)
    if v is None: return None
    return torch.sigmoid(
        model(x3d_transform(v).unsqueeze(0).to(DEVICE))).item()

@torch.no_grad()
def infer_aasist(model, path):
    a = load_audio(path)
    if a is None: return None
    _, out = model(a.unsqueeze(0).to(DEVICE))
    return torch.softmax(out, dim=1)[0,1].item()

@torch.no_grad()
def infer_nlp(nlp_model, tokenizer, whisper_model, path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    ok = extract_audio_16k(path, tmp_wav)
    if not ok:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)
        return None, None
    try:
        result     = whisper_model.transcribe(tmp_wav, task="transcribe",
                                               verbose=False)
        transcript = result["text"].strip()
        lang       = result["language"]
    except: return None, None
    finally:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)

    # AVDF1M은 영어 화자 → 언어 불일치 없음, classifier 사용
    if not transcript:
        return 0.5, lang

    enc  = tokenizer(transcript, return_tensors="pt", truncation=True,
                     max_length=128, padding="max_length")
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    logits = nlp_model(ids, mask)
    probs  = torch.sigmoid(logits)
    topk   = min(NLP_CFG["topk"], probs.shape[1])
    st     = probs.topk(topk, dim=1).values.mean().item()
    return st, lang

def prob_or3(a,b,c): return 1-(1-a)*(1-b)*(1-c)

# ── 샘플 구성 ─────────────────────────────────────────────────────
print("메타데이터 로드 중...")
with open(AVDF1M_VAL_META) as f:
    meta = json.load(f)

samples = []
for m in meta:
    path = os.path.join(AVDF1M_VAL_ROOT, m["file"])
    if not os.path.exists(path): continue

    # 최종 fusion 레이블 (모든 조작 = fake)
    fusion_label = 0 if m["modify_type"] == "real" else 1

    # 모달리티별 레이블
    x3d_label    = 1 if m["modify_type"] in \
                   ("visual_modified","both_modified") else 0
    aasist_label = 1 if m["modify_type"] in \
                   ("audio_modified","both_modified") else 0
    nlp_label    = 1 if m["modify_type"] in \
                   ("audio_modified","both_modified") else 0

    samples.append({
        "path":          path,
        "file":          m["file"],
        "modify_type":   m["modify_type"],
        "fusion_label":  fusion_label,
        "x3d_label":     x3d_label,
        "aasist_label":  aasist_label,
        "nlp_label":     nlp_label,
        "audio_fake_segs": m.get("audio_fake_segments",[]),
    })

from collections import Counter
dist = Counter(s["modify_type"] for s in samples)
print(f"전체: {len(samples):,}개")
for t,n in dist.items():
    fl = 0 if t=="real" else 1
    print(f"  {t:<20}: {n:,}개  (fusion_label={fl})")

# ── 모델 로드 ─────────────────────────────────────────────────────
print("\n모델 로드 중...")
for name, path in [("X3D",    X3D_SAVE_PATH),
                   ("AASIST", AASIST_SAVE_PATH),
                   ("NLP",    NLP_CKPT)]:
    if not os.path.exists(path):
        print(f"  ❌ {name} 가중치 없음: {path}")
        sys.exit(1)
    print(f"  ✅ {name}: {path}")

x3d_model            = load_x3d(X3D_SAVE_PATH)
aasist_model         = load_aasist(AASIST_SAVE_PATH)
nlp_model, tokenizer = load_nlp(NLP_CKPT)
whisper_model        = whisper.load_model("medium")
print("로드 완료\n")

# ── 중단 재개 ─────────────────────────────────────────────────────
if os.path.exists(RESULT_PATH):
    with open(RESULT_PATH) as f:
        results = json.load(f)
    done = {r["file"] for r in results}
    todo = [s for s in samples if s["file"] not in done]
    print(f"기존 {len(results)}개 → 남은 {len(todo)}개\n")
else:
    results, todo = [], samples

# ── 메인 루프 ─────────────────────────────────────────────────────
t_start = time.time()
for idx, s in enumerate(tqdm(todo, desc="AVDF1M In-domain")):
    sv = infer_x3d(x3d_model, s["path"])
    sa = infer_aasist(aasist_model, s["path"])
    st, lang = infer_nlp(nlp_model, tokenizer, whisper_model, s["path"])

    if sv is None or sa is None or st is None: continue

    results.append({
        "file":         s["file"],
        "modify_type":  s["modify_type"],
        "fusion_label": s["fusion_label"],
        "x3d_label":    s["x3d_label"],
        "aasist_label": s["aasist_label"],
        "nlp_label":    s["nlp_label"],
        "sv":           round(sv, 4),
        "sa":           round(sa, 4),
        "st":           round(st, 4),
        "final":        round(prob_or3(sv, sa, st), 4),
        "whisper_lang": lang,
    })

    if (idx+1) % 500 == 0:
        with open(RESULT_PATH,"w") as f:
            json.dump(results, f, ensure_ascii=False)
        elapsed = time.time()-t_start
        eta     = elapsed/(idx+1)*(len(todo)-idx-1)
        print(f"\n[{idx+1}/{len(todo)}] "
              f"{elapsed/3600:.1f}h 경과 | 남은 {eta/3600:.1f}h")

with open(RESULT_PATH,"w") as f:
    json.dump(results, f, ensure_ascii=False)
print(f"\n추론 완료: {len(results)}개")

# ── 성능 평가 ─────────────────────────────────────────────────────
def met(labels, probs, name):
    preds = (probs>0.5).astype(int)
    auc = roc_auc_score(labels, probs)
    f1  = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    cm  = confusion_matrix(labels, preds)
    tn,fp,fn,tp = cm.ravel()
    print(f"  {name:<30} AUC={auc*100:6.2f}%  "
          f"F1={f1*100:6.2f}%  ACC={acc*100:6.2f}%  "
          f"TN={tn} FP={fp} FN={fn} TP={tp}")
    return auc*100

print(f"\n{'='*70}")
print("AVDF1M In-domain 평가 결과")
print(f"{'='*70}")

# 배열 추출
sv_arr  = np.array([r["sv"]    for r in results])
sa_arr  = np.array([r["sa"]    for r in results])
st_arr  = np.array([r["st"]    for r in results])
fin_arr = np.array([r["final"] for r in results])
fl_arr  = np.array([r["fusion_label"] for r in results])
xl_arr  = np.array([r["x3d_label"]    for r in results])
al_arr  = np.array([r["aasist_label"] for r in results])
nl_arr  = np.array([r["nlp_label"]    for r in results])

# ── 각 모달리티 자체 도메인 평가 ──────────────────────────────────
print("\n[모달리티별 자체 도메인 평가]")
print("  (각 모델이 학습한 레이블 기준으로 평가)")
met(xl_arr, sv_arr, "X3D    (visual 조작 탐지)")
met(al_arr, sa_arr, "AASIST (audio  조작 탐지)")
met(nl_arr, st_arr, "NLP    (audio  조작 탐지)")

# ── Fusion 평가 (최종 레이블 기준) ────────────────────────────────
print(f"\n[Fusion 평가 (fake=모든조작, real=real만)]")
met(fl_arr, sv_arr,                      "X3D 단독")
met(fl_arr, sa_arr,                      "AASIST 단독")
met(fl_arr, st_arr,                      "NLP 단독")
met(fl_arr, 1-(1-sv_arr)*(1-sa_arr),    "X3D+AASIST OR")
met(fl_arr, 1-(1-sv_arr)*(1-st_arr),    "X3D+NLP OR")
met(fl_arr, 1-(1-sa_arr)*(1-st_arr),    "AASIST+NLP OR")
met(fl_arr, fin_arr,                     "3방향 OR (최종)")

# ── modify_type별 상세 분석 ───────────────────────────────────────
print(f"\n[modify_type별 3방향 OR 분석]")
print(f"  {'type':<20} {'n':>6} {'AUC':>8} {'F1':>8} {'ACC':>8}")
print(f"  {'-'*52}")

# real vs 전체
for mtype in ["real","audio_modified","visual_modified","both_modified"]:
    idx = [i for i,r in enumerate(results) if r["modify_type"]==mtype]
    if not idx: continue
    n = len(idx)

    if mtype == "real":
        # real: 모두 label=0, 탐지율=FPR (낮을수록 좋음)
        fp_rate = sum(1 for i in idx if fin_arr[i]>0.5) / n
        print(f"  {mtype:<20} {n:>6}  FPR={fp_rate*100:.2f}% "
              f"(real 중 fake로 오판)")
    else:
        # fake type별: AUC/F1 계산 (real + 해당 type)
        real_idx = [i for i,r in enumerate(results) if r["modify_type"]=="real"]
        comb_idx = real_idx + idx
        lab = fl_arr[comb_idx]
        prb = fin_arr[comb_idx]
        if len(set(lab)) < 2: continue
        auc = roc_auc_score(lab, prb)
        f1  = f1_score(lab, (prb>0.5).astype(int), zero_division=0)
        acc = accuracy_score(lab, (prb>0.5).astype(int))
        print(f"  {mtype:<20} {n:>6}  "
              f"AUC={auc*100:.2f}%  F1={f1*100:.2f}%  ACC={acc*100:.2f}%")

# 저장
summary = {
    "total": len(results),
    "x3d_indomain_auc":    roc_auc_score(xl_arr, sv_arr)*100,
    "aasist_indomain_auc": roc_auc_score(al_arr, sa_arr)*100,
    "nlp_indomain_auc":    roc_auc_score(nl_arr, st_arr)*100,
    "fusion_3way_auc":     roc_auc_score(fl_arr, fin_arr)*100,
}
with open(os.path.join(RESULTS_DIR,"avdf1m_indomain_summary.json"),"w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ 평가 완료")
print(f"   결과: {RESULT_PATH}")
elapsed = time.time()-t_start
print(f"   소요: {elapsed/3600:.1f}h")
