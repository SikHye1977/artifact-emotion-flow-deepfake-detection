"""
eval_pgf.py
AVDF1M 학습 모델 → PolyGlotFake Zero-shot 평가
FakeAV 학습 baseline과 동시 비교
"""

import os, sys, json, time, functools
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import av, whisper, subprocess, tempfile
import scipy.io.wavfile as wavfile
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
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

# ── 모델 로드 ─────────────────────────────────────────────────────
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
def infer_nlp(nlp_model, tokenizer, whisper_model, path, raw_lang):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    ok = extract_audio_16k(path, tmp_wav)
    if not ok:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)
        return None, None, False
    try:
        result     = whisper_model.transcribe(tmp_wav, task="transcribe",
                                               verbose=False)
        transcript = result["text"].strip()
        lang       = result["language"]
    except: return None, None, False
    finally:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)

    # 언어 불일치 (PGF 핵심 신호)
    if lang != raw_lang:
        return 0.92, lang, True

    if not transcript:
        return 0.5, lang, False

    enc  = tokenizer(transcript, return_tensors="pt", truncation=True,
                     max_length=128, padding="max_length")
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    logits = nlp_model(ids, mask)          # (1, L)
    probs  = torch.sigmoid(logits)
    topk   = min(NLP_CFG["topk"], probs.shape[1])
    st     = probs.topk(topk, dim=1).values.mean().item()
    return st, lang, False

def prob_or3(a,b,c): return 1-(1-a)*(1-b)*(1-c)

# ── PGF 샘플 구성 ─────────────────────────────────────────────────
with open(os.path.join(PGF_JSON_DIR,"fake_Json_file/all_fake_video.json")) as f:
    all_fake = json.load(f)
with open(os.path.join(PGF_JSON_DIR,"real_json_file/all_real_video.json")) as f:
    all_real = json.load(f)

samples = []
for v in all_fake["video"]:
    path = os.path.join(PGF_ROOT, f'fake/to_{v["target_lang"]}', v["filename"])
    if not os.path.exists(path): continue
    samples.append({"path":path,"label":1,"filename":v["filename"],
                    "raw_lang":v["raw_lang"],"tgt_lang":v["target_lang"],
                    "tts":v["tts_technique"],"sync":v["sync_tech"]})
for v in all_real["videos"]:
    path = os.path.join(PGF_ROOT, f'real/{v["lang"]}', v["filename"])
    if not os.path.exists(path): continue
    samples.append({"path":path,"label":0,"filename":v["filename"],
                    "raw_lang":v["lang"],"tgt_lang":v["lang"],
                    "tts":"real","sync":"real"})

print(f"PGF: {len(samples)}개  "
      f"(real={sum(1 for s in samples if s['label']==0)}, "
      f"fake={sum(1 for s in samples if s['label']==1)})")

# ── 평가 실행 함수 ────────────────────────────────────────────────
def run_eval(tag, x3d_ckpt, aasist_ckpt, nlp_ckpt, result_path):
    print(f"\n{'='*60}")
    print(f"[{tag}] 모델 로드 중...")

    # 가중치 존재 확인
    for name, path in [("X3D",x3d_ckpt),("AASIST",aasist_ckpt),
                       ("NLP",nlp_ckpt)]:
        if not os.path.exists(path):
            print(f"  ❌ {name} 가중치 없음: {path}")
            return None

    x3d_model    = load_x3d(x3d_ckpt)
    aasist_model = load_aasist(aasist_ckpt)
    nlp_model, tokenizer = load_nlp(nlp_ckpt)
    whisper_model = whisper.load_model("medium")
    print("  로드 완료")

    # 중단 재개
    if os.path.exists(result_path):
        with open(result_path) as f:
            results = json.load(f)
        done = {r["filename"] for r in results}
        todo = [s for s in samples if s["filename"] not in done]
        print(f"  기존 {len(results)}개 → 남은 {len(todo)}개")
    else:
        results, todo = [], samples

    t_start = time.time()
    for idx, s in enumerate(tqdm(todo, desc=f"[{tag}]")):
        sv = infer_x3d(x3d_model, s["path"])
        sa = infer_aasist(aasist_model, s["path"])
        st, lang, mismatch = infer_nlp(
            nlp_model, tokenizer, whisper_model,
            s["path"], s["raw_lang"])

        if sv is None or sa is None or st is None: continue

        results.append({
            "filename":      s["filename"],
            "label":         s["label"],
            "raw_lang":      s["raw_lang"],
            "tgt_lang":      s["tgt_lang"],
            "tts":           s["tts"],
            "sync":          s["sync"],
            "sv":            round(sv, 4),
            "sa":            round(sa, 4),
            "st":            round(st, 4),
            "final":         round(prob_or3(sv,sa,st), 4),
            "whisper_lang":  lang,
            "lang_mismatch": mismatch,
        })

        if (idx+1) % 200 == 0:
            with open(result_path,"w") as f:
                json.dump(results, f, ensure_ascii=False)
            elapsed = time.time()-t_start
            eta     = elapsed/(idx+1)*(len(todo)-idx-1)
            print(f"\n  [{idx+1}/{len(todo)}] "
                  f"{elapsed/60:.1f}분 경과 | 남은 {eta/60:.1f}분")

    with open(result_path,"w") as f:
        json.dump(results, f, ensure_ascii=False)
    return results

# ── 성능 출력 함수 ────────────────────────────────────────────────
def print_report(tag, results):
    labels  = np.array([r["label"] for r in results])
    sv_arr  = np.array([r["sv"]    for r in results])
    sa_arr  = np.array([r["sa"]    for r in results])
    st_arr  = np.array([r["st"]    for r in results])
    fin_arr = np.array([r["final"] for r in results])

    def met(name, probs):
        preds = (probs>0.5).astype(int)
        auc = roc_auc_score(labels,probs)
        f1  = f1_score(labels,preds,zero_division=0)
        acc = accuracy_score(labels,preds)
        print(f"  {name:<25} AUC={auc*100:6.2f}%  "
              f"F1={f1*100:6.2f}%  ACC={acc*100:6.2f}%")
        return auc*100

    print(f"\n{'='*60}")
    print(f"[{tag}] PolyGlotFake Zero-shot")
    print(f"{'='*60}")
    met("X3D 단독",       sv_arr)
    met("AASIST 단독",    sa_arr)
    met("NLP 단독",       st_arr)
    met("X3D+AASIST OR", 1-(1-sv_arr)*(1-sa_arr))
    met("AASIST+NLP OR", 1-(1-sa_arr)*(1-st_arr))
    met("3방향 OR (최종)", fin_arr)

    # target_lang별
    print(f"\n  [target_lang별 3방향 OR]")
    for lang in sorted(set(r["tgt_lang"] for r in results)):
        idx = [i for i,r in enumerate(results) if r["tgt_lang"]==lang]
        lab = labels[idx]; prb = fin_arr[idx]
        if len(set(lab)) < 2: continue
        auc = roc_auc_score(lab,prb)
        f1  = f1_score(lab,(prb>0.5).astype(int),zero_division=0)
        print(f"    to_{lang:<4}: AUC={auc*100:.2f}%  "
              f"F1={f1*100:.2f}%  ({len(idx)}개)")

    mismatch = sum(1 for r in results if r.get("lang_mismatch"))
    print(f"\n  언어 불일치 감지: {mismatch}/{len(results)} "
          f"({mismatch/len(results)*100:.1f}%)")

    return {
        "tag": tag,
        "x3d":    roc_auc_score(labels,sv_arr)*100,
        "aasist": roc_auc_score(labels,sa_arr)*100,
        "nlp":    roc_auc_score(labels,st_arr)*100,
        "3way":   roc_auc_score(labels,fin_arr)*100,
    }

# ── 실행 ─────────────────────────────────────────────────────────
# NLP 가중치: AVDF1M 학습된 것 공통 사용
NLP_CKPT_AVDF1M = os.path.join(BASE, "NLP_architecture/token_nlp_best.pth")

# 기존 NLP 가중치 (NLP_architecture 폴더의 것)
NLP_CKPT_OLD = os.path.join(BASE, "NLP_architecture/token_nlp_best.pth")

# Baseline: FakeAV 학습 X3D+AASIST + AVDF1M 학습 NLP
r_baseline = run_eval(
    "Baseline(FakeAV X3D+AASIST / AVDF1M NLP)",
    FAKEAV_X3D_CKPT,
    FAKEAV_AASIST_CKPT,
    NLP_CKPT_OLD,
    PGF_RESULT_FAKEAV
)

# New: AVDF1M 학습 X3D+AASIST+NLP
r_new = run_eval(
    "New(AVDF1M X3D+AASIST+NLP)",
    X3D_SAVE_PATH,
    AASIST_SAVE_PATH,
    NLP_CKPT_AVDF1M,
    PGF_RESULT_AVDF1M
)

# ── 최종 비교 요약 ────────────────────────────────────────────────
summaries = []
for tag, results in [
    ("Baseline(FakeAV)", r_baseline),
    ("New(AVDF1M)",      r_new),
]:
    if results is None: continue
    summaries.append(print_report(tag, results))

print(f"\n{'='*60}")
print("최종 비교 요약 (AUC %)")
print(f"{'='*60}")
print(f"  {'설정':<35} {'X3D':>7} {'AASIST':>7} "
      f"{'NLP':>7} {'3방향OR':>8}")
print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
for s in summaries:
    print(f"  {s['tag']:<35} {s['x3d']:>6.2f}% {s['aasist']:>6.2f}% "
          f"{s['nlp']:>6.2f}% {s['3way']:>7.2f}%")

# JSON 저장
import json
with open(os.path.join(RESULTS_DIR,"pgf_comparison_summary.json"),"w") as f:
    json.dump(summaries, f, indent=2)
print(f"\n✅ 평가 완료 | 결과: {RESULTS_DIR}")
