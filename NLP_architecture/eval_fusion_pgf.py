"""
eval_fusion_pgf.py  ── Option B: PolyGlotFake zero-shot fusion 평가
─────────────────────────────────────────────────────────────────────
X3D + AASIST + NLP → 확률적 OR
비교 baseline: 기존 X3D+AASIST OR (인수인계 문서 기준 91.10%)
"""

import os, sys, json, time, functools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import av, whisper, subprocess, tempfile
import scipy.io.wavfile as wavfile
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, accuracy_score,
                              f1_score, confusion_matrix)
from transformers import AutoTokenizer, AutoModel

torch.load = functools.partial(torch.load, weights_only=False)

# torchvision 하위 호환 패치
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as _Ftv
    sys.modules["torchvision.transforms.functional_tensor"] = _Ftv

from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize

# ── 경로 ─────────────────────────────────────────────────────────
BASE        = os.path.expanduser("~/hsh/AIApplication")
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
X3D_CKPT    = os.path.join(BASE, "x3d_model_best_final.pth")
AASIST_CKPT = os.path.join(BASE, "aasist_model_best_final.pth")
AASIST_CFG  = os.path.join(BASE, "aasist/config/AASIST.conf")
NLP_CKPT    = os.path.join(OUT_DIR, "nlp_best.pth")
PGF_ROOT    = os.path.join(BASE, "PolyGlotFake")
JSON_DIR    = os.path.join(PGF_ROOT, "json_file")
RESULT_PATH = os.path.join(OUT_DIR, "fusion_pgf_results.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════
# 1. 전처리
# ══════════════════════════════════════════════════════════════════
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
        wav = torch.from_numpy(np.concatenate(frames)).unsqueeze(0)
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

# ══════════════════════════════════════════════════════════════════
# 2. 모델
# ══════════════════════════════════════════════════════════════════
sys.path.insert(0, BASE)
from aasist.models.AASIST import Model as AASISTModel

class NLPClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.config.hidden_size, 1)
        )
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :]).squeeze(-1)

def load_models():
    print("모델 로드 중...")
    # X3D
    x3d = x3d_m(pretrained=False)
    x3d.blocks[5].proj       = nn.Linear(2048, 1)
    x3d.blocks[5].activation = nn.Identity()
    x3d.load_state_dict(torch.load(X3D_CKPT, map_location=DEVICE))
    x3d = x3d.to(DEVICE).eval(); print("  ✅ X3D_m")

    # AASIST
    with open(AASIST_CFG) as f: cfg = json.load(f)
    aasist = AASISTModel(cfg['model_config'])
    aasist.load_state_dict(torch.load(AASIST_CKPT, map_location=DEVICE))
    aasist = aasist.to(DEVICE).eval(); print("  ✅ AASIST")

    # NLP
    ckpt = torch.load(NLP_CKPT, map_location=DEVICE)
    nlp  = NLPClassifier()
    nlp.load_state_dict(ckpt['state_dict'])
    nlp  = nlp.to(DEVICE).eval()
    tok  = AutoTokenizer.from_pretrained("xlm-roberta-base")
    print("  ✅ NLP Classifier")

    wmodel = whisper.load_model("medium"); print("  ✅ Whisper medium")
    return x3d, aasist, nlp, tok, wmodel

# ══════════════════════════════════════════════════════════════════
# 3. 샘플 구성
# ══════════════════════════════════════════════════════════════════
def build_samples():
    with open(os.path.join(JSON_DIR,"fake_Json_file/all_fake_video.json")) as f:
        all_fake = json.load(f)
    with open(os.path.join(JSON_DIR,"real_json_file/all_real_video.json")) as f:
        all_real = json.load(f)

    samples = []
    for v in all_fake["video"]:
        path = os.path.join(PGF_ROOT, f'fake/to_{v["target_lang"]}', v["filename"])
        if not os.path.exists(path): continue
        samples.append({
            "path":     path,
            "label":    1,
            "filename": v["filename"],
            "raw_lang": v["raw_lang"],
            "tgt_lang": v["target_lang"],
            "tts":      v["tts_technique"],
            "sync":     v["sync_tech"],
        })
    for v in all_real["videos"]:
        path = os.path.join(PGF_ROOT, f'real/{v["lang"]}', v["filename"])
        if not os.path.exists(path): continue
        samples.append({
            "path":     path,
            "label":    0,
            "filename": v["filename"],
            "raw_lang": v["lang"],
            "tgt_lang": v["lang"],
            "tts":      "real",
            "sync":     "real",
        })
    return samples

# ══════════════════════════════════════════════════════════════════
# 4. 추론
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer(x3d, aasist, nlp, tok, wmodel, s):
    path = s["path"]

    # X3D
    v = load_video(path)
    if v is None: return None
    sv = torch.sigmoid(x3d(x3d_transform(v).unsqueeze(0).to(DEVICE))).item()

    # AASIST
    a = load_audio_aasist(path)
    if a is None: return None
    _, out = aasist(a.unsqueeze(0).to(DEVICE))
    sa = torch.softmax(out, dim=1)[0,1].item()

    # NLP (Whisper 언어 감지 + classifier)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    ok = extract_audio_16k(path, tmp_wav)
    if not ok:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)
        return None
    try:
        result     = wmodel.transcribe(tmp_wav, task="transcribe", verbose=False)
        transcript = result["text"].strip()
        lang       = result["language"]
    except:
        return None
    finally:
        if os.path.exists(tmp_wav): os.remove(tmp_wav)

    # 언어 불일치 → 즉시 0.92 (PGF 핵심 탐지)
    if lang != s["raw_lang"]:
        st = 0.92
        lang_mismatch = True
    elif not transcript:
        st = 0.5
        lang_mismatch = False
    else:
        enc  = tok(transcript, return_tensors="pt", truncation=True,
                   max_length=128, padding="max_length")
        ids  = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)
        st   = torch.sigmoid(nlp(ids, mask)).item()
        lang_mismatch = False

    final = 1 - (1-sv)*(1-sa)*(1-st)
    return {
        "sv": sv, "sa": sa, "st": st,
        "final": final,
        "whisper_lang": lang,
        "lang_mismatch": lang_mismatch,
    }

def prob_or(a, b): return 1-(1-a)*(1-b)

# ══════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════
def main():
    samples = build_samples()
    print(f"PGF 샘플: {len(samples)}개  "
          f"(real={sum(1 for s in samples if s['label']==0)}, "
          f"fake={sum(1 for s in samples if s['label']==1)})")
    print(f"예상 시간: ~{len(samples)*1.3/3600:.1f}시간\n")

    x3d, aasist, nlp, tok, wmodel = load_models()

    # 중단 재개
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as f:
            results = json.load(f)
        done = {r["filename"] for r in results}
        todo = [s for s in samples if s["filename"] not in done]
        print(f"기존 {len(results)}개 → 남은 {len(todo)}개\n")
    else:
        results, todo = [], samples

    t_start = time.time()
    for idx, s in enumerate(tqdm(todo, desc="PGF Fusion")):
        out = infer(x3d, aasist, nlp, tok, wmodel, s)
        if out is None: continue
        results.append({
            "filename":     s["filename"],
            "label":        s["label"],
            "raw_lang":     s["raw_lang"],
            "tgt_lang":     s["tgt_lang"],
            "tts":          s["tts"],
            "sync":         s["sync"],
            **{k: round(v,4) if isinstance(v,float) else v
               for k,v in out.items()}
        })
        if (idx+1) % 100 == 0:
            with open(RESULT_PATH,"w") as f:
                json.dump(results, f, ensure_ascii=False)
            elapsed = time.time()-t_start
            eta     = elapsed/(idx+1)*(len(todo)-idx-1)
            print(f"\n[{idx+1}/{len(todo)}] {elapsed/60:.1f}분 경과 | "
                  f"남은 예상 {eta/60:.1f}분")

    with open(RESULT_PATH,"w") as f:
        json.dump(results, f, ensure_ascii=False)

    # ── 성능 평가 ──────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PolyGlotFake Zero-shot Fusion 결과")
    print(f"{'='*65}")

    labels  = np.array([r["label"] for r in results])
    sv_arr  = np.array([r["sv"]    for r in results])
    sa_arr  = np.array([r["sa"]    for r in results])
    st_arr  = np.array([r["st"]    for r in results])
    fin_arr = np.array([r["final"] for r in results])

    def report(name, probs):
        preds = (probs>0.5).astype(int)
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, zero_division=0)
        cm  = confusion_matrix(labels, preds)
        tn,fp,fn,tp = cm.ravel()
        print(f"  {name:<25} AUC={auc*100:6.2f}%  ACC={acc*100:6.2f}%  "
              f"F1={f1*100:6.2f}%  TN={tn} FP={fp} FN={fn} TP={tp}")
        return {"name":name,"auc":round(auc*100,2),
                "acc":round(acc*100,2),"f1":round(f1*100,2)}

    rows = [
        report("X3D 단독",          sv_arr),
        report("AASIST 단독",       sa_arr),
        report("NLP 단독",          st_arr),
        report("X3D+AASIST OR",    prob_or(sv_arr,sa_arr)),
        report("X3D+NLP OR",       prob_or(sv_arr,st_arr)),
        report("AASIST+NLP OR",    prob_or(sa_arr,st_arr)),
        report("3방향 OR (최종)",    fin_arr),
    ]

    # 인수인계 baseline과 비교
    print(f"\n{'='*65}")
    print("[기존 baseline 비교 (인수인계 문서 기준)]")
    print(f"  X3D+AASIST OR (기존):  AUC=91.10%  ACC=91.10%  F1=95.30%")
    new_art = next(r for r in rows if r["name"]=="X3D+AASIST OR")
    new_3way = next(r for r in rows if r["name"]=="3방향 OR (최종)")
    print(f"  X3D+AASIST OR (현재):  AUC={new_art['auc']:.2f}%  "
          f"ACC={new_art['acc']:.2f}%  F1={new_art['f1']:.2f}%")
    print(f"  3방향 OR   (NLP 추가): AUC={new_3way['auc']:.2f}%  "
          f"ACC={new_3way['acc']:.2f}%  F1={new_3way['f1']:.2f}%")
    print(f"  NLP 기여: AUC {new_3way['auc']-new_art['auc']:+.2f}%p  "
          f"F1 {new_3way['f1']-new_art['f1']:+.2f}%p")

    # target_lang별 분석
    print(f"\n[target_lang별 3방향 OR 탐지율]")
    for lang in sorted(set(r["tgt_lang"] for r in results)):
        idx = [i for i,r in enumerate(results) if r["tgt_lang"]==lang]
        lab = labels[idx]; prb = fin_arr[idx]
        if len(set(lab)) < 2: continue
        auc = roc_auc_score(lab, prb)
        f1  = f1_score(lab, (prb>0.5).astype(int), zero_division=0)
        n   = len(idx)
        print(f"  to_{lang:<4}: AUC={auc*100:6.2f}%  F1={f1*100:6.2f}%  ({n}개)")

    # lang_mismatch 탐지 통계
    mismatch_cnt = sum(1 for r in results if r.get("lang_mismatch"))
    print(f"\n[NLP 언어 불일치 탐지]")
    print(f"  전체 {len(results)}개 중 {mismatch_cnt}개 언어 불일치 감지 "
          f"({mismatch_cnt/len(results)*100:.1f}%)")

    # CSV 저장
    report_path = os.path.join(OUT_DIR, "fusion_pgf_report.csv")
    pd.DataFrame(rows).to_csv(report_path, index=False)
    print(f"\n✅ 완료 | {RESULT_PATH}")
    print(f"         {report_path}")

if __name__ == "__main__":
    main()
