"""
eval_pgf_pure_nlp.py
NLP 순수 문맥 탐지 방식으로 PGF 재평가 (정의 A)
언어 불일치 rule 제거 → 항상 XLM-RoBERTa classifier 사용
"""
import os, sys, json, time, functools, tempfile, subprocess
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import av, whisper
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

NLP_CKPT    = os.path.join(BASE_APP, "NLP_architecture/token_nlp_best.pth")
RESULT_PATH = os.path.join(RESULTS_DIR, "pgf_pure_nlp_results.json")

def rescale(x): return x / 255.0
def to_tc(x):   return x.permute(1,0,2,3)
def to_ct(x):   return x.permute(1,0,2,3)
x3d_transform = Compose([
    UniformTemporalSubsample(16), Lambda(rescale),
    Lambda(to_tc), Normalize([0.45,0.45,0.45],[0.225,0.225,0.225]),
    Lambda(to_ct), ShortSideScale(256), Resize((224,224))])

def load_video(path):
    try:
        c = av.open(path)
        frames = [f.to_rgb().to_ndarray() for f in c.decode(video=0)]
        c.close()
        if len(frames) < 16: return None
        if len(frames) > 128:
            idx = np.linspace(0, len(frames)-1, 128, dtype=int)
            frames = [frames[i] for i in idx]
        return torch.from_numpy(np.stack(frames)).permute(3,0,1,2).float()
    except: return None

def load_audio(path, sr=16000, max_len=64000):
    try:
        c = av.open(path)
        if not c.streams.audio: c.close(); return None
        osr = c.streams.audio[0].rate
        frames = []
        for fr in c.decode(audio=0):
            arr = fr.to_ndarray().astype(np.float32)
            if arr.ndim > 1: arr = arr.mean(axis=0)
            frames.append(arr)
        c.close()
        if not frames: return None
        wav = torch.from_numpy(np.concatenate(frames)).unsqueeze(0)
        if osr != sr: wav = torchaudio.transforms.Resample(osr,sr)(wav)
        if wav.shape[1] > max_len: wav = wav[:,:max_len]
        else: wav = torch.nn.functional.pad(wav,(0,max_len-wav.shape[1]))
        return wav.squeeze()
    except: return None

def extract_audio_16k(video_path, out_wav):
    try:
        c = av.open(video_path)
        if not c.streams.audio: c.close(); return False
        sr = c.streams.audio[0].rate
        frames = []
        for fr in c.decode(audio=0):
            arr = fr.to_ndarray().astype(np.float32)
            if arr.ndim > 1: arr = arr.mean(axis=0)
            frames.append(arr)
        c.close()
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

class TokenNLPClassifier(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", dropout=0.3):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.token_head = nn.Sequential(nn.Dropout(dropout),
            nn.Linear(self.encoder.config.hidden_size, 1))
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.token_head(out.last_hidden_state).squeeze(-1)

def load_x3d(p):
    m = x3d_m(pretrained=False)
    m.blocks[5].proj = nn.Linear(2048,1); m.blocks[5].activation = nn.Identity()
    ck = torch.load(p, map_location=DEVICE)
    m.load_state_dict(ck.get("state_dict", ck))
    return m.to(DEVICE).eval()

def load_aasist(p):
    with open(AASIST_CFG) as f: cfg = json.load(f)
    m = AASISTModel(cfg["model_config"])
    ck = torch.load(p, map_location=DEVICE)
    m.load_state_dict(ck.get("state_dict", ck))
    return m.to(DEVICE).eval()

def load_nlp(p):
    ck = torch.load(p, map_location=DEVICE)
    m = TokenNLPClassifier(); m.load_state_dict(ck["state_dict"])
    tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return m.to(DEVICE).eval(), tok

@torch.no_grad()
def infer_x3d(m, p):
    v = load_video(p)
    if v is None: return None
    return torch.sigmoid(m(x3d_transform(v).unsqueeze(0).to(DEVICE))).item()

@torch.no_grad()
def infer_aasist(m, p):
    a = load_audio(p)
    if a is None: return None
    _, out = m(a.unsqueeze(0).to(DEVICE))
    return torch.softmax(out, dim=1)[0,1].item()

@torch.no_grad()
def infer_nlp_pure(nlp, tok, wm, path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    if not extract_audio_16k(path, tmp):
        if os.path.exists(tmp): os.remove(tmp)
        return None, None
    try:
        res = wm.transcribe(tmp, task="transcribe", verbose=False)
        transcript = res["text"].strip(); lang = res["language"]
    except: return None, None
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    if not transcript: return 0.5, lang
    enc = tok(transcript, return_tensors="pt", truncation=True,
              max_length=128, padding="max_length")
    ids = enc["input_ids"].to(DEVICE); msk = enc["attention_mask"].to(DEVICE)
    probs = torch.sigmoid(nlp(ids, msk))
    k = min(NLP_CFG["topk"], probs.shape[1])
    st = probs.topk(k, dim=1).values.mean().item()
    return st, lang

def prob_or3(a,b,c): return 1-(1-a)*(1-b)*(1-c)

with open(os.path.join(PGF_JSON_DIR,"fake_Json_file/all_fake_video.json")) as f:
    af = json.load(f)
with open(os.path.join(PGF_JSON_DIR,"real_json_file/all_real_video.json")) as f:
    ar = json.load(f)

samples = []
for v in af["video"]:
    p = os.path.join(PGF_ROOT, f'fake/to_{v["target_lang"]}', v["filename"])
    if os.path.exists(p):
        samples.append({"path":p,"label":1,"filename":v["filename"],
                        "raw_lang":v["raw_lang"],"tgt_lang":v["target_lang"]})
for v in ar["videos"]:
    p = os.path.join(PGF_ROOT, f'real/{v["lang"]}', v["filename"])
    if os.path.exists(p):
        samples.append({"path":p,"label":0,"filename":v["filename"],
                        "raw_lang":v["lang"],"tgt_lang":v["lang"]})

print(f"PGF: {len(samples)}개")
print("모델 로드 중...")
x3d_model    = load_x3d(FAKEAV_X3D_CKPT)
aasist_model = load_aasist(FAKEAV_AASIST_CKPT)
nlp_model, tokenizer = load_nlp(NLP_CKPT)
whisper_model = whisper.load_model("medium")
print("로드 완료\n")

if os.path.exists(RESULT_PATH):
    with open(RESULT_PATH) as f: results = json.load(f)
    done = {r["filename"] for r in results}
    todo = [s for s in samples if s["filename"] not in done]
    print(f"기존 {len(results)}개 → 남은 {len(todo)}개\n")
else:
    results, todo = [], samples

t0 = time.time()
for idx, s in enumerate(tqdm(todo, desc="PGF Pure-NLP")):
    sv = infer_x3d(x3d_model, s["path"])
    sa = infer_aasist(aasist_model, s["path"])
    st, lang = infer_nlp_pure(nlp_model, tokenizer, whisper_model, s["path"])
    if sv is None or sa is None or st is None: continue
    results.append({
        "filename": s["filename"], "label": s["label"],
        "raw_lang": s["raw_lang"], "tgt_lang": s["tgt_lang"],
        "sv": round(sv,4), "sa": round(sa,4), "st": round(st,4),
        "final": round(prob_or3(sv,sa,st),4), "whisper_lang": lang,
    })
    if (idx+1) % 200 == 0:
        with open(RESULT_PATH,"w") as f: json.dump(results,f,ensure_ascii=False)

with open(RESULT_PATH,"w") as f: json.dump(results,f,ensure_ascii=False)

labels  = np.array([r["label"] for r in results])
sv_arr  = np.array([r["sv"] for r in results])
sa_arr  = np.array([r["sa"] for r in results])
st_arr  = np.array([r["st"] for r in results])
fin_arr = np.array([r["final"] for r in results])

def met(name, probs):
    preds = (probs>0.5).astype(int)
    auc = roc_auc_score(labels,probs)
    f1  = f1_score(labels,preds,zero_division=0)
    acc = accuracy_score(labels,preds)
    print(f"  {name:<22} AUC={auc*100:6.2f}%  F1={f1*100:6.2f}%  ACC={acc*100:6.2f}%")
    return auc*100

print(f"\n{'='*60}")
print("PGF 순수 문맥 탐지 NLP (정의 A) - rule-based 제거")
print(f"{'='*60}")
met("X3D 단독", sv_arr)
met("AASIST 단독", sa_arr)
met("NLP 단독", st_arr)
met("X3D+AASIST OR", 1-(1-sv_arr)*(1-sa_arr))
met("AASIST+NLP OR", 1-(1-sa_arr)*(1-st_arr))
met("3방향 OR", fin_arr)
print(f"\n[비교: rule-based 버전]")
print(f"  NLP 단독:  rule-based 95.32%")
print(f"  3방향 OR:  rule-based 95.60%")
print(f"\n✅ 완료: {RESULT_PATH}")
