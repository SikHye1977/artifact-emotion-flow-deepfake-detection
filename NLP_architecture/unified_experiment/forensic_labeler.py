"""
forensic_labeler.py
End-to-end deepfake forensic 레이블러 (GT 없이 모델 추론만)

입력: 영상 경로
처리: X3D + AASIST + Whisper + SyncNet + TTS분류기
출력: 자연어 forensic 리포트 (확실/추정 신뢰도 구분)

사용법:
  python3 forensic_labeler.py --dataset pgf --n 5
  python3 forensic_labeler.py --dataset avdf1m --n 5
"""
import os, sys, json, pickle, functools, argparse
import numpy as np
import torch
import torch.nn as nn
import torchaudio, av

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

torch.load = functools.partial(torch.load, weights_only=False)
try:
    import torchvision.transforms.functional_tensor
except ImportError:
    import torchvision.transforms.functional as _Ftv
    sys.modules["torchvision.transforms.functional_tensor"]=_Ftv
from pytorchvideo.models.hub import x3d_m
from pytorchvideo.transforms import UniformTemporalSubsample, ShortSideScale
from torchvision.transforms import Compose, Lambda, Normalize, Resize
BASE_APP=os.path.expanduser("~/hsh/AIApplication")
sys.path.insert(0, BASE_APP)
from aasist.models.AASIST import Model as AASISTModel

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
ML=NLP_CFG["max_len"]

# ── 전처리 (기존과 동일) ──
def rescale(x): return x/255.0
def to_tc(x): return x.permute(1,0,2,3)
def to_ct(x): return x.permute(1,0,2,3)
x3d_tf=Compose([UniformTemporalSubsample(16),Lambda(rescale),Lambda(to_tc),
    Normalize([0.45]*3,[0.225]*3),Lambda(to_ct),ShortSideScale(256),Resize((224,224))])

def load_video(p):
    try:
        c=av.open(p); fr=[f.to_rgb().to_ndarray() for f in c.decode(video=0)]; c.close()
        if len(fr)<16: return None
        if len(fr)>128:
            idx=np.linspace(0,len(fr)-1,128,dtype=int); fr=[fr[i] for i in idx]
        return torch.from_numpy(np.stack(fr)).permute(3,0,1,2).float()
    except: return None

def load_audio(p,sr=16000,ml=64000):
    try:
        c=av.open(p)
        if not c.streams.audio: c.close(); return None
        osr=c.streams.audio[0].rate; fr=[]
        for f in c.decode(audio=0):
            a=f.to_ndarray().astype(np.float32)
            if a.ndim>1: a=a.mean(axis=0)
            fr.append(a)
        c.close()
        if not fr: return None
        w=torch.from_numpy(np.concatenate(fr)).unsqueeze(0)
        if osr!=sr: w=torchaudio.transforms.Resample(osr,sr)(w)
        if w.shape[1]>ml: w=w[:,:ml]
        else: w=torch.nn.functional.pad(w,(0,ml-w.shape[1]))
        return w.squeeze()
    except: return None

def load_x3d(p):
    m=x3d_m(pretrained=False)
    m.blocks[5].proj=nn.Linear(2048,1); m.blocks[5].activation=nn.Identity()
    ck=torch.load(p,map_location=DEVICE); m.load_state_dict(ck.get("state_dict",ck))
    return m.to(DEVICE).eval()

def load_aasist(p):
    with open(AASIST_CFG) as f: cfg=json.load(f)
    m=AASISTModel(cfg["model_config"])
    ck=torch.load(p,map_location=DEVICE); m.load_state_dict(ck.get("state_dict",ck))
    return m.to(DEVICE).eval()

@torch.no_grad()
def infer_x3d(m,p):
    v=load_video(p)
    if v is None: return None
    return torch.sigmoid(m(x3d_tf(v).unsqueeze(0).to(DEVICE))).item()
@torch.no_grad()
def infer_aasist(m,p):
    a=load_audio(p)
    if a is None: return None
    _,out=m(a.unsqueeze(0).to(DEVICE))
    return torch.softmax(out,dim=1)[0,1].item()

def tts_feat(words, sc, sd):
    probs=np.array([w["prob"] for w in words]); durs=np.array([w["dur"] for w in words])
    if len(probs)<3: return None
    return [[probs.mean(),probs.std(),probs.min(),probs.max(),
        np.percentile(probs,10),np.percentile(probs,25),np.percentile(probs,50),
        np.percentile(probs,75),np.percentile(probs,90),
        (probs<0.3).mean(),(probs<0.5).mean(),(probs<0.7).mean(),(probs>0.9).mean(),
        durs.mean(),durs.std(),durs.min(),durs.max(),
        np.abs(np.diff(probs)).mean() if len(probs)>1 else 0,
        np.abs(np.diff(probs)).max() if len(probs)>1 else 0,
        len(probs), sc, sd]]

# 언어 코드 정상화
def detect_lang_mismatch(detected, expected_set):
    """Whisper 감지 언어가 예상 범위 밖이면 변환 의심"""
    return detected

def generate_report(sv, sa, whisper_entry, sync_entry, tts_clf, tts_classes, dataset, clf_label="음성합성 기법"):
    lines=[]
    av_or=1-(1-sv)*(1-sa)
    is_fake = av_or > 0.5
    verdict = "FAKE" if is_fake else "REAL"
    lines.append(f"[Deepfake Forensic Report]")
    lines.append(f"■ 판정: {verdict} (신뢰도 {av_or*100:.0f}%, X3D+AASIST)")
    lines.append(f"  - 시각 조작 점수(X3D): {sv:.2f}")
    lines.append(f"  - 음성 조작 점수(AASIST): {sa:.2f}")

    if not is_fake:
        lines.append("  → 조작 정황이 탐지되지 않음")
        return "\n".join(lines)

    # 조작 유형 추정 (sv, sa 패턴)
    vis = sv > 0.5; aud = sa > 0.5
    if vis and aud: mtype="음성+영상 모두 조작 가능성"
    elif vis: mtype="영상(입모양) 조작 가능성"
    elif aud: mtype="음성 조작 가능성"
    else: mtype="경계 사례"
    lines.append(f"■ 조작 유형 추정: {mtype}")

    # 언어 정보 (Whisper)
    lang = whisper_entry.get("language","?")
    probs=[w["prob"] for w in whisper_entry.get("words",[])]
    if probs:
        ac=np.mean(probs)
        qual="부자연스러운 합성 정황" if ac<0.75 else "자연스러운 합성"
        lines.append(f"■ ASR 분석:")
        lines.append(f"  - 감지 언어: {lang}")
        lines.append(f"  - ASR confidence: {ac:.2f} ({qual})")

    # sync
    if sync_entry and sync_entry.get("conf") is not None:
        sc=sync_entry["conf"]; sd=sync_entry["dist"]
        lines.append(f"  - 입-소리 sync: conf={sc:.2f}, dist={sd:.2f}")
    else:
        sc=sd=0
        lines.append(f"  - 입-소리 sync: 측정 불가")

    # TTS 기법 추정 (음성 조작 시)
    if aud and probs and tts_clf is not None:
        f=tts_feat(whisper_entry["words"], sc, sd)
        if f:
            proba=tts_clf.predict_proba(f)[0]
            top=np.argmax(proba)
            lines.append(f"■ {clf_label} 추정 [추정, 신뢰도 낮음]:")
            order=np.argsort(proba)[::-1][:3]
            for i in order:
                lines.append(f"  - {tts_classes[i]}: {proba[i]*100:.0f}%")
    return "\n".join(lines)

# ── 메인 ──
ap=argparse.ArgumentParser()
ap.add_argument("--dataset",default="pgf")
ap.add_argument("--n",type=int,default=5)
args=ap.parse_args()

print("모델 로드 중...")
if args.dataset=="pgf":
    x3d=load_x3d(X3D_PGF); aasist=load_aasist(AASIST_PGF)
    with open(PGF_CACHE) as f: cache=json.load(f)
    with open(os.path.join(RESULTS_DIR,"sync_cache_pgf.json")) as f: sync=json.load(f)
    with open(os.path.join(BASE_APP,"PolyGlotFake/json_file/fake_Json_file/all_fake_video.json")) as f:
        fake={v["filename"]:v for v in json.load(f)["video"]}
    def path_of(key):
        fn=key.split("/",1)[1]; info=fake.get(fn)
        return os.path.join(BASE_APP,f'PolyGlotFake/fake/to_{info["target_lang"]}',fn) if info else None
    keys=[k for k in cache if k.startswith("fake/") and cache[k].get("words")][:args.n]
else:
    x3d=load_x3d(X3D_AVDF1M); aasist=load_aasist(AASIST_AVDF1M)
    with open(AVDF1M_CACHE) as f: cache=json.load(f)
    with open(os.path.join(RESULTS_DIR,"sync_cache_avdf1m.json")) as f: sync=json.load(f)
    with open(AVDF1M_VAL_META) as f: meta=json.load(f)
    mmap={m["file"]:m for m in meta}
    def path_of(key): return os.path.join(BASE_APP,"AV-Deepfake1M_RootFiles/extracted_val/val",key)
    keys=[k for k in cache if mmap.get(k,{}).get("modify_type") in ("audio_modified","both_modified") and cache[k].get("words")][:args.n]

clf_file = "tts_rf_model.pkl" if args.dataset=="pgf" else "avdf1m_audio_rf_model.pkl"
with open(os.path.join(RESULTS_DIR,clf_file),"rb") as f:
    tts_data=pickle.load(f)
tts_clf=tts_data["model"]; tts_classes=tts_data["classes"]
clf_label = "음성합성 기법" if args.dataset=="pgf" else "음성합성 모델(audio_model)"
print("로드 완료\n")

for key in keys:
    p=path_of(key)
    if not p or not os.path.exists(p): continue
    sv=infer_x3d(x3d,p); sa=infer_aasist(aasist,p)
    if sv is None or sa is None: continue
    report=generate_report(sv,sa,cache[key],sync.get(key),tts_clf,tts_classes,args.dataset,clf_label)
    print(report)
    print("="*55)
