"""
verify_prosody.py
운율(prosody) 신호 검증
AVDF1M 조작 단어 구간 vs 정상 구간의 F0/energy/duration 비교
"""
import os, sys, json, subprocess, tempfile, random
import numpy as np
import librosa
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

random.seed(42)
BASE=os.path.expanduser("~/hsh/AIApplication")
AVDF_ROOT=os.path.join(BASE,"AV-Deepfake1M_RootFiles/extracted_val/val")

with open(AVDF1M_VAL_META) as f: meta=json.load(f)
mmap={m["file"]:m for m in meta}
with open(AVDF1M_CACHE) as f: cache=json.load(f)
with open(os.path.join(RESULTS_DIR,"avdf1m_split.json")) as f: split=json.load(f)

def extract_audio(path):
    """16k mono wav로 추출"""
    tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False).name
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",path,
        "-ac","1","-ar","16000",tmp],capture_output=True,timeout=30)
    return tmp

def prosody_in_segment(y, sr, t_start, t_end):
    """구간 내 운율 특성"""
    s=int(t_start*sr); e=int(t_end*sr)
    if e<=s or e>len(y): return None
    seg=y[s:e]
    if len(seg)<sr*0.05: return None  # 너무 짧음
    # F0 (pitch)
    try:
        f0=librosa.yin(seg, fmin=80, fmax=400, sr=sr)
        f0=f0[~np.isnan(f0)]
        f0_mean=np.mean(f0) if len(f0)>0 else 0
        f0_std=np.std(f0) if len(f0)>0 else 0
    except:
        f0_mean=f0_std=0
    # energy
    energy=np.sqrt(np.mean(seg**2))
    # zero crossing rate (음색 거칠기)
    zcr=np.mean(librosa.feature.zero_crossing_rate(seg))
    return {"f0_mean":f0_mean,"f0_std":f0_std,"energy":energy,"zcr":zcr}

# audio_modified, both_modified에서 조작 구간 vs 정상 구간
fake_pros={"f0_mean":[],"f0_std":[],"energy":[],"zcr":[]}
clean_pros={"f0_mean":[],"f0_std":[],"energy":[],"zcr":[]}

keys=[k for k in split["train"] if mmap.get(k,{}).get("modify_type") in ("audio_modified","both_modified")]
random.shuffle(keys)
keys=keys[:60]
print(f"분석 대상: {len(keys)}개 (audio/both_modified)")

for i,key in enumerate(keys):
    m=mmap.get(key)
    path=os.path.join(AVDF_ROOT,key)
    if not os.path.exists(path): continue
    segs=m.get("audio_fake_segments",[])
    if not segs: continue
    wav=extract_audio(path)
    try:
        y,sr=librosa.load(wav,sr=16000)
    except:
        os.remove(wav); continue
    os.remove(wav)
    # 조작 구간
    for seg in segs:
        p=prosody_in_segment(y,sr,seg[0],seg[1])
        if p:
            for k in fake_pros: fake_pros[k].append(p[k])
    # 정상 구간 (조작 구간 밖, words 활용)
    e=cache.get(key,{})
    for w in e.get("words",[]):
        is_fake=any(w["start"]<se and w["end"]>ss for ss,se in segs)
        if not is_fake:
            p=prosody_in_segment(y,sr,w["start"],w["end"])
            if p:
                for k in clean_pros: clean_pros[k].append(p[k])
    if (i+1)%20==0: print(f"  {i+1}/{len(keys)}")

print("\n=== 운율 신호: 조작 구간 vs 정상 구간 ===")
print(f"{'특성':<10} {'조작':>10} {'정상':>10} {'p-value':>12}")
for k in ["f0_mean","f0_std","energy","zcr"]:
    fm=np.mean(fake_pros[k]); cm=np.mean(clean_pros[k])
    if len(fake_pros[k])>10 and len(clean_pros[k])>10:
        _,p=stats.mannwhitneyu(fake_pros[k],clean_pros[k])
        sig="유의" if p<0.05 else "무의미"
        print(f"{k:<10} {fm:>10.3f} {cm:>10.3f} {p:>12.2e} {sig}")
    else:
        print(f"{k:<10} {fm:>10.3f} {cm:>10.3f}  (샘플부족)")
print(f"\n조작 구간 n={len(fake_pros['f0_mean'])}, 정상 구간 n={len(clean_pros['f0_mean'])}")
