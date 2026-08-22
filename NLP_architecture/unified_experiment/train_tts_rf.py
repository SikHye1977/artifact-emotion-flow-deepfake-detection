"""
train_tts_rf.py
TTS 기법 분류 RF 학습 → 저장 (레이블러 추론용)
PGF 전체 fake로 학습. confidence/duration 분포 22-feature.
"""
import os, sys, json, pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

BASE = os.path.expanduser("~/hsh/AIApplication")
with open(os.path.join(BASE,"PolyGlotFake/json_file/fake_Json_file/all_fake_video.json")) as f:
    fake={v["filename"]:v for v in json.load(f)["video"]}
with open(PGF_CACHE) as f: cache=json.load(f)
with open(os.path.join(RESULTS_DIR,"sync_cache_pgf.json")) as f: sync=json.load(f)

TTS_LIST=["Bark","MicroTts","Tacotron","Vall","Xtts"]
TTS_MAP={t:i for i,t in enumerate(TTS_LIST)}

def extract_feat(words, sync_entry):
    probs=np.array([w["prob"] for w in words])
    durs=np.array([w["dur"] for w in words])
    if len(probs)<3: return None
    sc=sync_entry.get("conf",0) or 0 if sync_entry else 0
    sd=sync_entry.get("dist",0) or 0 if sync_entry else 0
    return [
        probs.mean(),probs.std(),probs.min(),probs.max(),
        np.percentile(probs,10),np.percentile(probs,25),np.percentile(probs,50),
        np.percentile(probs,75),np.percentile(probs,90),
        (probs<0.3).mean(),(probs<0.5).mean(),(probs<0.7).mean(),(probs>0.9).mean(),
        durs.mean(),durs.std(),durs.min(),durs.max(),
        np.abs(np.diff(probs)).mean() if len(probs)>1 else 0,
        np.abs(np.diff(probs)).max() if len(probs)>1 else 0,
        len(probs), sc, sd,
    ]

X,y=[],[]
for key,e in cache.items():
    if not key.startswith("fake/") or not e.get("words"): continue
    fn=key.split("/",1)[1]
    info=fake.get(fn)
    if not info or info.get("tts_technique") not in TTS_MAP: continue
    f=extract_feat(e["words"], sync.get(key))
    if f is None: continue
    X.append(f); y.append(TTS_MAP[info["tts_technique"]])
X=np.array(X); y=np.array(y)
print(f"TTS 학습: {len(y)}개, 분포 {dict(Counter(y))}")

# 전체로 학습 (추론용), 성능은 CV로 확인
from sklearn.model_selection import cross_val_score
clf=RandomForestClassifier(n_estimators=200,max_depth=12,class_weight="balanced",random_state=42)
cv=cross_val_score(clf,X,y,cv=5,scoring="accuracy")
print(f"5-fold CV ACC: {cv.mean()*100:.2f}±{cv.std()*100:.2f}%")
clf.fit(X,y)
with open(os.path.join(RESULTS_DIR,"tts_rf_model.pkl"),"wb") as f:
    pickle.dump({"model":clf,"classes":TTS_LIST},f)
print(f"✅ 저장: tts_rf_model.pkl")
