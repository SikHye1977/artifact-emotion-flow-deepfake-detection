"""
AVDF1M audio_model 분류기 (vits/yourtts/vits_word/yourtts_word) 학습+저장
"""
import os, sys, json, pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

with open(AVDF1M_VAL_META) as f: meta=json.load(f)
mmap={m["file"]:m for m in meta}
with open(AVDF1M_CACHE) as f: cache=json.load(f)
with open(os.path.join(RESULTS_DIR,"sync_cache_avdf1m.json")) as f: sync=json.load(f)

AM_LIST=["vits","yourtts","vits_word","yourtts_word"]
AM_MAP={t:i for i,t in enumerate(AM_LIST)}

def feat(words, sc, sd):
    probs=np.array([w["prob"] for w in words]); durs=np.array([w["dur"] for w in words])
    if len(probs)<3: return None
    return [probs.mean(),probs.std(),probs.min(),probs.max(),
        np.percentile(probs,10),np.percentile(probs,25),np.percentile(probs,50),
        np.percentile(probs,75),np.percentile(probs,90),
        (probs<0.3).mean(),(probs<0.5).mean(),(probs<0.7).mean(),(probs>0.9).mean(),
        durs.mean(),durs.std(),durs.min(),durs.max(),
        np.abs(np.diff(probs)).mean() if len(probs)>1 else 0,
        np.abs(np.diff(probs)).max() if len(probs)>1 else 0,
        len(probs), sc, sd]

X,y=[],[]
for key,e in cache.items():
    if not e.get("words"): continue
    m=mmap.get(key)
    if not m or m.get("audio_model") not in AM_MAP: continue
    sc=sync.get(key,{}).get("conf",0) or 0
    sd=sync.get(key,{}).get("dist",0) or 0
    f=feat(e["words"],sc,sd)
    if f: X.append(f); y.append(AM_MAP[m["audio_model"]])
X=np.array(X); y=np.array(y)
print(f"AVDF1M audio_model: {len(y)}개, 분포 {dict(Counter(y))}")
clf=RandomForestClassifier(n_estimators=200,max_depth=12,class_weight="balanced",random_state=42)
cv=cross_val_score(clf,X,y,cv=5)
print(f"5-fold CV: {cv.mean()*100:.2f}±{cv.std()*100:.2f}%")
clf.fit(X,y)
with open(os.path.join(RESULTS_DIR,"avdf1m_audio_rf_model.pkl"),"wb") as f:
    pickle.dump({"model":clf,"classes":AM_LIST},f)
print("✅ 저장: avdf1m_audio_rf_model.pkl")
