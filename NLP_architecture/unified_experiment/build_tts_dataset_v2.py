"""
build_tts_dataset_v2.py
NLP attribution 데이터셋: 텍스트 토큰(transcript) + 22차원 confidence 통계
사용법:
  python3 build_tts_dataset_v2.py pgf
  python3 build_tts_dataset_v2.py avdf1m
"""
import os, sys, json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

mode = sys.argv[1] if len(sys.argv)>1 else "pgf"
tokenizer = AutoTokenizer.from_pretrained(NLP_CFG["model_name"])
BASE = os.path.expanduser("~/hsh/AIApplication")

def stat_feat(words, sc, sd):
    probs=np.array([w["prob"] for w in words])
    durs=np.array([w["dur"] for w in words])
    if len(probs)<3: return None
    return [
        float(probs.mean()),float(probs.std()),float(probs.min()),float(probs.max()),
        float(np.percentile(probs,10)),float(np.percentile(probs,25)),float(np.percentile(probs,50)),
        float(np.percentile(probs,75)),float(np.percentile(probs,90)),
        float((probs<0.3).mean()),float((probs<0.5).mean()),float((probs<0.7).mean()),float((probs>0.9).mean()),
        float(durs.mean()),float(durs.std()),float(durs.min()),float(durs.max()),
        float(np.abs(np.diff(probs)).mean()) if len(probs)>1 else 0.0,
        float(np.abs(np.diff(probs)).max()) if len(probs)>1 else 0.0,
        float(len(probs)), float(sc), float(sd),
    ]

if mode=="pgf":
    with open(os.path.join(BASE,"PolyGlotFake/json_file/fake_Json_file/all_fake_video.json")) as f:
        fake={v["filename"]:v for v in json.load(f)["video"]}
    with open(PGF_CACHE) as f: cache=json.load(f)
    with open(os.path.join(RESULTS_DIR,"sync_cache_pgf.json")) as f: sync=json.load(f)
    CLASSES=["Bark","MicroTts","Tacotron","Vall","Xtts"]
    CMAP={t:i for i,t in enumerate(CLASSES)}
    def get_label(key):
        fn=key.split("/",1)[1]; info=fake.get(fn)
        return CMAP.get(info.get("tts_technique")) if info else None
    keys=[k for k in cache if k.startswith("fake/") and cache[k].get("words")]
    OUT=os.path.join(RESULTS_DIR,"tts_nlp_dataset_pgf.csv")
elif mode=="avdf1m":
    with open(AVDF1M_VAL_META) as f: meta=json.load(f)
    mmap={m["file"]:m for m in meta}
    with open(AVDF1M_CACHE) as f: cache=json.load(f)
    with open(os.path.join(RESULTS_DIR,"sync_cache_avdf1m.json")) as f: sync=json.load(f)
    CLASSES=["vits","yourtts","vits_word","yourtts_word"]
    CMAP={t:i for i,t in enumerate(CLASSES)}
    def get_label(key):
        m=mmap.get(key)
        return CMAP.get(m.get("audio_model")) if m else None
    keys=[k for k in cache if cache[k].get("words") and mmap.get(k,{}).get("audio_model") in CMAP]
    OUT=os.path.join(RESULTS_DIR,"tts_nlp_dataset_avdf1m.csv")
else:
    print("사용법: build_tts_dataset_v2.py [pgf|avdf1m]"); sys.exit(1)

rows=[]
for key in keys:
    e=cache[key]
    lab=get_label(key)
    if lab is None: continue
    sc=sync.get(key,{}).get("conf",0) or 0
    sd=sync.get(key,{}).get("dist",0) or 0
    sf=stat_feat(e["words"], sc, sd)
    if sf is None: continue
    # 텍스트 토큰
    ids=tokenizer.encode(e["text"], add_special_tokens=True, max_length=NLP_CFG["max_len"], truncation=True)
    rows.append({"key":key,"token_ids":json.dumps(ids),
                 "stat_feats":json.dumps(sf),"label":lab,"label_name":CLASSES[lab]})

df=pd.DataFrame(rows)
print(f"[{mode}] {len(df)}개")
print(f"분포: {dict(Counter(df['label_name']))}")

# train/eval 70:30 계층적
np.random.seed(42)
tr_idx,ev_idx=[],[]
for lab in df["label"].unique():
    sub=df[df["label"]==lab].index.tolist()
    np.random.shuffle(sub)
    n=int(len(sub)*0.7)
    tr_idx+=sub[:n]; ev_idx+=sub[n:]
df["split"]="eval"; df.loc[tr_idx,"split"]="train"
df.to_csv(OUT,index=False)
print(f"train {len(tr_idx)}, eval {len(ev_idx)}")
print(f"✅ {OUT}")
