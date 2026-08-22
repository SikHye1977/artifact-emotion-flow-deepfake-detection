"""
build_tts_dataset.py
TTS 기법 분류용 데이터셋 (PGF fake만, 기법 5종 레이블)
NLP feature: 토큰 임베딩 + word별 (prob, dur, gap)
"""
import os, sys, json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

tokenizer = AutoTokenizer.from_pretrained(NLP_CFG["model_name"])
BASE = os.path.expanduser("~/hsh/AIApplication")

with open(os.path.join(BASE,"PolyGlotFake/json_file/fake_Json_file/all_fake_video.json")) as f:
    fake = {v["filename"]: v for v in json.load(f)["video"]}
with open(PGF_CACHE) as f: cache = json.load(f)
with open(os.path.join(RESULTS_DIR,"sync_cache_pgf.json")) as f: sync = json.load(f)

TTS_LIST = ["Bark","MicroTts","Tacotron","Vall","Xtts"]
TTS_MAP = {t:i for i,t in enumerate(TTS_LIST)}

def build_word_meta(words):
    n=len(words); metas=[]
    for i,w in enumerate(words):
        gp=abs(w["prob"]-words[i-1]["prob"]) if i>0 else 0.0
        gn=abs(w["prob"]-words[i+1]["prob"]) if i<n-1 else 0.0
        metas.append([w["prob"],w["dur"],gp,gn])
    return metas

rows=[]
for key,e in cache.items():
    if not key.startswith("fake/") or not e.get("words"): continue
    fn=key.split("/",1)[1]
    info=fake.get(fn)
    if not info: continue
    tts=info.get("tts_technique")
    if tts not in TTS_MAP: continue
    wm=build_word_meta(e["words"])
    ids,metas=[],[]
    for w,mt in zip(e["words"],wm):
        sub=tokenizer.encode(w["word"],add_special_tokens=False)
        for tid in sub:
            ids.append(tid); metas.append(mt)
    if not ids: continue
    rows.append({"key":key,"token_ids":json.dumps(ids),
                 "token_metas":json.dumps(metas),
                 "tts_label":TTS_MAP[tts],"tts_name":tts})

df=pd.DataFrame(rows)
print(f"TTS 분류 데이터: {len(df)}개")
print(f"분포: {dict(Counter(df['tts_name']))}")

# train/eval 분리 (70:30, 계층적)
np.random.seed(42)
train_idx,eval_idx=[],[]
for lab in df["tts_label"].unique():
    sub=df[df["tts_label"]==lab].index.tolist()
    np.random.shuffle(sub)
    n_tr=int(len(sub)*0.7)
    train_idx+=sub[:n_tr]; eval_idx+=sub[n_tr:]
df["split"]="eval"
df.loc[train_idx,"split"]="train"
OUT=os.path.join(RESULTS_DIR,"pgf_tts_dataset.csv")
df.to_csv(OUT,index=False)
print(f"train {len(train_idx)}, eval {len(eval_idx)}")
print(f"✅ {OUT}")
