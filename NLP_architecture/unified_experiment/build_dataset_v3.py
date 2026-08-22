"""
build_dataset_v3.py
v2 메타(prob,dur,gap_prev,gap_next 4개) + sync(conf,dist 2개) = 6개
sync 캐시 있는 샘플만 사용
"""
import os, sys, json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

mode = sys.argv[1] if len(sys.argv)>1 else "avdf1m"
tokenizer = AutoTokenizer.from_pretrained(NLP_CFG["model_name"])

def overlaps(ws,we,segs): return any(ws<e and we>s for s,e in segs)

def build_word_meta(words):
    n=len(words); metas=[]
    for i,w in enumerate(words):
        gp=abs(w["prob"]-words[i-1]["prob"]) if i>0 else 0.0
        gn=abs(w["prob"]-words[i+1]["prob"]) if i<n-1 else 0.0
        metas.append([w["prob"],w["dur"],gp,gn])
    return metas

# sync 정규화용 통계 (대략적 스케일)
SYNC_CONF_MEAN, SYNC_CONF_STD = 3.0, 1.5
SYNC_DIST_MEAN, SYNC_DIST_STD = 9.7, 1.0

def norm_sync(conf, dist):
    c = (conf - SYNC_CONF_MEAN)/SYNC_CONF_STD
    d = (dist - SYNC_DIST_MEAN)/SYNC_DIST_STD
    return c, d

if mode=="avdf1m":
    CACHE=AVDF1M_CACHE
    with open(CACHE) as f: cache=json.load(f)
    with open(os.path.join(RESULTS_DIR,"avdf1m_split.json")) as f: split=json.load(f)
    with open(AVDF1M_VAL_META) as f: meta=json.load(f)
    mmap={m["file"]:m for m in meta}
    with open(os.path.join(RESULTS_DIR,"sync_cache_avdf1m.json")) as f: sync=json.load(f)
    OUT=os.path.join(RESULTS_DIR,"avdf1m_dataset_v3.csv")
    rows=[]
    for sp in ["train","eval"]:
        for key in split[sp]:
            if key not in cache or key not in sync: continue
            if sync[key]["conf"] is None: continue
            m=mmap.get(key)
            if not m: continue
            e=cache[key]
            if not e["words"]: continue
            clip=0 if m["modify_type"]=="real" else 1
            segs=m.get("audio_fake_segments",[]) if m["modify_type"] in ("audio_modified","both_modified") else []
            sc,sd=norm_sync(sync[key]["conf"],sync[key]["dist"])
            wm=build_word_meta(e["words"])
            ids,metas,labels=[],[],[]
            for w,mt in zip(e["words"],wm):
                sub=tokenizer.encode(w["word"],add_special_tokens=False)
                if not sub: continue
                lab=1 if (segs and overlaps(w["start"],w["end"],segs)) else 0
                for tid in sub:
                    ids.append(tid); metas.append(mt+[sc,sd]); labels.append(lab)
            if not ids: continue
            rows.append({"key":key,"modify_type":m["modify_type"],
                "token_ids":json.dumps(ids),"token_metas":json.dumps(metas),
                "token_labels":json.dumps(labels),"clip_label":clip,"split":sp})
elif mode=="pgf":
    CACHE=PGF_CACHE
    with open(CACHE) as f: cache=json.load(f)
    with open(os.path.join(RESULTS_DIR,"pgf_split.json")) as f: split=json.load(f)
    with open(os.path.join(RESULTS_DIR,"sync_cache_pgf.json")) as f: sync=json.load(f)
    OUT=os.path.join(RESULTS_DIR,"pgf_dataset_v3.csv")
    rows=[]
    for sp in ["train","eval"]:
        for key in split[sp]:
            if key not in cache or key not in sync: continue
            if sync[key]["conf"] is None: continue
            e=cache[key]
            if not e["words"]: continue
            clip=0 if key.startswith("real/") else 1
            sc,sd=norm_sync(sync[key]["conf"],sync[key]["dist"])
            wm=build_word_meta(e["words"])
            ids,metas,labels=[],[],[]
            for w,mt in zip(e["words"],wm):
                sub=tokenizer.encode(w["word"],add_special_tokens=False)
                if not sub: continue
                for tid in sub:
                    ids.append(tid); metas.append(mt+[sc,sd]); labels.append(clip)
            if not ids: continue
            rows.append({"key":key,"modify_type":"real" if clip==0 else "lang_swap",
                "token_ids":json.dumps(ids),"token_metas":json.dumps(metas),
                "token_labels":json.dumps(labels),"clip_label":clip,"split":sp})
else:
    print("사용법: build_dataset_v3.py [avdf1m|pgf]"); sys.exit(1)

df=pd.DataFrame(rows)
train_df=df[df["split"]=="train"]; eval_df=df[df["split"]=="eval"]
ft=train_df[train_df["clip_label"]==1]; rt=train_df[train_df["clip_label"]==0]
n=min(len(ft),len(rt))
if n==0:
    print(f"⚠ 클래스 부족: fake={len(ft)} real={len(rt)}")
    sys.exit(1)
tb=pd.concat([ft.sample(n,random_state=42),rt.sample(n,random_state=42)])
final=pd.concat([tb,eval_df]).reset_index(drop=True)
final.to_csv(OUT,index=False)
print(f"[{mode}] train {len(tb)} (fake={n},real={n}) + eval {len(eval_df)} = {len(final)}")
print(f"✅ {OUT}")
