"""
train_tts_v2.py
NLP 기반 TTS 기법 attribution 학습 (텍스트 + confidence 통계 융합)
사용법:
  python3 train_tts_v2.py pgf
  python3 train_tts_v2.py avdf1m
  옵션: --no-text (통계만), --no-stat (텍스트만) → ablation
"""
import os, sys, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *
from model_tts_v2 import TTSClassifierV2

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(42)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

mode = sys.argv[1] if len(sys.argv)>1 else "pgf"
use_text = "--no-text" not in sys.argv
use_stat = "--no-stat" not in sys.argv
abl = "" if (use_text and use_stat) else (" [통계만]" if not use_text else " [텍스트만]")

if mode=="pgf":
    CSV=os.path.join(RESULTS_DIR,"tts_nlp_dataset_pgf.csv")
    CLASSES=["Bark","MicroTts","Tacotron","Vall","Xtts"]
elif mode=="avdf1m":
    CSV=os.path.join(RESULTS_DIR,"tts_nlp_dataset_avdf1m.csv")
    CLASSES=["vits","yourtts","vits_word","yourtts_word"]
else:
    print("사용법: train_tts_v2.py [pgf|avdf1m]"); sys.exit(1)

N_CLASS=len(CLASSES); N_STAT=22; ML=NLP_CFG["max_len"]; EPOCHS=20
SAVE=os.path.join(RESULTS_DIR,f"tts_nlp_{mode}_best.pth")

class TDS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        ids=json.loads(r["token_ids"])[:ML]
        stat=json.loads(r["stat_feats"])
        if not use_text: ids=[0,2]  # 빈 텍스트 (CLS,SEP만)
        if not use_stat: stat=[0.0]*N_STAT
        L=len(ids); pad=ML-L
        return {
            "input_ids":torch.tensor(ids+[1]*pad,dtype=torch.long),
            "attention_mask":torch.tensor([1]*L+[0]*pad,dtype=torch.long),
            "stat_feats":torch.tensor(stat,dtype=torch.float),
            "label":torch.tensor(int(r["label"]),dtype=torch.long),
        }

df=pd.read_csv(CSV)
train_df=df[df["split"]=="train"]; eval_df=df[df["split"]=="eval"]
print(f"[{mode}]{abl} train {len(train_df)}, eval {len(eval_df)}")

cnt=Counter(train_df["label"])
weights=torch.tensor([len(train_df)/(N_CLASS*cnt[i]) for i in range(N_CLASS)],dtype=torch.float).to(DEVICE)

tl=DataLoader(TDS(train_df),batch_size=NLP_CFG["batch_size"],shuffle=True,num_workers=4)
ev=DataLoader(TDS(eval_df),batch_size=NLP_CFG["batch_size"],shuffle=False,num_workers=4)

model=TTSClassifierV2(NLP_CFG["model_name"],n_class=N_CLASS,n_stat=N_STAT).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=NLP_CFG["lr"],weight_decay=0.01)
total=len(tl)*EPOCHS
sched=get_cosine_schedule_with_warmup(opt,int(total*0.1),total)
crit=nn.CrossEntropyLoss(weight=weights)

@torch.no_grad()
def evaluate():
    model.eval(); P,Y=[],[]
    for b in ev:
        out=model(b["input_ids"].to(DEVICE),b["attention_mask"].to(DEVICE),b["stat_feats"].to(DEVICE))
        P.extend(out.argmax(1).cpu().numpy()); Y.extend(b["label"].numpy())
    P=np.array(P); Y=np.array(Y)
    return accuracy_score(Y,P)*100, f1_score(Y,P,average="macro")*100, (Y,P)

best=0; best_acc=0
for ep in range(1,EPOCHS+1):
    model.train()
    for b in tqdm(tl,desc=f"Ep{ep}/{EPOCHS} [{mode}]{abl}"):
        opt.zero_grad()
        out=model(b["input_ids"].to(DEVICE),b["attention_mask"].to(DEVICE),b["stat_feats"].to(DEVICE))
        loss=crit(out,b["label"].to(DEVICE))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step()
    acc,f1,_=evaluate()
    if f1>best: best=f1; best_acc=acc; torch.save({"state_dict":model.state_dict()},SAVE)
    print(f"Ep{ep} ACC={acc:.2f}% macroF1={f1:.2f}%"+(" ✅" if f1==best else ""))

acc,f1,(Y,P)=evaluate()
print(f"\n✅ [{mode}]{abl} Best: ACC={best_acc:.2f}% macroF1={best:.2f}% (랜덤={100/N_CLASS:.0f}%)")
print(f"[참고] RF(통계만): {'52.7%' if mode=='pgf' else '47.9%'}")
cm=confusion_matrix(Y,P,labels=range(N_CLASS))
print("혼동행렬(행=실제):")
print("        "+" ".join(f"{c[:5]:>6}" for c in CLASSES))
for i,row in enumerate(cm):
    print(f"{CLASSES[i][:7]:<8}"+" ".join(f"{x:>6}" for x in row))
