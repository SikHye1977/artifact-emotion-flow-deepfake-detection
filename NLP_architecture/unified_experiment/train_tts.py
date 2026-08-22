"""
train_tts.py
TTS 기법 5분류 학습 (NLP feature 유무 ablation 가능)
  python3 train_tts.py            # 텍스트+메타
  python3 train_tts.py --no-text  # 메타만 (텍스트 임베딩 무력화 비교용)
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
from model_tts import TTSClassifier

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(42)
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

ML=NLP_CFG["max_len"]; N_META=4; N_CLASS=5
TTS_LIST=["Bark","MicroTts","Tacotron","Vall","Xtts"]
CSV=os.path.join(RESULTS_DIR,"pgf_tts_dataset.csv")
SAVE=os.path.join(RESULTS_DIR,"tts_classifier_best.pth")
EPOCHS=20

class TDS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        ids=json.loads(r["token_ids"])[:ML]
        metas=json.loads(r["token_metas"])[:ML]
        lab=int(r["tts_label"]); L=len(ids); pad=ML-L
        return {
            "input_ids":torch.tensor(ids+[1]*pad,dtype=torch.long),
            "attention_mask":torch.tensor([1]*L+[0]*pad,dtype=torch.long),
            "meta_feats":torch.tensor(metas+[[0.0]*N_META]*pad,dtype=torch.float),
            "label":torch.tensor(lab,dtype=torch.long),
        }

df=pd.read_csv(CSV)
train_df=df[df["split"]=="train"]; eval_df=df[df["split"]=="eval"]
print(f"train {len(train_df)}, eval {len(eval_df)}")

# 클래스 가중치 (불균형 보정)
cnt=Counter(train_df["tts_label"])
weights=torch.tensor([len(train_df)/(N_CLASS*cnt[i]) for i in range(N_CLASS)],dtype=torch.float).to(DEVICE)
print(f"클래스 가중치: {weights.cpu().numpy().round(2)}")

train_loader=DataLoader(TDS(train_df),batch_size=NLP_CFG["batch_size"],shuffle=True,num_workers=4)
eval_loader=DataLoader(TDS(eval_df),batch_size=NLP_CFG["batch_size"],shuffle=False,num_workers=4)

model=TTSClassifier(NLP_CFG["model_name"],n_class=N_CLASS,n_meta=N_META).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=NLP_CFG["lr"],weight_decay=0.01)
total=len(train_loader)*EPOCHS
sched=get_cosine_schedule_with_warmup(opt,int(total*0.1),total)
crit=nn.CrossEntropyLoss(weight=weights)

@torch.no_grad()
def evaluate():
    model.eval(); preds,labels=[],[]
    for b in eval_loader:
        out=model(b["input_ids"].to(DEVICE),b["attention_mask"].to(DEVICE),b["meta_feats"].to(DEVICE))
        preds.extend(out.argmax(1).cpu().numpy()); labels.extend(b["label"].numpy())
    preds=np.array(preds); labels=np.array(labels)
    return accuracy_score(labels,preds)*100, f1_score(labels,preds,average="macro")*100, (labels,preds)

best=0
for ep in range(1,EPOCHS+1):
    model.train()
    for b in tqdm(train_loader,desc=f"Ep{ep}/{EPOCHS}"):
        opt.zero_grad()
        out=model(b["input_ids"].to(DEVICE),b["attention_mask"].to(DEVICE),b["meta_feats"].to(DEVICE))
        loss=crit(out,b["label"].to(DEVICE))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step()
    acc,f1,_=evaluate()
    print(f"Ep{ep} ACC={acc:.2f}% macroF1={f1:.2f}%")
    if f1>best:
        best=f1
        torch.save({"state_dict":model.state_dict()},SAVE)
        print(f"  ✅ Best F1 {best:.2f}%")

acc,f1,(labels,preds)=evaluate()
print(f"\n✅ TTS 분류 Best macroF1: {best:.2f}%")
print(f"(이전 GBM confidence통계: 57% ACC, 49% F1)")
print(f"\n혼동행렬 (행=실제):")
cm=confusion_matrix(labels,preds)
print("        "+" ".join(f"{t[:5]:>6}" for t in TTS_LIST))
for i,row in enumerate(cm):
    print(f"{TTS_LIST[i][:7]:<8}"+" ".join(f"{x:>6}" for x in row))
