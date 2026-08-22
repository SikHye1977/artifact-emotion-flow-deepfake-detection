"""
train_nlp_v3.py
메타 신호 6종(prob,dur,gap_prev,gap_next,sync_conf,sync_dist) NLP 학습
사용법:
  python3 train_nlp_v3.py avdf1m
  python3 train_nlp_v3.py pgf
"""
import os, sys, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *
from model_v3 import UnifiedNLPModelV3

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(NLP_CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

mode = sys.argv[1] if len(sys.argv)>1 else "avdf1m"
if mode=="avdf1m":
    CSV=os.path.join(RESULTS_DIR,"avdf1m_dataset_v3.csv")
    SAVE=os.path.join(RESULTS_DIR,"nlp_v3_avdf1m_best.pth")
elif mode=="pgf":
    CSV=os.path.join(RESULTS_DIR,"pgf_dataset_v3.csv")
    SAVE=os.path.join(RESULTS_DIR,"nlp_v3_pgf_best.pth")
else:
    print("사용법: train_nlp_v3.py [avdf1m|pgf]"); sys.exit(1)

ML=NLP_CFG["max_len"]; N_META=6

class TDS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]
        ids=json.loads(r["token_ids"])[:ML]
        metas=json.loads(r["token_metas"])[:ML]
        tlab=json.loads(r["token_labels"])[:ML]
        clip=int(r["clip_label"])
        L=len(ids); pad=ML-L
        return {
            "input_ids":torch.tensor(ids+[1]*pad,dtype=torch.long),
            "attention_mask":torch.tensor([1]*L+[0]*pad,dtype=torch.long),
            "meta_feats":torch.tensor(metas+[[0.0]*N_META]*pad,dtype=torch.float),
            "token_labels":torch.tensor(tlab+[-1]*pad,dtype=torch.long),
            "clip_label":torch.tensor(clip,dtype=torch.float),
        }

def compute_loss(logits,tl,cl,topk=3):
    mask=(tl!=-1)
    tloss=nn.functional.binary_cross_entropy_with_logits(
        logits[mask].float(),tl[mask].float()) if mask.sum()>0 else torch.tensor(0.0,device=logits.device)
    probs=torch.sigmoid(logits); vm=(tl!=-1).float(); pv=probs*vm
    k=min(topk,logits.shape[1])
    tp,_=torch.topk(pv,k,dim=1)
    cp=tp.mean(dim=1).clamp(1e-6,1-1e-6)
    closs=nn.functional.binary_cross_entropy_with_logits(
        torch.log(cp/(1-cp)),cl.float())
    return tloss+closs

@torch.no_grad()
def evaluate(model,loader):
    model.eval(); labels,scores=[],[]
    for b in loader:
        logits=model(b["input_ids"].to(DEVICE),b["attention_mask"].to(DEVICE),
                     b["meta_feats"].to(DEVICE))
        probs=torch.sigmoid(logits); vm=(b["token_labels"].to(DEVICE)!=-1).float()
        pv=probs*vm; k=min(NLP_CFG["topk"],pv.shape[1])
        tp,_=torch.topk(pv,k,dim=1)
        scores.extend(tp.mean(dim=1).cpu().numpy()); labels.extend(b["clip_label"].numpy())
    labels=np.array(labels); scores=np.array(scores); preds=(scores>0.5).astype(int)
    return {"auc":roc_auc_score(labels,scores),"f1":f1_score(labels,preds,zero_division=0),
            "acc":accuracy_score(labels,preds)}

df=pd.read_csv(CSV)
train_df=df[df["split"]=="train"]; eval_df=df[df["split"]=="eval"]
print(f"[{mode}] train {len(train_df)}, eval {len(eval_df)}")
train_loader=DataLoader(TDS(train_df),batch_size=NLP_CFG["batch_size"],shuffle=True,num_workers=4)
eval_loader=DataLoader(TDS(eval_df),batch_size=NLP_CFG["batch_size"],shuffle=False,num_workers=4)

model=UnifiedNLPModelV3(NLP_CFG["model_name"],n_meta=N_META).to(DEVICE)
opt=torch.optim.AdamW(model.parameters(),lr=NLP_CFG["lr"],weight_decay=0.01)
total=len(train_loader)*NLP_CFG["epochs"]
sched=get_cosine_schedule_with_warmup(opt,int(total*NLP_CFG["warmup_ratio"]),total)

best,log=0.0,[]
for ep in range(1,NLP_CFG["epochs"]+1):
    model.train(); tl=0.0
    for b in tqdm(train_loader,desc=f"Ep{ep}/{NLP_CFG['epochs']} [{mode}-v3]"):
        opt.zero_grad()
        logits=model(b["input_ids"].to(DEVICE),b["attention_mask"].to(DEVICE),
                     b["meta_feats"].to(DEVICE))
        loss=compute_loss(logits,b["token_labels"].to(DEVICE),
                          b["clip_label"].to(DEVICE),NLP_CFG["topk"])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sched.step(); tl+=loss.item()
    tl/=len(train_loader); vm=evaluate(model,eval_loader)
    print(f"Ep{ep} loss={tl:.4f} AUC={vm['auc']*100:.2f}% F1={vm['f1']*100:.2f}% ACC={vm['acc']*100:.2f}%")
    log.append({"epoch":ep,"train_loss":tl,**vm})
    if vm["auc"]>best:
        best=vm["auc"]
        torch.save({"state_dict":model.state_dict(),"val_auc":best,"n_meta":N_META},SAVE)
        print(f"  ✅ Best {best*100:.2f}%")

with open(os.path.join(RESULTS_DIR,f"nlp_v3_{mode}_log.json"),"w") as f:
    json.dump(log,f,indent=2)
v2={"avdf1m":62.12,"pgf":86.74}
print(f"\n✅ [{mode}-v3] Best AUC: {best*100:.2f}%")
print(f"[비교] v2(메타4): {v2[mode]:.2f}%  →  v3(메타6+sync): {best*100:.2f}%  ({best*100-v2[mode]:+.2f}%p)")
