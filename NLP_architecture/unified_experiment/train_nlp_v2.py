"""
train_nlp_v2.py
메타 신호 4종(prob, dur, gap_prev, gap_next) 포함 통합 NLP 학습

사용법:
  python3 train_nlp_v2.py avdf1m   → nlp_v2_avdf1m_best.pth
  python3 train_nlp_v2.py pgf      → nlp_v2_pgf_best.pth
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
from model_v2 import UnifiedNLPModelV2

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(NLP_CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

mode = sys.argv[1] if len(sys.argv) > 1 else "avdf1m"
if mode == "avdf1m":
    CSV = os.path.join(RESULTS_DIR, "avdf1m_dataset_v2.csv")
    SAVE = os.path.join(RESULTS_DIR, "nlp_v2_avdf1m_best.pth")
elif mode == "pgf":
    CSV = os.path.join(RESULTS_DIR, "pgf_dataset_v2.csv")
    SAVE = os.path.join(RESULTS_DIR, "nlp_v2_pgf_best.pth")
else:
    print("사용법: python3 train_nlp_v2.py [avdf1m|pgf]"); sys.exit(1)

ML = NLP_CFG["max_len"]
N_META = 4

class TokenDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        ids   = json.loads(r["token_ids"])[:ML]
        metas = json.loads(r["token_metas"])[:ML]
        tlab  = json.loads(r["token_labels"])[:ML]
        clip  = int(r["clip_label"])
        L = len(ids); pad = ML - L
        return {
            "input_ids":      torch.tensor(ids+[1]*pad, dtype=torch.long),
            "attention_mask": torch.tensor([1]*L+[0]*pad, dtype=torch.long),
            "meta_feats":     torch.tensor(metas+[[0.0]*N_META]*pad, dtype=torch.float),
            "token_labels":   torch.tensor(tlab+[-1]*pad, dtype=torch.long),
            "clip_label":     torch.tensor(clip, dtype=torch.float),
        }

def compute_loss(logits, tlabels, clabels, topk=3):
    mask = (tlabels != -1)
    if mask.sum() > 0:
        tloss = nn.functional.binary_cross_entropy_with_logits(
            logits[mask].float(), tlabels[mask].float())
    else:
        tloss = torch.tensor(0.0, device=logits.device)
    probs = torch.sigmoid(logits)
    vmask = (tlabels != -1).float()
    pv = probs * vmask
    k = min(topk, logits.shape[1])
    topk_p,_ = torch.topk(pv, k, dim=1)
    clip_prob = topk_p.mean(dim=1).clamp(1e-6, 1-1e-6)
    clip_logit = torch.log(clip_prob/(1-clip_prob))
    closs = nn.functional.binary_cross_entropy_with_logits(clip_logit, clabels.float())
    return tloss + closs

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    labels, scores = [], []
    for b in loader:
        logits = model(b["input_ids"].to(DEVICE),
                       b["attention_mask"].to(DEVICE),
                       b["meta_feats"].to(DEVICE))
        probs = torch.sigmoid(logits)
        vmask = (b["token_labels"].to(DEVICE) != -1).float()
        pv = probs * vmask
        k = min(NLP_CFG["topk"], pv.shape[1])
        topk_p,_ = torch.topk(pv, k, dim=1)
        scores.extend(topk_p.mean(dim=1).cpu().numpy())
        labels.extend(b["clip_label"].numpy())
    labels = np.array(labels); scores = np.array(scores)
    preds = (scores>0.5).astype(int)
    return {"auc": roc_auc_score(labels,scores),
            "f1": f1_score(labels,preds,zero_division=0),
            "acc": accuracy_score(labels,preds)}

df = pd.read_csv(CSV)
train_df = df[df["split"]=="train"]; eval_df = df[df["split"]=="eval"]
print(f"[{mode}] train {len(train_df)}, eval {len(eval_df)}")

train_loader = DataLoader(TokenDataset(train_df),
    batch_size=NLP_CFG["batch_size"], shuffle=True, num_workers=4)
eval_loader  = DataLoader(TokenDataset(eval_df),
    batch_size=NLP_CFG["batch_size"], shuffle=False, num_workers=4)

model = UnifiedNLPModelV2(NLP_CFG["model_name"], n_meta=N_META).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=NLP_CFG["lr"], weight_decay=0.01)
total = len(train_loader)*NLP_CFG["epochs"]
sched = get_cosine_schedule_with_warmup(
    optimizer, int(total*NLP_CFG["warmup_ratio"]), total)

best_auc, log = 0.0, []
for epoch in range(1, NLP_CFG["epochs"]+1):
    model.train(); tl = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NLP_CFG['epochs']} [{mode}-v2]")
    for b in pbar:
        optimizer.zero_grad()
        logits = model(b["input_ids"].to(DEVICE),
                       b["attention_mask"].to(DEVICE),
                       b["meta_feats"].to(DEVICE))
        loss = compute_loss(logits, b["token_labels"].to(DEVICE),
                            b["clip_label"].to(DEVICE), NLP_CFG["topk"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); sched.step()
        tl += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    tl /= len(train_loader)
    vm = evaluate(model, eval_loader)
    print(f"\nEpoch {epoch} | train_loss={tl:.4f} | "
          f"AUC={vm['auc']*100:.2f}% F1={vm['f1']*100:.2f}% ACC={vm['acc']*100:.2f}%")
    log.append({"epoch":epoch,"train_loss":tl,**vm})
    if vm["auc"] > best_auc:
        best_auc = vm["auc"]
        torch.save({"epoch":epoch,"state_dict":model.state_dict(),
                    "val_auc":best_auc,"cfg":NLP_CFG,"n_meta":N_META}, SAVE)
        print(f"  ✅ Best 저장 (AUC={best_auc*100:.2f}%)")

with open(os.path.join(RESULTS_DIR,f"nlp_v2_{mode}_train_log.json"),"w") as f:
    json.dump(log, f, indent=2)
print(f"\n✅ [{mode}-v2] 학습 완료 | Best AUC: {best_auc*100:.2f}%")

# v1과 비교
v1_auc = {"avdf1m": 61.61, "pgf": 85.19}
print(f"\n[v1 vs v2 비교]")
print(f"  v1 (prob+dur):           {v1_auc[mode]:.2f}%")
print(f"  v2 (prob+dur+gap):       {best_auc*100:.2f}%")
print(f"  개선: {best_auc*100 - v1_auc[mode]:+.2f}%p")
