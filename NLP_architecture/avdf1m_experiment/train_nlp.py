"""
train_nlp.py
AVDF1M val set transcript → XLM-RoBERTa Token Classifier fine-tune
기존 token_dataset_train/val.csv 재사용
"""

import os, sys, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import *

import functools
torch.load = functools.partial(torch.load, weights_only=False)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(NLP_CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# 기존 dataset CSV 경로
NLP_ARCH = os.path.join(BASE, "NLP_architecture")
TRAIN_CSV = os.path.join(NLP_ARCH, "token_dataset_train.csv")
VAL_CSV   = os.path.join(NLP_ARCH, "token_dataset_val.csv")

# ── Dataset ───────────────────────────────────────────────────────
class TokenDataset(Dataset):
    def __init__(self, df, max_len):
        self.records = df.reset_index(drop=True)
        self.max_len = max_len
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        row         = self.records.iloc[idx]
        token_ids   = json.loads(row["token_ids"])[:self.max_len]
        token_labels= json.loads(row["token_labels"])[:self.max_len]
        clip_label  = int(row["clip_label"])
        L           = len(token_ids)
        pad         = self.max_len - L
        return {
            "input_ids":      torch.tensor(token_ids+[1]*pad,     dtype=torch.long),
            "attention_mask": torch.tensor([1]*L+[0]*pad,         dtype=torch.long),
            "token_labels":   torch.tensor(token_labels+[-1]*pad, dtype=torch.long),
            "clip_label":     torch.tensor(clip_label,            dtype=torch.float),
        }

# ── Model ─────────────────────────────────────────────────────────
class TokenNLPClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size
        self.token_head = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(hidden, 1))
    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask)
        return self.token_head(out.last_hidden_state).squeeze(-1)  # (B,L)

# ── Loss ─────────────────────────────────────────────────────────
def compute_loss(logits, token_labels, clip_labels, topk=3):
    mask = (token_labels != -1)
    token_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[mask].float(), token_labels[mask].float()
    ) if mask.sum() > 0 else torch.tensor(0.0, device=logits.device)

    probs       = torch.sigmoid(logits)
    valid_mask  = (token_labels != -1).float()
    probs_valid = probs * valid_mask
    k           = min(topk, logits.shape[1])
    topk_probs, _ = torch.topk(probs_valid, k, dim=1)
    clip_prob   = topk_probs.mean(dim=1).clamp(1e-6, 1-1e-6)
    clip_logit  = torch.log(clip_prob / (1-clip_prob))
    clip_loss   = nn.functional.binary_cross_entropy_with_logits(
        clip_logit, clip_labels.float())
    return token_loss + clip_loss

# ── 평가 ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_clip, all_max, all_topk = [], [], []
    for batch in loader:
        ids    = batch["input_ids"].to(DEVICE)
        mask   = batch["attention_mask"].to(DEVICE)
        tlabels= batch["token_labels"].to(DEVICE)
        logits = model(ids, mask)
        probs  = torch.sigmoid(logits)
        vm     = (tlabels != -1).float()
        pv     = probs * vm
        max_p, _   = pv.max(dim=1)
        k          = min(NLP_CFG["topk"], pv.shape[1])
        topk_p, _  = torch.topk(pv, k, dim=1)
        all_clip.extend(batch["clip_label"].numpy())
        all_max.extend(max_p.cpu().numpy())
        all_topk.extend(topk_p.mean(dim=1).cpu().numpy())
    labels = np.array(all_clip)
    max_p  = np.array(all_max)
    topk_p = np.array(all_topk)
    def m(p): return {"auc": roc_auc_score(labels,p),
                      "f1":  f1_score(labels,(p>0.5).astype(int),zero_division=0),
                      "acc": accuracy_score(labels,(p>0.5).astype(int))}
    return m(max_p), m(topk_p)

# ── 메인 ─────────────────────────────────────────────────────────
df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
print(f"train: {len(df_train):,}  val: {len(df_val):,}")

tokenizer    = AutoTokenizer.from_pretrained(NLP_CFG["model_name"])
train_loader = DataLoader(TokenDataset(df_train, NLP_CFG["max_len"]),
    batch_size=NLP_CFG["batch_size"], shuffle=True,
    num_workers=4, pin_memory=True)
val_loader   = DataLoader(TokenDataset(df_val, NLP_CFG["max_len"]),
    batch_size=NLP_CFG["batch_size"], shuffle=False,
    num_workers=4, pin_memory=True)

print("XLM-RoBERTa 로드 중...")
model     = TokenNLPClassifier(NLP_CFG["model_name"]).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(),
                lr=NLP_CFG["lr"], weight_decay=0.01)
total_steps  = len(train_loader) * NLP_CFG["epochs"]
warmup_steps = int(total_steps * NLP_CFG["warmup_ratio"])
scheduler    = get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps)

best_auc = 0.0
log      = []
for epoch in range(1, NLP_CFG["epochs"]+1):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader,
                desc=f"Epoch {epoch}/{NLP_CFG['epochs']} [NLP]")
    for batch in pbar:
        ids    = batch["input_ids"].to(DEVICE)
        mask   = batch["attention_mask"].to(DEVICE)
        tlabels= batch["token_labels"].to(DEVICE)
        clabels= batch["clip_label"].to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = compute_loss(logits, tlabels, clabels, NLP_CFG["topk"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")
    train_loss /= len(train_loader)
    vm_max, vm_topk = evaluate(model, val_loader)
    print(f"\nEpoch {epoch} | train_loss={train_loss:.4f}")
    print(f"  [max]  AUC={vm_max['auc']*100:.4f}  "
          f"F1={vm_max['f1']*100:.4f}  ACC={vm_max['acc']*100:.4f}")
    print(f"  [topk] AUC={vm_topk['auc']*100:.4f}  "
          f"F1={vm_topk['f1']*100:.4f}  ACC={vm_topk['acc']*100:.4f}")
    log.append({"epoch":epoch,"train_loss":train_loss,
                **{f"max_{k}":v for k,v in vm_max.items()},
                **{f"topk_{k}":v for k,v in vm_topk.items()}})
    if vm_topk["auc"] > best_auc:
        best_auc = vm_topk["auc"]
        torch.save({"epoch":epoch,"state_dict":model.state_dict(),
                    "val_auc":best_auc,"cfg":NLP_CFG}, NLP_SAVE_PATH)
        print(f"  ✅ Best 저장 (topk AUC={best_auc*100:.4f}%)")

with open(os.path.join(RESULTS_DIR,"nlp_train_log.json"),"w") as f:
    json.dump(log, f, indent=2)
print(f"\n✅ NLP 학습 완료 | Best topk AUC: {best_auc*100:.4f}%")
print(f"   저장: {NLP_SAVE_PATH}")
