"""
train_token_nlp.py
─────────────────────────────────────────────────────────────────────
XLM-RoBERTa Token Classification fine-tune

입력 : token_dataset_train.csv / token_dataset_val.csv
출력 : token_nlp_best.pth  (best val AUC 기준)
       token_train_log.json

clip-level score 산출:
  방법 A: max(유효 토큰 fake prob)
  방법 B: top-k mean (k=3)
  → 둘 다 계산해서 비교
"""

import json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

# ── 설정 ─────────────────────────────────────────────────────────
CFG = {
    "model_name"  : "xlm-roberta-base",
    "max_len"     : 128,
    "batch_size"  : 32,
    "epochs"      : 5,
    "lr"          : 2e-5,
    "warmup_ratio": 0.1,
    "seed"        : 42,
    "topk"        : 3,       # top-k mean score
    "out_dir"     : os.path.dirname(os.path.abspath(__file__)),
}

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(CFG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════
class TokenDataset(Dataset):
    def __init__(self, df, max_len):
        self.records  = df.reset_index(drop=True)
        self.max_len  = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]

        token_ids    = json.loads(row["token_ids"])[:self.max_len]
        token_labels = json.loads(row["token_labels"])[:self.max_len]
        clip_label   = int(row["clip_label"])

        L = len(token_ids)
        pad_len = self.max_len - L

        input_ids      = token_ids + [1] * pad_len          # pad_token_id=1
        attention_mask = [1]*L + [0]*pad_len
        labels         = token_labels + [-1]*pad_len         # -1 = ignore

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_labels":   torch.tensor(labels,         dtype=torch.long),
            "clip_label":     torch.tensor(clip_label,     dtype=torch.float),
        }

# ══════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════
class TokenNLPClassifier(nn.Module):
    """
    XLM-RoBERTa → 각 토큰 fake 확률 + clip-level 확률

    두 가지 loss 동시 학습:
      1. token_loss : 각 토큰 BCE (label=-1은 ignore)
      2. clip_loss  : max(token fake prob) 기반 clip BCE
    """
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(model_name)
        hidden        = self.encoder.config.hidden_size
        self.token_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, input_ids, attention_mask):
        out        = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask)
        hidden     = out.last_hidden_state          # (B, L, H)
        token_logits = self.token_head(hidden).squeeze(-1)  # (B, L)
        return token_logits

# ══════════════════════════════════════════════════════════════════
# Loss
# ══════════════════════════════════════════════════════════════════
def compute_loss(token_logits, token_labels, clip_labels, topk=3):
    """
    token_loss : 레이블 있는 토큰만 BCE
    clip_loss  : top-k mean prob → clip BCE
    """
    B, L = token_logits.shape

    # ── token BCE ───────────────────────────────────────────────
    mask = (token_labels != -1)
    if mask.sum() > 0:
        token_loss = nn.functional.binary_cross_entropy_with_logits(
            token_logits[mask].float(),
            token_labels[mask].float()
        )
    else:
        token_loss = torch.tensor(0.0, device=token_logits.device)

    # ── clip BCE via top-k mean ──────────────────────────────────
    probs      = torch.sigmoid(token_logits)             # (B, L)
    # attention_mask 역할: label=-1인 special token 제외
    valid_mask = (token_labels != -1).float()
    probs_valid = probs * valid_mask + (1-valid_mask)*0.0  # special → 0

    # top-k
    k = min(topk, L)
    topk_probs, _ = torch.topk(probs_valid, k, dim=1)    # (B, k)
    clip_prob = topk_probs.mean(dim=1)                    # (B,)
    clip_logit = torch.log(clip_prob.clamp(1e-6, 1-1e-6) /
                           (1 - clip_prob.clamp(1e-6, 1-1e-6)))

    clip_loss = nn.functional.binary_cross_entropy_with_logits(
        clip_logit, clip_labels.float()
    )

    return token_loss + clip_loss, token_loss.item(), clip_loss.item()

# ══════════════════════════════════════════════════════════════════
# 평가
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, topk=3):
    model.eval()
    all_clip_labels, all_max_probs, all_topk_probs = [], [], []
    total_loss = 0.0

    for batch in loader:
        ids    = batch["input_ids"].to(DEVICE)
        mask   = batch["attention_mask"].to(DEVICE)
        tlabels= batch["token_labels"].to(DEVICE)
        clabels= batch["clip_label"].to(DEVICE)

        logits = model(ids, mask)
        loss, _, _ = compute_loss(logits, tlabels, clabels, topk)
        total_loss += loss.item()

        probs       = torch.sigmoid(logits)               # (B, L)
        valid_mask  = (tlabels != -1).float()
        probs_valid = probs * valid_mask

        # max score
        max_prob, _  = probs_valid.max(dim=1)

        # top-k mean score
        k = min(topk, probs_valid.shape[1])
        topk_p, _    = torch.topk(probs_valid, k, dim=1)
        topk_prob    = topk_p.mean(dim=1)

        all_clip_labels.extend(clabels.cpu().numpy())
        all_max_probs.extend(max_prob.cpu().numpy())
        all_topk_probs.extend(topk_prob.cpu().numpy())

    labels   = np.array(all_clip_labels)
    max_p    = np.array(all_max_probs)
    topk_p   = np.array(all_topk_probs)

    def metrics(probs, name):
        preds = (probs > 0.5).astype(int)
        auc = roc_auc_score(labels, probs)
        f1  = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        return auc, f1, acc

    auc_max,  f1_max,  acc_max  = metrics(max_p,  "max")
    auc_topk, f1_topk, acc_topk = metrics(topk_p, "topk")

    return {
        "loss":     total_loss / len(loader),
        "auc_max":  auc_max,  "f1_max":  f1_max,  "acc_max":  acc_max,
        "auc_topk": auc_topk, "f1_topk": f1_topk, "acc_topk": acc_topk,
    }

# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    out_dir = CFG["out_dir"]
    df_train = pd.read_csv(os.path.join(out_dir, "token_dataset_train.csv"))
    df_val   = pd.read_csv(os.path.join(out_dir, "token_dataset_val.csv"))
    print(f"train: {len(df_train):,}  val: {len(df_val):,}")

    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    train_ds = TokenDataset(df_train, CFG["max_len"])
    val_ds   = TokenDataset(df_val,   CFG["max_len"])
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    print("모델 로드...")
    model     = TokenNLPClassifier(CFG["model_name"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=CFG["lr"], weight_decay=0.01)

    total_steps  = len(train_loader) * CFG["epochs"]
    warmup_steps = int(total_steps * CFG["warmup_ratio"])
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    best_auc  = 0.0
    log       = []
    best_path = os.path.join(out_dir, "token_nlp_best.pth")

    for epoch in range(1, CFG["epochs"]+1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{CFG['epochs']} [train]")

        for batch in pbar:
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            tlabels= batch["token_labels"].to(DEVICE)
            clabels= batch["clip_label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss, tl, cl = compute_loss(logits, tlabels, clabels, CFG["topk"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                tok=f"{tl:.3f}", clip=f"{cl:.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        train_loss /= len(train_loader)

        # ── Val ────────────────────────────────────────────────
        vm = evaluate(model, val_loader, CFG["topk"])

        print(f"\nEpoch {epoch} | train_loss={train_loss:.4f} | "
              f"val_loss={vm['loss']:.4f}")
        print(f"  [max score]  AUC={vm['auc_max']:.4f}  "
              f"F1={vm['f1_max']:.4f}  ACC={vm['acc_max']:.4f}")
        print(f"  [topk score] AUC={vm['auc_topk']:.4f}  "
              f"F1={vm['f1_topk']:.4f}  ACC={vm['acc_topk']:.4f}")

        log.append({"epoch": epoch, "train_loss": train_loss, **vm})

        # best = topk AUC 기준
        if vm["auc_topk"] > best_auc:
            best_auc = vm["auc_topk"]
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_auc":    best_auc,
                "cfg":        CFG,
            }, best_path)
            print(f"  ✅ Best 저장 (topk AUC={best_auc:.4f})")

    with open(os.path.join(out_dir, "token_train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ 학습 완료 | Best topk AUC: {best_auc:.4f}")
    print(f"  모델: {best_path}")

    # ── Binary classifier와 성능 비교 ──────────────────────────
    print(f"\n[Binary vs Token-level 비교]")
    print(f"  Binary classifier (이전): AUC=0.7481")
    print(f"  Token-level (이번):       AUC={best_auc:.4f}")
    print(f"  개선: {(best_auc-0.7481)*100:+.2f}%p")

if __name__ == "__main__":
    main()
