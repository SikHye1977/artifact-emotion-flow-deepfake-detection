"""
train_nlp.py
─────────────────────────────────────────────────────────────────────
XLM-RoBERTa Binary Classifier fine-tune

입력 : dataset_train.csv / dataset_val.csv
출력 : nlp_best.pth  (best val AUC 기준)
       train_log.json
"""

import json, os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

# ── 설정 ─────────────────────────────────────────────────────────
CFG = {
    "model_name"  : "xlm-roberta-base",
    "max_len"     : 128,        # transcript 평균 길이 고려
    "batch_size"  : 32,
    "epochs"      : 5,
    "lr"          : 2e-5,       # RoBERTa fine-tune 표준
    "warmup_ratio": 0.1,
    "seed"        : 42,
    "out_dir"     : os.path.dirname(os.path.abspath(__file__)),
}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────
class TranscriptDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ── Model ─────────────────────────────────────────────────────────
class NLPClassifier(nn.Module):
    """
    XLM-RoBERTa [CLS] 토큰 → Dropout → Linear → sigmoid
    """
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size   # 768
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, input_ids, attention_mask):
        out   = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask)
        cls   = out.last_hidden_state[:, 0, :]   # [CLS]
        logit = self.head(cls).squeeze(-1)        # (B,)
        return logit

# ── 평가 함수 ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    criterion  = nn.BCEWithLogitsLoss()

    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        y    = batch["label"].to(DEVICE)

        logits = model(ids, mask)
        loss   = criterion(logits, y)
        total_loss += loss.item()

        all_logits.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    probs  = np.array(all_logits)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    return {
        "loss": total_loss / len(loader),
        "auc":  roc_auc_score(labels, probs),
        "f1":   f1_score(labels, preds, zero_division=0),
        "acc":  accuracy_score(labels, preds)
    }

# ── 메인 ─────────────────────────────────────────────────────────
def main():
    out_dir    = CFG["out_dir"]
    train_path = os.path.join(out_dir, "dataset_train.csv")
    val_path   = os.path.join(out_dir, "dataset_val.csv")

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    print(f"train: {len(df_train):,}  val: {len(df_val):,}")

    # 토크나이저
    print(f"토크나이저 로드: {CFG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

    train_ds = TranscriptDataset(df_train, tokenizer, CFG["max_len"])
    val_ds   = TranscriptDataset(df_val,   tokenizer, CFG["max_len"])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    # 모델
    print("모델 로드...")
    model     = NLPClassifier(CFG["model_name"]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"],
                                  weight_decay=0.01)

    # Warmup + Cosine LR 스케줄러
    total_steps   = len(train_loader) * CFG["epochs"]
    warmup_steps  = int(total_steps * CFG["warmup_ratio"])
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)

    best_auc  = 0.0
    log       = []
    best_path = os.path.join(out_dir, "nlp_best.pth")

    for epoch in range(1, CFG["epochs"] + 1):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['epochs']} [train]")

        for batch in pbar:
            ids    = batch["input_ids"].to(DEVICE)
            mask   = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss /= len(train_loader)

        # ── Val ────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader)

        print(f"\nEpoch {epoch} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} | "
              f"AUC={val_metrics['auc']:.4f} | "
              f"F1={val_metrics['f1']:.4f} | "
              f"ACC={val_metrics['acc']:.4f}")

        log.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        # best 저장
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_auc":    best_auc,
                "cfg":        CFG
            }, best_path)
            print(f"  ✅ Best 저장 (AUC={best_auc:.4f})")

    # 로그 저장
    log_path = os.path.join(out_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n✅ 학습 완료 | Best AUC: {best_auc:.4f}")
    print(f"  모델: {best_path}")
    print(f"  로그: {log_path}")

if __name__ == "__main__":
    main()
