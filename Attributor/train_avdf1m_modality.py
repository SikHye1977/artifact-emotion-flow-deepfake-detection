"""
AVDF1M in-domain modality attribution training.

목표:
- AVDF1M feature로 modality_label만 학습
- target:
  0 real
  1 visual_modified
  2 audio_modified
  3 both_modified

지원 모델:
1. score
   - p_v, p_a score statistics만 사용

2. embedding
   - mean/std(z_v), mean/std(z_a), score statistics 사용

3. aat
   - clip-level evidence tokens 사용
   - [CLS], [V1..VK], [A1..AK]

실행 예시:

Score-only MLP:
python train_avdf1m_modality.py \
  --model-type score \
  --feature-path features/avdf1m_n1000.pt \
  --epochs 100

Embedding MLP:
python train_avdf1m_modality.py \
  --model-type embedding \
  --feature-path features/avdf1m_n1000.pt \
  --epochs 100

AAT:
python train_avdf1m_modality.py \
  --model-type aat \
  --feature-path features/avdf1m_n1000.pt \
  --epochs 100 \
  --d-model 64 \
  --num-layers 1 \
  --dropout 0.3
"""

import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class AVDF1MModalityDataset(Dataset):
    def __init__(
        self,
        feature_path: str,
        model_type: str,
        normalize_embeddings: bool = True,
        max_tokens_per_modality: int = 4,
    ):
        assert model_type in ["score", "embedding", "aat"]

        self.feature_path = feature_path
        self.model_type = model_type
        self.normalize_embeddings = normalize_embeddings
        self.max_tokens_per_modality = max_tokens_per_modality

        data = torch.load(feature_path, map_location="cpu", weights_only=False)

        self.data = data
        self.features = data["features"]

        if len(self.features) == 0:
            raise RuntimeError("feature file에 features가 비어 있습니다.")

        # modality_label이 유효한 샘플만 사용
        self.features = [
            x for x in self.features
            if int(x.get("modality_label", -1)) >= 0
        ]

        if len(self.features) == 0:
            raise RuntimeError("유효한 modality_label을 가진 샘플이 없습니다.")

        self.binary_map = data.get("binary_map", {})
        self.modality_map = data.get("modality_map", {})

        print("[INFO] loaded:", feature_path)
        print("[INFO] model_type:", model_type)
        print("[INFO] num valid samples:", len(self.features))
        print("[INFO] binary map:", self.binary_map)
        print("[INFO] modality map:", self.modality_map)

        self._print_distribution()

        if self.model_type == "score":
            self.input_dim = 8
            print("[INFO] score input_dim:", self.input_dim)

        elif self.model_type == "embedding":
            self.input_dim = self._infer_embedding_dim()
            print("[INFO] embedding input_dim:", self.input_dim)

        elif self.model_type == "aat":
            self.video_token_dim, self.audio_token_dim = self._infer_token_dims()
            print("[INFO] video_token_dim:", self.video_token_dim)
            print("[INFO] audio_token_dim:", self.audio_token_dim)
            print("[INFO] max_tokens_per_modality:", self.max_tokens_per_modality)

    def _print_distribution(self):
        print("[INFO] binary distribution:")
        print(Counter([int(x["binary_label"]) for x in self.features]))

        print("[INFO] modality distribution:")
        print(Counter([int(x["modality_label"]) for x in self.features]))

        print("[INFO] modify_type distribution:")
        print(Counter([x.get("modify_type", "") for x in self.features]))

        print("[INFO] audio_model distribution:")
        print(Counter([x.get("audio_model", "") for x in self.features]))

    def _score_feature(self, item):
        p_v = item["p_v"].float().view(-1)
        p_a = item["p_a"].float().view(-1)

        pv_mean = p_v.mean()
        pv_std = p_v.std(unbiased=False)
        pv_max = p_v.max()

        pa_mean = p_a.mean()
        pa_std = p_a.std(unbiased=False)
        pa_max = p_a.max()

        diff = torch.abs(pv_mean - pa_mean)
        prod = pv_mean * pa_mean

        x = torch.stack([
            pv_mean,
            pv_std,
            pv_max,
            pa_mean,
            pa_std,
            pa_max,
            diff,
            prod,
        ], dim=0)

        return x

    def _embedding_feature(self, item):
        z_v = item["z_v"].float()
        z_a = item["z_a"].float()

        zv_mean = z_v.mean(dim=0)
        zv_std = z_v.std(dim=0, unbiased=False)

        za_mean = z_a.mean(dim=0)
        za_std = z_a.std(dim=0, unbiased=False)

        if self.normalize_embeddings:
            zv_mean = torch.nn.functional.normalize(zv_mean, dim=0)
            zv_std = torch.nn.functional.normalize(zv_std, dim=0)
            za_mean = torch.nn.functional.normalize(za_mean, dim=0)
            za_std = torch.nn.functional.normalize(za_std, dim=0)

        score_feat = self._score_feature(item)

        x = torch.cat([
            zv_mean,
            zv_std,
            za_mean,
            za_std,
            score_feat,
        ], dim=0)

        return x

    def _infer_embedding_dim(self):
        x = self._embedding_feature(self.features[0])
        return int(x.numel())

    def _pad_or_trim(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        K, D = x.shape

        if K == target_len:
            return x

        if K > target_len:
            return x[:target_len]

        pad = torch.zeros(target_len - K, D, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    def _make_tokens(self, item):
        z_v = item["z_v"].float()
        logit_v = item["logit_v"].float()
        p_v = item["p_v"].float()

        z_a = item["z_a"].float()
        logits_a = item["logits_a"].float()
        p_a = item["p_a"].float()

        if logit_v.ndim == 1:
            logit_v = logit_v.unsqueeze(-1)
        if p_v.ndim == 1:
            p_v = p_v.unsqueeze(-1)
        if logits_a.ndim == 1:
            logits_a = logits_a.unsqueeze(0)
        if p_a.ndim == 1:
            p_a = p_a.unsqueeze(-1)

        if self.normalize_embeddings:
            z_v = torch.nn.functional.normalize(z_v, dim=-1)
            z_a = torch.nn.functional.normalize(z_a, dim=-1)

        v_tokens = torch.cat([z_v, logit_v, p_v], dim=-1)
        a_tokens = torch.cat([z_a, logits_a, p_a], dim=-1)

        v_tokens = self._pad_or_trim(v_tokens, self.max_tokens_per_modality)
        a_tokens = self._pad_or_trim(a_tokens, self.max_tokens_per_modality)

        return v_tokens, a_tokens

    def _infer_token_dims(self):
        v_tokens, a_tokens = self._make_tokens(self.features[0])
        return int(v_tokens.size(-1)), int(a_tokens.size(-1))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]

        y = torch.tensor(int(item["modality_label"]), dtype=torch.long)

        meta = {
            "sample_id": item.get("sample_id", ""),
            "video_path": item.get("video_path", ""),
            "modify_type": item.get("modify_type", ""),
            "audio_model": item.get("audio_model", ""),
        }

        if self.model_type == "score":
            x = self._score_feature(item)
            return x, y, meta

        if self.model_type == "embedding":
            x = self._embedding_feature(item)
            return x, y, meta

        if self.model_type == "aat":
            v_tokens, a_tokens = self._make_tokens(item)
            return v_tokens, a_tokens, y, meta

        raise ValueError(self.model_type)


# ---------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------

def stratified_split_by_modality(
    dataset,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
):
    rng = random.Random(seed)

    label_to_indices = defaultdict(list)

    for i, item in enumerate(dataset.features):
        label = int(item["modality_label"])
        label_to_indices[label].append(i)

    train_idx = []
    val_idx = []
    test_idx = []

    for label, indices in sorted(label_to_indices.items()):
        indices = indices[:]
        rng.shuffle(indices)

        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    print("[INFO] split sizes:")
    print("  train:", len(train_idx))
    print("  val:  ", len(val_idx))
    print("  test: ", len(test_idx))

    def print_dist(name, indices):
        labels = [int(dataset.features[i]["modality_label"]) for i in indices]
        print(f"[INFO] {name} modality distribution:", Counter(labels))

    print_dist("train", train_idx)
    print_dist("val", val_idx)
    print_dist("test", test_idx)

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class ScoreModalityMLP(nn.Module):
    def __init__(
        self,
        in_dim=8,
        hidden_dim=64,
        num_classes=4,
        dropout=0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EmbeddingModalityMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=512,
        bottleneck_dim=256,
        num_classes=4,
        dropout=0.4,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(bottleneck_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class AATModalityTransformer(nn.Module):
    def __init__(
        self,
        video_token_dim,
        audio_token_dim,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.3,
        max_tokens_per_modality=4,
        num_classes=4,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_tokens_per_modality = max_tokens_per_modality
        self.seq_len = 1 + max_tokens_per_modality * 2

        self.video_proj = nn.Sequential(
            nn.Linear(video_token_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_token_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 0 CLS, 1 video, 2 audio
        self.type_embed = nn.Embedding(3, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, v_tokens, a_tokens):
        B = v_tokens.size(0)
        K = self.max_tokens_per_modality

        v = self.video_proj(v_tokens)
        a = self.audio_proj(a_tokens)

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, v, a], dim=1)

        type_ids = torch.cat([
            torch.zeros(1, dtype=torch.long, device=x.device),
            torch.ones(K, dtype=torch.long, device=x.device),
            torch.full((K,), 2, dtype=torch.long, device=x.device),
        ], dim=0)

        x = x + self.type_embed(type_ids).unsqueeze(0)
        x = x + self.pos_embed[:, :x.size(1), :]

        h = self.encoder(x)
        cls_h = self.norm(h[:, 0])

        return self.head(cls_h)


# ---------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------

def class_weights_from_dataset(dataset, train_idx, device):
    labels = [int(dataset.features[i]["modality_label"]) for i in train_idx]
    counts = Counter(labels)

    num_classes = 4
    total = sum(counts.values())

    weights = []
    for c in range(num_classes):
        count = counts.get(c, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("[INFO] class weights:", weights.detach().cpu().tolist())
    return weights


def train_one_epoch(model, loader, optimizer, criterion, device, model_type, grad_clip=1.0):
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        if model_type in ["score", "embedding"]:
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

        elif model_type == "aat":
            v_tokens, a_tokens, y, _ = batch
            v_tokens = v_tokens.to(device)
            a_tokens = a_tokens.to(device)
            y = y.to(device)

            logits = model(v_tokens, a_tokens)

        else:
            raise ValueError(model_type)

        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, model_type):
    model.eval()

    all_true = []
    all_pred = []

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        if model_type in ["score", "embedding"]:
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

        elif model_type == "aat":
            v_tokens, a_tokens, y, _ = batch
            v_tokens = v_tokens.to(device)
            a_tokens = a_tokens.to(device)
            y = y.to(device)

            logits = model(v_tokens, a_tokens)

        else:
            raise ValueError(model_type)

        loss = criterion(logits, y)

        pred = logits.argmax(dim=1)

        bs = y.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs

        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())

    avg_loss = total_loss / max(total_count, 1)

    metrics = {
        "loss": avg_loss,
        "acc": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "true": all_true,
        "pred": all_pred,
    }

    return metrics


def print_report(metrics, title):
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(f"Loss:        {metrics['loss']:.4f}")
    print(f"Accuracy:    {metrics['acc']:.4f}")
    print(f"Macro-F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print()

    target_names = [
        "real",
        "visual_modified",
        "audio_modified",
        "both_modified",
    ]

    print(classification_report(
        metrics["true"],
        metrics["pred"],
        labels=[0, 1, 2, 3],
        target_names=target_names,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(metrics["true"], metrics["pred"], labels=[0, 1, 2, 3]))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-type", required=True, choices=["score", "embedding", "aat"])
    parser.add_argument("--feature-path", required=True)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--bottleneck-dim", type=int, default=256)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dim-feedforward", type=int, default=128)

    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max-tokens-per-modality", type=int, default=4)
    parser.add_argument("--no-normalize-embeddings", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    print("[INFO] args:", vars(args))

    # model별 기본값
    if args.lr is None:
        if args.model_type == "score":
            args.lr = 1e-3
        elif args.model_type == "embedding":
            args.lr = 5e-4
        else:
            args.lr = 3e-4

    if args.dropout is None:
        if args.model_type == "score":
            args.dropout = 0.2
        elif args.model_type == "embedding":
            args.dropout = 0.4
        else:
            args.dropout = 0.3

    if args.save_path is None:
        args.save_path = f"runs_avdf1m_modality/best_{args.model_type}.pt"

    dataset = AVDF1MModalityDataset(
        feature_path=args.feature_path,
        model_type=args.model_type,
        normalize_embeddings=not args.no_normalize_embeddings,
        max_tokens_per_modality=args.max_tokens_per_modality,
    )

    train_idx, val_idx, test_idx = stratified_split_by_modality(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.model_type == "score":
        model = ScoreModalityMLP(
            in_dim=dataset.input_dim,
            hidden_dim=64,
            num_classes=4,
            dropout=args.dropout,
        ).to(device)

    elif args.model_type == "embedding":
        model = EmbeddingModalityMLP(
            in_dim=dataset.input_dim,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.bottleneck_dim,
            num_classes=4,
            dropout=args.dropout,
        ).to(device)

    elif args.model_type == "aat":
        model = AATModalityTransformer(
            video_token_dim=dataset.video_token_dim,
            audio_token_dim=dataset.audio_token_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            max_tokens_per_modality=args.max_tokens_per_modality,
            num_classes=4,
        ).to(device)

    else:
        raise ValueError(args.model_type)

    print("[INFO] model:")
    print(model)

    class_weights = class_weights_from_dataset(dataset, train_idx, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    best_val = -1.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            model_type=args.model_type,
            grad_clip=1.0,
        )

        scheduler.step()

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            model_type=args.model_type,
        )

        score = val_metrics["macro_f1"]

        if score > best_val:
            best_val = score
            ckpt = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "model_type": args.model_type,
                "best_val_macro_f1": best_val,
            }

            if args.model_type == "embedding":
                ckpt["input_dim"] = dataset.input_dim

            if args.model_type == "aat":
                ckpt["video_token_dim"] = dataset.video_token_dim
                ckpt["audio_token_dim"] = dataset.audio_token_dim

            torch.save(ckpt, save_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"lr={current_lr:.6f} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['acc']:.4f} | "
                f"val_macro_f1={val_metrics['macro_f1']:.4f}"
            )

    print("[INFO] best val macro-F1:", best_val)
    print("[INFO] loading best model:", save_path)

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        model_type=args.model_type,
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        model_type=args.model_type,
    )

    print_report(val_metrics, "Validation Report")
    print_report(test_metrics, "Test Report")

    print("=" * 80)
    print("[SUMMARY]")
    print(f"Val  Acc:        {val_metrics['acc']:.4f}")
    print(f"Val  Macro-F1:   {val_metrics['macro_f1']:.4f}")
    print(f"Val  Weighted-F1:{val_metrics['weighted_f1']:.4f}")
    print(f"Test Acc:        {test_metrics['acc']:.4f}")
    print(f"Test Macro-F1:   {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted-F1:{test_metrics['weighted_f1']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()