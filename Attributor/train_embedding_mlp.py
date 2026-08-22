"""
Step 3. Embedding MLP baseline for Attribution

입력 feature:
- mean(z_v) over clips      : 2048 dim
- std(z_v) over clips       : 2048 dim
- mean(z_a) over chunks     : 160 dim
- std(z_a) over chunks      : 160 dim
- score features            : 8 dim
  - mean(p_v)
  - std(p_v)
  - max(p_v)
  - mean(p_a)
  - std(p_a)
  - max(p_a)
  - abs(mean(p_v) - mean(p_a))
  - mean(p_v) * mean(p_a)

총 입력 차원:
2048*2 + 160*2 + 8 = 4424

출력 head:
- binary_label: real / fake
- modality_label: real / visual / audio / both
- fav_tech_label: real / faceswap / fsgan / wav2lip / rtvc / faceswap-wav2lip / fsgan-wav2lip

실행:
python train_embedding_mlp.py \
  --feature-path features/fakeavceleb_balanced_20pm.pt \
  --epochs 100
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


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class EmbeddingFeatureDataset(Dataset):
    def __init__(
        self,
        feature_path: str,
        normalize_embeddings: bool = True,
    ):
        self.feature_path = feature_path
        self.normalize_embeddings = normalize_embeddings

        data = torch.load(feature_path, map_location="cpu", weights_only=False)

        self.data = data
        self.features = data["features"]

        if len(self.features) == 0:
            raise RuntimeError("feature file에 features가 비어 있습니다.")

        self.binary_map = data.get("binary_map", {})
        self.modality_map = data.get("modality_map", {})
        self.fav_tech_map = data.get("fav_tech_map", {})

        print("[INFO] loaded:", feature_path)
        print("[INFO] num samples:", len(self.features))
        print("[INFO] binary map:", self.binary_map)
        print("[INFO] modality map:", self.modality_map)
        print("[INFO] fav tech map:", self.fav_tech_map)

        self._print_distribution()

        self.input_dim = self._infer_input_dim()
        print("[INFO] inferred input_dim:", self.input_dim)

    def _print_distribution(self):
        print("[INFO] method distribution:")
        print(Counter([x["method"] for x in self.features]))

        print("[INFO] binary distribution:")
        print(Counter([int(x["binary_label"]) for x in self.features]))

        print("[INFO] modality distribution:")
        print(Counter([int(x["modality_label"]) for x in self.features]))

        print("[INFO] fav tech distribution:")
        print(Counter([int(x["fav_tech_label"]) for x in self.features]))

    def _make_feature(self, item):
        """
        하나의 sample에서 embedding MLP 입력 feature 생성.
        """
        z_v = item["z_v"].float()  # (K, Dv)
        z_a = item["z_a"].float()  # (K, Da)

        p_v = item["p_v"].float().view(-1)  # (K,)
        p_a = item["p_a"].float().view(-1)  # (K,)

        # embedding statistics
        zv_mean = z_v.mean(dim=0)
        zv_std = z_v.std(dim=0, unbiased=False)

        za_mean = z_a.mean(dim=0)
        za_std = z_a.std(dim=0, unbiased=False)

        if self.normalize_embeddings:
            zv_mean = torch.nn.functional.normalize(zv_mean, dim=0)
            zv_std = torch.nn.functional.normalize(zv_std, dim=0)
            za_mean = torch.nn.functional.normalize(za_mean, dim=0)
            za_std = torch.nn.functional.normalize(za_std, dim=0)

        # score statistics
        pv_mean = p_v.mean()
        pv_std = p_v.std(unbiased=False)
        pv_max = p_v.max()

        pa_mean = p_a.mean()
        pa_std = p_a.std(unbiased=False)
        pa_max = p_a.max()

        diff = torch.abs(pv_mean - pa_mean)
        prod = pv_mean * pa_mean

        score_feat = torch.stack([
            pv_mean,
            pv_std,
            pv_max,
            pa_mean,
            pa_std,
            pa_max,
            diff,
            prod,
        ], dim=0)

        x = torch.cat([
            zv_mean,
            zv_std,
            za_mean,
            za_std,
            score_feat,
        ], dim=0)

        return x

    def _infer_input_dim(self):
        x = self._make_feature(self.features[0])
        return int(x.numel())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]

        x = self._make_feature(item)

        y_binary = torch.tensor(int(item["binary_label"]), dtype=torch.long)
        y_modality = torch.tensor(int(item["modality_label"]), dtype=torch.long)
        y_tech = torch.tensor(int(item["fav_tech_label"]), dtype=torch.long)

        meta = {
            "sample_id": item["sample_id"],
            "method": item["method"],
            "video_path": item["video_path"],
        }

        return x, y_binary, y_modality, y_tech, meta


# ---------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------

def stratified_split_by_label(
    dataset,
    label_name="fav_tech_label",
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
):
    """
    fav_tech_label 기준 stratified split.
    method별 20개 같은 소량 데이터에서 class 분포를 유지하기 위해 사용.
    """
    rng = random.Random(seed)

    label_to_indices = defaultdict(list)

    for i, item in enumerate(dataset.features):
        label = int(item[label_name])
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

    def print_split_dist(name, indices):
        labels = [int(dataset.features[i][label_name]) for i in indices]
        print(f"[INFO] {name} {label_name} distribution:", Counter(labels))

    print_split_dist("train", train_idx)
    print_split_dist("val", val_idx)
    print_split_dist("test", test_idx)

    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=512,
        bottleneck_dim=256,
        num_binary=2,
        num_modality=4,
        num_tech=7,
        dropout=0.4,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.binary_head = nn.Linear(bottleneck_dim, num_binary)
        self.modality_head = nn.Linear(bottleneck_dim, num_modality)
        self.tech_head = nn.Linear(bottleneck_dim, num_tech)

    def forward(self, x):
        h = self.encoder(x)

        return {
            "binary": self.binary_head(h),
            "modality": self.modality_head(h),
            "tech": self.tech_head(h),
        }


# ---------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------

def compute_loss(outputs, y_binary, y_modality, y_tech, weights):
    ce = nn.CrossEntropyLoss()

    loss_binary = ce(outputs["binary"], y_binary)
    loss_modality = ce(outputs["modality"], y_modality)
    loss_tech = ce(outputs["tech"], y_tech)

    loss = (
        weights["binary"] * loss_binary
        + weights["modality"] * loss_modality
        + weights["tech"] * loss_tech
    )

    return loss, {
        "binary": float(loss_binary.detach().cpu()),
        "modality": float(loss_modality.detach().cpu()),
        "tech": float(loss_tech.detach().cpu()),
    }


def train_one_epoch(model, loader, optimizer, device, loss_weights, grad_clip=1.0):
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x, y_binary, y_modality, y_tech, _ = batch

        x = x.to(device)
        y_binary = y_binary.to(device)
        y_modality = y_modality.to(device)
        y_tech = y_tech.to(device)

        outputs = model(x)

        loss, _ = compute_loss(
            outputs,
            y_binary,
            y_modality,
            y_tech,
            loss_weights,
        )

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_binary_true = []
    all_binary_pred = []

    all_mod_true = []
    all_mod_pred = []

    all_tech_true = []
    all_tech_pred = []

    total_loss = 0.0
    total_count = 0

    weights = {
        "binary": 1.0,
        "modality": 1.0,
        "tech": 1.0,
    }

    for batch in loader:
        x, y_binary, y_modality, y_tech, _ = batch

        x = x.to(device)
        y_binary = y_binary.to(device)
        y_modality = y_modality.to(device)
        y_tech = y_tech.to(device)

        outputs = model(x)

        loss, _ = compute_loss(outputs, y_binary, y_modality, y_tech, weights)

        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_count += bs

        binary_pred = outputs["binary"].argmax(dim=1)
        mod_pred = outputs["modality"].argmax(dim=1)
        tech_pred = outputs["tech"].argmax(dim=1)

        all_binary_true.extend(y_binary.cpu().numpy().tolist())
        all_binary_pred.extend(binary_pred.cpu().numpy().tolist())

        all_mod_true.extend(y_modality.cpu().numpy().tolist())
        all_mod_pred.extend(mod_pred.cpu().numpy().tolist())

        all_tech_true.extend(y_tech.cpu().numpy().tolist())
        all_tech_pred.extend(tech_pred.cpu().numpy().tolist())

    avg_loss = total_loss / max(total_count, 1)

    metrics = {
        "loss": avg_loss,

        "binary_acc": accuracy_score(all_binary_true, all_binary_pred),
        "binary_macro_f1": f1_score(
            all_binary_true,
            all_binary_pred,
            average="macro",
            zero_division=0,
        ),

        "modality_acc": accuracy_score(all_mod_true, all_mod_pred),
        "modality_macro_f1": f1_score(
            all_mod_true,
            all_mod_pred,
            average="macro",
            zero_division=0,
        ),

        "tech_acc": accuracy_score(all_tech_true, all_tech_pred),
        "tech_macro_f1": f1_score(
            all_tech_true,
            all_tech_pred,
            average="macro",
            zero_division=0,
        ),

        "binary_true": all_binary_true,
        "binary_pred": all_binary_pred,
        "mod_true": all_mod_true,
        "mod_pred": all_mod_pred,
        "tech_true": all_tech_true,
        "tech_pred": all_tech_pred,
    }

    return metrics


def print_final_report(metrics, title):
    print("=" * 80)
    print(title)
    print("=" * 80)

    print("[Binary]")
    print(classification_report(
        metrics["binary_true"],
        metrics["binary_pred"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(metrics["binary_true"], metrics["binary_pred"]))

    print("\n[Modality]")
    print(classification_report(
        metrics["mod_true"],
        metrics["mod_pred"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(metrics["mod_true"], metrics["mod_pred"]))

    print("\n[Technique]")
    print(classification_report(
        metrics["tech_true"],
        metrics["tech_pred"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(metrics["tech_true"], metrics["tech_pred"]))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--bottleneck-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="embedding L2 normalize 비활성화",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="runs_embedding_mlp/best_embedding_mlp.pt",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    dataset = EmbeddingFeatureDataset(
        args.feature_path,
        normalize_embeddings=not args.no_normalize_embeddings,
    )

    train_idx, val_idx, test_idx = stratified_split_by_label(
        dataset,
        label_name="fav_tech_label",
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

    model = EmbeddingMLP(
        in_dim=dataset.input_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_binary=2,
        num_modality=4,
        num_tech=7,
        dropout=args.dropout,
    ).to(device)

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

    loss_weights = {
        "binary": 0.5,
        "modality": 1.0,
        "tech": 1.0,
    }

    best_val = -1.0
    best_state = None

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            grad_clip=1.0,
        )

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)

        # technique attribution이 핵심이므로 tech macro-F1 기준 저장
        score = val_metrics["tech_macro_f1"]

        if score > best_val:
            best_val = score
            best_state = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "input_dim": dataset.input_dim,
                "best_val_tech_macro_f1": best_val,
            }
            torch.save(best_state, save_path)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"lr={current_lr:.6f} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"bin_f1={val_metrics['binary_macro_f1']:.4f} | "
                f"mod_f1={val_metrics['modality_macro_f1']:.4f} | "
                f"tech_f1={val_metrics['tech_macro_f1']:.4f}"
            )

    print("[INFO] best val tech macro-F1:", best_val)
    print("[INFO] loading best model:", save_path)

    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    print_final_report(val_metrics, "Validation Report")
    print_final_report(test_metrics, "Test Report")

    print("=" * 80)
    print("[SUMMARY]")
    print(f"Val  binary F1:   {val_metrics['binary_macro_f1']:.4f}")
    print(f"Val  modality F1: {val_metrics['modality_macro_f1']:.4f}")
    print(f"Val  tech F1:     {val_metrics['tech_macro_f1']:.4f}")

    print(f"Test binary F1:   {test_metrics['binary_macro_f1']:.4f}")
    print(f"Test modality F1: {test_metrics['modality_macro_f1']:.4f}")
    print(f"Test tech F1:     {test_metrics['tech_macro_f1']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()