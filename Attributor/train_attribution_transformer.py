"""
Step 4. Artifact Attribution Transformer (AAT)

목표:
- X3D / AASIST가 추출한 clip-level evidence token을 Transformer에 입력
- binary / modality / technique attribution을 multi-task로 학습

입력 feature:
- z_v:      (K, 2048)
- logit_v:  (K, 1)
- p_v:      (K, 1)

- z_a:      (K, 160)
- logits_a: (K, 2)
- p_a:      (K, 1)

Transformer 입력:
[CLS], [V1], [V2], [V3], [V4], [A1], [A2], [A3], [A4]

Video token:
concat(z_v_i, logit_v_i, p_v_i) -> 2050 dim -> d_model

Audio token:
concat(z_a_i, logits_a_i, p_a_i) -> 163 dim -> d_model

출력 head:
- binary_label: real / fake
- modality_label: real / visual / audio / both
- fav_tech_label: real / faceswap / fsgan / wav2lip / rtvc / faceswap-wav2lip / fsgan-wav2lip

실행:
python train_attribution_transformer.py \
  --feature-path features/fakeavceleb_balanced_20pm.pt \
  --epochs 100

기준 baseline:
Score-only MLP
- Test binary F1   = 0.9143
- Test modality F1 = 0.9138
- Test tech F1     = 0.6497

Embedding MLP
- Test binary F1   = 0.9143
- Test modality F1 = 0.9138
- Test tech F1     = 0.7041
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

    # 작은 실험에서 재현성 우선
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class EvidenceTokenDataset(Dataset):
    def __init__(
        self,
        feature_path: str,
        normalize_embeddings: bool = True,
        max_tokens_per_modality: int = 4,
    ):
        self.feature_path = feature_path
        self.normalize_embeddings = normalize_embeddings
        self.max_tokens_per_modality = max_tokens_per_modality

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

        self.video_token_dim, self.audio_token_dim = self._infer_token_dims()
        print("[INFO] video_token_dim:", self.video_token_dim)
        print("[INFO] audio_token_dim:", self.audio_token_dim)
        print("[INFO] max_tokens_per_modality:", self.max_tokens_per_modality)

    def _print_distribution(self):
        print("[INFO] method distribution:")
        print(Counter([x["method"] for x in self.features]))

        print("[INFO] binary distribution:")
        print(Counter([int(x["binary_label"]) for x in self.features]))

        print("[INFO] modality distribution:")
        print(Counter([int(x["modality_label"]) for x in self.features]))

        print("[INFO] fav tech distribution:")
        print(Counter([int(x["fav_tech_label"]) for x in self.features]))

    def _infer_token_dims(self):
        item = self.features[0]
        v_tokens, a_tokens = self._make_tokens(item)
        return int(v_tokens.size(-1)), int(a_tokens.size(-1))

    def _pad_or_trim(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        x: (K, D)
        return: (target_len, D)
        """
        K, D = x.shape

        if K == target_len:
            return x

        if K > target_len:
            return x[:target_len]

        pad = torch.zeros(target_len - K, D, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    def _make_tokens(self, item):
        """
        return:
        - v_tokens: (K, video_token_dim)
        - a_tokens: (K, audio_token_dim)
        """
        z_v = item["z_v"].float()              # (K, 2048)
        logit_v = item["logit_v"].float()      # (K, 1)
        p_v = item["p_v"].float()              # (K, 1)

        z_a = item["z_a"].float()              # (K, 160)
        logits_a = item["logits_a"].float()    # (K, 2)
        p_a = item["p_a"].float()              # (K, 1)

        # shape 안전화
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

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]

        v_tokens, a_tokens = self._make_tokens(item)

        y_binary = torch.tensor(int(item["binary_label"]), dtype=torch.long)
        y_modality = torch.tensor(int(item["modality_label"]), dtype=torch.long)
        y_tech = torch.tensor(int(item["fav_tech_label"]), dtype=torch.long)

        meta = {
            "sample_id": item["sample_id"],
            "method": item["method"],
            "video_path": item["video_path"],
        }

        return v_tokens, a_tokens, y_binary, y_modality, y_tech, meta


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

class ArtifactAttributionTransformer(nn.Module):
    def __init__(
        self,
        video_token_dim: int,
        audio_token_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_tokens_per_modality: int = 4,
        num_binary: int = 2,
        num_modality: int = 4,
        num_tech: int = 7,
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

        # 0: CLS, 1: video, 2: audio
        self.type_embed = nn.Embedding(3, d_model)

        # position: CLS + K video + K audio
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

        self.pre_head_norm = nn.LayerNorm(d_model)

        self.binary_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_binary),
        )

        self.modality_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_modality),
        )

        self.tech_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_tech),
        )

    def forward(self, v_tokens, a_tokens):
        """
        v_tokens: (B, K, video_token_dim)
        a_tokens: (B, K, audio_token_dim)
        """
        B = v_tokens.size(0)
        K = self.max_tokens_per_modality

        v = self.video_proj(v_tokens)
        a = self.audio_proj(a_tokens)

        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, v, a], dim=1)  # (B, 1+K+K, d_model)

        # type embedding
        type_ids = torch.cat([
            torch.zeros(1, dtype=torch.long, device=x.device),
            torch.ones(K, dtype=torch.long, device=x.device),
            torch.full((K,), 2, dtype=torch.long, device=x.device),
        ], dim=0)

        x = x + self.type_embed(type_ids).unsqueeze(0)
        x = x + self.pos_embed[:, :x.size(1), :]

        h = self.encoder(x)
        cls_h = self.pre_head_norm(h[:, 0])

        return {
            "binary": self.binary_head(cls_h),
            "modality": self.modality_head(cls_h),
            "tech": self.tech_head(cls_h),
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
        v_tokens, a_tokens, y_binary, y_modality, y_tech, _ = batch

        v_tokens = v_tokens.to(device)
        a_tokens = a_tokens.to(device)

        y_binary = y_binary.to(device)
        y_modality = y_modality.to(device)
        y_tech = y_tech.to(device)

        outputs = model(v_tokens, a_tokens)

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

        bs = v_tokens.size(0)
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
        v_tokens, a_tokens, y_binary, y_modality, y_tech, _ = batch

        v_tokens = v_tokens.to(device)
        a_tokens = a_tokens.to(device)

        y_binary = y_binary.to(device)
        y_modality = y_modality.to(device)
        y_tech = y_tech.to(device)

        outputs = model(v_tokens, a_tokens)

        loss, _ = compute_loss(
            outputs,
            y_binary,
            y_modality,
            y_tech,
            weights,
        )

        bs = v_tokens.size(0)
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
        default=3e-4,
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max-tokens-per-modality",
        type=int,
        default=4,
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
        default="runs_aat/best_aat.pt",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    dataset = EvidenceTokenDataset(
        args.feature_path,
        normalize_embeddings=not args.no_normalize_embeddings,
        max_tokens_per_modality=args.max_tokens_per_modality,
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

    model = ArtifactAttributionTransformer(
        video_token_dim=dataset.video_token_dim,
        audio_token_dim=dataset.audio_token_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_tokens_per_modality=args.max_tokens_per_modality,
        num_binary=2,
        num_modality=4,
        num_tech=7,
    ).to(device)

    print("[INFO] model:")
    print(model)

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
                "video_token_dim": dataset.video_token_dim,
                "audio_token_dim": dataset.audio_token_dim,
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