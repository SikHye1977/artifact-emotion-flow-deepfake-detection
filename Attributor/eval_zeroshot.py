"""
Zero-shot evaluation for attribution heads.

지원 모델:
- score
- embedding
- aat

예시:

python eval_zeroshot.py \
  --model-type score \
  --checkpoint runs_score_mlp_100pm/best_score_mlp_seed42.pt \
  --feature-path features/avdf1m_n1000.pt

python eval_zeroshot.py \
  --model-type embedding \
  --checkpoint runs_embedding_mlp_100pm/best_embedding_mlp_seed42.pt \
  --feature-path features/avdf1m_n1000.pt

python eval_zeroshot.py \
  --model-type aat \
  --checkpoint runs_aat_100pm/best_aat_seed42.pt \
  --feature-path features/avdf1m_n1000.pt
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from train_score_mlp import ScoreFeatureDataset, ScoreMLP
from train_embedding_mlp import EmbeddingFeatureDataset, EmbeddingMLP
from train_attribution_transformer import EvidenceTokenDataset, ArtifactAttributionTransformer


def filter_valid(y_true, y_pred):
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if int(t) >= 0]
    if len(pairs) == 0:
        return [], []
    yt, yp = zip(*pairs)
    return list(yt), list(yp)


@torch.no_grad()
def eval_score(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    binary_true, binary_pred = [], []
    mod_true, mod_pred = [], []
    tech_true, tech_pred = [], []

    model.eval()

    for batch in loader:
        x, y_binary, y_modality, y_tech, _ = batch
        x = x.to(device)

        out = model(x)

        pb = out["binary"].argmax(dim=1).cpu().tolist()
        pm = out["modality"].argmax(dim=1).cpu().tolist()
        pt = out["tech"].argmax(dim=1).cpu().tolist()

        binary_true.extend(y_binary.tolist())
        binary_pred.extend(pb)

        mod_true.extend(y_modality.tolist())
        mod_pred.extend(pm)

        tech_true.extend(y_tech.tolist())
        tech_pred.extend(pt)

    return binary_true, binary_pred, mod_true, mod_pred, tech_true, tech_pred


@torch.no_grad()
def eval_embedding(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    binary_true, binary_pred = [], []
    mod_true, mod_pred = [], []
    tech_true, tech_pred = [], []

    model.eval()

    for batch in loader:
        x, y_binary, y_modality, y_tech, _ = batch
        x = x.to(device)

        out = model(x)

        pb = out["binary"].argmax(dim=1).cpu().tolist()
        pm = out["modality"].argmax(dim=1).cpu().tolist()
        pt = out["tech"].argmax(dim=1).cpu().tolist()

        binary_true.extend(y_binary.tolist())
        binary_pred.extend(pb)

        mod_true.extend(y_modality.tolist())
        mod_pred.extend(pm)

        tech_true.extend(y_tech.tolist())
        tech_pred.extend(pt)

    return binary_true, binary_pred, mod_true, mod_pred, tech_true, tech_pred


@torch.no_grad()
def eval_aat(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    binary_true, binary_pred = [], []
    mod_true, mod_pred = [], []
    tech_true, tech_pred = [], []

    model.eval()

    for batch in loader:
        v_tokens, a_tokens, y_binary, y_modality, y_tech, _ = batch

        v_tokens = v_tokens.to(device)
        a_tokens = a_tokens.to(device)

        out = model(v_tokens, a_tokens)

        pb = out["binary"].argmax(dim=1).cpu().tolist()
        pm = out["modality"].argmax(dim=1).cpu().tolist()
        pt = out["tech"].argmax(dim=1).cpu().tolist()

        binary_true.extend(y_binary.tolist())
        binary_pred.extend(pb)

        mod_true.extend(y_modality.tolist())
        mod_pred.extend(pm)

        tech_true.extend(y_tech.tolist())
        tech_pred.extend(pt)

    return binary_true, binary_pred, mod_true, mod_pred, tech_true, tech_pred


def print_report(name, y_true, y_pred):
    y_true, y_pred = filter_valid(y_true, y_pred)

    print("=" * 80)
    print(name)
    print("=" * 80)

    if len(y_true) == 0:
        print("[SKIP] no valid labels")
        return None

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("Accuracy:", f"{acc:.4f}")
    print("Macro-F1:", f"{macro_f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc, macro_f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-type", required=True, choices=["score", "embedding", "aat"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] device:", device)
    print("[INFO] model_type:", args.model_type)
    print("[INFO] checkpoint:", args.checkpoint)
    print("[INFO] feature_path:", args.feature_path)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})

    if args.model_type == "score":
        dataset = ScoreFeatureDataset(args.feature_path)

        model = ScoreMLP(
            in_dim=8,
            hidden_dim=train_args.get("hidden_dim", 64),
            num_binary=2,
            num_modality=4,
            num_tech=7,
            dropout=train_args.get("dropout", 0.2),
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])

        results = eval_score(model, dataset, device, args.batch_size)

    elif args.model_type == "embedding":
        dataset = EmbeddingFeatureDataset(
            args.feature_path,
            normalize_embeddings=not train_args.get("no_normalize_embeddings", False),
        )

        model = EmbeddingMLP(
            in_dim=ckpt.get("input_dim", dataset.input_dim),
            hidden_dim=train_args.get("hidden_dim", 512),
            bottleneck_dim=train_args.get("bottleneck_dim", 256),
            num_binary=2,
            num_modality=4,
            num_tech=7,
            dropout=train_args.get("dropout", 0.4),
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])

        results = eval_embedding(model, dataset, device, args.batch_size)

    elif args.model_type == "aat":
        dataset = EvidenceTokenDataset(
            args.feature_path,
            normalize_embeddings=not train_args.get("no_normalize_embeddings", False),
            max_tokens_per_modality=train_args.get("max_tokens_per_modality", 4),
        )

        model = ArtifactAttributionTransformer(
            video_token_dim=ckpt.get("video_token_dim", dataset.video_token_dim),
            audio_token_dim=ckpt.get("audio_token_dim", dataset.audio_token_dim),
            d_model=train_args.get("d_model", 128),
            nhead=train_args.get("nhead", 4),
            num_layers=train_args.get("num_layers", 1),
            dim_feedforward=train_args.get("dim_feedforward", 256),
            dropout=train_args.get("dropout", 0.3),
            max_tokens_per_modality=train_args.get("max_tokens_per_modality", 4),
            num_binary=2,
            num_modality=4,
            num_tech=7,
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])

        results = eval_aat(model, dataset, device, args.batch_size)

    else:
        raise ValueError(args.model_type)

    binary_true, binary_pred, mod_true, mod_pred, tech_true, tech_pred = results

    b = print_report("Binary zero-shot", binary_true, binary_pred)
    m = print_report("Modality zero-shot", mod_true, mod_pred)
    t = print_report("Technique zero-shot", tech_true, tech_pred)

    print("=" * 80)
    print("[SUMMARY]")
    if b is not None:
        print(f"Binary Acc:      {b[0]:.4f}")
        print(f"Binary Macro-F1: {b[1]:.4f}")
    if m is not None:
        print(f"Modality Acc:      {m[0]:.4f}")
        print(f"Modality Macro-F1: {m[1]:.4f}")
    if t is not None:
        print(f"Technique Acc:      {t[0]:.4f}")
        print(f"Technique Macro-F1: {t[1]:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()