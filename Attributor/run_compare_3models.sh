#!/usr/bin/env bash
set -e

FEATURE_PATH="features/fakeavceleb_balanced_100pm.pt"
EPOCHS=100
SEED=42
BATCH_SIZE=32

mkdir -p logs_compare
mkdir -p runs_score_mlp_100pm
mkdir -p runs_embedding_mlp_100pm
mkdir -p runs_aat_100pm

echo "============================================================"
echo "Feature path: ${FEATURE_PATH}"
echo "Epochs: ${EPOCHS}"
echo "Seed: ${SEED}"
echo "Batch size: ${BATCH_SIZE}"
echo "============================================================"

echo ""
echo "============================================================"
echo "[1/3] Training Score-only MLP"
echo "============================================================"

PYTHONUNBUFFERED=1 python train_score_mlp.py \
  --feature-path "${FEATURE_PATH}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --save-path runs_score_mlp_100pm/best_score_mlp_seed${SEED}.pt \
  2>&1 | tee logs_compare/score_mlp_seed${SEED}.log


echo ""
echo "============================================================"
echo "[2/3] Training Embedding MLP"
echo "============================================================"

PYTHONUNBUFFERED=1 python train_embedding_mlp.py \
  --feature-path "${FEATURE_PATH}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --save-path runs_embedding_mlp_100pm/best_embedding_mlp_seed${SEED}.pt \
  2>&1 | tee logs_compare/embedding_mlp_seed${SEED}.log


echo ""
echo "============================================================"
echo "[3/3] Training Artifact Attribution Transformer"
echo "============================================================"

PYTHONUNBUFFERED=1 python train_attribution_transformer.py \
  --feature-path "${FEATURE_PATH}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --d-model 128 \
  --nhead 4 \
  --num-layers 1 \
  --dim-feedforward 256 \
  --dropout 0.3 \
  --save-path runs_aat_100pm/best_aat_seed${SEED}.pt \
  2>&1 | tee logs_compare/aat_seed${SEED}.log


echo ""
echo "============================================================"
echo "All experiments finished."
echo "Logs saved to logs_compare/"
echo "============================================================"