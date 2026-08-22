#!/usr/bin/env bash
set -e

FEATURE_PATH="features/fakeavceleb_balanced_100pm.pt"
SEED=42
EPOCHS=150
BATCH_SIZE=32

mkdir -p logs_aat_tuning
mkdir -p runs_aat_tuning

echo "============================================================"
echo "AAT tuning"
echo "Feature: ${FEATURE_PATH}"
echo "============================================================"

echo ""
echo "[1/4] AAT small: d_model=64, 1 layer"
PYTHONUNBUFFERED=1 python train_attribution_transformer.py \
  --feature-path "${FEATURE_PATH}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --d-model 64 \
  --nhead 4 \
  --num-layers 1 \
  --dim-feedforward 128 \
  --dropout 0.3 \
  --save-path runs_aat_tuning/best_aat_small.pt \
  2>&1 | tee logs_aat_tuning/aat_small.log


echo ""
echo "[2/4] AAT base regularized: d_model=128, 1 layer, dropout=0.4"
PYTHONUNBUFFERED=1 python train_attribution_transformer.py \
  --feature-path "${FEATURE_PATH}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --d-model 128 \
  --nhead 4 \
  --num-layers 1 \
  --dim-feedforward 256 \
  --dropout 0.4 \
  --save-path runs_aat_tuning/best_aat_base_dropout04.pt \
  2>&1 | tee logs_aat_tuning/aat_base_dropout04.log


echo ""
echo "[3/4] AAT 2-layer: d_model=128, 2 layers"
PYTHONUNBUFFERED=1 python train_attribution_transformer.py \
  --feature-path "${FEATURE_PATH}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --d-model 128 \
  --nhead 4 \
  --num-layers 2 \
  --dim-feedforward 256 \
  --dropout 0.4 \
  --save-path runs_aat_tuning/best_aat_2layer_dropout04.pt \
  2>&1 | tee logs_aat_tuning/aat_2layer_dropout04.log


echo ""
echo "[4/4] AAT wider: d_model=256, 1 layer"
PYTHONUNBUFFERED=1 python train_attribution_transformer.py \
  --feature-path "${FEATURE_PATH}" \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --seed ${SEED} \
  --d-model 256 \
  --nhead 4 \
  --num-layers 1 \
  --dim-feedforward 512 \
  --dropout 0.4 \
  --save-path runs_aat_tuning/best_aat_wide_dropout04.pt \
  2>&1 | tee logs_aat_tuning/aat_wide_dropout04.log


echo ""
echo "============================================================"
echo "AAT tuning finished."
echo "============================================================"
