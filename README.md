# Deepfake Fusion Research

## Main paper experiment

Train: FakeAVCeleb  
Zero-shot evaluation: PolyGlotFake, AV-Deepfake1M

Branches:
- X3D-m
- AASIST
- HSEmotion + GRU + Attention
- Audio CRNN + GRU + Attention

## Important entry points

- train_HSEmotion.py
- train_CRNN.py
- evaluate_avdf1m_zeroshot.py
- polyglotfake_zeroshot.py
- reverse_zero_shot/evaluate_v2.py

## External assets

Datasets and checkpoints are not included in Git.

The workstation migration layout, pinned external repositories, and validation
order are documented in `docs/WORKSTATION_MIGRATION.md`. On that workstation,
the repository belongs on the SSD at:

```text
/home/sikhye/workspace/artifact-emotion-flow-deepfake-detection
```

Datasets, checkpoints, caches, and extracted features belong under the dedicated
research directory on the 4 TB HDD:

```text
/home/hdd1/sikhye/deepfake-research
```

Some legacy entry points may still expect dataset directories named
`FakeAVCeleb_v1.2`, `PolyGlotFake`, or `AV-Deepfake1M_RootFiles` beneath the
project root. Prefer configurable dataset roots; use symlinks only when needed
for compatibility.
