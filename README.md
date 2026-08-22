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

Expected project path:
~/hsh/AIApplication

Expected dataset paths:
~/hsh/AIApplication/FakeAVCeleb_v1.2
~/hsh/AIApplication/PolyGlotFake
~/hsh/AIApplication/AV-Deepfake1M_RootFiles