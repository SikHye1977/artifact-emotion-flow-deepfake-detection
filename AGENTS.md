# Repository Working Instructions

This repository contains research code for cross-dataset audiovisual deepfake
detection. Before changing the environment, paths, training code, or evaluation
protocol, read these files:

1. `docs/WORKSTATION_MIGRATION.md`
2. `docs/HANDOFF.md`
3. `README.md`

## Storage policy

- Keep the Git repository and Python environment on the workstation SSD.
- Keep datasets, trained weights, preprocessing caches, extracted features, and
  backups on the 4 TB HDD.
- Do not commit datasets, media, checkpoints, caches, credentials, unpublished
  manuscripts, per-sample evaluation dumps, or machine-specific logs.
- Do not modify or delete data owned by other users under `/home/hdd1`.
- Use paths below as the workstation defaults unless the user reports a change:
  - repository: `/home/sikhye/workspace/artifact-emotion-flow-deepfake-detection`
  - research storage: `/home/hdd1/sikhye/deepfake-research`
  - datasets: `/home/hdd1/sikhye/deepfake-research/datasets`
  - checkpoints: `/home/hdd1/sikhye/deepfake-research/checkpoints`
  - caches: `/home/hdd1/sikhye/deepfake-research/caches`
  - extracted features: `/home/hdd1/sikhye/deepfake-research/extracted_features`

## Environment policy

- Treat `requirements-legacy.txt` as a record of the old machine, not as a
  directly installable workstation lock file.
- Inspect the workstation OS, Python, NVIDIA driver, and all four GPUs before
  selecting PyTorch and CUDA packages.
- The driver's reported CUDA support is not proof that a matching CUDA toolkit
  or PyTorch runtime is installed.
- Build a clean, reproducible environment and record it in a new environment
  file before running full training.

## External repositories

Run `bash integrations/setup_external_repos.sh` to restore the pinned AASIST,
MAE-DFER, and SyncNet repositories. They are intentionally excluded from this
repository. The script does not download pretrained weights or research
checkpoints.

## Safe continuation order

1. Inspect storage, OS, GPU, driver, Python, Conda, FFmpeg, and Git.
2. Verify the HDD directory is writable without changing `/home/hdd1` globally.
3. Restore pinned external repositories.
4. Create and validate a clean Python environment.
5. Transfer datasets and checkpoints separately.
6. Verify transferred checkpoints with `checkpoint_sha256.txt` after mapping old
   manifest paths to their new locations.
7. Run import, data-loading, and single-sample inference smoke tests.
8. Run a short single-GPU experiment before attempting multi-GPU training.

Preserve existing user changes. Ask before deleting, overwriting, or moving any
material dataset, checkpoint, or experiment output.
