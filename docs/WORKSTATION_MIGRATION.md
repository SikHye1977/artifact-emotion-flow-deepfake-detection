# Workstation Migration Handoff

## Purpose

This document transfers the repository from the original development machine to
the workstation used for continued research and experiments. It records what is
known, what is intentionally absent from Git, and the order in which the new
environment should be established.

For the research architecture, experiment history, datasets, and main entry
points, also read `docs/HANDOFF.md`.

## Repository

- GitHub repository: `https://github.com/SikHye1977/artifact-emotion-flow-deepfake-detection.git`
- Target branch: `main`
- Workstation user: `sikhye`
- Target clone path: `/home/sikhye/workspace/artifact-emotion-flow-deepfake-detection`
- Original development path: `<ORIGINAL_PROJECT_ROOT>`

The original path may still appear in historical documentation or generated
artifacts. New executable configuration must not depend on it.

## Workstation hardware and storage

Known hardware:

- CPU: Intel Xeon W-2235, 6 cores / 12 threads
- RAM: approximately 128 GB
- GPU: four NVIDIA RTX 2080 Ti cards, 11 GB VRAM each
- NVIDIA driver: 535.261.03
- Driver-reported CUDA compatibility: 12.2
- Main drive: Intel 480 GB SSD
- Additional drive: Seagate 4 TB HDD
- Mainboard: GIGABYTE MW51-HP0-00

Observed storage state before migration:

| Device | Mount | Filesystem | Capacity | Used | Available |
| --- | --- | --- | ---: | ---: | ---: |
| `/dev/sda2` | `/` | ext4 | 439 GB | 309 GB | 108 GB |
| `/dev/sdb1` | `/home/hdd1` | ext4 | 3.6 TB | 991 GB | about 2.5 TB |

The HDD already contains directories owned by another user (`hyjin`). Do not
modify, move, recursively chmod, or recursively chown those directories.

## Target directory layout

Keep source code and the Python environment on the SSD:

```text
/home/sikhye/workspace/
└── artifact-emotion-flow-deepfake-detection/
```

Keep large and generated research assets on the HDD:

```text
/home/hdd1/sikhye/deepfake-research/
├── datasets/
├── checkpoints/
├── caches/
├── extracted_features/
└── backups/
```

Create the HDD layout as `sikhye`:

```bash
mkdir -p /home/hdd1/sikhye/deepfake-research/{datasets,checkpoints,caches,extracted_features,backups}
chmod 700 /home/hdd1/sikhye
```

Do not run `chmod -R` or `chown -R` against `/home/hdd1` itself.

## Items intentionally absent from Git

The public repository does not carry:

- original datasets or video/audio media;
- trained and pretrained model weights (`.pth`, `.pt`, `.ckpt`, and similar);
- preprocessing caches, extracted frames, and extracted features;
- local virtual environments;
- external repository working trees;
- the unpublished manuscript PDF;
- per-sample evaluation JSON dumps containing local paths;
- machine-specific console logs.

These exclusions are expected and are defined in `.gitignore`. Do not weaken
them merely to simplify transfer.

`checkpoint_manifest_raw.txt` records the original locations and sizes of files
found during migration preparation. `checkpoint_sha256.txt` records hashes from
the original machine. The manifest is deliberately raw and may contain entries
from ignored virtual environments; transfer only the research weights actually
needed by the project.

## External source dependencies

The following projects are restored by `integrations/setup_external_repos.sh`:

| Project | Repository | Pinned commit |
| --- | --- | --- |
| AASIST | `https://github.com/clovaai/aasist.git` | `a04c9863f63d44471dde8a6abcb3b082b07cd1d1` |
| MAE-DFER | `https://github.com/sunlicai/MAE-DFER.git` | `81fcf589bb584a7a255e47af0e802c12b25f9eba` |
| SyncNet | `https://github.com/joonson/syncnet_python.git` | `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623` |

The script also installs the project-specific SyncNet Python files stored under
`integrations/syncnet_custom/`. It does not install Python packages or download
weights.

## Migration procedure

### 1. Clone the public repository to the SSD

```bash
mkdir -p /home/sikhye/workspace
cd /home/sikhye/workspace
git clone https://github.com/SikHye1977/artifact-emotion-flow-deepfake-detection.git
cd artifact-emotion-flow-deepfake-detection
git status
git log -1 --format='%h %an <%ae> %s'
```

### 2. Record the workstation baseline

Run these before installing or upgrading GPU software:

```bash
uname -a
cat /etc/os-release
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python3 --version
conda --version
ffmpeg -version
git --version
df -hT / /home/hdd1
```

An absent `conda` or `ffmpeg` command is a setup item, not a reason to alter the
system CUDA installation. Save the output in migration notes without committing
host secrets.

### 3. Restore external repositories

```bash
cd /home/sikhye/workspace/artifact-emotion-flow-deepfake-detection
bash integrations/setup_external_repos.sh
```

Confirm the pinned revisions:

```bash
git -C aasist rev-parse HEAD
git -C mae_dfer rev-parse HEAD
git -C syncnet_python rev-parse HEAD
```

### 4. Build a clean software environment

Do not install `requirements-legacy.txt` directly. It is a historical snapshot
and may contain platform-specific or obsolete packages. Select a Python and
PyTorch combination only after recording the OS and driver baseline. Create a
new reproducible `environment.yml` or workstation requirements file and test
imports before installing optional experiment dependencies.

The driver reporting CUDA 12.2 means the driver can support compatible CUDA
applications; it does not establish that a CUDA 12.2 toolkit or a CUDA 12.2
PyTorch build is installed.

### 5. Transfer data and weights outside Git

Place datasets under:

```text
/home/hdd1/sikhye/deepfake-research/datasets/
```

Place trained and pretrained weights under:

```text
/home/hdd1/sikhye/deepfake-research/checkpoints/
```

Prefer resumable transfer (`rsync --partial --progress`) over copying large
trees with a non-resumable command. Preserve the original source until sizes,
file counts, and hashes have been verified on the workstation.

Dataset directories expected by existing code include:

- AV-Deepfake1M (`AV-Deepfake1M_RootFiles` in legacy paths)
- FakeAVCeleb (`FakeAVCeleb_v1.2` in legacy paths)
- PolyGlotFake

Do not create repository symlinks until the actual transferred directory names
are known. Prefer configurable dataset roots; use symlinks only as a compatibility
bridge for legacy entry points.

### 6. Validate before full experiments

Validation order:

1. Confirm all four GPUs are visible in `nvidia-smi`.
2. Confirm PyTorch imports and reports four CUDA devices.
3. Confirm FFmpeg can decode one permitted sample.
4. Verify checkpoint hashes for transferred research weights.
5. Load one sample from each dataset without training.
6. Run one checkpoint inference on a single GPU.
7. Run a short single-GPU training smoke test.
8. Only then configure multi-GPU experiments.

Do not assume existing scripts support distributed training merely because four
GPUs are present. Inspect each training entry point before choosing DDP,
DataParallel, or independent per-GPU experiment scheduling. With 11 GB VRAM per
GPU, validate batch size on one GPU first.

## Current migration status

At the time this handoff was written:

- the Git repository had been prepared for public hosting;
- datasets and trained weights were intentionally excluded from Git;
- the workstation SSD and HDD mount points and free space had been identified;
- `/home/hdd1` was writable, but it also contained another user's data;
- the dedicated `/home/hdd1/sikhye/deepfake-research` layout was planned;
- repository cloning, environment construction, data transfer, checksum
  validation, and smoke tests still needed to be completed on the workstation.

Update this section as each stage is completed.

## Prompt for a new Codex task

After opening the cloned repository on the workstation, start with:

```text
Read AGENTS.md, docs/WORKSTATION_MIGRATION.md, docs/HANDOFF.md, and README.md.
Continue the workstation migration using the documented storage policy. First
inspect the actual OS, GPU, Python, Conda, FFmpeg, Git, disk permissions, and Git
status. Do not install packages, move data, or change CUDA until you report the
baseline and propose the next verified step.
```
