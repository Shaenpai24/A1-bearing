# Team 06 — Bearing Fault Diagnosis (Card A1)

**Strategy:** S4  
**Required fusion pair:** 1-D raw-vibration CNN + 2-D spectrogram CNN

This repository contains the implementation, split manifests, and development-run evidence for the Team 06 A1 bearing-fault diagnosis project.

## Project scope

The project evaluates bearing-fault classification using the Case Western Reserve University (CWRU) and Paderborn bearing datasets.

The required model components are:

- CWRU 1-D raw-vibration CNN
- CWRU 2-D spectrogram CNN
- Paderborn 1-D raw-vibration CNN
- Paderborn envelope-feature baseline
- Frequency-attention image model and its no-attention ablation
- Validation-weighted fusion of the 1-D and 2-D model outputs

The required fusion pair combines the 1-D raw-vibration CNN with the 2-D spectrogram CNN. Fusion is performed by validation-based weighting; it does not train a separate set of model weights.

## Repository layout

```text
.
├── cwru_recordings.json
├── T06_A1_cwru.csv
├── T06_A1_paderborn.csv
├── T06_A1_split_summary.csv
├── check_cwru.py
├── check_cwru_guards.py
├── generate_cwru_split.py
├── generate_split_ids.py
├── generate_split_summary.py
├── inspect_paderborn*.py
├── scripts/
│   ├── cwru_envelope_baseline.py
│   ├── eda_paderborn.py
│   ├── export_split_ids.py
│   ├── paderborn_envelope_baseline.py
│   ├── preprocess_paderborn.py
│   ├── train_cwru_spectrogram_week1.py
│   ├── train_paderborn_1d_week1.py
│   └── validate_paderborn.py
└── src/
    └── preprocess_cwru.py
```

Raw datasets, extracted recordings, generated arrays, and trained model checkpoints are not committed to this repository.

## Data setup

Place the datasets under the following paths:

```text
data/
└── raw/
    ├── cwru/
    │   ├── 97.mat
    │   ├── 98.mat
    │   ├── 99.mat
    │   └── ...
    └── paderborn/
        ├── K001.rar
        ├── K002.rar
        ├── ...
        └── extracted/
            ├── K001/
            │   └── *.mat
            ├── K002/
            │   └── *.mat
            └── ...
```

The CWRU recording manifest is `cwru_recordings.json`. The split manifests are:

- `T06_A1_cwru.csv`
- `T06_A1_paderborn.csv`
- `T06_A1_split_summary.csv`

Do not regenerate or edit the approved split manifests casually: they define the train, validation, and test assignments used by the experiments.

### Paderborn condition

The approved Paderborn experiment condition is **N15_M07_F10**. Use the approved bearing-level split and only recordings from that condition. Confirm the extracted directory contents before running preprocessing or training.

## Environment

Use Python 3.10 or newer in a virtual environment. On systems where the `python` command is unavailable, use `python3`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The current `requirements.txt` is not yet a complete environment lock for every script. The training and analysis scripts also require their imported scientific Python packages, including pandas, scikit-learn, matplotlib, and PyTorch. Install compatible versions for your platform before running those components. PyTorch installation may need a platform-specific wheel; use the official PyTorch installation selector for CPU/CUDA compatibility.

Record the exact installed package versions with:

```bash
python3 -m pip freeze
```

## Running components

Run commands from the repository root.

### CWRU envelope baseline

```bash
python3 scripts/cwru_envelope_baseline.py
```

### Paderborn preprocessing

```bash
python3 scripts/preprocess_paderborn.py
```

### Paderborn envelope baseline

```bash
python3 scripts/paderborn_envelope_baseline.py
```

### CWRU spectrogram model

```bash
python3 scripts/train_cwru_spectrogram_week1.py
```

### Paderborn 1-D model

```bash
python3 scripts/train_paderborn_1d_week1.py
```

Some scripts use fixed project-relative input/output paths. Inspect each script's command-line options and configuration before changing paths or rerunning an experiment. Keep run outputs separate from the raw data and preserve the original run records.

## Experimental protocol and reporting

- Fit preprocessing transforms and feature scalers on training data only; apply the fitted transforms unchanged to validation data.
- Use validation data for model selection, calibration, and fusion-weight selection.
- Keep bearing identities disjoint between Paderborn train, validation, and test partitions as specified by the approved split.
- CWRU uses guarded within-recording time-block splits. It does **not** establish unseen-bearing generalisation.
- Treat windows from the same bearing/recording as correlated observations. Report both window support and the number of independent bearings/units.
- At the checkpoint stage, report validation results only. Do not score or report test-partition metrics before the project result freeze.
- Test-shaped inference, if required for a smoke test, must not calculate or expose test metrics.

## Checkpoint smoke test

The required checkpoint smoke-test entry point and its exact command must be added and verified before the repository is tagged `checkpoint`. The final README must put that single smoke-test command first and the submitted `T06_A1_smoke_log.txt` must contain its complete output.

Do not treat individual script execution as proof that the complete pipeline—including attention, ablation, fusion, and both core experiments—runs end to end.

## Reproducibility

For every fit, preserve:

- run ID and date
- source commit
- component and experimental condition
- random seed
- training sample count
- epochs, updates, and stopping reason
- elapsed and CPU time
- peak memory, threads, and workers
- trainable parameter count
- validation metric, value, and best epoch
- explicit confirmation that no test partition was scored

Keep the run log and checkpoint note consistent with the exact source commit used for the checkpoint archive.

## Data and artifact policy

Do not commit:

- raw CWRU or Paderborn recordings
- extracted dataset directories
- generated large arrays
- model weights/checkpoints
- credentials, tokens, or local environment files
- virtual environments or Python cache files

The checkpoint source archive must be generated from the final tagged commit and remain below the assignment's 25 MB limit.
