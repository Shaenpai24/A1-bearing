#!/usr/bin/env python3

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIG
# ============================================================

SEED = 22
THREADS = 6
WORKERS = 4
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCH_LIMIT = 30
TIME_LIMIT_SECONDS = 30 * 60

WINDOW_SIZE = 2048
N_CLASSES = 4

LABELS = {
    "healthy": 0,
    "inner": 1,
    "outer": 2,
    "ball": 3,
}

DATA_DIR = Path("data/processed/cwru")
RUN_DIR = Path("runs/cwru_spectrogram_week1")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# SPECTROGRAM
# ============================================================

def make_spectrograms(x):
    """
    Convert (N, 2048) vibration windows into log-power
    spectrograms suitable for a small 2-D CNN.

    Output shape:
        (N, 1, frequency_bins, time_bins)
    """

    x = torch.from_numpy(np.asarray(x, dtype=np.float32))

    spec = torch.stft(
        x,
        n_fft=128,
        hop_length=32,
        win_length=128,
        window=torch.hann_window(128),
        center=True,
        return_complex=True,
    )

    power = spec.abs().pow(2)

    # Log compression.
    power = torch.log1p(power)

    # Per-window normalization.
    mean = power.mean(dim=(1, 2), keepdim=True)
    std = power.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)

    power = (power - mean) / std

    return power.unsqueeze(1).float()


# ============================================================
# DATA LOADING
# ============================================================

def load_source_load(source_load):
    train_x = []
    val_x = []
    test_x = []

    train_y = []
    val_y = []

    files = sorted(DATA_DIR.glob("*.npz"))

    for path in files:
        with np.load(path, allow_pickle=True) as data:
            load_hp = int(data["load_hp"])
            label = str(data["label"])

            if load_hp != source_load:
                continue

            if label not in LABELS:
                raise ValueError(
                    f"Unknown label '{label}' in {path}"
                )

            y = LABELS[label]

            train_x.append(data["train"])
            val_x.append(data["validation"])
            test_x.append(data["test"])

            train_y.extend([y] * len(data["train"]))
            val_y.extend([y] * len(data["validation"]))

    if not train_x:
        raise RuntimeError(
            f"No CWRU files found for source load {source_load} hp."
        )

    train_x = np.concatenate(train_x)
    val_x = np.concatenate(val_x)
    test_x = np.concatenate(test_x)

    train_y = np.asarray(train_y, dtype=np.int64)
    val_y = np.asarray(val_y, dtype=np.int64)

    return train_x, train_y, val_x, val_y, test_x


# ============================================================
# MODEL
# ============================================================

class FrequencyAttention(nn.Module):
    """
    Learns weights along the frequency axis of a spectrogram.

    Input:  (batch, channels, frequency, time)
    Output: same shape, frequency-weighted.
    """

    def __init__(self, kernel_size=5):
        super().__init__()

        self.attention = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Average over channels and time, retaining frequency.
        frequency_summary = x.mean(dim=(1, 3)).unsqueeze(1)

        weights = self.activation(
            self.attention(frequency_summary)
        )

        return x * weights.unsqueeze(-1)


class Small2DCNN(nn.Module):

    def __init__(self, use_attention=False):
        super().__init__()

        self.use_attention = use_attention

        if use_attention:
            self.frequency_attention = FrequencyAttention()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(64, N_CLASSES)

    def forward(self, x):

        if self.use_attention:
            x = self.frequency_attention(x)

        x = self.features(x)
        x = x.flatten(1)

        return self.classifier(x)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-load",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
    )
    args = parser.parse_args()

    set_seed(SEED)

    torch.set_num_threads(THREADS)

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    print("=" * 70)
    print("CWRU A1 WEEK-1 SPECTROGRAM CNN TIMING RUN")
    print("=" * 70)
    print(f"Source load          : {args.source_load} hp")
    print(f"Seed                 : {SEED}")
    print(f"Threads              : {THREADS}")
    print(f"DataLoader workers   : {WORKERS}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Learning rate        : {LEARNING_RATE}")
    print(f"Epoch limit          : {EPOCH_LIMIT}")
    print(f"Elapsed time limit   : {TIME_LIMIT_SECONDS / 60:.0f} minutes")
    print(f"PyTorch              : {torch.__version__}")
    print(f"Python               : {os.sys.version.split()[0]}")
    print(f"Device               : {device}")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD EXISTING PREPROCESSED DATA
    # --------------------------------------------------------

    print("\nLOADING PREPROCESSED CWRU DATA")

    train_x, train_y, val_x, val_y, test_x = load_source_load(
        args.source_load
    )

    print(f"Train windows        : {len(train_x)}")
    print(f"Validation windows   : {len(val_x)}")
    print(f"Test windows         : {len(test_x)}")

    print("\nCLASS COUNTS")

    for name, y in [
        ("train", train_y),
        ("validation", val_y),
    ]:
        counts = np.bincount(y, minlength=N_CLASSES)
        print(
            f"{name:<12}: "
            f"healthy={counts[0]} "
            f"inner={counts[1]} "
            f"outer={counts[2]} "
            f"ball={counts[3]}"
        )

    # --------------------------------------------------------
    # SPECTROGRAM CONVERSION
    # --------------------------------------------------------

    print("\nCONVERTING TO SPECTROGRAMS")

    prep_start = time.perf_counter()

    train_spec = make_spectrograms(train_x)
    val_spec = make_spectrograms(val_x)

    # Test data are converted only to test-shaped inputs.
    # Labels are never loaded or used.
    test_spec = make_spectrograms(test_x)

    prep_elapsed = time.perf_counter() - prep_start

    print(f"Spectrogram shape   : {tuple(train_spec.shape)}")
    print(f"Preparation time     : {prep_elapsed:.3f} s")

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_ds = TensorDataset(
        train_spec,
        torch.from_numpy(train_y),
    )

    val_ds = TensorDataset(
        val_spec,
        torch.from_numpy(val_y),
    )

    # Deliberately no labels for test.
    test_ds = TensorDataset(test_spec)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=False,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = Small2DCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nMODEL")
    print(f"Trainable parameters : {parameters}")
    print(model)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\nTRAINING")

    best_val_loss = float("inf")
    best_epoch = 0
    updates = 0
    epochs_completed = 0
    epochs_without_improvement = 0

    history = []

    fit_start = time.perf_counter()

    for epoch in range(1, EPOCH_LIMIT + 1):

        model.train()

        epoch_start = time.perf_counter()

        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:

            # D0 elapsed-time clock starts with training.
            elapsed = time.perf_counter() - fit_start

            if elapsed >= TIME_LIMIT_SECONDS:
                break

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            updates += 1

            running_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

        # If the time limit was hit during training.
        elapsed = time.perf_counter() - fit_start

        if total == 0:
            stop_reason = "time limit"
            break

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        epochs_completed = epoch

        epoch_elapsed = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - fit_start

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                RUN_DIR / "best_validation_checkpoint.pt",
            )
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "epoch_seconds": epoch_elapsed,
            "elapsed_seconds": elapsed,
            "updates": updates,
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.6f} | "
            f"val loss {val_loss:.6f} | "
            f"train acc {train_acc:.4f} | "
            f"val acc {val_acc:.4f} | "
            f"{epoch_elapsed:.2f}s | "
            f"elapsed {elapsed / 60:.2f}m | "
            f"updates {updates}"
        )

        if epochs_without_improvement >= 5:
            stop_reason = (
                "early stopping "
                "(5 validation checks without improvement)"
            )
            break

        if elapsed >= TIME_LIMIT_SECONDS:
            stop_reason = "30-minute elapsed time limit"
            break

    else:
        stop_reason = "30-epoch limit"

    fit_elapsed = time.perf_counter() - fit_start

    # --------------------------------------------------------
    # LOAD BEST CHECKPOINT
    # --------------------------------------------------------

    checkpoint = RUN_DIR / "best_validation_checkpoint.pt"

    if checkpoint.exists():
        model.load_state_dict(
            torch.load(
                checkpoint,
                map_location=device,
            )
        )

    # --------------------------------------------------------
    # TIMED TEST-SHAPED INFERENCE
    # --------------------------------------------------------

    print("\nTIMED INFERENCE ON TEST-SHAPED DATA")

    model.eval()

    inference_start = time.perf_counter()

    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            _ = model(x)

    inference_elapsed = time.perf_counter() - inference_start

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    report = {
        "source_load_hp": args.source_load,
        "seed": SEED,
        "threads": THREADS,
        "dataloader_workers": WORKERS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epoch_limit": EPOCH_LIMIT,
        "elapsed_time_limit_seconds": TIME_LIMIT_SECONDS,
        "pytorch": torch.__version__,
        "python": os.sys.version.split()[0],
        "train_windows": int(len(train_x)),
        "validation_windows": int(len(val_x)),
        "test_windows": int(len(test_x)),
        "spectrogram_shape": list(train_spec.shape),
        "trainable_parameters": parameters,
        "preparation_seconds": prep_elapsed,
        "epochs_completed": epochs_completed,
        "optimization_updates": updates,
        "fit_elapsed_seconds": fit_elapsed,
        "best_validation_loss": best_val_loss,
        "best_epoch": best_epoch,
        "stop_reason": stop_reason,
        "training_loss_moved": (
            len(history) >= 2
            and history[-1]["train_loss"]
            != history[0]["train_loss"]
        ),
        "validation_loss_moved": (
            len(history) >= 2
            and history[-1]["val_loss"]
            != history[0]["val_loss"]
        ),
        "inference_windows": int(len(test_x)),
        "inference_elapsed_seconds": inference_elapsed,
        "history": history,
        "test_labels_used": False,
        "test_metrics_computed": False,
    }

    with open(
        RUN_DIR / "timing_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("SPECTROGRAM REFERENCE RUN COMPLETE")
    print("=" * 70)
    print(f"Stop reason          : {stop_reason}")
    print(f"Epochs completed     : {epochs_completed}")
    print(f"Optimization updates : {updates}")
    print(f"Fit elapsed          : {fit_elapsed:.3f} s")
    print(f"Best validation loss : {best_val_loss:.6f}")
    print(f"Training loss moved  : {report['training_loss_moved']}")
    print(f"Validation loss moved: {report['validation_loss_moved']}")
    print(f"Inference windows    : {len(test_x)}")
    print(f"Inference elapsed    : {inference_elapsed:.3f} s")
    print(
        f"Checkpoint            : "
        f"{RUN_DIR / 'best_validation_checkpoint.pt'}"
    )
    print(
        f"Timing report         : "
        f"{RUN_DIR / 'timing_report.json'}"
    )
    print()
    print("No test labels or test metrics were evaluated.")


if __name__ == "__main__":
    main()
