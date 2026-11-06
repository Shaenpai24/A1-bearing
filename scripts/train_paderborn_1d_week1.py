#!/usr/bin/env python3
"""
Paderborn A1 Week-1 reference run
Small 1-D CNN on raw 2048-sample vibration windows.

Protocol:
- bearing-disjoint Paderborn A1 split
- train/validation only for model selection
- seed 22
- max 30 epochs OR 30 minutes elapsed, whichever comes first
- best validation checkpoint retained
- final test pass is inference-only: NO test labels or metrics
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 22
WINDOW_SIZE = 2048
NUM_CLASSES = 3
MAX_EPOCHS = 30
TIME_LIMIT_SECONDS = 30 * 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEFAULT_WORKERS = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Small1DCNN(nn.Module):
    """
    Small from-scratch raw-vibration CNN.
    ~56k trainable parameters, comfortably below the 250k D0 ceiling.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


LABEL_MAP = {
    "healthy": 0,
    "outer_artificial": 1,
    "outer_natural": 1,
    "inner_artificial": 2,
    "inner_natural": 2,
}


def load_split(root: Path, split: str):
    xs = []
    ys = []
    bearings = []
    npz_files = sorted((root / split).glob("*/*.npz"))

    if not npz_files:
        raise RuntimeError(f"No NPZ files found under {root / split}")

    for path in npz_files:
        with np.load(path, allow_pickle=True) as data:
            X = np.asarray(data["X"], dtype=np.float32)
            label = str(data["label"].item())

            if X.ndim != 2 or X.shape[1] != WINDOW_SIZE:
                raise ValueError(f"{path}: unexpected X shape {X.shape}")

            if label not in LABEL_MAP:
                raise ValueError(f"{path}: unsupported label {label}")

            xs.append(X)
            ys.extend([LABEL_MAP[label]] * len(X))
            bearings.extend([str(data["bearing_id"].item())] * len(X))

    X = np.concatenate(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)

    return X, y, bearings, len(npz_files)


def fit_scaling(X_train: np.ndarray):
    # Training-only global scaling. Statistics never use validation/test samples.
    mean = float(X_train.mean(dtype=np.float64))
    std = float(X_train.std(dtype=np.float64))
    if std == 0:
        raise RuntimeError("Training standard deviation is zero.")
    return mean, std


def make_loader(X, y, batch_size, workers, shuffle):
    X_t = torch.from_numpy(X[:, None, :])
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)

    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
    )
    if workers > 0:
        kwargs["persistent_workers"] = True

    return DataLoader(**kwargs)


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/paderborn"),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--threads", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paderborn_week1"))
    args = parser.parse_args()

    seed_everything(SEED)
    torch.set_num_threads(args.threads)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PADERBORN A1 WEEK-1 REFERENCE RUN")
    print("=" * 70)
    print(f"Seed                 : {SEED}")
    print(f"Threads              : {args.threads}")
    print(f"DataLoader workers   : {args.workers}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Learning rate        : {LEARNING_RATE}")
    print(f"Epoch limit          : {MAX_EPOCHS}")
    print(f"Elapsed time limit   : {TIME_LIMIT_SECONDS / 60:.0f} minutes")
    print(f"PyTorch              : {torch.__version__}")
    print(f"Python               : {platform.python_version()}")
    print(f"OS                   : {platform.platform()}")
    print(f"CPU                  : {platform.processor() or platform.machine()}")
    print(f"CPU cores visible    : {os.cpu_count()}")
    print("=" * 70)

    X_train, y_train, _, n_train_files = load_split(args.data_root, "train")
    X_val, y_val, _, n_val_files = load_split(args.data_root, "validation")
    X_test, y_test, _, n_test_files = load_split(args.data_root, "test")

    print("\nDATA")
    print(f"Train windows        : {len(X_train)}")
    print(f"Validation windows   : {len(X_val)}")
    print(f"Test windows         : {len(X_test)}")
    print(f"Train NPZ files      : {n_train_files}")
    print(f"Validation NPZ files : {n_val_files}")
    print(f"Test NPZ files       : {n_test_files}")

    if len(X_train) != 3116:
        print(f"WARNING: expected 3,116 training windows; found {len(X_train)}")

    mean, std = fit_scaling(X_train)
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_val = ((X_val - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)

    train_loader = make_loader(
        X_train, y_train, args.batch_size, args.workers, shuffle=True
    )
    val_loader = make_loader(
        X_val, y_val, args.batch_size, args.workers, shuffle=False
    )
    test_loader = make_loader(
        X_test, y_test, args.batch_size, args.workers, shuffle=False
    )

    model = Small1DCNN().cpu()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nMODEL")
    print(f"Trainable parameters : {trainable}")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    updates = 0
    history = []

    fit_start = time.perf_counter()
    stop_reason = "epoch_limit"

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for xb, yb in train_loader:
            if time.perf_counter() - fit_start >= TIME_LIMIT_SECONDS:
                stop_reason = "30-minute elapsed limit"
                break

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            updates += 1
            n = len(yb)
            train_loss_sum += loss.item() * n
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_count += n

        if train_count == 0:
            stop_reason = "30-minute elapsed limit before a complete training epoch"
            break

        # Validation is part of the D0 timed fit.
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                n = len(yb)
                val_loss_sum += loss.item() * n
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_count += n

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        train_acc = train_correct / train_count
        val_acc = val_correct / val_count
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - fit_start

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "epoch_elapsed_seconds": epoch_time,
            "elapsed_seconds": elapsed,
            "optimization_updates": updates,
            "checkpoint_improved": improved,
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.6f} | "
            f"val loss {val_loss:.6f} | "
            f"train acc {train_acc:.4f} | "
            f"val acc {val_acc:.4f} | "
            f"{epoch_time:.2f}s | "
            f"elapsed {elapsed/60:.2f}m | "
            f"updates {updates}"
        )

        # D0 early stopping: five validation checks without improvement.
        if epochs_without_improvement >= 5:
            stop_reason = "early stopping (5 validation checks without improvement)"
            break

        if elapsed >= TIME_LIMIT_SECONDS:
            stop_reason = "30-minute elapsed limit"
            break

    fit_elapsed = time.perf_counter() - fit_start

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced.")

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "seed": SEED,
            "trainable_parameters": trainable,
            "label_map": LABEL_MAP,
            "window_size": WINDOW_SIZE,
        },
        args.output_dir / "best_validation_checkpoint.pt",
    )

    # Required inference pass over test-shaped data, without reading labels.
    model.eval()
    inference_start = time.perf_counter()
    n_inference = 0
    with torch.no_grad():
        for xb, _ in test_loader:
            _ = model(xb)
            n_inference += len(xb)
    inference_elapsed = time.perf_counter() - inference_start

    # Save timing record. CPU time is intentionally reported separately by
    # /usr/bin/time -v in the command used for the readiness measurement,
    # because it includes DataLoader worker processes.
    report = {
        "seed": SEED,
        "threads": args.threads,
        "dataloader_workers": args.workers,
        "batch_size": args.batch_size,
        "max_epochs": MAX_EPOCHS,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "stop_reason": stop_reason,
        "epochs_completed": len(history),
        "optimization_updates": updates,
        "fit_elapsed_seconds": fit_elapsed,
        "inference_windows": n_inference,
        "inference_elapsed_seconds": inference_elapsed,
        "train_windows": len(X_train),
        "validation_windows": len(X_val),
        "test_windows": len(X_test),
        "trainable_parameters": trainable,
        "best_validation_loss": best_val_loss,
        "training_loss_moved": (
            len(history) >= 2 and history[-1]["train_loss"] != history[0]["train_loss"]
        ),
        "validation_loss_moved": (
            len(history) >= 2 and history[-1]["validation_loss"] != history[0]["validation_loss"]
        ),
        "hardware": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "os": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "visible_cpu_cores": os.cpu_count(),
            "ram_note": "Record physical RAM separately in readiness note.",
        },
        "history": history,
    }

    (args.output_dir / "timing_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("REFERENCE RUN COMPLETE")
    print("=" * 70)
    print(f"Stop reason          : {stop_reason}")
    print(f"Epochs completed     : {len(history)}")
    print(f"Optimization updates : {updates}")
    print(f"Fit elapsed          : {fit_elapsed:.3f} s")
    print(f"Best validation loss : {best_val_loss:.6f}")
    print(f"Training loss moved  : {report['training_loss_moved']}")
    print(f"Validation loss moved: {report['validation_loss_moved']}")
    print(f"Inference windows    : {n_inference}")
    print(f"Inference elapsed    : {inference_elapsed:.3f} s")
    print(f"Checkpoint            : {args.output_dir / 'best_validation_checkpoint.pt'}")
    print(f"Timing report         : {args.output_dir / 'timing_report.json'}")
    print("\nNo test labels or test metrics were evaluated.")


if __name__ == "__main__":
    main()
