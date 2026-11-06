#!/usr/bin/env python3

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/cwru"
OUT_DIR = ROOT / "runs/cwru_1d_core"

SEED = 22
THREADS = 6
WORKERS = 0
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4

LABELS = ["healthy", "inner", "outer", "ball"]
LABEL_MAP = {name: i for i, name in enumerate(LABELS)}
LOADS = [0, 1, 2, 3]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Small1DCNN(nn.Module):
    def __init__(self, num_classes=4):
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

    def forward(self, x):
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


def load_source_data(source_load):
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x = []

    for path in sorted(DATA_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=True) as d:
            load_hp = int(d["load_hp"].item())
            if load_hp != source_load:
                continue

            label = str(d["label"].item())
            if label not in LABEL_MAP:
                raise ValueError(f"Unknown label {label}: {path}")

            label_id = LABEL_MAP[label]

            a = np.asarray(d["train"], dtype=np.float32)
            b = np.asarray(d["validation"], dtype=np.float32)

            if len(a):
                train_x.append(a)
                train_y.extend([label_id] * len(a))

            if len(b):
                val_x.append(b)
                val_y.extend([label_id] * len(b))

    # Load test signals for inference only. Never access their labels.
    for path in sorted(DATA_DIR.glob("*.npz")):
        with np.load(path, allow_pickle=True) as d:
            if int(d["load_hp"].item()) != source_load:
                continue
            x = np.asarray(d["test"], dtype=np.float32)
            if len(x):
                test_x.append(x)

    if not train_x or not val_x:
        raise RuntimeError(f"Missing train/validation data for {source_load} hp")

    return (
        np.concatenate(train_x),
        np.asarray(train_y, dtype=np.int64),
        np.concatenate(val_x),
        np.asarray(val_y, dtype=np.int64),
        test_x,
    )


def make_loader(x, y=None, shuffle=False):
    xt = torch.from_numpy(x[:, None, :].astype(np.float32))

    if y is None:
        ds = TensorDataset(xt)
    else:
        yt = torch.from_numpy(y.astype(np.int64))
        ds = TensorDataset(xt, yt)

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=WORKERS,
    )


def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    actual, predicted = [], []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            losses.append(criterion(logits, y).item() * len(y))
            actual.extend(y.numpy().tolist())
            predicted.extend(logits.argmax(1).numpy().tolist())

    return (
        sum(losses) / len(actual),
        np.asarray(actual),
        np.asarray(predicted),
    )


def run_source_load(source_load):
    print(f"\n{'=' * 65}")
    print(f"CWRU 1-D CNN — SOURCE LOAD {source_load} hp")
    print("=" * 65)

    seed_everything(SEED)

    x_train, y_train, x_val, y_val, test_arrays = load_source_data(
        source_load
    )

    train_loader = make_loader(x_train, y_train, shuffle=True)
    val_loader = make_loader(x_val, y_val)

    model = Small1DCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    updates = 0
    history = []

    start = time.perf_counter()
    stop_reason = "epoch_limit"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            updates += 1
            train_loss_sum += loss.item() * len(yb)
            train_count += len(yb)

        train_loss = train_loss_sum / train_count
        val_loss, y_true, y_pred = evaluate(model, val_loader, criterion)

        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(
            y_true,
            y_pred,
            labels=list(range(4)),
            average="macro",
            zero_division=0,
        )

        improved = val_loss < best_loss

        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "validation_macro_f1": val_f1,
            "updates": updates,
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_loss:.5f} | "
            f"val loss={val_loss:.5f} | "
            f"val acc={val_acc:.4f} | "
            f"val macro-F1={val_f1:.4f}"
        )

        if stale_epochs >= PATIENCE:
            stop_reason = "early_stopping_5_checks"
            break

    fit_seconds = time.perf_counter() - start

    if best_state is None:
        raise RuntimeError("No best validation checkpoint was produced.")

    model.load_state_dict(best_state)

    # Re-evaluate the best checkpoint on source validation.
    val_loss, y_true, y_pred = evaluate(model, val_loader, criterion)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(4)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(4)),
    )

    # Test signals are used for forward inference only.
    model.eval()
    inference_start = time.perf_counter()
    inference_windows = 0

    with torch.no_grad():
        for array in test_arrays:
            loader = make_loader(array)
            for (xb,) in loader:
                _ = model(xb)
                inference_windows += len(xb)

    inference_seconds = time.perf_counter() - inference_start

    run_dir = OUT_DIR / f"source_{source_load}hp"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_load_hp": source_load,
            "seed": SEED,
            "label_map": LABEL_MAP,
            "trainable_parameters": params,
            "best_epoch": best_epoch,
        },
        run_dir / "best.pt",
    )

    result = {
        "component": "1d",
        "experiment": "CWRU cross-load",
        "source_load_hp": source_load,
        "seed": SEED,
        "train_windows": len(x_train),
        "validation_windows": len(x_val),
        "epochs": len(history),
        "updates": updates,
        "stop_reason": stop_reason,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "inference_windows": inference_windows,
        "trainable_parameters": params,
        "best_epoch": best_epoch,
        "validation_loss": val_loss,
        "validation_accuracy": accuracy_score(y_true, y_pred),
        "validation_macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(4)),
            average="macro",
            zero_division=0,
        ),
        "classification_report": report,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "test_scored": False,
    }

    (run_dir / "validation_report.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nSOURCE VALIDATION")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(4)),
        target_names=LABELS,
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(cm)
    print(f"Best epoch: {best_epoch}")
    print(f"Fit time: {fit_seconds:.2f}s")
    print(f"Unscored inference windows: {inference_windows}")

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for load in LOADS:
        results.append(run_source_load(load))

    summary = {
        "experiment": "CWRU cross-load 1-D CNN",
        "source_loads_hp": LOADS,
        "results": results,
        "test_scored": False,
        "note": "Test partition was not scored; inference only.",
    }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nALL FOUR SOURCE-LOAD FITS COMPLETE")
    print("Test partition: inference only; no test metrics calculated.")


if __name__ == "__main__":
    torch.set_num_threads(THREADS)
    main()
