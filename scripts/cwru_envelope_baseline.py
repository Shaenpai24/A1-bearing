#!/usr/bin/env python3
"""CWRU A1 envelope-analysis baseline.

Fits one fixed RandomForest baseline per source load (0, 1, 2, 3 hp).
Reports source-load validation metrics only. It runs inference on every
load's test partition but does NOT read test labels for scoring.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "cwru"
OUTPUT_DIR = DATA_DIR / "baseline"

FS = 12_000
WINDOW_SIZE = 2_048
HARMONICS = (1, 2, 3)
RANDOM_STATE = 22

# Nominal CWRU drive-end shaft speeds by load (RPM). These are fixed metadata,
# not estimated from held-out signals.
RPM_BY_LOAD = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}

# CWRU 6205 bearing geometry: 9 rolling elements, ball diameter / pitch
# diameter, and zero nominal contact angle. The ratios are theoretical.
N_BALLS = 9
BALL_DIAMETER_IN = 0.3125
PITCH_DIAMETER_IN = 1.537
CONTACT_ANGLE_RAD = 0.0
BD_PD = BALL_DIAMETER_IN / PITCH_DIAMETER_IN

BPFO_RATIO = (N_BALLS / 2.0) * (1.0 - BD_PD * np.cos(CONTACT_ANGLE_RAD))
BPFI_RATIO = (N_BALLS / 2.0) * (1.0 + BD_PD * np.cos(CONTACT_ANGLE_RAD))
BSF_RATIO = (PITCH_DIAMETER_IN / (2.0 * BALL_DIAMETER_IN)) * (
    1.0 - (BD_PD**2) * (np.cos(CONTACT_ANGLE_RAD) ** 2)
)
FAULT_RATIOS = {"BPFO": BPFO_RATIO, "BPFI": BPFI_RATIO, "BSF": BSF_RATIO}

LABELS = ["healthy", "inner", "outer", "ball"]
FEATURE_NAMES = [
    "rms",
    "kurtosis",
    *[
        f"{name}_h{harmonic}_energy"
        for name in FAULT_RATIOS
        for harmonic in HARMONICS
    ],
]


def scalar_from_npz(data, key):
    value = data[key]
    return value.item() if np.asarray(value).shape == () else value


def characteristic_frequencies(load_hp: int) -> dict[str, float]:
    shaft_hz = RPM_BY_LOAD[load_hp] / 60.0
    return {name: ratio * shaft_hz for name, ratio in FAULT_RATIOS.items()}


def envelope_features(window: np.ndarray, load_hp: int) -> np.ndarray:
    x = np.asarray(window, dtype=np.float64)
    x = x - np.mean(x)

    rms = float(np.sqrt(np.mean(x**2)))
    kurt = float(kurtosis(x, fisher=False, bias=False))
    if not np.isfinite(kurt):
        kurt = 0.0

    envelope = np.abs(hilbert(x))
    envelope -= np.mean(envelope)
    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), d=1.0 / FS)

    # One FFT-bin on either side, fixed by the sampling rate/window length.
    # This avoids an excessively narrow band for 2048 samples at 12 kHz.
    bandwidth_hz = FS / WINDOW_SIZE
    features = [rms, kurt]
    for base_frequency in characteristic_frequencies(load_hp).values():
        for harmonic in HARMONICS:
            target = base_frequency * harmonic
            mask = (freqs >= target - bandwidth_hz) & (
                freqs <= target + bandwidth_hz
            )
            energy = float(np.sum(spectrum[mask] ** 2)) if np.any(mask) else 0.0
            features.append(energy)

    result = np.asarray(features, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite envelope feature encountered")
    return result


def load_recordings():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"{DATA_DIR} does not exist. Run src/preprocess_cwru.py first."
        )

    records = []
    for path in sorted(DATA_DIR.glob("*.npz")):
        # Skip this script's own output folder.
        if path.parent.name == "baseline":
            continue
        with np.load(path, allow_pickle=True) as data:
            file_number = int(scalar_from_npz(data, "file_number"))
            label = str(scalar_from_npz(data, "label"))
            load_hp = int(scalar_from_npz(data, "load_hp"))
            if label not in LABELS:
                raise ValueError(f"Unexpected CWRU label {label!r} in {path}")
            if load_hp not in RPM_BY_LOAD:
                raise ValueError(f"Unexpected load {load_hp} hp in {path}")

            # Copy arrays before closing the NPZ handle.
            parts = {
                part: np.asarray(data[part], dtype=np.float32).copy()
                for part in ("train", "validation", "test")
            }
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "file_number": file_number,
                "label": label,
                "load_hp": load_hp,
                "parts": parts,
            }
        )

    if not records:
        raise FileNotFoundError(f"No per-recording NPZ files found in {DATA_DIR}")
    return records


def build_partition(records, partition):
    feature_rows, labels, metadata = [], [], []
    start = time.perf_counter()
    for rec in records:
        windows = rec["parts"][partition]
        for window_index, window in enumerate(windows):
            feature_rows.append(envelope_features(window, rec["load_hp"]))
            labels.append(rec["label"])
            metadata.append(
                {
                    "file_number": rec["file_number"],
                    "load_hp": rec["load_hp"],
                    "label": rec["label"],
                    "partition": partition,
                    "window_index": window_index,
                    "sample_id": (
                        f'{rec["file_number"]}.mat:{partition}:{window_index}'
                    ),
                }
            )

    if not feature_rows:
        raise ValueError(f"No windows found for partition {partition!r}")
    X = np.vstack(feature_rows)
    y = np.asarray(labels, dtype=str)
    meta = pd.DataFrame(metadata)
    elapsed = time.perf_counter() - start
    return X, y, meta, elapsed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = load_recordings()

    print("=" * 72)
    print("CWRU A1 CLASSICAL ENVELOPE BASELINE")
    print("=" * 72)
    print(f"Input recordings: {len(records)}")
    print(f"Sampling rate: {FS} Hz; window: {WINDOW_SIZE}; stride inherited from preprocessing")
    print("Test-label scoring: DISABLED")
    print("Characteristic frequency ratios:")
    for name, ratio in FAULT_RATIOS.items():
        print(f"  {name}: {ratio:.5f} x shaft frequency")

    partitions = {}
    extraction_seconds = {}
    for part in ("train", "validation", "test"):
        X, y, meta, elapsed = build_partition(records, part)
        partitions[part] = (X, y, meta)
        extraction_seconds[part] = elapsed
        print(f"{part:10s}: {len(y):5d} windows, features={X.shape}, extraction={elapsed:.2f}s")

    X_train_all, y_train_all, meta_train_all = partitions["train"]
    X_val_all, y_val_all, meta_val_all = partitions["validation"]
    X_test_all, _y_test_unused, meta_test_all = partitions["test"]

    all_predictions = []
    source_validation_summary = []
    fit_records = []

    for source_load in sorted(RPM_BY_LOAD):
        train_mask = meta_train_all["load_hp"].to_numpy() == source_load
        val_mask = meta_val_all["load_hp"].to_numpy() == source_load

        X_train = X_train_all[train_mask]
        y_train = y_train_all[train_mask]
        X_val = X_val_all[val_mask]
        y_val = y_val_all[val_mask]

        if len(y_train) == 0 or len(y_val) == 0:
            raise ValueError(
                f"Missing train/validation windows for source load {source_load} hp"
            )

        fit_start = time.perf_counter()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X_train_scaled, y_train)
        fit_seconds = time.perf_counter() - fit_start

        val_pred = model.predict(scaler.transform(X_val))
        val_accuracy = float(accuracy_score(y_val, val_pred))
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        report = classification_report(
            y_val,
            val_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_val, val_pred, labels=LABELS).tolist()

        print()
        print("-" * 72)
        print(f"SOURCE LOAD: {source_load} hp")
        print(f"Training windows: {len(y_train)}; source-validation windows: {len(y_val)}")
        print(f"Fit time: {fit_seconds:.3f}s")
        print(f"Source-validation accuracy: {val_accuracy:.6f}")
        print(f"Source-validation macro-F1: {val_macro_f1:.6f}")
        print("Source-validation classification report:")
        print(classification_report(y_val, val_pred, labels=LABELS, zero_division=0))
        print(f"Confusion matrix (labels={LABELS}):")
        print(np.asarray(cm))

        source_validation_summary.append(
            {
                "source_load_hp": source_load,
                "validation_windows": int(len(y_val)),
                "accuracy": val_accuracy,
                "macro_f1": val_macro_f1,
                "per_class": {
                    label: {
                        "precision": float(report[label]["precision"]),
                        "recall": float(report[label]["recall"]),
                        "f1": float(report[label]["f1-score"]),
                        "support": int(report[label]["support"]),
                    }
                    for label in LABELS
                },
                "confusion_matrix_labels": LABELS,
                "confusion_matrix": cm,
                "fit_seconds": fit_seconds,
            }
        )
        fit_records.append(
            {
                "source_load_hp": source_load,
                "train_windows": int(len(y_train)),
                "validation_windows": int(len(y_val)),
                "fit_seconds": fit_seconds,
            }
        )

        # Unscored inference over every target-load test partition.
        # Do not index, compare, or save test labels.
        for target_load in sorted(RPM_BY_LOAD):
            target_mask = meta_test_all["load_hp"].to_numpy() == target_load
            target_indices = np.flatnonzero(target_mask)
            if len(target_indices) == 0:
                raise ValueError(f"No test windows for target load {target_load} hp")

            infer_start = time.perf_counter()
            target_pred = model.predict(scaler.transform(X_test_all[target_indices]))
            inference_seconds = time.perf_counter() - infer_start
            target_meta = meta_test_all.iloc[target_indices]

            for row, prediction in zip(target_meta.to_dict("records"), target_pred):
                all_predictions.append(
                    {
                        "source_load_hp": source_load,
                        "target_load_hp": target_load,
                        "file_number": int(row["file_number"]),
                        "sample_id": row["sample_id"],
                        "partition": "test",
                        "predicted_label": str(prediction),
                        "test_scored": "no",
                    }
                )
            print(
                f"Unscored test inference: source={source_load} hp -> "
                f"target={target_load} hp; windows={len(target_indices)}; "
                f"time={inference_seconds:.3f}s"
            )

    predictions_path = OUTPUT_DIR / "cwru_envelope_test_predictions.csv"
    pd.DataFrame(all_predictions).to_csv(predictions_path, index=False)

    metadata_path = OUTPUT_DIR / "cwru_envelope_features.csv"
    feature_rows = []
    for part, (X, _y_unused, meta) in partitions.items():
        for idx, row in meta.iterrows():
            item = row.to_dict()
            # Store labels for train/validation only. Do not expose test labels.
            if part != "test":
                item["label"] = str(item["label"])
            else:
                item.pop("label", None)
            for col_index, name in enumerate(FEATURE_NAMES):
                item[name] = float(X[idx, col_index])
            item["test_scored"] = "no" if part == "test" else ""
            feature_rows.append(item)
    pd.DataFrame(feature_rows).to_csv(metadata_path, index=False)

    total_seconds = time.perf_counter() - started
    summary = {
        "experiment": "CWRU cross-load envelope baseline",
        "sampling_rate_hz": FS,
        "window_size": WINDOW_SIZE,
        "rpm_by_load": RPM_BY_LOAD,
        "fault_frequency_ratios": {k: float(v) for k, v in FAULT_RATIOS.items()},
        "harmonics": list(HARMONICS),
        "bandwidth_hz": FS / WINDOW_SIZE,
        "feature_names": FEATURE_NAMES,
        "labels": LABELS,
        "source_validation": source_validation_summary,
        "fit_records": fit_records,
        "partition_windows": {
            part: int(len(values[1])) for part, values in partitions.items()
        },
        "feature_extraction_seconds": extraction_seconds,
        "total_elapsed_seconds": total_seconds,
        "test_scored": "no",
        "test_metrics_status": "withheld; predictions are unscored pending results freeze",
        "outputs": {
            "features": str(metadata_path.relative_to(ROOT)),
            "unscored_test_predictions": str(predictions_path.relative_to(ROOT)),
        },
    }
    summary_path = OUTPUT_DIR / "cwru_envelope_baseline.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("COMPLETE")
    print(f"Source-validation summary: {summary_path.relative_to(ROOT)}")
    print(f"Feature matrix: {metadata_path.relative_to(ROOT)}")
    print(f"Unscored test predictions: {predictions_path.relative_to(ROOT)}")
    print(f"Total elapsed: {total_seconds:.2f}s")
    print("Test labels were not scored. Do not report target-test metrics before results freeze.")


if __name__ == "__main__":
    main()
