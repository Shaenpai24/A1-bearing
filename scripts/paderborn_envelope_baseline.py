#!/usr/bin/env python3

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "paderborn"
OUTPUT_DIR = DATA_DIR / "baseline"

FS = 16_000
WINDOW_SIZE = 2048

# Paderborn 6203 bearing at N15 = 1500 RPM
SHAFT_HZ = 1500.0 / 60.0

# Documented characteristic-frequency ratios for the 6203 bearing.
# These correspond approximately to:
# BPFO = 76.35 Hz
# BPFI = 123.64 Hz
BPFO_RATIO = 3.054
BPFI_RATIO = 4.946

BPFO = BPFO_RATIO * SHAFT_HZ
BPFI = BPFI_RATIO * SHAFT_HZ

HARMONICS = (1, 2, 3)

# Frequency neighbourhood used around each characteristic frequency.
# This is intentionally fixed and is NOT tuned using test data.
BANDWIDTH_HZ = 5.0

RANDOM_STATE = 22


# ============================================================
# A1 BEARING DEFINITIONS
# ============================================================

TRAIN_BEARINGS = {
    "K001",
    "K002",
    "KA01",
    "KA03",
    "KI01",
}

VALIDATION_BEARINGS = {
    "K003",
    "KA05",
    "KI03",
}

TEST_BEARINGS = {
    "K004",
    "KA04",
    "KA15",
    "KA22",
    "KI04",
    "KI16",
}


# Map detailed dataset labels to the required three classes.
LABEL_MAP = {
    "healthy": "healthy",

    "outer_artificial": "outer",
    "outer_natural": "outer",

    "inner_artificial": "inner",
    "inner_natural": "inner",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_partition(bearing_id):
    if bearing_id in TRAIN_BEARINGS:
        return "train"

    if bearing_id in VALIDATION_BEARINGS:
        return "validation"

    if bearing_id in TEST_BEARINGS:
        return "test"

    raise ValueError(f"Unknown bearing: {bearing_id}")


def characteristic_frequencies():
    return {
        "BPFO": BPFO,
        "BPFI": BPFI,
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def envelope_features(x):
    """
    Extract the classical envelope-analysis features required
    by the A1 card.

    Features:
      - RMS
      - kurtosis
      - envelope-band energy around BPFO harmonics
      - envelope-band energy around BPFI harmonics
    """

    x = np.asarray(x, dtype=np.float64)

    # Remove DC component.
    x = x - np.mean(x)

    # --------------------------------------------------------
    # Time-domain features
    # --------------------------------------------------------

    rms = np.sqrt(np.mean(x ** 2))

    kurt = kurtosis(
        x,
        fisher=False,
        bias=False,
    )

    # --------------------------------------------------------
    # Hilbert envelope
    # --------------------------------------------------------

    analytic = hilbert(x)
    envelope = np.abs(analytic)

    # Remove envelope DC before FFT.
    envelope = envelope - np.mean(envelope)

    # --------------------------------------------------------
    # Envelope spectrum
    # --------------------------------------------------------

    spectrum = np.abs(
        np.fft.rfft(envelope)
    )

    freqs = np.fft.rfftfreq(
        len(envelope),
        d=1.0 / FS,
    )

    features = [
        rms,
        kurt,
    ]

    # --------------------------------------------------------
    # Characteristic-frequency energies
    # --------------------------------------------------------

    for name, base_frequency in characteristic_frequencies().items():

        for harmonic in HARMONICS:

            target = base_frequency * harmonic

            lower = target - BANDWIDTH_HZ
            upper = target + BANDWIDTH_HZ

            mask = (
                (freqs >= lower)
                & (freqs <= upper)
            )

            if np.any(mask):
                energy = np.sum(
                    spectrum[mask] ** 2
                )
            else:
                energy = 0.0

            features.append(energy)

    return np.asarray(features, dtype=np.float64)


# ============================================================
# LOAD NPZ DATA
# ============================================================

def load_dataset():

    rows = []

    npz_files = sorted(
        DATA_DIR.rglob("*.npz")
    )

    print("=" * 70)
    print("LOADING PADERBORN NPZ DATA")
    print("=" * 70)
    print(f"NPZ files: {len(npz_files)}")
    print()

    for index, path in enumerate(npz_files, 1):

        data = np.load(
            path,
            allow_pickle=True,
        )

        X = data["X"]

        bearing_id = str(
            data["bearing_id"].item()
        )

        original_label = str(
            data["label"].item()
        )

        if original_label not in LABEL_MAP:
            raise ValueError(
                f"Unknown label '{original_label}' "
                f"in {path}"
            )

        label = LABEL_MAP[original_label]

        split = get_partition(
            bearing_id
        )

        for window_index, window in enumerate(X):

            rows.append(
                {
                    "bearing_id": bearing_id,
                    "label": label,
                    "original_label": original_label,
                    "split": split,
                    "source": str(path),
                    "window_index": window_index,
                    "signal": window,
                }
            )

        if index % 25 == 0:
            print(
                f"Loaded {index}/{len(npz_files)} files"
            )

    return rows


# ============================================================
# FEATURE MATRIX
# ============================================================

def build_features(rows):

    print()
    print("=" * 70)
    print("EXTRACTING ENVELOPE FEATURES")
    print("=" * 70)

    feature_rows = []

    start = time.perf_counter()

    for i, row in enumerate(rows, 1):

        features = envelope_features(
            row["signal"]
        )

        feature_rows.append(
            features
        )

        if i % 500 == 0:
            print(
                f"Processed {i}/{len(rows)} windows"
            )

    elapsed = time.perf_counter() - start

    X = np.vstack(feature_rows)

    metadata = pd.DataFrame(
        [
            {
                "bearing_id": row["bearing_id"],
                "label": row["label"],
                "original_label": row["original_label"],
                "split": row["split"],
                "source": row["source"],
                "window_index": row["window_index"],
            }
            for row in rows
        ]
    )

    print()
    print(f"Feature matrix: {X.shape}")
    print(f"Extraction time: {elapsed:.3f} s")

    return X, metadata, elapsed


# ============================================================
# CLASSIFIER
# ============================================================

def fit_classifier(X_train, y_train):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    classifier.fit(
        X_train_scaled,
        y_train,
    )

    return scaler, classifier


def predict(
    scaler,
    classifier,
    X,
):
    return classifier.predict(
        scaler.transform(X)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, y_true, y_pred):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    print(
        f"Accuracy : {accuracy:.6f}"
    )

    print(
        f"Macro-F1 : {macro_f1:.6f}"
    )

    print()
    print(
        classification_report(
            y_true,
            y_pred,
            labels=["healthy", "outer", "inner"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["healthy", "outer", "inner"],
    )

    print(cm)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("PADERBORN A1 CLASSICAL ENVELOPE BASELINE")
    print("=" * 70)

    print(
        f"Sampling rate       : {FS} Hz"
    )

    print(
        f"Window size         : {WINDOW_SIZE}"
    )

    print(
        f"Shaft frequency     : {SHAFT_HZ:.6f} Hz"
    )

    print(
        f"BPFO                : {BPFO:.4f} Hz"
    )

    print(
        f"BPFI                : {BPFI:.4f} Hz"
    )

    print(
        f"Harmonics           : {HARMONICS}"
    )

    print(
        f"Band width          : ±{BANDWIDTH_HZ} Hz"
    )

    rows = load_dataset()

    X, metadata, extraction_time = (
        build_features(rows)
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_mask = (
        metadata["split"] == "train"
    )

    validation_mask = (
        metadata["split"] == "validation"
    )

    test_mask = (
        metadata["split"] == "test"
    )

    X_train = X[train_mask]
    X_validation = X[validation_mask]
    X_test = X[test_mask]

    y_train = metadata.loc[
        train_mask, "label"
    ].to_numpy()

    y_validation = metadata.loc[
        validation_mask, "label"
    ].to_numpy()

    y_test = metadata.loc[
        test_mask, "label"
    ].to_numpy()

    print()
    print("=" * 70)
    print("SPLIT")
    print("=" * 70)

    print(
        f"Train      : {len(y_train)}"
    )

    print(
        f"Validation : {len(y_validation)}"
    )

    print(
        f"Test       : {len(y_test)}"
    )

    # --------------------------------------------------------
    # Fit ONLY on training data
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FITTING BASELINE")
    print("=" * 70)

    fit_start = time.perf_counter()

    scaler, classifier = fit_classifier(
        X_train,
        y_train,
    )

    fit_time = (
        time.perf_counter()
        - fit_start
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_predictions = predict(
        scaler,
        classifier,
        X_validation,
    )

    validation_metrics = evaluate(
        "VALIDATION PERFORMANCE",
        y_validation,
        validation_predictions,
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_start = time.perf_counter()

    test_predictions = predict(
        scaler,
        classifier,
        X_test,
    )

    test_time = (
        time.perf_counter()
        - test_start
    )

    test_metrics = evaluate(
        "TEST PERFORMANCE",
        y_test,
        test_predictions,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    test_output = metadata.loc[
        test_mask
    ].copy()

    test_output[
        "prediction"
    ] = test_predictions

    test_output.to_csv(
        OUTPUT_DIR
        / "paderborn_envelope_test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save feature matrix
    # --------------------------------------------------------

    feature_names = [
        "rms",
        "kurtosis",

        "bpfo_h1_energy",
        "bpfo_h2_energy",
        "bpfo_h3_energy",

        "bpfi_h1_energy",
        "bpfi_h2_energy",
        "bpfi_h3_energy",
    ]

    feature_df = pd.DataFrame(
        X,
        columns=feature_names,
    )

    feature_df = pd.concat(
        [
            metadata.drop(
                columns=["source"]
            ).reset_index(drop=True),
            feature_df,
        ],
        axis=1,
    )

    feature_df.to_csv(
        OUTPUT_DIR
        / "paderborn_envelope_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "sampling_rate": FS,
        "window_size": WINDOW_SIZE,
        "shaft_frequency_hz": SHAFT_HZ,
        "BPFO_hz": BPFO,
        "BPFI_hz": BPFI,
        "harmonics": list(HARMONICS),
        "bandwidth_hz": BANDWIDTH_HZ,

        "feature_names": feature_names,

        "npz_files": len(
            sorted(DATA_DIR.rglob("*.npz"))
        ),

        "windows": len(metadata),

        "train_windows": int(train_mask.sum()),
        "validation_windows": int(
            validation_mask.sum()
        ),
        "test_windows": int(
            test_mask.sum()
        ),

        "feature_extraction_seconds":
            extraction_time,

        "fit_seconds":
            fit_time,

        "test_inference_seconds":
            test_time,

        "validation":
            validation_metrics,

        "test":
            test_metrics,
    }

    with (
        OUTPUT_DIR
        / "paderborn_envelope_baseline.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("BASELINE COMPLETE")
    print("=" * 70)

    print(
        f"Validation Macro-F1 : "
        f"{validation_metrics['macro_f1']:.6f}"
    )

    print(
        f"Test Macro-F1       : "
        f"{test_metrics['macro_f1']:.6f}"
    )

    print()
    print(
        f"Output directory:"
        f"\n  {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
