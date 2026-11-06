#!/usr/bin/env python3

"""
Paderborn Dataset Preprocessing
================================

A1 / Annex D1 preprocessing pipeline.

Source:
    data/raw/paderborn/extracted/<bearing_id>/*.mat

Signal:
    Y.vibration_1

Declared source sampling rate:
    64,000 Hz

Target sampling rate:
    16,000 Hz

Window:
    2,048 samples
    Non-overlapping

The bearing-level train/validation/test split is fixed and must
not be changed.
"""

from pathlib import Path
import csv
import math

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "paderborn" / "extracted"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "paderborn"

MANIFEST_PATH = OUTPUT_ROOT / "paderborn_manifest.csv"


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

SOURCE_FS = 64_000
TARGET_FS = 16_000

WINDOW_SIZE = 2_048

# D1 block boundaries
A_RATIO = 0.60
B_RATIO = 0.80

# Guard region around block boundaries.
#
# D1 specifies guard regions around the 60% / 80% boundaries.
# Keep this as an explicit parameter rather than hiding it in the code.
GUARD_SAMPLES = WINDOW_SIZE
PADERBORN_SPLITS = {
    "K001": "train",
    "K002": "train",
    "KA01": "train",
    "KA03": "train",
    "KI01": "train",

    "K003": "validation",
    "KA05": "validation",
    "KI03": "validation",

    "K004": "test",
    "KA04": "test",
    "KA15": "test",
    "KA22": "test",
    "KI04": "test",
    "KI16": "test",
}

# ============================================================================
# APPROVED A1 BEARING SPLIT
# ============================================================================

SPLITS = {
    "train": [
        "K001",
        "K002",
        "KA01",
        "KA03",
        "KI01",
    ],
    "validation": [
        "K003",
        "KA05",
        "KI03",
    ],
    "test": [
        "K004",
        "KA04",
        "KA15",
        "KA22",
        "KI04",
        "KI16",
    ],
}


# ============================================================================
# BEARING LABELS
# ============================================================================

LABELS = {
    "K001": "healthy",
    "K002": "healthy",
    "K003": "healthy",
    "K004": "healthy",

    "KA01": "outer_artificial",
    "KA03": "outer_artificial",
    "KA05": "outer_artificial",

    "KI01": "inner_artificial",
    "KI03": "inner_artificial",

    "KA04": "outer_natural",
    "KA15": "outer_natural",
    "KA22": "outer_natural",

    "KI04": "inner_natural",
    "KI16": "inner_natural",
}


# ============================================================================
# EXPECTED INVENTORY
# ============================================================================

EXPECTED_BEARINGS = sorted(LABELS.keys())
EXPECTED_FILES_PER_BEARING = 20


# ============================================================================
# HELPERS
# ============================================================================


def get_split(bearing_id: str) -> str:
    """Return train / validation / test for a bearing."""

    for split, bearings in SPLITS.items():
        if bearing_id in bearings:
            return split

    raise ValueError(
        f"Bearing '{bearing_id}' is not present in the approved A1 split."
    )


def extract_vibration(path: Path) -> np.ndarray:
    """
    Load a Paderborn MAT file and extract Y.vibration_1.
    """

    data = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False,
    )

    root_name = path.stem

    if root_name not in data:
        raise ValueError(
            f"{path.name}: root variable '{root_name}' not found."
        )

    root = data[root_name]

    if not hasattr(root, "Y"):
        raise ValueError(
            f"{path.name}: root structure has no Y field."
        )

    signals = np.atleast_1d(root.Y)

    for signal in signals:

        name = getattr(signal, "Name", None)

        if name == "vibration_1":

            values = getattr(signal, "Data", None)

            if values is None:
                raise ValueError(
                    f"{path.name}: vibration_1 has no Data field."
                )

            values = np.asarray(values, dtype=np.float64).reshape(-1)

            if values.size == 0:
                raise ValueError(
                    f"{path.name}: vibration_1 contains zero samples."
                )

            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{path.name}: vibration_1 contains NaN/Inf values."
                )

            return values

    raise ValueError(
        f"{path.name}: vibration_1 was not found."
    )


def resample_signal(signal: np.ndarray) -> np.ndarray:
    """
    Resample 64 kHz -> 16 kHz.

    64,000 / 16,000 = 4 exactly, so polyphase
    decimation by 4 is sufficient.

    resample_poly performs anti-alias filtering internally.
    """

    if SOURCE_FS % TARGET_FS != 0:
        raise RuntimeError(
            "Source/target sampling-rate relationship is not an "
            "integer decimation."
        )

    down = SOURCE_FS // TARGET_FS

    output = resample_poly(
        signal,
        up=1,
        down=down,
    )

    return np.asarray(output, dtype=np.float64)



def make_windows(signal: np.ndarray, split: str):
    """Generate non-overlapping 2048-sample windows."""
    windows = []

    n_windows = len(signal) // WINDOW_SIZE

    for i in range(n_windows):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window = signal[start:end]

        windows.append((window,split,start,end))

    return windows


def save_recording(
    output_path: Path,
    windows,
    bearing_id: str,
    label: str,
    split: str,
    measurement_id: int,
    source_file: str,
):
    """
    Save all windows belonging to one recording.
    """

    X = np.stack(
        [item[0] for item in windows],
        axis=0,
    )

    region = np.array(
        [item[1] for item in windows],
        dtype=object,
    )

    starts = np.array(
        [item[2] for item in windows],
        dtype=np.int64,
    )

    ends = np.array(
        [item[3] for item in windows],
        dtype=np.int64,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        X=X,
        region=region,
        start_sample=starts,
        end_sample=ends,
        bearing_id=bearing_id,
        label=label,
        split=split,
        measurement_id=measurement_id,
        source_file=source_file,
        source_sampling_rate=SOURCE_FS,
        sampling_rate=TARGET_FS,
        window_size=WINDOW_SIZE,
    )


# ============================================================================
# VALIDATE INVENTORY
# ============================================================================


def validate_inventory():
    print("=" * 70)
    print("PADERBORN INVENTORY VALIDATION")
    print("=" * 70)

    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw directory does not exist:\n{RAW_ROOT}"
        )

    found_bearings = sorted(
        p.name
        for p in RAW_ROOT.iterdir()
        if p.is_dir()
    )

    print(f"Expected bearings: {len(EXPECTED_BEARINGS)}")
    print(f"Found bearings:    {len(found_bearings)}")

    if found_bearings != EXPECTED_BEARINGS:
        missing = sorted(
            set(EXPECTED_BEARINGS) - set(found_bearings)
        )
        unexpected = sorted(
            set(found_bearings) - set(EXPECTED_BEARINGS)
        )

        print("Missing:", missing)
        print("Unexpected:", unexpected)

        raise RuntimeError(
            "Paderborn bearing inventory does not match A1."
        )

    total = 0

    for bearing in EXPECTED_BEARINGS:

        files = sorted(
            (RAW_ROOT / bearing).glob("*.mat")
        )

        count = len(files)

        print(
            f"{bearing:<6} "
            f"{LABELS[bearing]:<20} "
            f"{count:>3} MAT files"
        )

        if count != EXPECTED_FILES_PER_BEARING:
            raise RuntimeError(
                f"{bearing}: expected "
                f"{EXPECTED_FILES_PER_BEARING} MAT files, "
                f"found {count}."
            )

        total += count

    print("-" * 70)
    print(f"Total MAT files: {total}")

    if total != 280:
        raise RuntimeError(
            f"Expected 280 MAT files, found {total}."
        )

    print("INVENTORY: PASS")
    print()


# ============================================================================
# VALIDATE SPLIT
# ============================================================================


def validate_split():

    assigned = []

    for split, bearings in SPLITS.items():

        for bearing in bearings:

            if bearing not in LABELS:
                raise RuntimeError(
                    f"{bearing} appears in split but has no label."
                )

            assigned.append(bearing)

    if sorted(assigned) != EXPECTED_BEARINGS:
        raise RuntimeError(
            "A1 split does not contain every expected bearing exactly once."
        )

    if len(assigned) != len(set(assigned)):
        raise RuntimeError(
            "A1 split contains duplicate bearing assignments."
        )

    print("=" * 70)
    print("A1 SPLIT VALIDATION")
    print("=" * 70)

    for split in ("train", "validation", "test"):
        print(
            f"{split:<12}: "
            + ", ".join(SPLITS[split])
        )

    print("SPLIT: PASS")
    print()


# ============================================================================
# MAIN PROCESSING
# ============================================================================


def main():

    validate_inventory()
    validate_split()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old manifest.
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

    manifest_rows = []

    total_windows = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    print("=" * 70)
    print("PADERBORN PREPROCESSING")
    print("=" * 70)

    for bearing_id in EXPECTED_BEARINGS:

        label = LABELS[bearing_id]
        split = get_split(bearing_id)

        bearing_dir = RAW_ROOT / bearing_id

        files = sorted(
            bearing_dir.glob("*.mat")
        )

        print()
        print("-" * 70)
        print(
            f"{bearing_id} | "
            f"{label} | "
            f"{split}"
        )
        print("-" * 70)

        bearing_windows = 0

        for file_index, path in enumerate(files, 1):

            print(
                f"[{file_index:02d}/{len(files):02d}] "
                f"{path.name}"
            )

            # ------------------------------------------------------------
            # Extract vibration_1
            # ------------------------------------------------------------

            signal = extract_vibration(path)

            original_samples = len(signal)

            # ------------------------------------------------------------
            # Resample 64 kHz -> 16 kHz
            # ------------------------------------------------------------

            resampled = resample_signal(signal)

            resampled_samples = len(resampled)

            # ------------------------------------------------------------
            # Window
            # ------------------------------------------------------------

            windows = make_windows(resampled, split)

            n_windows = len(windows)

            if n_windows == 0:
                raise RuntimeError(
                    f"{path.name}: produced zero windows."
                )

            # ------------------------------------------------------------
            # Measurement ID
            # ------------------------------------------------------------

            try:
                measurement_id = int(
                    path.stem.rsplit("_", 1)[1]
                )
            except ValueError:
                raise RuntimeError(
                    f"Could not determine measurement ID "
                    f"from {path.name}"
                )

            # ------------------------------------------------------------
            # Save
            # ------------------------------------------------------------

            output_path = (
                OUTPUT_ROOT
                / split
                / bearing_id
                / f"{path.stem}.npz"
            )

            save_recording(
                output_path=output_path,
                windows=windows,
                bearing_id=bearing_id,
                label=label,
                split=split,
                measurement_id=measurement_id,
                source_file=path.name,
            )

            # ------------------------------------------------------------
            # Manifest
            # ------------------------------------------------------------

            manifest_rows.append(
                {
                    "bearing_id": bearing_id,
                    "label": label,
                    "split": split,
                    "file": path.name,
                    "measurement_id": measurement_id,
                    "original_sampling_rate": SOURCE_FS,
                    "sampling_rate": TARGET_FS,
                    "original_samples": original_samples,
                    "resampled_samples": resampled_samples,
                    "window_size": WINDOW_SIZE,
                    "windows": n_windows,
                    "output": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                }
            )

            total_windows[split] += n_windows
            bearing_windows += n_windows

            print(
                f"      "
                f"{original_samples:,} -> "
                f"{resampled_samples:,} samples | "
                f"{n_windows} windows"
            )

        print(
            f"TOTAL {bearing_id}: "
            f"{bearing_windows} windows"
        )

    # =========================================================================
    # WRITE MANIFEST
    # =========================================================================

    fieldnames = [
        "bearing_id",
        "label",
        "split",
        "file",
        "measurement_id",
        "original_sampling_rate",
        "sampling_rate",
        "original_samples",
        "resampled_samples",
        "window_size",
        "windows",
        "output",
    ]

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Train windows:      {total_windows['train']}"
    )
    print(
        f"Validation windows: {total_windows['validation']}"
    )
    print(
        f"Test windows:       {total_windows['test']}"
    )

    total = sum(total_windows.values())

    print(
        f"Total windows:      {total}"
    )

    print()
    print(f"Manifest:")
    print(f"  {MANIFEST_PATH}")

    print()
    print("Output:")
    print(f"  {OUTPUT_ROOT}")

    print()
    print("=" * 70)
    print("NEXT STEP: VALIDATE THE PROCESSED DATASET")
    print("=" * 70)


if __name__ == "__main__":
    main()

