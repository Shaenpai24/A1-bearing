#!/usr/bin/env python3

from pathlib import Path
import sys
import pandas as pd

# scripts/validate_paderborn.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MANIFEST = PROJECT_ROOT / "data" / "processed" / "paderborn" / "paderborn_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "paderborn"

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn"
)

# A1 bearing-disjoint split
TRAIN_BEARINGS = {
    "K001", "K002", "KA01", "KA03", "KI01"
}

VALIDATION_BEARINGS = {
    "K003", "KA05", "KI03"
}

TEST_BEARINGS = {
    "K004", "KA04", "KA15", "KA22", "KI04", "KI16"
}

EXPECTED_BEARINGS = (
    TRAIN_BEARINGS
    | VALIDATION_BEARINGS
    | TEST_BEARINGS
)

EXPECTED_LABELS = {
    # Healthy
    "K001": "healthy",
    "K002": "healthy",
    "K003": "healthy",
    "K004": "healthy",

    # Outer race faults
    "KA01": "outer_artificial",
    "KA03": "outer_artificial",
    "KA04": "outer_natural",
    "KA05": "outer_artificial",
    "KA15": "outer_natural",
    "KA22": "outer_natural",

    # Inner race faults
    "KI01": "inner_artificial",
    "KI03": "inner_artificial",
    "KI04": "inner_natural",
    "KI16": "inner_natural",
}

EXPECTED_FILES_PER_BEARING = 20
EXPECTED_WINDOW_SIZE = 2048
EXPECTED_FS = 16000

# ============================================================
# Helpers
# ============================================================

errors = []
warnings = []


def check(condition, message):
    if condition:
        print(f"  PASS  {message}")
    else:
        print(f"  FAIL  {message}")
        errors.append(message)


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Start
# ============================================================

print("=" * 70)
print("PADERBORN A1 PROCESSED DATASET VALIDATION")
print("=" * 70)

print(f"Project root : {PROJECT_ROOT}")
print(f"Manifest     : {MANIFEST}")
print(f"Output dir   : {PROCESSED_DIR}")

# ============================================================
# 1. Manifest exists
# ============================================================

section("1. MANIFEST")

check(MANIFEST.exists(), f"Manifest exists: {MANIFEST}")

if not MANIFEST.exists():
    print()
    print("Cannot continue without manifest.")
    sys.exit(1)

df = pd.read_csv(MANIFEST)

print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")

required_columns = {
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
}

missing_columns = required_columns - set(df.columns)

check(
    not missing_columns,
    f"Required columns present: {sorted(required_columns)}"
)

if missing_columns:
    print(f"Missing columns: {sorted(missing_columns)}")
    sys.exit(1)

# ============================================================
# 2. Bearing inventory
# ============================================================

section("2. BEARING INVENTORY")

found_bearings = set(df["bearing_id"].astype(str))

print(f"Expected bearings: {len(EXPECTED_BEARINGS)}")
print(f"Found bearings:    {len(found_bearings)}")

missing = EXPECTED_BEARINGS - found_bearings
unexpected = found_bearings - EXPECTED_BEARINGS

check(not missing, f"No expected bearings missing: {sorted(missing)}")
check(not unexpected, f"No unexpected bearings: {sorted(unexpected)}")

print()
for bearing in sorted(EXPECTED_BEARINGS):
    count = int((df["bearing_id"] == bearing).sum())

    status = "PASS" if count == EXPECTED_FILES_PER_BEARING else "FAIL"

    print(
        f"  {status}  {bearing:<5} "
        f"{count:>3} MAT recordings "
        f"(expected {EXPECTED_FILES_PER_BEARING})"
    )

    if count != EXPECTED_FILES_PER_BEARING:
        errors.append(
            f"{bearing} has {count} recordings; "
            f"expected {EXPECTED_FILES_PER_BEARING}"
        )

# ============================================================
# 3. Labels
# ============================================================

section("3. LABEL VALIDATION")

for bearing in sorted(EXPECTED_BEARINGS):

    expected = EXPECTED_LABELS[bearing]

    subset = df[df["bearing_id"] == bearing]

    labels = set(subset["label"].astype(str)) if "label" in df.columns else set()

    if "label" not in df.columns:
        print(
            "  WARNING  Manifest has no 'label' column; "
            "label validation skipped."
        )
        warnings.append("Manifest has no label column.")
        break

    check(
        labels == {expected},
        f"{bearing}: expected label '{expected}', found {sorted(labels)}"
    )

# ============================================================
# 4. Bearing-disjoint split
# ============================================================

section("4. BEARING-DISJOINT A1 SPLIT")

split_sets = {
    "train": TRAIN_BEARINGS,
    "validation": VALIDATION_BEARINGS,
    "test": TEST_BEARINGS,
}

for split_name, expected_bearings in split_sets.items():

    if "split" not in df.columns:
        print(
            "  WARNING  Manifest has no 'split' column; "
            "split validation will use bearing IDs only."
        )
        warnings.append("Manifest has no split column.")
        break

    subset = df[df["split"].astype(str) == split_name]
    actual = set(subset["bearing_id"].astype(str))

    print(f"\n{split_name.upper()}")
    print(f"  Expected: {sorted(expected_bearings)}")
    print(f"  Found:    {sorted(actual)}")

    check(
        actual == expected_bearings,
        f"{split_name}: bearing assignment matches A1"
    )

# Explicit leakage check
for name_a, set_a in split_sets.items():
    for name_b, set_b in split_sets.items():

        if name_a >= name_b:
            continue

        overlap = set_a & set_b

        check(
            not overlap,
            f"No bearing leakage: {name_a} ↔ {name_b}"
        )

# ============================================================
# 5. Window size
# ============================================================

section("5. WINDOW SIZE")

if "window_size" in df.columns:

    values = set(df["window_size"].dropna().astype(int))

    print(f"Window sizes found: {sorted(values)}")

    check(
        values == {EXPECTED_WINDOW_SIZE},
        f"All windows are {EXPECTED_WINDOW_SIZE} samples"
    )

else:
    print(
        "  WARNING  Manifest has no 'window_size' column."
    )
    warnings.append("Manifest has no window_size column.")

# ============================================================
# 6. Sampling rate
# ============================================================

section("6. SAMPLING RATE")

possible_fs_columns = [
    "sampling_rate",
    "target_fs",
    "fs",
]

fs_column = None

for column in possible_fs_columns:
    if column in df.columns:
        fs_column = column
        break

if fs_column:

    values = set(
        df[fs_column]
        .dropna()
        .astype(float)
    )

    print(f"Sampling rates found: {sorted(values)}")

    check(
        values == {float(EXPECTED_FS)},
        f"All processed windows use {EXPECTED_FS} Hz"
    )

else:
    print(
        "  WARNING  No sampling-rate column found."
    )
    warnings.append("No sampling-rate column found.")

# ============================================================
# 7. Vibration metadata
# ============================================================

section("7. VIBRATION METADATA")

if "vibration_raster" in df.columns:

    rasters = set(
        df["vibration_raster"]
        .dropna()
        .astype(str)
    )

    print(f"Vibration rasters: {sorted(rasters)}")

if "vibration_samples" in df.columns:

    samples = pd.to_numeric(
        df["vibration_samples"],
        errors="coerce"
    )

    print(
        f"Vibration sample counts: "
        f"min={samples.min():,.0f}, "
        f"max={samples.max():,.0f}"
    )

# ============================================================
# 8. Duplicate recording/window checks
# ============================================================

section("8. DUPLICATE CHECKS")

duplicate_file_rows = df.duplicated(
    subset=["bearing_id", "file"],
    keep=False
)

duplicate_measurements = df.duplicated(
    subset=["bearing_id", "measurement_id"],
    keep=False
)

check(
    not duplicate_file_rows.any(),
    "No duplicate bearing/file records"
)

check(
    not duplicate_measurements.any(),
    "No duplicate bearing/measurement IDs"
)

if duplicate_file_rows.any():

    print("\nDuplicate files:")
    print(
        df.loc[
            duplicate_file_rows,
            ["bearing_id", "file"]
        ].to_string(index=False)
    )

if duplicate_measurements.any():

    print("\nDuplicate measurements:")
    print(
        df.loc[
            duplicate_measurements,
            ["bearing_id", "measurement_id"]
        ].to_string(index=False)
    )

# ======================================================================
# 9. PROCESSED OUTPUT FILES
# ======================================================================

print("=" * 70)
print("9. PROCESSED OUTPUT FILES")
print("=" * 70)

npz_files = sorted(OUTPUT_DIR.rglob("*.npz"))

print(f"NPZ files found: {len(npz_files)}")

if npz_files:
    print(f"First NPZ: {npz_files[0]}")
    print(f"Last NPZ : {npz_files[-1]}")

    # Show distribution by split
    for split in ["train", "validation", "test"]:
        split_dir = OUTPUT_DIR / split
        count = len(list(split_dir.rglob("*.npz")))
        print(f"  {split:12s}: {count} NPZ files")
else:
    print("  WARNING: No NPZ files found.")

# ======================================================================
# 10. WINDOW COUNTS
# ======================================================================

print()
print("=" * 70)
print("10. WINDOW COUNTS")
print("=" * 70)

# The manifest has one row per source MAT recording.
# The `windows` column contains the actual number of processed
# 2048-sample windows generated from that recording.

df["windows"] = pd.to_numeric(
    df["windows"],
    errors="coerce"
).fillna(0).astype(int)

split_window_counts = (
    df.groupby("split")["windows"]
    .sum()
    .reindex(["train", "validation", "test"], fill_value=0)
)

train_windows = int(split_window_counts["train"])
validation_windows = int(split_window_counts["validation"])
test_windows = int(split_window_counts["test"])
total_windows = train_windows + validation_windows + test_windows

print(f"  train       : {train_windows:6d} windows")
print(f"  validation  : {validation_windows:6d} windows")
print(f"  test        : {test_windows:6d} windows")
print(f"  TOTAL       : {total_windows:6d} windows")

if total_windows > 0:
    print("  PASS  Processed window counts are present in manifest")
else:
    print("  FAIL  No processed windows recorded in manifest")
    errors.append("No processed windows recorded in manifest")


# ======================================================================
# 11. WINDOWS PER BEARING
# ======================================================================

print()
print("=" * 70)
print("11. WINDOWS PER BEARING")
print("=" * 70)

windows_per_bearing = (
    df.groupby("bearing_id")["windows"]
    .sum()
    .sort_index()
)

for bearing_id, count in windows_per_bearing.items():
    print(f"  {bearing_id:<6} : {int(count):6d} windows")


# ======================================================================
# 12. WINDOWS BY BEARING AND SPLIT
# ======================================================================

print()
print("=" * 70)
print("12. WINDOWS BY BEARING AND SPLIT")
print("=" * 70)

window_table = (
    df.pivot_table(
        index="bearing_id",
        columns="split",
        values="windows",
        aggfunc="sum",
        fill_value=0
    )
    .reindex(columns=["test", "train", "validation"], fill_value=0)
)

print(window_table.to_string())

# ============================================================
# FINAL RESULT
# ============================================================

section("FINAL RESULT")

if errors:

    print("❌ VALIDATION FAILED")
    print()
    print(f"Errors: {len(errors)}")

    for error in errors:
        print(f"  - {error}")

else:

    print("✅ VALIDATION PASSED")
    print("   No hard validation errors found.")

if warnings:

    print()
    print(f"Warnings: {len(warnings)}")

    for warning in warnings:
        print(f"  - {warning}")

print()
print("=" * 70)
print("END OF PADERBORN VALIDATION")
print("=" * 70)

sys.exit(1 if errors else 0)

