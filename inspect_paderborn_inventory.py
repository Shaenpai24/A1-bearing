from pathlib import Path
from collections import Counter

ROOT = Path("data/raw/paderborn/extracted")

# Expected bearing folders from the downloaded dataset subset
EXPECTED = [
    "K001",
    "K002",
    "K003",
    "K004",
    "KA01",
    "KA03",
    "KA04",
    "KA05",
    "KA15",
    "KA22",
    "KI01",
    "KI03",
    "KI04",
    "KI16",
]

print("=" * 70)
print("PADERBORN DATASET — BEARING INVENTORY")
print("=" * 70)

if not ROOT.exists():
    print(f"\nERROR: Directory does not exist:")
    print(f"  {ROOT.resolve()}")
    raise SystemExit(1)

# ------------------------------------------------------------
# Find bearing directories
# ------------------------------------------------------------

directories = sorted(
    d for d in ROOT.iterdir()
    if d.is_dir()
)

print(f"\nRoot directory:")
print(f"  {ROOT.resolve()}")

print(f"\nBearing directories found: {len(directories)}")

# ------------------------------------------------------------
# Check each bearing
# ------------------------------------------------------------

total_files = 0
actual_ids = []

print("\n" + "-" * 70)
print(f"{'BEARING':<10} {'CLASS':<10} {'MAT FILES':>10} {'STATUS':<10}")
print("-" * 70)

for directory in directories:

    bearing_id = directory.name
    actual_ids.append(bearing_id)

    mat_files = sorted(directory.glob("*.mat"))
    count = len(mat_files)

    total_files += count

    # Coarse class based ONLY on the Paderborn naming convention.
    # Detailed damage information will be validated separately.
    if bearing_id.startswith("KI"):
        coarse_class = "inner"
    elif bearing_id.startswith("KA"):
        coarse_class = "outer"
    elif bearing_id.startswith("K"):
        coarse_class = "healthy"
    else:
        coarse_class = "unknown"

    status = "OK" if count == 20 else "CHECK"

    print(
        f"{bearing_id:<10} "
        f"{coarse_class:<10} "
        f"{count:>10} "
        f"{status:<10}"
    )

print("-" * 70)

# ------------------------------------------------------------
# Check expected IDs
# ------------------------------------------------------------

expected_set = set(EXPECTED)
actual_set = set(actual_ids)

missing = sorted(expected_set - actual_set)
unexpected = sorted(actual_set - expected_set)

print("\nEXPECTED DATASET SUBSET")
print("-" * 70)

print(f"Expected bearing IDs : {len(EXPECTED)}")
print(f"Found bearing IDs    : {len(actual_ids)}")
print(f"Total MAT files      : {total_files}")

print("\nExpected IDs:")
print("  " + ", ".join(EXPECTED))

if missing:
    print("\nMISSING:")
    for bearing in missing:
        print(f"  ❌ {bearing}")
else:
    print("\nMissing:")
    print("  None")

if unexpected:
    print("\nUNEXPECTED:")
    for bearing in unexpected:
        print(f"  ⚠️ {bearing}")
else:
    print("\nUnexpected:")
    print("  None")

# ------------------------------------------------------------
# File-count validation
# ------------------------------------------------------------

counts = {}

for directory in directories:
    counts[directory.name] = len(
        list(directory.glob("*.mat"))
    )

bad_counts = {
    bearing: count
    for bearing, count in counts.items()
    if count != 20
}

print("\nFILE COUNT VALIDATION")
print("-" * 70)

if bad_counts:
    print("Bearings with file counts other than 20:")

    for bearing, count in bad_counts.items():
        print(f"  ⚠️ {bearing}: {count} files")
else:
    print("✅ Every bearing has exactly 20 MAT files.")

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

if (
    not missing
    and not unexpected
    and not bad_counts
    and len(actual_ids) == len(EXPECTED)
    and total_files == 280
):
    print("✅ INVENTORY VALIDATION PASSED")
    print("   14 bearing IDs")
    print("   20 MAT files per bearing")
    print("   280 MAT files total")
else:
    print("⚠️ INVENTORY REQUIRES REVIEW")

print("=" * 70)
