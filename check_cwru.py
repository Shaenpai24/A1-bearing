from pathlib import Path
from scipy.io import whosmat

DATA_DIR = Path("data/raw/cwru")

# File -> expected drive-end variable
EXPECTED = {
    97: "X097_DE_time",
    98: "X098_DE_time",
    99: "X099_DE_time",   # IMPORTANT: do not use the first DE variable
    100: "X100_DE_time",

    105: "X105_DE_time",
    106: "X106_DE_time",
    107: "X107_DE_time",
    108: "X108_DE_time",

    118: "X118_DE_time",
    119: "X119_DE_time",
    120: "X120_DE_time",
    121: "X121_DE_time",

    130: "X130_DE_time",
    131: "X131_DE_time",
    132: "X132_DE_time",
    133: "X133_DE_time",
}

print("=" * 70)
print("CWRU A1 DATA VALIDATION")
print("=" * 70)

total_bytes = 0
passed = 0

for number, expected_variable in EXPECTED.items():

    path = DATA_DIR / f"{number}.mat"

    print(f"\n[{number}.mat]")

    # Check file exists
    if not path.exists():
        print("  ❌ FILE MISSING")
        continue

    size = path.stat().st_size
    total_bytes += size

    print(f"  Size: {size:,} bytes")

    try:
        variables = whosmat(path)
    except Exception as e:
        print(f"  ❌ MATLAB READ FAILED")
        print(f"     {e}")
        continue

    variable_names = {name for name, shape, dtype in variables}

    print(f"  Variables: {sorted(variable_names)}")

    # Check expected DE signal
    if expected_variable not in variable_names:
        print(f"  ❌ MISSING EXPECTED VARIABLE: {expected_variable}")
        continue

    # Find shape/dtype
    info = next(
        (shape, dtype)
        for name, shape, dtype in variables
        if name == expected_variable
    )

    shape, dtype = info

    print(f"  DE signal: {expected_variable}")
    print(f"  Shape: {shape}")
    print(f"  Type: {dtype}")

    # Special check for 99.mat
    if number == 99:
        if "X098_DE_time" in variable_names:
            print("  ⚠️  X098_DE_time also present — EXPECTED TRAP")
        print("  ✓ Explicitly selecting X099_DE_time")

    print("  ✅ PASS")
    passed += 1


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Files passed: {passed}/{len(EXPECTED)}")
print(f"Total bytes:  {total_bytes:,}")

EXPECTED_TOTAL = 70_056_712

print(f"Manifest total: {EXPECTED_TOTAL:,}")
print(f"Difference:     {total_bytes - EXPECTED_TOTAL:+,} bytes")

if passed == len(EXPECTED):
    print("\n✅ ALL 16 CWRU FILES PASSED CONTENT VALIDATION")
else:
    print("\n❌ SOME FILES FAILED — DO NOT PREPROCESS YET")
