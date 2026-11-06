from pathlib import Path
import pandas as pd

MANIFEST = Path("data/processed/paderborn/paderborn_manifest.csv")
OUTPUT = Path("data/processed/paderborn/split_ids")

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

required = {"bearing_id", "file", "measurement_id", "split"}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

for split in ["train", "validation", "test"]:
    subset = df[df["split"] == split].copy()

    # Keep the identifiers needed to reproduce the partition.
    subset[
        ["bearing_id", "file", "measurement_id"]
    ].to_csv(
        OUTPUT / f"{split}_ids.csv",
        index=False
    )

    print(
        f"{split:10s}: "
        f"{len(subset):4d} recordings | "
        f"{subset['bearing_id'].nunique():2d} bearings | "
        f"{OUTPUT / f'{split}_ids.csv'}"
    )

print("\nSplit-ID export complete.")
