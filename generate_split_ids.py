
from pathlib import Path
import pandas as pd

manifest_path = Path("/home/shaenpai/Documents/A1-bearing/data/processed/paderborn/paderborn_manifest.csv")
output_path = Path("T06_A1_paderborn.csv")

manifest = pd.read_csv(manifest_path)

label_map = {
    "healthy": "healthy",
    "outer_artificial": "outer",
    "outer_natural": "outer",
    "inner_artificial": "inner",
    "inner_natural": "inner",
}

rows = []

for _, record in manifest.iterrows():
    bearing = record["bearing_id"]
    filename = record["file"]
    label = label_map[record["label"]]
    partition = record["split"]

    # The manifest records the number of full windows.
    n_windows = int(record["windows"])
    window_size = int(record["window_size"])

    for i in range(n_windows):
        start = i * window_size
        rows.append({
            "sample_id": f"{bearing}/{filename}:{start}",
            "unit_id": bearing,
            "partition": "val" if partition == "validation" else partition,
            "label": label,
            "reason": "",
        })

result = pd.DataFrame(
    rows,
    columns=["sample_id", "unit_id", "partition", "label", "reason"],
)

result.to_csv(output_path, index=False, encoding="utf-8")

print(result["partition"].value_counts())
print(f"Saved {output_path}")

