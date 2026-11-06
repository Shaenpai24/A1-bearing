
import csv
import hashlib
from pathlib import Path

import pandas as pd

files = [
    "T06_A1_cwru.csv",
    "T06_A1_paderborn.csv",
]
partitions = ["train", "val", "test", "excluded"]

summary_rows = []

for filename in files:
    path = Path(filename)
    data = pd.read_csv(path, keep_default_na=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    for partition in partitions:
        subset = data[data["partition"] == partition]
        summary_rows.append({
            "file": filename,
            "partition": partition,
            "n_samples": len(subset),
            "n_units": subset["unit_id"].nunique(),
            "sha256": digest,
        })

with open(
    "T06_A1_split_summary.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file", "partition", "n_samples", "n_units", "sha256"
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("Created T06_A1_split_summary.csv")
print(pd.DataFrame(summary_rows).to_string(index=False))

