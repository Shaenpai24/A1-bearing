
import csv
import json
import math
from pathlib import Path

L = 2048
metadata_path = Path("cwru_recordings.json")
output_path = Path("T06_A1_cwru.csv")

# The JSON is a list of recording metadata objects.
recordings = json.loads(metadata_path.read_text(encoding="utf-8"))

label_map = {
    "Normal": "healthy",
    "IR007": "inner",
    "B007": "ball",
    "OR007": "outer",
}

rows = []

for rec in recordings:
    filename = rec["file"]
    recording_id = rec["label"]
    n = int(rec["n_12k"])

    # Convert recording IDs such as Normal_2 and IR007_0
    # into the required class labels.
    if recording_id.startswith("Normal"):
        prefix = "Normal"
    elif recording_id.startswith("IR007"):
        prefix = "IR007"
    elif recording_id.startswith("B007"):
        prefix = "B007"
    elif recording_id.startswith("OR007"):
        prefix = "OR007"
    else:
        raise ValueError(f"Unknown recording label: {recording_id}")
    label = label_map[prefix]

    a = math.floor(0.6 * n)
    b = math.floor(0.8 * n)

    regions = {
        "train": (0, a - L),
        "val": (a + L, b - L),
        "test": (b + L, n),
    }

    for partition, (start, end) in regions.items():
        offset = start
        while offset + L <= end:
            rows.append({
                "sample_id": f"{filename}:{offset}",
                "unit_id": recording_id,
                "partition": partition,
                "label": label,
                "reason": "",
            })
            offset += L

    # Guard gaps: [a-L, a+L) and [b-L, b+L).
    # Include only complete, non-overlapping windows.
    for gap_start in (a - L, b - L):
        for offset in (gap_start, gap_start + L):
            if offset >= 0 and offset + L <= n:
                rows.append({
                    "sample_id": f"{filename}:{offset}",
                    "unit_id": recording_id,
                    "partition": "excluded",
                    "label": label,
                    "reason": "guard",
                })

columns = [
    "sample_id", "unit_id", "partition", "label", "reason"
]

# Check sample IDs are unique.
sample_ids = [row["sample_id"] for row in rows]
assert len(sample_ids) == len(set(sample_ids)), "Duplicate sample IDs!"

with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved: {output_path}")
for partition in ("train", "val", "test", "excluded"):
    count = sum(r["partition"] == partition for r in rows)
    print(f"{partition}: {count}")
