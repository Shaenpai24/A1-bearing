
import json
import math
import pandas as pd
from pathlib import Path

L = 2048

with open("cwru_recordings.json", encoding="utf-8") as f:
    recordings = json.load(f)

df = pd.read_csv("T06_A1_cwru.csv", keep_default_na=False)

expected = {}
for rec in recordings:
    filename = rec["file"]
    unit = rec["label"]
    n = int(rec["n_12k"])

    a = math.floor(0.6 * n)
    b = math.floor(0.8 * n)

    # Two complete windows in each 4096-sample guard gap.
    starts = [a - L, a, b - L, b]

    for start in starts:
        if start >= 0 and start + L <= n:
            expected[f"{filename}:{start}"] = unit

excluded = df[df["partition"] == "excluded"].copy()
actual = dict(zip(excluded["sample_id"], excluded["unit_id"]))

missing = sorted(set(expected) - set(actual))
unexpected = sorted(set(actual) - set(expected))
wrong_units = sorted(
    (sample_id, expected[sample_id], actual[sample_id])
    for sample_id in set(expected) & set(actual)
    if expected[sample_id] != actual[sample_id]
)

bad_reasons = excluded[excluded["reason"] != "guard"]
bad_labels = excluded[excluded["label"] == ""]

print("Expected guard windows:", len(expected))
print("Actual excluded rows:", len(excluded))
print("Missing guard windows:", missing)
print("Unexpected exclusions:", unexpected)
print("Wrong unit IDs:", wrong_units)
print("Rows with wrong exclusion reason:", len(bad_reasons))
print("Rows with empty labels:", len(bad_labels))

if not any([missing, unexpected, wrong_units, len(bad_reasons), len(bad_labels)]):
    print("\nPASS: all excluded rows match the expected guard offsets.")
else:
    print("\nFAIL: inspect the discrepancies above.")

