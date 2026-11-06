from scipy.io import loadmat
from pathlib import Path
import csv

ROOT = Path("data/raw/paderborn/extracted")
OUTPUT = Path("data/raw/paderborn/paderborn_metadata.csv")

mat_files = sorted(ROOT.glob("*/*.mat"))

rows = []

for path in mat_files:

    bearing_id = path.parent.name

    data = loadmat(
        path,
        squeeze_me=True,
        struct_as_record=False
    )

    root = data[path.stem]

    vibration = None
    vibration_raster = None
    vibration_samples = None

    if hasattr(root, "Y"):

        for signal in root.Y:

            name = getattr(signal, "Name", None)

            if name == "vibration_1":

                vibration = signal.Data
                vibration_raster = getattr(signal, "Raster", None)
                vibration_samples = vibration.size
                break

    # --------------------------------------------------------------
    # Extract measurement metadata
    # --------------------------------------------------------------

    measurement_id = None
    start_timestamp = None
    stop_timestamp = None

    if hasattr(root, "Info"):
        measurement_id = getattr(
            root.Info,
            "MeasurementID",
            None
        )

    if hasattr(root, "Description"):
        if hasattr(root.Description, "Measurement"):

            measurement = root.Description.Measurement

            start_timestamp = getattr(
                measurement,
                "StartTimestamp",
                None
            )

            stop_timestamp = getattr(
                measurement,
                "StopTimestamp",
                None
            )

    rows.append({
        "bearing_id": bearing_id,
        "file": path.name,
        "measurement_id": measurement_id,
        "vibration_samples": vibration_samples,
        "vibration_raster": vibration_raster,
        "start_timestamp": start_timestamp,
        "stop_timestamp": stop_timestamp,
    })


# --------------------------------------------------------------
# Write CSV
# --------------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

fieldnames = [
    "bearing_id",
    "file",
    "measurement_id",
    "vibration_samples",
    "vibration_raster",
    "start_timestamp",
    "stop_timestamp",
]

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


print("=" * 60)
print("PADERBORN METADATA EXTRACTION")
print("=" * 60)
print(f"MAT files: {len(mat_files)}")
print(f"Rows written: {len(rows)}")
print(f"Output: {OUTPUT}")
print("=" * 60)

