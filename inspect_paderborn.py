from scipy.io import loadmat
from pathlib import Path
import numpy as np

ROOT = Path("data/raw/paderborn/extracted")
OUTPUT = Path("data/raw/paderborn/paderborn_inspection.txt")

mat_files = sorted(ROOT.glob("*/*.mat"))

mat_files = sorted(ROOT.glob("*/*.mat"))

with OUTPUT.open("w", encoding="utf-8") as out:

    def p(text=""):
        out.write(str(text) + "\n")

    p("=" * 100)
    p("PADERBORN DATASET INSPECTION")
    p("=" * 100)
    p(f"Root directory: {ROOT.resolve()}")
    p(f"MAT files found: {len(mat_files)}")
    p()

    # ------------------------------------------------------------------
    # Global counters
    # ------------------------------------------------------------------

    vibration_found = 0
    vibration_missing = 0
    unexpected_root = 0
    load_errors = 0

    vibration_lengths = {}
    vibration_rasters = {}
    vibration_dtypes = {}

    signal_names = {}

    # ------------------------------------------------------------------
    # Inspect every file
    # ------------------------------------------------------------------

    for index, path in enumerate(mat_files, 1):

        p("=" * 100)
        p(f"FILE {index}/{len(mat_files)}")
        p("=" * 100)

        p(f"Path: {path}")
        p(f"Size: {path.stat().st_size:,} bytes")

        try:
            data = loadmat(
                path,
                squeeze_me=True,
                struct_as_record=False
            )
        except Exception as e:
            load_errors += 1
            p(f"LOAD ERROR: {type(e).__name__}: {e}")
            continue

        user_keys = [
            key for key in data.keys()
            if not key.startswith("__")
        ]

        p(f"MAT variables: {user_keys}")

        if len(user_keys) != 1:
            unexpected_root += 1
            p("WARNING: Expected one root structure.")

        root_name = path.stem

        if root_name not in data:
            p(f"WARNING: Root variable '{root_name}' not found.")
            continue

        root = data[root_name]

        p()
        p("--- ROOT STRUCTURE ---")
        p(f"Type: {type(root)}")
        p(f"Fields: {getattr(root, '_fieldnames', None)}")

        # ==============================================================
        # INFO
        # ==============================================================

        p()
        p("--- INFO ---")

        if hasattr(root, "Info"):
            info = root.Info

            for field in getattr(info, "_fieldnames", []):
                value = getattr(info, field)
                p(
                    f"{field}: "
                    f"type={type(value).__name__}, "
                    f"value={value}"
                )
        else:
            p("MISSING")

        # ==============================================================
        # X
        # ==============================================================

        p()
        p("--- X STRUCTURES ---")

        if hasattr(root, "X"):

            X = np.atleast_1d(root.X)

            p(f"Number of X structures: {len(X)}")

            for i, x in enumerate(X):

                p(f"X[{i}]")

                fields = getattr(x, "_fieldnames", [])

                for field in fields:

                    value = getattr(x, field)

                    if isinstance(value, np.ndarray):
                        p(
                            f"  {field}: "
                            f"type={type(value).__name__}, "
                            f"shape={value.shape}, "
                            f"dtype={value.dtype}"
                        )
                    else:
                        p(
                            f"  {field}: "
                            f"type={type(value).__name__}, "
                            f"value={value}"
                        )

        else:
            p("MISSING")

        # ==============================================================
        # Y
        # ==============================================================

        p()
        p("--- Y SIGNALS ---")

        vibration = None

        if hasattr(root, "Y"):

            Y = np.atleast_1d(root.Y)

            p(f"Number of Y signals: {len(Y)}")

            for i, y in enumerate(Y):

                name = getattr(y, "Name", "<NO NAME>")
                signal_names[name] = signal_names.get(name, 0) + 1

                values = getattr(y, "Data", None)

                p()
                p(f"Y[{i}]")
                p(f"  Name: {name}")
                p(f"  Type: {getattr(y, 'Type', None)}")
                p(f"  Raster: {getattr(y, 'Raster', None)}")
                p(f"  Device: {getattr(y, 'Device', None)}")
                p(f"  XIndex: {getattr(y, 'XIndex', None)}")
                p(f"  DownSampling: {getattr(y, 'DownSampling', None)}")
                p(f"  Path: {getattr(y, 'Path', None)}")
                p(f"  Flags: {getattr(y, 'Flags', None)}")

                if isinstance(values, np.ndarray):

                    p(f"  Data shape: {values.shape}")
                    p(f"  Data dtype: {values.dtype}")
                    p(f"  Data samples: {values.size}")

                    if values.size > 0 and np.issubdtype(
                        values.dtype, np.number
                    ):
                        finite = np.isfinite(values)

                        p(f"  Finite samples: {finite.sum()}")
                        p(f"  NaN samples: {np.isnan(values).sum()}")
                        p(f"  Inf samples: {np.isinf(values).sum()}")

                        if finite.any():
                            finite_values = values[finite]

                            p(f"  Min: {finite_values.min()}")
                            p(f"  Max: {finite_values.max()}")
                            p(f"  Mean: {finite_values.mean()}")
                            p(f"  Std: {finite_values.std()}")

                    # --------------------------------------------------
                    # VIBRATION
                    # --------------------------------------------------

                    if name == "vibration_1":

                        vibration_found += 1
                        vibration = values

                        length = values.size
                        raster = getattr(y, "Raster", None)
                        dtype = str(values.dtype)

                        vibration_lengths[length] = (
                            vibration_lengths.get(length, 0) + 1
                        )

                        vibration_rasters[raster] = (
                            vibration_rasters.get(raster, 0) + 1
                        )

                        vibration_dtypes[dtype] = (
                            vibration_dtypes.get(dtype, 0) + 1
                        )

        else:
            p("MISSING")

        if vibration is None:
            vibration_missing += 1
            p()
            p("!!! WARNING: vibration_1 NOT FOUND !!!")

        # ==============================================================
        # DESCRIPTION
        # ==============================================================

        p()
        p("--- DESCRIPTION ---")

        if hasattr(root, "Description"):

            desc = root.Description

            p(f"Fields: {getattr(desc, '_fieldnames', [])}")

            for section_name in getattr(
                desc, "_fieldnames", []
            ):

                section = getattr(desc, section_name)

                p()
                p(f"[{section_name}]")
                p(f"Fields: {getattr(section, '_fieldnames', [])}")

                for field in getattr(
                    section, "_fieldnames", []
                ):

                    value = getattr(section, field)

                    p(
                        f"  {field}: "
                        f"type={type(value).__name__}, "
                        f"value={value}"
                    )

        else:
            p("MISSING")

    # ==================================================================
    # SUMMARY
    # ==================================================================

    p()
    p("=" * 100)
    p("GLOBAL SUMMARY")
    p("=" * 100)

    p(f"MAT files found:        {len(mat_files)}")
    p(f"Successfully loaded:    {len(mat_files) - load_errors}")
    p(f"Load errors:            {load_errors}")
    p(f"vibration_1 found:      {vibration_found}")
    p(f"vibration_1 missing:    {vibration_missing}")
    p(f"Unexpected root files:  {unexpected_root}")

    p()
    p("--- VIBRATION LENGTHS ---")

    for length, count in sorted(vibration_lengths.items()):
        p(f"{length:,} samples : {count} files")

    p()
    p("--- VIBRATION RASTERS ---")

    for raster, count in sorted(
        vibration_rasters.items(),
        key=lambda x: str(x[0])
    ):
        p(f"{raster} : {count} files")

    p()
    p("--- VIBRATION DTYPES ---")

    for dtype, count in sorted(vibration_dtypes.items()):
        p(f"{dtype} : {count} files")

    p()
    p("--- ALL Y SIGNAL NAMES ---")

    for name, count in sorted(signal_names.items()):
        p(f"{name} : {count} occurrences")

    p()
    p("=" * 100)
    p("END OF REPORT")
    p("=" * 100)

print(f"Inspection complete.")
print(f"Files inspected: {len(mat_files)}")
print(f"Report: {OUTPUT}")

