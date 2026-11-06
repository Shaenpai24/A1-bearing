from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("data/processed/paderborn")
EDA_DIR = ROOT / "eda"
REPORT = ROOT / "eda_report.txt"

EDA_DIR.mkdir(parents=True, exist_ok=True)

npz_files = sorted(ROOT.rglob("*.npz"))

if not npz_files:
    raise SystemExit("ERROR: No NPZ files found.")

records = []
region_values = {}
quality_issues = []

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def crest_factor(x):
    r = rms(x)
    return float(np.max(np.abs(x)) / r) if r > 0 else np.nan


def spectral_features(x, fs):
    x = x - np.mean(x)

    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)

    if len(spectrum) > 1:
        spectrum[0] = 0

    dominant_idx = int(np.argmax(spectrum))

    power = spectrum ** 2
    total_power = float(np.sum(power))

    return {
        "dominant_frequency": float(freqs[dominant_idx]),
        "spectral_energy": total_power,
    }


# ----------------------------------------------------------------------
# Load every NPZ
# ----------------------------------------------------------------------

for file_index, path in enumerate(npz_files, 1):

    with np.load(path, allow_pickle=True) as data:

        X = np.asarray(data["X"], dtype=np.float64)

        bearing = str(data["bearing_id"].item())
        label = str(data["label"].item())
        split = str(data["split"].item())
        measurement_id = int(data["measurement_id"].item())
        source_file = str(data["source_file"].item())
        source_fs = int(data["source_sampling_rate"].item())
        fs = int(data["sampling_rate"].item())
        window_size = int(data["window_size"].item())

        # --------------------------------------------------------------
        # Region metadata
        # --------------------------------------------------------------

        if "region" in data.files:
            regions = data["region"]

            for region in regions:
                region = str(region)
                region_values[region] = region_values.get(region, 0) + 1

        # --------------------------------------------------------------
        # Window statistics
        # --------------------------------------------------------------

        for window_index, x in enumerate(X):

            x = np.asarray(x, dtype=np.float64)

            finite = np.isfinite(x)

            if not finite.all():
                quality_issues.append(
                    f"{path}: window {window_index} contains "
                    f"{np.size(x) - np.sum(finite)} non-finite samples"
                )
                continue

            stats = {
                "bearing_id": bearing,
                "label": label,
                "split": split,
                "measurement_id": measurement_id,
                "source_file": source_file,
                "window_index": window_index,
                "samples": len(x),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "rms": rms(x),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "peak_to_peak": float(np.ptp(x)),
                "crest_factor": crest_factor(x),
            }

            stats.update(spectral_features(x, fs))

            records.append(stats)


df = pd.DataFrame(records)

# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

with REPORT.open("w", encoding="utf-8") as out:

    def p(text=""):
        out.write(str(text) + "\n")

    p("=" * 80)
    p("PADERBORN A1 EXPLORATORY DATA ANALYSIS")
    p("=" * 80)

    p(f"NPZ files: {len(npz_files):,}")
    p(f"Windows:   {len(df):,}")

    p()
    p("DATASET ROOT")
    p(str(ROOT.resolve()))

    # ------------------------------------------------------------------
    # Bearings
    # ------------------------------------------------------------------

    p()
    p("=" * 80)
    p("1. BEARING / CLASS DISTRIBUTION")
    p("=" * 80)

    p("\nBearings:")
    for bearing in sorted(df["bearing_id"].unique()):
        sub = df[df["bearing_id"] == bearing]

        p(
            f"{bearing:5s} "
            f"{str(sub['label'].iloc[0]):18s} "
            f"{len(sub):5d} windows"
        )

    p("\nLabels:")
    p(df["label"].value_counts().to_string())

    p("\nSplits:")
    p(df["split"].value_counts().to_string())

    # ------------------------------------------------------------------
    # Numerical summary
    # ------------------------------------------------------------------

    p()
    p("=" * 80)
    p("2. WINDOW-LEVEL FEATURE SUMMARY")
    p("=" * 80)

    feature_cols = [
        "mean",
        "std",
        "rms",
        "peak_to_peak",
        "crest_factor",
        "dominant_frequency",
        "spectral_energy",
    ]

    p(
        df[feature_cols]
        .describe()
        .round(6)
        .to_string()
    )

    # ------------------------------------------------------------------
    # By label
    # ------------------------------------------------------------------

    p()
    p("=" * 80)
    p("3. FEATURES BY LABEL")
    p("=" * 80)

    grouped = (
        df.groupby("label")[feature_cols]
        .agg(["mean", "std"])
        .round(6)
    )

    p(grouped.to_string())

    # ------------------------------------------------------------------
    # By bearing
    # ------------------------------------------------------------------

    p()
    p("=" * 80)
    p("4. FEATURES BY BEARING")
    p("=" * 80)

    bearing_summary = (
        df.groupby(["bearing_id", "label"])[
            ["std", "rms", "peak_to_peak", "crest_factor",
             "dominant_frequency", "spectral_energy"]
        ]
        .agg(["mean", "std"])
        .round(6)
    )

    p(bearing_summary.to_string())

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    p()
    p("=" * 80)
    p("5. SAMPLING CONFIGURATION")
    p("=" * 80)

    p(f"Processed sampling rates: {sorted(df['samples'].unique())}")
    p("Expected window size: 2048 samples")
    p("Processed sampling rate: 16000 Hz")

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    p()
    p("=" * 80)
    p("6. SIGNAL QUALITY")
    p("=" * 80)

    p(f"Non-finite windows: {len(quality_issues)}")

    if quality_issues:
        for issue in quality_issues[:50]:
            p(issue)

    p("\nConstant windows:")
    constant = df[df["std"] == 0]
    p(str(len(constant)))

    # ------------------------------------------------------------------
    # Region metadata
    # ------------------------------------------------------------------

    p()
    p("=" * 80)
    p("7. REGION METADATA")
    p("=" * 80)

    if region_values:
        for region, count in sorted(region_values.items()):
            p(f"{region}: {count:,}")

    # ------------------------------------------------------------------
    # Split x label
    # ------------------------------------------------------------------

    p()
    p("=" * 80)
    p("8. SPLIT × LABEL")
    p("=" * 80)

    p(
        pd.crosstab(
            df["split"],
            df["label"]
        ).to_string()
    )

    # ------------------------------------------------------------------
    # Split x bearing
    # ------------------------------------------------------------------

    p()
    p("=" * 80)
    p("9. SPLIT × BEARING")
    p("=" * 80)

    p(
        pd.crosstab(
            df["split"],
            df["bearing_id"]
        ).to_string()
    )

    p()
    p("=" * 80)
    p("END OF EDA REPORT")
    p("=" * 80)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

# 1. RMS by label
plt.figure(figsize=(10, 6))
df.boxplot(column="rms", by="label", grid=False)
plt.title("RMS by Bearing Class")
plt.suptitle("")
plt.ylabel("RMS")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(EDA_DIR / "rms_by_label.png", dpi=160)
plt.close()

# 2. Standard deviation by label
plt.figure(figsize=(10, 6))
df.boxplot(column="std", by="label", grid=False)
plt.title("Standard Deviation by Bearing Class")
plt.suptitle("")
plt.ylabel("Standard deviation")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(EDA_DIR / "std_by_label.png", dpi=160)
plt.close()

# 3. Crest factor
plt.figure(figsize=(10, 6))
df.boxplot(column="crest_factor", by="label", grid=False)
plt.title("Crest Factor by Bearing Class")
plt.suptitle("")
plt.ylabel("Crest factor")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(EDA_DIR / "crest_factor_by_label.png", dpi=160)
plt.close()

# 4. Dominant frequency
plt.figure(figsize=(10, 6))
df.boxplot(column="dominant_frequency", by="label", grid=False)
plt.title("Dominant Frequency by Bearing Class")
plt.suptitle("")
plt.ylabel("Frequency (Hz)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(EDA_DIR / "dominant_frequency_by_label.png", dpi=160)
plt.close()

# ----------------------------------------------------------------------
# Representative waveforms
# ----------------------------------------------------------------------

representatives = (
    df.groupby("label", sort=True)
    .head(1)
)

plt.figure(figsize=(12, 7))

for label in sorted(representatives["label"].unique()):

    row = representatives[
        representatives["label"] == label
    ].iloc[0]

    matching = npz_files

    target = None

    for candidate in matching:
        if candidate.name == (
            Path(row["source_file"]).stem + ".npz"
        ):
            target = candidate
            break

    if target is None:
        continue

    with np.load(target, allow_pickle=True) as data:
        X = data["X"]

        x = X[int(row["window_index"])]

        t = np.arange(len(x)) / 16000

        plt.plot(
            t,
            x,
            label=label,
            alpha=0.8
        )

plt.title("Representative Vibration Windows")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(
    EDA_DIR / "representative_waveforms.png",
    dpi=160
)
plt.close()

print("=" * 80)
print("EDA COMPLETE")
print("=" * 80)
print(f"NPZ files: {len(npz_files):,}")
print(f"Windows:   {len(df):,}")
print(f"Report:    {REPORT}")
print(f"Plots:     {EDA_DIR}")
print("=" * 80)
