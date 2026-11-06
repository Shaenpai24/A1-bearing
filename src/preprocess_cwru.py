from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly


RAW_DIR = Path("data/raw/cwru")
OUT_DIR = Path("data/processed/cwru")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# file_number: (label, load_hp, original_sampling_rate)
FILES = {
    # Healthy
    97:  ("healthy", 0, 48000),
    98:  ("healthy", 1, 48000),
    99:  ("healthy", 2, 48000),
    100: ("healthy", 3, 48000),

    # Inner race fault
    105: ("inner", 0, 12000),
    106: ("inner", 1, 12000),
    107: ("inner", 2, 12000),
    108: ("inner", 3, 12000),

    # Ball fault
    118: ("ball", 0, 12000),
    119: ("ball", 1, 12000),
    120: ("ball", 2, 12000),
    121: ("ball", 3, 12000),

    # Outer race fault at 6 o'clock
    130: ("outer", 0, 12000),
    131: ("outer", 1, 12000),
    132: ("outer", 2, 12000),
    133: ("outer", 3, 12000),
}

def get_de_signal(file_number: int) -> np.ndarray:
    """
    Load the required drive-end vibration signal from a CWRU .mat file.

    Special case:
    99.mat contains both X098_DE_time and X099_DE_time.
    D1 requires X099_DE_time.
    """

    path = RAW_DIR / f"{file_number}.mat"

    data = loadmat(path)

    expected_name = f"X{file_number:03d}_DE_time"

    if expected_name not in data:
        raise KeyError(
            f"{path} does not contain expected variable "
            f"{expected_name}"
        )

    signal = np.asarray(data[expected_name]).squeeze()

    if signal.ndim != 1:
        raise ValueError(
            f"{expected_name} has unexpected shape: {signal.shape}"
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            f"{expected_name} contains NaN or infinite values"
        )

    return signal.astype(np.float64)
TARGET_FS = 12000


def resample_if_needed(
    signal: np.ndarray,
    original_fs: int
) -> np.ndarray:

    if original_fs == TARGET_FS:
        return signal

    if original_fs == 48000 and TARGET_FS == 12000:
        # 48 kHz -> 12 kHz = 1/4
        signal = resample_poly(signal, up=1, down=4)
        return signal

    raise ValueError(
        f"Unsupported resampling: {original_fs} -> {TARGET_FS}"
    )
WINDOW = 2048
STRIDE = 2048
GUARD = 2048


def make_blocks(signal: np.ndarray):
    """
    Create the D1 guarded train/validation/test regions.

    a = floor(0.6N)
    b = floor(0.8N)

    train = [0, a-L)
    validation = [a+L, b-L)
    test = [b+L, N)

    L = 2048
    """

    n = len(signal)

    a = int(np.floor(0.6 * n))
    b = int(np.floor(0.8 * n))
    L = GUARD

    train = signal[0 : a - L]
    validation = signal[a + L : b - L]
    test = signal[b + L : n]

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }

def cut_windows(signal: np.ndarray) -> np.ndarray:
    """
    Cut non-overlapping 2048-sample windows.

    No window is allowed to cross a block boundary.
    """

    n_windows = len(signal) // WINDOW

    if n_windows == 0:
        return np.empty((0, WINDOW), dtype=np.float32)

    usable = n_windows * WINDOW

    windows = signal[:usable].reshape(
        n_windows,
        WINDOW
    )

    return windows.astype(np.float32)

def process_recording(file_number: int):
    label, load_hp, original_fs = FILES[file_number]

    print(f"\nProcessing {file_number}.mat")
    print(f"  Label: {label}")
    print(f"  Load: {load_hp} hp")
    print(f"  Original sampling rate: {original_fs} Hz")

    signal = get_de_signal(file_number)

    print(f"  Original samples: {len(signal):,}")

    signal = resample_if_needed(signal, original_fs)

    print(f"  Resampled samples: {len(signal):,}")
    print(f"  Final sampling rate: {TARGET_FS} Hz")

    blocks = make_blocks(signal)

    result = {}

    for partition, block in blocks.items():
        windows = cut_windows(block)

        result[partition] = windows

        print(
            f"  {partition:10s}: "
            f"{len(windows):4d} windows"
        )

    return {
        "file_number": file_number,
        "label": label,
        "load_hp": load_hp,
        "train": result["train"],
        "validation": result["validation"],
        "test": result["test"],
    }

def main():

    all_recordings = []

    for file_number in FILES:
        result = process_recording(file_number)
        all_recordings.append(result)

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    total_train = sum(
        len(r["train"]) for r in all_recordings
    )

    total_validation = sum(
        len(r["validation"]) for r in all_recordings
    )

    total_test = sum(
        len(r["test"]) for r in all_recordings
    )

    print(f"Train windows:      {total_train}")
    print(f"Validation windows: {total_validation}")
    print(f"Test windows:       {total_test}")

    # Save each recording separately for now.
    # This preserves the recording identity, which is important for A1.
    for r in all_recordings:

        file_number = r["file_number"]

        output = OUT_DIR / f"{file_number}.npz"

        np.savez_compressed(
            output,
            train=r["train"],
            validation=r["validation"],
            test=r["test"],
            label=r["label"],
            load_hp=r["load_hp"],
            file_number=file_number,
            sampling_rate=TARGET_FS,
        )

        print(f"Saved: {output}")


if __name__ == "__main__":
    main()


