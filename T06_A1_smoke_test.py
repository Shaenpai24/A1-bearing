#!/usr/bin/env python3
"""
Team 06 / Card A1 bounded smoke test.

Run from the repository root:
    ./venv/bin/python3.13 T06_A1_smoke_test.py

The script uses a small sample from the EXISTING processed split arrays. It does
not alter preprocessing, regenerate split manifests, fit full models, or score
the test partition. Test arrays are used only for forward-pass inference; test
labels are never loaded.

It checks the required model/baseline/fusion code paths with one optimizer
update per neural model, validation forward passes, and test-shaped inference.
This is a smoke test, not a substitute for the full experiments.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
LOG = ROOT / "T06_A1_smoke_log.txt"
MAX_SECONDS = 15 * 60
BATCH = 8
SEED = 22

lines: list[str] = []
results: list[tuple[str, str, float, str]] = []
started_all = time.perf_counter()


def emit(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def timed(name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn()
        elapsed = time.perf_counter() - t0
        results.append((name, "PASS", elapsed, str(detail)))
        emit(f"[PASS] {name}: {elapsed:.2f}s — {detail}")
        return detail
    except Exception as e:
        elapsed = time.perf_counter() - t0
        results.append((name, "FAIL", elapsed, f"{type(e).__name__}: {e}"))
        emit(f"[FAIL] {name}: {elapsed:.2f}s — {type(e).__name__}: {e}")
        emit(traceback.format_exc())
        return None


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required project module missing: {path.relative_to(ROOT)}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def first_npz(folder: Path):
    files = sorted(folder.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No processed .npz files found under {folder}")
    return files[0]


def sample_cwru(load_hp: int):
    import numpy as np

    folder = ROOT / "data/processed/cwru"
    xs, ys, vxs, vys = [], [], [], []
    test_x = None
    labels = {"healthy": 0, "inner": 1, "outer": 2, "ball": 3}

    for f in sorted(folder.glob("*.npz")):
        with np.load(f, allow_pickle=True) as d:
            if int(d["load_hp"]) != load_hp:
                continue

            label = str(d["label"])
            if label not in labels:
                raise ValueError(f"Unknown CWRU label {label} in {f}")

            a = np.asarray(d["train"], dtype=np.float32)
            b = np.asarray(d["validation"], dtype=np.float32)
            c = np.asarray(d["test"], dtype=np.float32)

            if len(a):
                xs.append(a[:BATCH])
                ys.extend([labels[label]] * min(len(a), BATCH))

            if len(b):
                vxs.append(b[:BATCH])
                vys.extend([labels[label]] * min(len(b), BATCH))

            if test_x is None and len(c):
                test_x = c[:BATCH]

    if not xs or not vxs or test_x is None:
        raise RuntimeError(
            f"CWRU load {load_hp}: train/validation/test arrays missing"
        )

    return (
        torch_np_concat(xs),
        torch_np(ys, "int64"),
        torch_np_concat(vxs),
        torch_np(vys, "int64"),
        test_x,
    )

def torch_np_concat(parts):
    import numpy as np
    return np.concatenate(parts, axis=0).astype(np.float32)


def torch_np(values, dtype):
    import numpy as np
    return np.asarray(values, dtype=dtype)

def sample_paderborn():
    import numpy as np

    root = ROOT / "data/processed/paderborn"

    def collect(split):
        files = sorted((root / split).glob("*/*.npz"))

        if not files:
            raise FileNotFoundError(
                f"No Paderborn {split} arrays under {root / split}"
            )

        xx, yy = [], []

        label_map = {
            "healthy": 0,
            "outer_artificial": 1,
            "outer_natural": 1,
            "inner_artificial": 2,
            "inner_natural": 2,
        }

        for f in files:
            with np.load(f, allow_pickle=True) as d:
                raw_label = str(
                    d["label"].item()
                    if d["label"].ndim == 0
                    else d["label"]
                )

                if raw_label not in label_map:
                    raise ValueError(
                        f"Unknown Paderborn label {raw_label} in {f}"
                    )

                x = d["X"]

                if len(x):
                    xx.append(x[:BATCH].astype("float32"))
                    yy.extend(
                        [label_map[raw_label]] * min(len(x), BATCH)
                    )

        if not xx:
            raise RuntimeError(f"No Paderborn samples in {split}")

        return torch_np_concat(xx), torch_np(yy, "int64")

    train_x, train_y = collect("train")
    val_x, val_y = collect("validation")

    # Test signals only. Do not load test labels.
    test_files = sorted((root / "test").glob("*/*.npz"))

    if not test_files:
        raise FileNotFoundError(
            f"No Paderborn test arrays under {root / 'test'}"
        )

    with np.load(test_files[0], allow_pickle=True) as d:
        test_x = d["X"][:BATCH].astype("float32")

    return (
        train_x[:BATCH],
        train_y[:BATCH],
        val_x[:BATCH],
        val_y[:BATCH],
        test_x,
    )

def np_load(path):
    import numpy as np
    return np.load(path, allow_pickle=True)


def one_update(model, x_train, y_train, x_val, y_val, x_test, *, spectrogram=False):
    import numpy as np
    import torch
    from torch import nn

    if spectrogram:
        x_train = model_module.make_spectrograms(x_train)
        x_val = model_module.make_spectrograms(x_val)
        x_test = model_module.make_spectrograms(x_test)
    else:
        x_train = torch.from_numpy(np.asarray(x_train, dtype=np.float32)[:, None, :])
        x_val = torch.from_numpy(np.asarray(x_val, dtype=np.float32)[:, None, :])
        x_test = torch.from_numpy(np.asarray(x_test, dtype=np.float32)[:, None, :])
    y_train = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    y_val = torch.from_numpy(np.asarray(y_val, dtype=np.int64))
    # Train on one batch; this checks gradients and optimizer update.
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    opt.zero_grad(set_to_none=True)
    logits = model(x_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    opt.step()
    # Validation: labels are permitted here.
    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        val_loss = float(loss_fn(val_logits, y_val).item())
        # Test-shaped inference: no test labels are loaded or scored.
        test_logits = model(x_test)
    if not torch.isfinite(test_logits).all():
        raise RuntimeError("Non-finite test-shaped inference output")
    return {
        "train_loss": float(loss.item()),
        "validation_loss": val_loss,
        "validation_samples": int(len(y_val)),
        "unscored_inference_samples": int(len(x_test)),
        "test_labels_loaded": False,
    }


def main():
    global model_module
    emit("TEAM 06 — CARD A1 CHECKPOINT SMOKE TEST")
    emit(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    emit(f"Python: {platform.python_version()} | Platform: {platform.platform()}")
    emit(f"Repository: {ROOT}")
    emit("Policy: validation only; test partition is inference-only; no test labels/metrics.")
    emit("")

    def prereq():
        import numpy, scipy, sklearn, torch
        emit(f"NumPy {numpy.__version__}; SciPy {scipy.__version__}; scikit-learn {sklearn.__version__}; PyTorch {torch.__version__}")
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        numpy.random.seed(SEED)
        return "scientific stack imports and CPU torch initialization passed"
    timed("environment", prereq)

    def split_manifests():
        required = ["T06_A1_cwru.csv", "T06_A1_paderborn.csv", "T06_A1_split_summary.csv"]
        for name in required:
            path = ROOT / name
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty split manifest: {name}")
        with (ROOT / "T06_A1_cwru.csv").open(newline="") as f:
            cwru_rows = sum(1 for _ in csv.DictReader(f))
        with (ROOT / "T06_A1_paderborn.csv").open(newline="") as f:
            pb_rows = sum(1 for _ in csv.DictReader(f))
        return f"manifests readable; CWRU rows={cwru_rows}, Paderborn rows={pb_rows}"
    timed("split manifests", split_manifests)

    def raw_data_presence():
        cwru = list((ROOT / "data/raw/cwru").glob("*.mat"))
        pb_archives = list((ROOT / "data/raw/paderborn").glob("*.rar"))
        pb_extracted = ROOT / "data/raw/paderborn/extracted"
        if len(cwru) < 16:
            raise RuntimeError(f"CWRU raw files: expected at least 16 .mat, found {len(cwru)}")
        if len(pb_archives) < 1 or not pb_extracted.exists():
            raise RuntimeError("Paderborn .rar archives or extracted/ directory missing")
        return f"CWRU MAT files={len(cwru)}; Paderborn RAR archives={len(pb_archives)}; extracted directory present"
    timed("raw dataset inventory", raw_data_presence)

    def cwru_arrays():
        data = sample_cwru(0)
        return f"source load 0 hp arrays: train={len(data[0])}, val={len(data[2])}, test-inference={len(data[4])}"
    cwru_data = timed("CWRU processed split arrays", cwru_arrays)

    def paderborn_arrays():
        data = sample_paderborn()
        return f"bearing-disjoint arrays available; smoke subset train={len(data[0])}, val={len(data[2])}, test-inference={len(data[4])}"
    pb_data = timed("Paderborn processed split arrays", paderborn_arrays)

    # CWRU envelope baseline: use the project's actual envelope feature function
    # and classifier, on a bounded sample from the approved source-load split.
    def cwru_baseline():
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        mod = load_module("a1_cwru_baseline_smoke", ROOT / "scripts/cwru_envelope_baseline.py")
        x, y, vx, vy, _ = sample_cwru(0)
        fx = np.vstack([mod.envelope_features(w, 0) for w in x[:min(24, len(x))]])
        fy = y[:len(fx)]
        scaler = StandardScaler().fit(fx)
        clf = RandomForestClassifier(n_estimators=8, random_state=SEED, n_jobs=1).fit(scaler.transform(fx), fy)
        fv = np.vstack([mod.envelope_features(w, 0) for w in vx[:min(8, len(vx))]])
        pred = clf.predict(scaler.transform(fv))
        return f"feature shape={fx.shape}; validation predictions={len(pred)}; no test labels accessed"
    timed("CWRU envelope baseline", cwru_baseline)

    # Paderborn envelope baseline: actual envelope feature function and a small
    # train-only scaler/classifier fit. This checks the feature/classifier path.
    def paderborn_baseline():
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        mod = load_module("a1_pb_baseline_smoke", ROOT / "scripts/paderborn_envelope_baseline.py")
        x, y, vx, vy, _ = sample_paderborn()
        fx = np.vstack([mod.envelope_features(w) for w in x])
        scaler = StandardScaler().fit(fx)
        clf = RandomForestClassifier(n_estimators=8, random_state=SEED, n_jobs=1).fit(scaler.transform(fx), y)
        fv = np.vstack([mod.envelope_features(w) for w in vx])
        pred = clf.predict(scaler.transform(fv))
        return f"feature shape={fx.shape}; validation predictions={len(pred)}; no test labels accessed"
    timed("Paderborn envelope baseline", paderborn_baseline)

    # Paderborn 1-D raw CNN.
    def paderborn_1d():
        import torch
        mod = load_module("a1_pb_1d_smoke", ROOT / "scripts/train_paderborn_1d_week1.py")
        x, y, vx, vy, tx = sample_paderborn()
        model = mod.Small1DCNN(num_classes=3)
        return one_update(model, x, y, vx, vy, tx)
    timed("Paderborn 1-D CNN", paderborn_1d)

    # CWRU 1-D raw CNN from the added A1 implementation.
    def cwru_1d():
        mod = load_module(
            "a1_cwru_1d_smoke",
            ROOT / "scripts/train_paderborn_1d_week1.py"
        )

        x, y, vx, vy, tx = sample_cwru(0)

        # Reuse the existing 1-D CNN architecture,
        # but configure it for CWRU's four classes.
        model = mod.Small1DCNN(num_classes=4)

        return one_update(model, x, y, vx, vy, tx)

    # CWRU 2-D spectrogram CNN: both ablation and attention variants.
    def cwru_2d(use_attention):
        global model_module
        import torch
        model_module = load_module("a1_cwru_2d_smoke", ROOT / "scripts/train_cwru_spectrogram_week1.py")
        x, y, vx, vy, tx = sample_cwru(0)
        model = model_module.Small2DCNN(use_attention=use_attention)
        return one_update(model, x, y, vx, vy, tx, spectrogram=True)
    timed("CWRU 2-D CNN no-attention ablation", lambda: cwru_2d(False))
    timed("CWRU 2-D CNN frequency-attention", lambda: cwru_2d(True))

    # Fusion: use validation logits only to select alpha, then fuse validation
    # logits. Test arrays are never supplied to the weight-selection function.
    def fusion_smoke():
        import numpy as np
        import torch
        fusion = load_module("a1_fusion_smoke", ROOT / "src/a1_fusion.py")
        x, y, vx, vy, _ = sample_cwru(0)
        # Independently initialized models are sufficient for an interface smoke.
        m1 = load_module(
            "paderborn_1d_for_fusion",
            ROOT / "scripts/train_paderborn_1d_week1.py",
        )
        m2 = load_module("a1_cwru_2d_fusion_smoke", ROOT / "scripts/train_cwru_spectrogram_week1.py")
        a = m1.Small1DCNN(num_classes=4).cpu()
        b = m2.Small2DCNN(use_attention=True).eval()
        with torch.no_grad():
            l1 = a(torch.from_numpy(vx[:, None, :]))
            sx = m2.make_spectrograms(vx)
            l2 = b(sx)
        selected = fusion.choose_validation_weight(l1.numpy(), l2.numpy(), vy)
        alpha = selected["alpha_1d"]
        fused = fusion.fuse_logits(l1.numpy(), l2.numpy(), alpha)
        if fused.shape != tuple(l1.shape) or not np.isfinite(fused).all():
            raise RuntimeError("Fusion returned invalid probabilities")
        return (f"validation-selected alpha_1d={alpha:.3f}; "
                f"validation macro-F1={selected['validation_macro_f1']:.4f}; "
                f"output={fused.shape}")
    timed("1-D + 2-D validation-weighted fusion", fusion_smoke)

    elapsed = time.perf_counter() - started_all
    emit("")
    emit("COMPONENT SUMMARY")
    for name, status, seconds, detail in results:
        emit(f"{status:4s} | {seconds:8.2f}s | {name} | {detail}")
    emit(f"TOTAL ELAPSED: {elapsed:.2f}s")
    emit("TEST PARTITION: no labels loaded; inference-only forwards only.")
    emit("NOTE: this is a bounded smoke test, not a scored test evaluation or full experiment.")
    if elapsed > MAX_SECONDS:
        emit(f"[FAIL] Time budget exceeded: {MAX_SECONDS}s")
    failed = [r for r in results if r[1] != "PASS"]
    if elapsed > MAX_SECONDS:
        failed.append(("15-minute budget", "FAIL", elapsed, "time budget exceeded"))
    emit(f"OVERALL: {'FAIL' if failed else 'PASS'} ({len(results)-len(failed)} passed, {len(failed)} failed)")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
