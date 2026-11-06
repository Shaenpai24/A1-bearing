from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/cwru"
RUNS = ROOT / "runs/cwru_1d_core"
OUT = RUNS / "crossload_inference"
OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(1)


class Small1DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))


def load_target_windows(target_load):
    parts = []
    for path in sorted(DATA.glob("*.npz")):
        with np.load(path, allow_pickle=True) as d:
            if int(np.asarray(d["load_hp"]).item()) != target_load:
                continue
            x = np.asarray(d["test"], dtype=np.float32)
            if len(x):
                parts.append(x)

    if not parts:
        raise RuntimeError(f"No test windows for target load {target_load} hp")

    return np.concatenate(parts, axis=0)


def main():
    rows = []
    started = time.perf_counter()

    for source in range(4):
        checkpoint = RUNS / f"source_{source}hp" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        model = Small1DCNN()
        checkpoint_data = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True
        )

        state = checkpoint_data["model_state_dict"]
        model.load_state_dict(state)
        model.eval()

        for target in range(4):
            x = load_target_windows(target)
            tensor = torch.from_numpy(x[:, None, :])

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(tensor)
                predictions = logits.argmax(dim=1).numpy()
            elapsed = time.perf_counter() - t0

            if not np.isfinite(logits.numpy()).all():
                raise RuntimeError("Non-finite inference output")

            # Predictions only: deliberately no target labels or metrics.
            pred_path = OUT / f"source_{source}hp_target_{target}hp_predictions.npy"
            np.save(pred_path, predictions)

            rows.append({
                "source_load_hp": source,
                "target_load_hp": target,
                "windows": len(x),
                "inference_seconds": elapsed,
                "predictions_file": str(pred_path.relative_to(ROOT)),
                "test_scored": "no",
            })

            print(
                f"source={source} hp -> target={target} hp | "
                f"windows={len(x)} | inference={elapsed:.3f}s | "
                "metrics=NOT CALCULATED"
            )

    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_target_combinations": len(rows),
        "total_windows": sum(r["windows"] for r in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "test_labels_loaded": False,
        "test_metrics_calculated": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nCOMPLETE:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
