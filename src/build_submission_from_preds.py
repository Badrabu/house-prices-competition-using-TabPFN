"""Join TabPFN prediction JSON with test Ids into a Kaggle submission CSV.

Usage:
  python src/build_submission_from_preds.py \
    --predictions artifacts/tabpfn/predictions.json \
    --ids artifacts/tabpfn/ids.csv \
    --out house-prices-submission.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_predictions(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "mean" in raw:
        arr = np.asarray(raw["mean"], dtype=float)
    elif isinstance(raw, list):
        arr = np.asarray(raw, dtype=float)
    else:
        raise ValueError(f"Unrecognized prediction format in {path}")
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D predictions, got shape {arr.shape}")
    return np.clip(arr, 1.0, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ids = pd.read_csv(args.ids)
    preds = load_predictions(args.predictions)
    if len(preds) != len(ids):
        raise SystemExit(f"Row mismatch: {len(preds)} preds vs {len(ids)} ids")

    sub = pd.DataFrame({"Id": ids["Id"], "SalePrice": preds})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(sub)} rows)")
    print(sub["SalePrice"].describe())


if __name__ == "__main__":
    main()
