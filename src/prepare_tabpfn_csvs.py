"""Split Kaggle House Prices train/test into TabPFN-ready CSVs.

Produces:
  artifacts/tabpfn/x_train.csv  — features only (no Id / SalePrice)
  artifacts/tabpfn/y_train.csv  — SalePrice column
  artifacts/tabpfn/x_test.csv   — same feature columns as x_train
  artifacts/tabpfn/ids.csv      — test Id column for submission join
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "artifacts" / "tabpfn"


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    feature_cols = [c for c in train.columns if c not in ("Id", "SalePrice")]
    missing = set(feature_cols) - set(test.columns)
    if missing:
        raise SystemExit(f"test.csv missing columns: {sorted(missing)}")

    OUT.mkdir(parents=True, exist_ok=True)
    train[feature_cols].to_csv(OUT / "x_train.csv", index=False)
    train[["SalePrice"]].to_csv(OUT / "y_train.csv", index=False)
    test[feature_cols].to_csv(OUT / "x_test.csv", index=False)
    test[["Id"]].to_csv(OUT / "ids.csv", index=False)

    print(f"Wrote {OUT}")
    print(f"x_train={train.shape[0]}x{len(feature_cols)}  x_test={test.shape[0]}x{len(feature_cols)}")


if __name__ == "__main__":
    main()
