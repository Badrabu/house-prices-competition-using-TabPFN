"""Local, fully reproducible House Prices baseline + Kaggle-format submission.

Competition metric (official):
  RMSLE = sqrt(mean((log(pred) - log(true))^2))
  https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

This script:
  1) loads data/train.csv + data/test.csv
  2) reports 5-fold CV RMSLE (local estimate of leaderboard score)
  3) fits on all train rows and writes Id,SalePrice submission CSV

The TabPFN submission shipped in this repo was produced via the Prior Labs
TabPFN MCP server (see src/run_tabpfn_pipeline.py). Use this script when you
want a self-contained sklearn run with no external MCP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 1.0, None)
    y_true = np.clip(y_true, 1.0, None)
    return float(np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred))))


def build_xy(train: pd.DataFrame, test: pd.DataFrame):
    feature_cols = [c for c in train.columns if c not in ("Id", "SalePrice")]
    x_train = train[feature_cols].copy()
    y_train = train["SalePrice"].astype(float).to_numpy()
    x_test = test[feature_cols].copy()
    ids = test["Id"].to_numpy()

    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(x_train[c])]
    categorical = [c for c in feature_cols if c not in numeric]

    # Ordinal for tree models (HistGradientBoosting supports native categorical
    # via ordinal encoding + categorical_features mask).
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return x_train, y_train, x_test, ids, preprocess, len(numeric)


def make_model(n_num: int, n_features_after: int | None = None) -> HistGradientBoostingRegressor:
    # categorical feature indices come after numeric block in ColumnTransformer
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=500,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
    )


def cross_val_rmsle(x, y, preprocess, n_splits: int = 5) -> tuple[float, list[float]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores: list[float] = []
    for fold, (tr, va) in enumerate(kf.split(x), start=1):
        pipe = Pipeline(
            steps=[
                ("prep", preprocess),
                ("model", make_model(0)),
            ]
        )
        # Train on log target to align with RMSLE
        pipe.fit(x.iloc[tr], np.log1p(y[tr]))
        pred = np.expm1(pipe.predict(x.iloc[va]))
        score = rmsle(y[va], pred)
        scores.append(score)
        print(f"fold {fold}: RMSLE={score:.5f}")
    return float(np.mean(scores)), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "local-baseline-submission.csv",
    )
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    x_train, y_train, x_test, ids, preprocess, n_num = build_xy(train, test)

    mean_cv = None
    if not args.skip_cv:
        print("=== 5-fold CV (log-target HistGradientBoosting) ===")
        mean_cv, scores = cross_val_rmsle(x_train, y_train, preprocess)
        print(f"CV RMSLE mean={mean_cv:.5f}  std={np.std(scores):.5f}")
        print("Lower is better. Sample-submission ~0.41; strong public ~0.11–0.12.")

    pipe = Pipeline(steps=[("prep", preprocess), ("model", make_model(n_num))])
    pipe.fit(x_train, np.log1p(y_train))
    preds = np.clip(np.expm1(pipe.predict(x_test)), 1.0, None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"Id": ids, "SalePrice": preds})
    sub.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(sub)} rows)")
    if mean_cv is not None:
        metrics_path = args.out.with_suffix(".metrics.txt")
        metrics_path.write_text(
            f"model=HistGradientBoostingRegressor(log1p target)\n"
            f"cv_folds=5\n"
            f"cv_rmsle_mean={mean_cv:.6f}\n"
            f"metric=RMSLE (Kaggle House Prices official)\n"
        )
        print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
