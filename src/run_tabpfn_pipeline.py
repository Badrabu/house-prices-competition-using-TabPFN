"""Reproduce the TabPFN House Prices submission produced for this project.

This is the exact logical pipeline used to generate:
  /agent/house-prices-submission.csv
  house-prices-submission.csv

In the Cursor cloud agent that created the file, steps 2–5 were executed via
the Prior Labs TabPFN MCP tools (`prepare_train_set_upload`, `fit`,
`prepare_test_set_upload`, `predict`). The same flow is spelled out here so
you can re-run it from shell + MCP, or adapt it to the `tabpfn` Python API.

Competition: House Prices - Advanced Regression Techniques
URL: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
Metric: RMSLE on SalePrice (RMSE of log predictions vs log truth)
Train rows: 1460  |  Test rows: 1459  |  Features used: 79 (all except Id)
Model: TabPFN v3.5 regression (package_version 9.2.0, n_estimators=8)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Local helpers in this repo
from build_submission_from_preds import load_predictions  # noqa: E402


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    """Prepare CSVs, then print the MCP / curl steps used for the submission."""
    run([sys.executable, str(ROOT / "src" / "prepare_tabpfn_csvs.py")])

    x_train = ROOT / "artifacts" / "tabpfn" / "x_train.csv"
    y_train = ROOT / "artifacts" / "tabpfn" / "y_train.csv"
    x_test = ROOT / "artifacts" / "tabpfn" / "x_test.csv"
    ids = ROOT / "artifacts" / "tabpfn" / "ids.csv"
    preds_json = ROOT / "artifacts" / "tabpfn" / "predictions.json"
    out = ROOT / "house-prices-submission.csv"

    print(
        """
========================================================================
TABPFN MCP STEPS (what the agent actually ran)
========================================================================
1) prepare_train_set_upload(x_format='csv', y_format='csv',
     description='Kaggle House Prices ... SalePrice regression')
   -> train_set_upload_id + signed PUT URLs

2) Upload features + target:
     curl --fail-with-body -X PUT -H 'Content-Type: text/csv' \\
       --upload-file artifacts/tabpfn/x_train.csv '<signed_x_url>'
     curl --fail-with-body -X PUT -H 'Content-Type: text/csv' \\
       --upload-file artifacts/tabpfn/y_train.csv '<signed_y_url>'

3) fit(train_set_upload_id, task='regression')
   -> fitted_train_set_id
   poll get_fit_status until status == 'completed'

4) prepare_test_set_upload(fitted_train_set_id)
   upload artifacts/tabpfn/x_test.csv to the signed URL

5) predict(test_set_upload_id, fitted_train_set_id, task='regression')
   download prediction JSON to artifacts/tabpfn/predictions.json

6) Build Kaggle CSV:
"""
    )
    print(
        f"   {sys.executable} src/build_submission_from_preds.py \\\n"
        f"     --predictions {preds_json} \\\n"
        f"     --ids {ids} \\\n"
        f"     --out {out}"
    )

    if preds_json.exists() and ids.exists():
        import pandas as pd
        import numpy as np

        arr = load_predictions(preds_json)
        id_df = pd.read_csv(ids)
        assert len(arr) == len(id_df)
        sub = pd.DataFrame({"Id": id_df["Id"], "SalePrice": arr})
        sub.to_csv(out, index=False)
        (ROOT / "artifacts" / "house-prices-submission.csv").write_text(out.read_text())
        print(f"\nRebuilt submission from cached TabPFN predictions -> {out}")
        print(sub["SalePrice"].describe().to_string())
    else:
        print("\nNo cached predictions.json yet — run the MCP steps above first.")
        print(f"Expected files: {x_train}, {y_train}, {x_test}")


if __name__ == "__main__":
    main()
