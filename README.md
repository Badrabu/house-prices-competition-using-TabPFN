# house-prices-competition-using-TabPFN

Kaggle competition submission for **[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)** using **Prior Labs TabPFN** (tabular foundation model) for regression.

**Repository:** https://github.com/Badrabu/house-prices-competition-using-TabPFN

Primary deliverable: `house-prices-submission.csv` (Kaggle `Id,SalePrice` format).

---

## Competition details

| Item | Value |
|------|--------|
| Competition | House Prices - Advanced Regression Techniques |
| Dataset | Ames, Iowa residential sales |
| Goal | Predict `SalePrice` for each test `Id` |
| Train size | 1,460 rows × 80 features (+ Id, SalePrice) |
| Test size | 1,459 rows |
| Features used | 79 (every column except `Id`; `SalePrice` is the target) |
| Submission format | CSV with header `Id,SalePrice` |
| Official metric | **RMSLE** — RMSE between `log(prediction)` and `log(actual)` |
| Metric note | Log-space so expensive and cheap homes weigh equally |
| Leaderboard | Public ~50% of test; private = remaining (revealed at end) |

Reference scores (approximate, public LB):

- Sample mean submission ≈ **0.41** RMSLE  
- Competitive / strong public entries ≈ **0.11–0.12** RMSLE  
- Top solutions often ensemble stacked GBMs + careful feature engineering  

**Official Kaggle score for this TabPFN file:** not yet submitted from this environment (no Kaggle API credentials). Upload `house-prices-submission.csv` on the competition Submit page to get the public LB RMSLE.

---

## What was produced

| File | Description |
|------|-------------|
| `house-prices-submission.csv` | **TabPFN** predictions (main submission) |
| `artifacts/house-prices-submission.csv` | Same TabPFN submission |
| `artifacts/tabpfn/predictions.json` | Raw TabPFN regression outputs |
| `artifacts/local-baseline-submission.csv` | Sklearn HGB baseline (reproducible offline) |
| `artifacts/local-baseline-submission.metrics.txt` | Local 5-fold CV RMSLE |

### TabPFN submission summary

- Model: TabPFN **v3.5** regression (`package_version` 9.2.0, `n_estimators=8`)
- Predicted SalePrice mean ≈ **179,401** (train mean ≈ 180,921)
- Range ≈ **44,935 – 587,194**
- Correlates **0.98** with the local HGB baseline (sanity check)

### Local baseline CV (proxy score)

`HistGradientBoostingRegressor` on `log1p(SalePrice)`, 5-fold CV:

| Fold | RMSLE |
|------|-------|
| 1 | 0.14526 |
| 2 | 0.12652 |
| 3 | 0.16770 |
| 4 | 0.13699 |
| 5 | 0.11773 |
| **Mean** | **0.13884** (±0.017) |

This is a **local estimate**, not the Kaggle public score. TabPFN is expected to be similar or better; confirm by submitting on Kaggle.

---

## Full code used for the TabPFN submission

Pipeline scripts live under `src/`:

1. `src/prepare_tabpfn_csvs.py` — split train/test into `x_train`, `y_train`, `x_test`, `ids`
2. TabPFN MCP: upload → `fit(task='regression')` → upload test → `predict`
3. `src/build_submission_from_preds.py` — join predictions with Ids → Kaggle CSV
4. `src/run_tabpfn_pipeline.py` — documents and rebuilds the submission from cached preds
5. `src/train_local_baseline.py` — fully offline sklearn baseline + CV RMSLE

### Rebuild the TabPFN CSV from cached predictions

```bash
pip install -r requirements.txt
python src/prepare_tabpfn_csvs.py
python src/build_submission_from_preds.py \
  --predictions artifacts/tabpfn/predictions.json \
  --ids artifacts/tabpfn/ids.csv \
  --out house-prices-submission.csv
```

### Re-run the offline baseline (with CV)

```bash
python src/train_local_baseline.py --out artifacts/local-baseline-submission.csv
```

### End-to-end TabPFN MCP flow (agent / Prior Labs)

```text
prepare_train_set_upload → PUT x_train.csv + y_train.csv
fit(task='regression') → poll get_fit_status
prepare_test_set_upload → PUT x_test.csv
predict(task='regression') → download predictions.json
build_submission_from_preds.py → house-prices-submission.csv
```

See `src/run_tabpfn_pipeline.py` for the full annotated walkthrough.

---

## Data

`data/train.csv`, `data/test.csv`, and `data/sample_submission.csv` are the standard Kaggle competition files (Ames housing). Replace with a fresh download from Kaggle if you prefer the official copies.

---

## Repository

Canonical remote: https://github.com/Badrabu/house-prices-competition-using-TabPFN (`main`)

### Kaggle

Official LB score requires uploading `house-prices-submission.csv` on the competition Submit page (or configuring Kaggle API credentials).
