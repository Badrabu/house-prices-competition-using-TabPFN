# Score & competition notes

## Official evaluation

Competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Kaggle evaluates submissions with **RMSLE**:

```text
RMSLE = sqrt( mean( (log(y_pred) - log(y_true))^2 ) )
```

- Lower is better  
- Taking logs makes relative errors matter, not absolute dollar errors  
- Submission must contain every test `Id` exactly once with a positive `SalePrice`

## This project’s scores

| Source | Score | Status |
|--------|-------|--------|
| TabPFN submission (`house-prices-submission.csv`) | *pending Kaggle upload* | File ready (1,459 rows) |
| Local HGB 5-fold CV | **0.13884** RMSLE | Computed in-repo |
| Sample submission (mean price) | ~0.41 RMSLE | Typical weak baseline |
| Strong public LB solutions | ~0.11–0.12 | Reference only |

## How to get the official TabPFN score

1. Open the competition → **Submit Prediction**
2. Upload `house-prices-submission.csv`
3. Record the **public** RMSLE here after submit:

```text
Public LB RMSLE: ___________
Private LB RMSLE: ___________  (filled when competition finalizes / for permanent comps, when revealed)
Submitted at (America/Los_Angeles): ___________
```

## Model card (TabPFN run)

- Task: regression  
- Target: `SalePrice` (raw dollars; TabPFN trained directly on SalePrice)  
- Features: 79 columns (Id dropped)  
- Engine: TabPFN v3.5 / package 9.2.0 / 8 estimators  
- Device used by service: CUDA  
- No manual feature engineering beyond dropping Id  

## Sanity checks already run

- Row count matches test set (1459)  
- All SalePrice > 0  
- Prediction mean close to train mean  
- Correlation with independent HGB baseline ≈ 0.98  
