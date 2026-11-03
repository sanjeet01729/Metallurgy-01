# Titanium Young's Modulus: Model Training and Comparison

## Summary

- Dataset: 754 composition groups; training split: 603 rows; held-out test split: 151 rows.
- Best model by training-only 5-fold grouped CV MAE: **XGBoost + Random Forest mean**.
- Held-out test: R² = **0.710**, MAE = **9.42 GPa**, RMSE = **12.44 GPa**.
- The test split was not used for model selection. Only the CV-selected candidate was evaluated on it.
- App model artifact: `ti_ym_ensemble_v2.joblib`.

## Model comparison

Metrics are the mean and sample standard deviation across five GroupKFold validation folds on `train_v2.csv`. MAE was the selection metric; lower is better. R² is unitless; MAE and RMSE are in GPa.

| Model | CV R² (mean ± SD) | CV MAE (GPa, mean ± SD) | CV RMSE (GPa, mean ± SD) |
|---|---:|---:|---:|
| XGBoost + Random Forest mean | 0.527 ± 0.112 | 10.03 ± 0.90 | 13.47 ± 1.51 |
| XGBoost (absolute error) | 0.509 ± 0.088 | 10.19 ± 0.63 | 13.77 ± 1.28 |
| Random Forest | 0.512 ± 0.133 | 10.22 ± 1.13 | 13.65 ± 1.74 |
| Gradient Boosting | 0.494 ± 0.127 | 10.54 ± 1.08 | 13.90 ± 1.53 |
| CatBoost | 0.485 ± 0.111 | 10.59 ± 1.04 | 14.08 ± 1.50 |
| Extra Trees | 0.474 ± 0.108 | 10.70 ± 1.21 | 14.24 ± 1.51 |
| LightGBM | 0.453 ± 0.164 | 11.10 ± 1.40 | 14.44 ± 2.09 |
| Ridge | 0.315 ± 0.046 | 12.32 ± 0.93 | 16.32 ± 0.89 |
| SVR (RBF) | 0.249 ± 0.038 | 12.54 ± 1.25 | 17.10 ± 1.22 |
| Mean baseline | -0.007 ± 0.007 | 15.39 ± 1.24 | 19.82 ± 1.28 |

## Selected model scores

| Split | Rows | R² | MAE (GPa) | RMSE (GPa) |
|---|---:|---:|---:|---:|
| Training (in-sample) | 603 | 0.947 | 2.82 | 4.55 |
| Held-out test | 151 | 0.710 | 9.42 | 12.44 |

### Existing saved v2 model (test-set reference)

The pre-existing `ti_ym_model_v2.joblib` scored R² = 0.684, MAE = 9.87 GPa, and RMSE = 13.00 GPa on the same held-out test split. This is a reference evaluation of the existing model, not a candidate used to choose the newly trained model.

## Training method

1. Used the existing `train_v2.csv` and `test_v2.csv` split; kept the test set separate during validation and selection.
2. Used all `*_wt` columns, the stored engineered descriptors (`VEC`, `delta_r`, `delta_S_mix`, `Mo_eq`, `Al_eq`, `Bo`, `Md`), and one-hot indicators for `phase_est`.
3. Excluded `E_GPa` from inputs, along with `Ti_at`, `ti_based`, and `phase_note`. No feature scaling or imputation was fitted before a fold.
4. Compared ten regressors using 5-fold `GroupKFold` on the training set. Groups were all composition columns rounded to 1 decimal place.
5. XGBoost, LightGBM, and CatBoost received missing values directly. Models that required complete numeric inputs used median imputation fitted separately inside each fold; SVR and Ridge additionally used fold-fitted standardization.
6. Chose the candidate with the lowest mean validation MAE, refit it on all training rows, and evaluated it on the held-out test rows. The pre-existing saved model was also evaluated on test only as a historical reference.
7. Saved the selected fitted pipeline to `ti_ym_ensemble_v2.joblib` and its encoded feature order to `ensemble_feature_columns_v2.json` for app inference.

The in-sample training scores are optimistic and are not the preferred estimate of generalization. The grouped CV results and held-out test metrics are more informative. The data are modest in size and combine measurements from different sources and methods, so results should not be treated as a guarantee for new alloys or processing conditions.

## Data checks and software versions

- Train/test composition overlap at 0.1 wt% grouping tolerance: **0 groups**.
- Model input columns: 30 (including one-hot phase indicators).
- Versions:
  - Python: 3.13.5
  - NumPy: 2.5.4
  - pandas: 3.1.0
  - scikit-learn: 1.9.1
  - XGBoost: 3.4.1
  - LightGBM: 4.7.0
  - CatBoost: 1.2.10
