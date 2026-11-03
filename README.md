# Titanium Alloy Young's Modulus Prediction

This project predicts Young's modulus (`E_GPa`) from titanium-alloy
composition. It includes a Streamlit app, shared composition-feature
calculations, the saved XGBoost + Random Forest ensemble, train/test data,
model-comparison code, and generated plots.

> **Use note:** Predictions are estimates from composition alone. The model
> does not account for processing, heat treatment, or measured phase fractions;
> treat predictions outside the represented training compositions with caution.

## Run the app

Use Python 3.13 and install the listed dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

In the sidebar, enter alloying-element concentrations in wt%. Titanium is
calculated as the balance to 100 wt%. Select **Calculate Young's Modulus** to
get the ensemble prediction in GPa. The app warns when the input is outside the
training range or when a supplied element is not represented in the trained
composition columns.

## Model and features

The app loads `ti_ym_ensemble_v2.joblib`, a mean ensemble of:

- XGBoost regression (`reg:absoluteerror`)
- Random Forest regression (500 trees)

The model uses the element `*_wt` columns, seven engineered descriptors
(`VEC`, `delta_r`, `delta_S_mix`, `Mo_eq`, `Al_eq`, `Bo`, and `Md`), and
one-hot encoded `phase_est`. `features.py` normalizes input composition and
calculates these descriptors using its atomic-mass and element-property
tables. The `phase_est` value is a composition-based screening label, not a
measured phase fraction.

The app's model input schema is stored in `feature_columns_v2.json` and
`ensemble_feature_columns_v2.json`. `train_v2.csv` is also used to read the
trained composition columns. Keep these files alongside `app.py` and the model
artifact.

## Evaluation

The supplied split contains 603 training rows and 151 held-out test rows. Model
comparison uses five-fold `GroupKFold` cross-validation on the training data,
grouping compositions rounded to 0.1 wt% so repeated compositions stay within
the same fold. The model was selected by mean CV MAE; the test set was held out
for final evaluation.

For the selected ensemble, the recorded results were:

| Evaluation | R² | MAE (GPa) | RMSE (GPa) |
|---|---:|---:|---:|
| Grouped training CV (mean ± SD for R²) | 0.527 ± 0.112 | 10.03 ± 0.90 | 13.47 ± 1.51 |
| Held-out test | 0.710 | 9.42 | 12.44 |

The in-sample training score is not a generalization estimate and is not shown
as the main result. Scores apply to this dataset and split and are not a
guarantee for new compositions.

## Reproduce model comparison

```bash
python3 compare_train_models_v2.py
```

The script compares ten regressors using grouped cross-validation, selects the
lowest-CV-MAE candidate, fits it on all training rows, evaluates it on the
held-out test rows, and writes `model_training_comparison.md`. It also saves
the selected fitted ensemble and feature order as `ti_ym_ensemble_v2.joblib`
and `ensemble_feature_columns_v2.json`, which are used by the app. Running
this script replaces those two generated model artifacts.

## Feature documentation

See [`feature_formula_report.md`](feature_formula_report.md) for the equations,
one-line definitions, phase-screening rules, and missing-value behavior of
the engineered features.

## Plot scripts and figures

The prediction-curve scripts use the saved ensemble and generate plots for
binary alloys with titanium as the balance:

| Script | Figure | Composition series |
|---|---|---|
| `plot_ti_nb_modulus.py` | `ti_nb_modulus_vs_nb.png` | Ti–Nb, Nb = 1 to 40 wt% |
| `plot_ti_sn_modulus.py` | `ti_sn_modulus_vs_sn.png` | Ti–Sn, Sn = 1 to 40 wt% |
| `plot_ti_v_modulus.py` | `ti_v_modulus_vs_v.png` | Ti–V, V = 1 to 40 wt% |
| `plot_ti_ta_modulus.py` | `ti_ta_modulus_vs_ta.png` | Ti–Ta, Ta = 1 to 40 wt% |
| `plot_ti_zr_modulus.py` | `ti_zr_modulus_vs_zr.png` | Ti–Zr, Zr = 1 to 50 wt% |
| `plot_ti_nb_ta_modulus.py` | `ti_nb_ta_modulus_vs_nb.png` | Nb series at fixed Ta levels |
| `plot_test_residuals.py` | `test_residuals_v2.png` | Residual diagnostics on the test set |

For example:

```bash
python3 plot_ti_nb_modulus.py
python3 plot_test_residuals.py
```

The binary curves use 4.5 wt% intervals starting at 1 wt%, with the specified
upper endpoint added where needed. These are model predictions, not measured
experimental values; sharp changes should be interpreted cautiously.

The repository also includes `training_fit_graph.png`, `shap_ensemble_summary_v2.png`,
`nb_variation_graph.png`, and `nb_zr_vs_real.png` as previously generated
figures. Refer to `model_training_comparison.md` for the candidate-model
comparison and `feature_formula_report.md` for feature definitions.

## Main files

| File | Purpose |
|---|---|
| `app.py` | Streamlit composition-input and prediction interface |
| `features.py` | Composition normalization and engineered feature functions |
| `ti_ym_ensemble_v2.joblib` | Ensemble model currently used by the app |
| `ensemble_feature_columns_v2.json` | Exact encoded feature order expected by the ensemble |
| `feature_columns_v2.json` | Raw model feature schema |
| `train_v2.csv`, `test_v2.csv` | Fixed training and held-out test splits |
| `compare_train_models_v2.py` | Cross-validated model comparison and ensemble saving |
| `model_training_comparison.md` | Recorded comparison results and training method |
| `feature_formula_report.md` | Feature formulas and definitions |
| `requirements.txt` | Python dependency versions |

`ti_ym_model_v2.joblib` is the earlier single-model artifact; the app uses
`ti_ym_ensemble_v2.joblib`.
