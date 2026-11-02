from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from features import build_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ti_ym_model_v2.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns_v2.json"
TRAIN_PATH = BASE_DIR / "train_v2.csv"
TEST_PATH = BASE_DIR / "test_v2.csv"
REPORT_PATH = BASE_DIR / "r2_report.txt"


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    with FEATURE_COLUMNS_PATH.open(encoding="utf-8") as file:
        feature_columns = json.load(file)
    model = joblib.load(MODEL_PATH)

    composition_columns = [
        column for column in train.columns if column.endswith("_wt")
    ]
    if not composition_columns:
        raise ValueError("train_v2.csv has no columns ending in '_wt'.")
    if "E_GPa" not in train or "E_GPa" not in test:
        raise ValueError("Both input CSV files must contain E_GPa.")

    expected_model_columns = list(
        getattr(model, "feature_names_in_", feature_columns)
    )
    if expected_model_columns != feature_columns:
        raise ValueError(
            "feature_columns_v2.json does not match the saved model input schema."
        )

    def make_X(df: pd.DataFrame) -> pd.DataFrame:
        composition = df.reindex(
            columns=composition_columns, fill_value=0.0
        ).copy()
        composition = composition.apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0.0)
        totals = composition.sum(axis=1)
        if not np.isfinite(totals.to_numpy()).all() or (totals <= 0).any():
            raise ValueError("Every composition must have a positive finite total.")
        normalized = composition.div(totals, axis=0) * 100.0
        engineered = build_features(normalized)

        # The saved pipeline expects phase_est and encodes it internally.
        if "phase_est" not in feature_columns:
            phase_columns = [
                column for column in feature_columns
                if column.startswith("phase_est_")
            ]
            if phase_columns:
                one_hot = pd.get_dummies(
                    engineered["phase_est"], prefix="phase_est", dtype=float
                ).reindex(columns=phase_columns, fill_value=0.0)
                engineered = pd.concat(
                    [engineered.drop(columns=["phase_est"]), one_hot], axis=1
                )

        absent = [column for column in feature_columns if column not in engineered]
        if absent:
            raise ValueError(f"Feature builder is missing model columns: {absent}")
        return engineered.loc[:, feature_columns]

    def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | int]:
        return {
            "n": int(len(y_true)),
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    train_X = make_X(train)
    test_X = make_X(test)
    train_y = pd.to_numeric(train["E_GPa"], errors="raise")
    test_y = pd.to_numeric(test["E_GPa"], errors="raise")
    train_predictions = np.asarray(model.predict(train_X), dtype=float)
    test_predictions = np.asarray(model.predict(test_X), dtype=float)
    train_metrics = metrics(train_y, train_predictions)
    test_metrics = metrics(test_y, test_predictions)

    expected_test = {"r2": 0.684, "mae": 9.88, "rmse": 13.0}
    tolerances = {"r2": 0.02, "mae": 0.2, "rmse": 0.2}
    failed_checks = [
        name for name, expected in expected_test.items()
        if abs(float(test_metrics[name]) - expected) > tolerances[name]
    ]
    if failed_checks:
        lines = [
            "FEATURE PIPELINE CHECK FAILED",
            "The saved-model test metrics differ from the expected values; "
            "no CV results were computed.",
            *(f"- Test {name.upper()}: {test_metrics[name]:.6f}" for name in expected_test),
            "Expected approximately: R2=0.684, MAE=9.88 GPa, RMSE=13.0 GPa.",
            "Check make_X, build_features, and feature column ordering.",
        ]
        REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        raise SystemExit(1)

    test_groups = {
        "phase_est == beta": test_X["phase_est"].eq("beta").to_numpy(),
        "E_GPa < 60": test_y.lt(60).to_numpy(),
        "Ti_wt < 70": test["Ti_wt"].lt(70).to_numpy(),
    }
    subset_metrics: dict[str, dict[str, float | int | None]] = {}
    for label, mask in test_groups.items():
        subset_y = test_y.iloc[np.flatnonzero(mask)]
        subset_pred = test_predictions[mask]
        if len(subset_y) == 0:
            subset_metrics[label] = {"n": 0, "r2": None, "mae": None}
            continue
        subset_metrics[label] = {
            "n": int(len(subset_y)),
            "r2": (
                float(r2_score(subset_y, subset_pred))
                if len(subset_y) >= 2 else None
            ),
            "mae": float(mean_absolute_error(subset_y, subset_pred)),
        }

    groups_frame = train[composition_columns].apply(
        pd.to_numeric, errors="raise"
    )
    groups = groups_frame.round(1).astype(str).agg("|".join, axis=1)
    cv = GroupKFold(n_splits=5)
    fold_results = []
    for fold_number, (fit_indices, validation_indices) in enumerate(
        cv.split(train_X, train_y, groups=groups), start=1
    ):
        fit_groups = set(groups.iloc[fit_indices])
        validation_groups = set(groups.iloc[validation_indices])
        if fit_groups & validation_groups:
            raise AssertionError(f"Composition leakage detected in fold {fold_number}.")
        fold_model = clone(model)
        fold_model.fit(train_X.iloc[fit_indices], train_y.iloc[fit_indices])
        fold_prediction = np.asarray(
            fold_model.predict(train_X.iloc[validation_indices]), dtype=float
        )
        fold_y = train_y.iloc[validation_indices]
        fold_results.append(
            {
                "r2": float(r2_score(fold_y, fold_prediction)),
                "mae": float(mean_absolute_error(fold_y, fold_prediction)),
            }
        )

    cv_summary = {}
    for name in ("r2", "mae"):
        values = np.array([fold[name] for fold in fold_results], dtype=float)
        cv_summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "folds": values.tolist(),
        }

    r2_gap = float(test_metrics["r2"] - train_metrics["r2"])
    mae_gap = float(test_metrics["mae"] - train_metrics["mae"])
    report_lines = [
        "FINAL MODEL R2 AND ERROR REPORT",
        "================================",
        "Saved model: ti_ym_model_v2.joblib (used without refitting for train/test metrics).",
        "Feature set: B; XGBoost objective: reg:absoluteerror; saved default parameters.",
        "Feature pipeline: raw *_wt columns -> zero-fill absent columns/cells -> normalize to 100 wt% -> build_features -> saved feature column order.",
        "phase_est is passed as a raw category because the saved pipeline one-hot encodes it internally.",
        "",
        "Software versions:",
        f"- pandas {pd.__version__}",
        f"- numpy {np.__version__}",
        f"- scikit-learn {sklearn.__version__}",
        f"- joblib {joblib.__version__}",
        "",
        "Train data metrics (saved model):",
        f"- n rows: {train_metrics['n']}",
        f"- R2: {train_metrics['r2']:.6f}",
        f"- MAE: {train_metrics['mae']:.4f} GPa",
        f"- RMSE: {train_metrics['rmse']:.4f} GPa",
        "",
        "Test data metrics (saved model):",
        f"- n rows: {test_metrics['n']}",
        f"- R2: {test_metrics['r2']:.6f}",
        f"- MAE: {test_metrics['mae']:.4f} GPa",
        f"- RMSE: {test_metrics['rmse']:.4f} GPa",
        "",
        "Test subgroup metrics:",
    ]
    for label, values in subset_metrics.items():
        r2_text = "undefined (fewer than 2 rows)" if values["r2"] is None else f"{values['r2']:.6f}"
        mae_text = "undefined" if values["mae"] is None else f"{values['mae']:.4f} GPa"
        report_lines.append(
            f"- {label}: n={values['n']}, R2={r2_text}, MAE={mae_text}"
        )
    report_lines.extend(
        [
            "",
            "Train-test gap (test minus train):",
            f"- R2 gap: {r2_gap:+.6f}",
            f"- MAE gap: {mae_gap:+.4f} GPa",
            "",
            "Test sanity check:",
            f"- R2={test_metrics['r2']:.6f} (expected about 0.684)",
            f"- MAE={test_metrics['mae']:.4f} GPa (expected about 9.88 GPa)",
            f"- RMSE={test_metrics['rmse']:.4f} GPa (expected about 13.0 GPa)",
            "- PASS: all test metrics are within the configured tolerances.",
            "",
            "Five-fold GroupKFold on train_v2.csv (same saved pipeline settings; "
            "a fresh clone was fitted on each CV fold only):",
            "- Group key: all *_wt columns rounded to one decimal place.",
            "- Fold composition overlap: zero (asserted).",
            f"- R2 mean +/- sample SD: {cv_summary['r2']['mean']:.6f} +/- {cv_summary['r2']['std']:.6f}",
            f"- MAE mean +/- sample SD: {cv_summary['mae']['mean']:.4f} +/- {cv_summary['mae']['std']:.4f} GPa",
            "- Per-fold R2: " + ", ".join(f"{value:.6f}" for value in cv_summary["r2"]["folds"]),
            "- Per-fold MAE (GPa): " + ", ".join(f"{value:.4f}" for value in cv_summary["mae"]["folds"]),
            "",
            "Interpretation: train R2 is an in-sample fit statistic; grouped CV is the "
            "more appropriate train-side generalization estimate.",
            "Standard-deviation convention: sample SD (ddof=1) across the five fold scores.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
