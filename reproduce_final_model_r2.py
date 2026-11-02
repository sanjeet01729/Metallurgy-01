from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from features import build_features


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train_v2.csv"
TEST_PATH = BASE_DIR / "test_v2.csv"
DATASET_PATH = BASE_DIR / "dataset_v2.csv"
REPORT_PATH = BASE_DIR / "r2_reproduction.txt"
TARGET = "E_GPa"
ENGINEERED_NUMERIC = [
    "VEC", "delta_r", "delta_S_mix", "Mo_eq", "Al_eq", "Bo", "Md",
]
CATEGORICAL_FEATURES = ["phase_est"]
EXPECTED_TEST = {"r2": 0.687, "mae": 9.84, "rmse": 12.93}
EXPECTED_TRAIN_R2 = 0.940


def build_feature_frame(
    frame: pd.DataFrame, composition_columns: list[str]
) -> pd.DataFrame:
    composition = frame.reindex(
        columns=composition_columns, fill_value=0.0
    ).copy()
    composition = composition.apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0)
    totals = composition.sum(axis=1)
    if not np.isfinite(totals.to_numpy()).all() or (totals <= 0).any():
        raise ValueError("Every composition must have a finite positive total.")
    normalized = composition.div(totals, axis=0) * 100.0
    engineered = build_features(normalized)
    feature_columns = composition_columns + ENGINEERED_NUMERIC + CATEGORICAL_FEATURES
    missing_columns = [
        column for column in feature_columns if column not in engineered
    ]
    if missing_columns:
        raise ValueError(f"Feature builder is missing columns: {missing_columns}")
    return engineered.loc[:, feature_columns]


def make_model(feature_columns: list[str], random_state: int) -> Pipeline:
    categorical = [
        column for column in feature_columns
        if column in CATEGORICAL_FEATURES
    ]
    numeric = [
        column for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]
    preprocess = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    estimator = XGBRegressor(
        objective="reg:absoluteerror",
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def score(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
    }


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    dataset = pd.read_csv(DATASET_PATH)
    if TARGET not in train or TARGET not in test or TARGET not in dataset:
        raise ValueError("All three data files must contain E_GPa.")

    composition_columns = [
        column for column in train.columns if column.endswith("_wt")
    ]
    if not composition_columns:
        raise ValueError("train_v2.csv has no *_wt composition columns.")
    forbidden = {TARGET, "Ti_at", "ti_based", "phase_note"}
    train_x = build_feature_frame(train, composition_columns)
    test_x = build_feature_frame(test, composition_columns)
    dataset_x = build_feature_frame(dataset, composition_columns)
    feature_columns = train_x.columns.tolist()
    if set(feature_columns) & forbidden:
        raise AssertionError("A forbidden feature entered the model.")

    train_y = pd.to_numeric(train[TARGET], errors="raise")
    test_y = pd.to_numeric(test[TARGET], errors="raise")
    dataset_y = pd.to_numeric(dataset[TARGET], errors="raise")

    final_model = make_model(feature_columns, random_state=42)
    final_model.fit(train_x, train_y)
    train_predictions = np.asarray(final_model.predict(train_x), dtype=float)
    test_predictions = np.asarray(final_model.predict(test_x), dtype=float)
    train_metrics = score(train_y, train_predictions)
    test_metrics = score(test_y, test_predictions)

    tolerances = {"r2": 0.02, "mae": 0.25, "rmse": 0.25}
    checks_pass = all(
        abs(test_metrics[name] - EXPECTED_TEST[name]) <= tolerances[name]
        for name in EXPECTED_TEST
    )
    train_r2_matches = abs(train_metrics["r2"] - EXPECTED_TRAIN_R2) <= 0.02

    seed_sensitivity = []
    for seed in range(5):
        seed_model = make_model(feature_columns, random_state=seed)
        seed_model.fit(train_x, train_y)
        seed_prediction = np.asarray(seed_model.predict(test_x), dtype=float)
        seed_sensitivity.append(
            (seed, float(r2_score(test_y, seed_prediction)))
        )

    dataset_compositions = dataset.reindex(
        columns=composition_columns, fill_value=0.0
    ).apply(pd.to_numeric, errors="raise")
    group_keys = (
        dataset_compositions.round(1).astype(str).agg("|".join, axis=1)
    )
    robust_split_results = []
    for seed in range(30):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=seed
        )
        fit_indices, test_indices = next(
            splitter.split(dataset_x, dataset_y, groups=group_keys)
        )
        fit_groups = set(group_keys.iloc[fit_indices])
        heldout_groups = set(group_keys.iloc[test_indices])
        if fit_groups & heldout_groups:
            raise AssertionError(
                f"Composition leakage detected for robustness split seed {seed}."
            )
        split_model = make_model(feature_columns, random_state=42)
        split_model.fit(
            dataset_x.iloc[fit_indices], dataset_y.iloc[fit_indices]
        )
        split_prediction = np.asarray(
            split_model.predict(dataset_x.iloc[test_indices]), dtype=float
        )
        split_score = score(
            dataset_y.iloc[test_indices], split_prediction
        )
        robust_split_results.append(
            {
                "seed": seed,
                "n_test": int(len(test_indices)),
                "r2": split_score["r2"],
                "mae": split_score["mae"],
            }
        )

    def distribution_summary(metric: str) -> dict[str, float]:
        values = np.array(
            [row[metric] for row in robust_split_results], dtype=float
        )
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }

    r2_distribution = distribution_summary("r2")
    mae_distribution = distribution_summary("mae")
    seed42_r2 = test_metrics["r2"]
    below_seed42 = sum(row["r2"] < seed42_r2 for row in robust_split_results)
    equal_seed42 = sum(
        np.isclose(row["r2"], seed42_r2, rtol=0, atol=1e-12)
        for row in robust_split_results
    )
    empirical_percentile = (
        below_seed42 + 0.5 * equal_seed42
    ) / len(robust_split_results) * 100.0

    lines = [
        "INDEPENDENT FINAL MODEL REPRODUCTION",
        "====================================",
        "No saved model file was loaded. The XGBRegressor was fitted from scratch.",
        "Model: XGBRegressor(objective='reg:absoluteerror', random_state=42); all other estimator parameters used library defaults.",
        "Features: *_wt composition + VEC, delta_r, delta_S_mix, Mo_eq, Al_eq, Bo, Md, phase_est.",
        "Preprocessing: normalize composition to 100 wt%; build_features; one-hot encode phase_est with unknown categories ignored; leave feature NaNs unchanged for XGBoost.",
        "Forbidden features excluded: E_GPa, Ti_at, ti_based, phase_note, and identifiers.",
        "",
        "Library versions:",
        f"- pandas: {pd.__version__}",
        f"- numpy: {np.__version__}",
        f"- scikit-learn: {sklearn.__version__}",
        f"- xgboost: {xgboost.__version__}",
        "",
        "PART 1 — train_v2.csv fit; test_v2.csv evaluation",
        f"Train: n={len(train)}, R2={train_metrics['r2']:.6f}, MAE={train_metrics['mae']:.4f} GPa, RMSE={train_metrics['rmse']:.4f} GPa.",
        f"Test: n={len(test)}, R2={test_metrics['r2']:.6f}, MAE={test_metrics['mae']:.4f} GPa, RMSE={test_metrics['rmse']:.4f} GPa.",
        f"Reported-value check: test targets R2 about 0.687, MAE about 9.84 GPa, RMSE about 12.93 GPa; train R2 about 0.940.",
        f"Test metric tolerances: R2 +/-{tolerances['r2']:.2f}, MAE/RMSE +/-{tolerances['mae']:.2f} GPa.",
        f"TRAIN R2 CHECK: {'PASS' if train_r2_matches else 'DIFFERS'} (absolute tolerance 0.02).",
        f"TEST R2 CHECK: {'PASS' if abs(test_metrics['r2'] - EXPECTED_TEST['r2']) <= tolerances['r2'] else 'FAIL'} (required tolerance +/-0.02).",
        f"OVERALL REPORTED-METRIC CHECK: {'PASS' if checks_pass and train_r2_matches else 'FAIL'}.",
        "",
        "Test R2 by estimator random_state (all other parameters unchanged):",
    ]
    lines.extend(
        f"- random_state={seed}: R2={value:.6f}"
        for seed, value in seed_sensitivity
    )
    lines.extend(
        [
            "",
            "PART 2 — 30 grouped train/test splits of dataset_v2.csv",
            "Every split used GroupShuffleSplit(test_size=0.2, random_state=seed), "
            "seed=0 through 29, grouped by all *_wt columns rounded to one decimal.",
            "For each split, the model used random_state=42 and default values for all other estimator parameters.",
            "seed | n_test | test_R2 | test_MAE_GPa",
            "-----|--------|---------|------------",
        ]
    )
    lines.extend(
        f"{row['seed']:>4} | {row['n_test']:>6} | {row['r2']:>7.6f} | {row['mae']:>11.4f}"
        for row in robust_split_results
    )
    lines.extend(
        [
            "",
            "Grouped-split distribution (standard deviation is sample SD; percentiles use linear interpolation):",
            f"- Test R2: mean={r2_distribution['mean']:.6f}, std={r2_distribution['std']:.6f}, min={r2_distribution['min']:.6f}, max={r2_distribution['max']:.6f}, p05={r2_distribution['p05']:.6f}, p95={r2_distribution['p95']:.6f}.",
            f"- Test MAE (GPa): mean={mae_distribution['mean']:.4f}, std={mae_distribution['std']:.4f}, min={mae_distribution['min']:.4f}, max={mae_distribution['max']:.4f}, p05={mae_distribution['p05']:.4f}, p95={mae_distribution['p95']:.4f}.",
            f"- The 603/151 seed-42 split R2 of {seed42_r2:.6f} is at empirical mid-rank percentile {empirical_percentile:.1f} among the 30 dataset_v2 splits ({below_seed42} lower, {equal_seed42} tied).",
        ]
    )
    lines.extend(
        [
            "",
            "Plain-language conclusion:",
            (
                f"The seed-42 test result {'was' if checks_pass and train_r2_matches else 'was not'} reproduced within the stated checks. "
                f"Across 30 composition-grouped splits, test R2 ranged from "
                f"{r2_distribution['min']:.3f} to {r2_distribution['max']:.3f} "
                f"(mean {r2_distribution['mean']:.3f}, SD {r2_distribution['std']:.3f}); "
                f"MAE ranged from {mae_distribution['min']:.2f} to "
                f"{mae_distribution['max']:.2f} GPa. Thus, R2 around 0.68 "
                f"{'was not a typical robust value across splits' if empirical_percentile >= 90 else 'should be interpreted in the context of split-to-split variability'}."
            ),
            "",
            "Choices and diagnostic notes:",
            "- The `features.py` build_features function was applied to the composition columns after row normalization; precomputed engineered columns in the CSVs were not used.",
            "- OneHotEncoder(handle_unknown='ignore', sparse_output=False) was used because phase_est is categorical and dense input supports XGBoost with NaNs.",
            "- No imputation was applied to engineered features; missing values were passed through to XGBoost.",
            "- The reproduction uses the explicitly requested default XGBoost parameters. Earlier project fitting code additionally specified eval_metric='mae', n_jobs=1, tree_method='hist', and verbosity=0; those overrides were intentionally not used here.",
        ]
    )
    if not checks_pass:
        lines.extend(
            [
                "",
                "Likely causes of any mismatch and checks performed:",
                "- Feature columns: checked the prescribed composition and seven numeric descriptors plus phase_est; excluded target, Ti_at, ti_based, phase_note, and identifiers.",
                "- phase_est encoding: applied one-hot encoding within the scikit-learn pipeline, fitted on the training partition only.",
                "- Row order: predictions were scored against the original CSV row order; no sorting, filtering, or deduplication was performed.",
                f"- Library versions: pandas {pd.__version__}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}, XGBoost {xgboost.__version__}; these can alter XGBoost default behavior.",
                "- Parameter differences: the explicit defaults requested here omit project-specific eval_metric, n_jobs, tree_method, and verbosity overrides.",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
