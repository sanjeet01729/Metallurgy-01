from __future__ import annotations

from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from features import build_features


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_v2.csv"
REPORT_PATH = BASE_DIR / "model_comparison_v2.txt"
TARGET = "E_GPa"
ENGINEERED_NUMERIC = [
    "VEC", "delta_r", "delta_S_mix", "Mo_eq", "Al_eq", "Bo", "Md",
]
CATEGORICAL_FEATURES = ["phase_est"]


def build_feature_frame(
    frame: pd.DataFrame, composition_columns: list[str]
) -> pd.DataFrame:
    composition = frame[composition_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    totals = composition.sum(axis=1, min_count=1)
    if totals.isna().any() or (totals <= 0).any():
        raise ValueError("Composition rows must have positive non-missing totals.")
    normalized = composition.div(totals, axis=0) * 100.0
    engineered = build_features(normalized)
    columns = composition_columns + ENGINEERED_NUMERIC + CATEGORICAL_FEATURES
    missing = [column for column in columns if column not in engineered]
    if missing:
        raise ValueError(f"Feature builder is missing required columns: {missing}")
    return engineered.loc[:, columns]


def make_preprocessor(
    feature_columns: list[str],
    *,
    impute_numeric: bool,
    scale_all: bool,
) -> ColumnTransformer:
    categorical = [
        column for column in feature_columns if column in CATEGORICAL_FEATURES
    ]
    numeric = [
        column for column in feature_columns if column not in CATEGORICAL_FEATURES
    ]
    if impute_numeric:
        numeric_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                )
            ]
        )
    else:
        numeric_transformer = "passthrough"
    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, numeric),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def make_pipeline(
    model_name: str, feature_columns: list[str]
) -> Pipeline | None:
    model_factories = {
        "XGBoost": lambda: XGBRegressor(
            objective="reg:absoluteerror",
            random_state=42,
        ),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=500,
            random_state=42,
        ),
        "Extra Trees": lambda: ExtraTreesRegressor(
            n_estimators=500,
            random_state=42,
        ),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            random_state=42,
        ),
        "LightGBM": lambda: LGBMRegressor(
            random_state=42,
            verbosity=-1,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            verbose=0,
            random_seed=42,
        ),
        "SVR (RBF)": lambda: SVR(kernel="rbf"),
        "Ridge": lambda: Ridge(),
    }
    if model_name == "Mean baseline":
        return None
    if model_name == "Mean ensemble (XGBoost + RF)":
        return None
    if model_name not in model_factories:
        raise ValueError(f"Unknown model: {model_name}")

    scale = model_name in {"SVR (RBF)", "Ridge"}
    impute = model_name in {"Gradient Boosting", "SVR (RBF)", "Ridge"}
    steps = [
        (
            "preprocess",
            make_preprocessor(
                feature_columns,
                impute_numeric=impute,
                scale_all=False,
            ),
        )
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model_factories[model_name]()))
    return Pipeline(steps)


def fit_predict(
    model_name: str,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_score: pd.DataFrame,
) -> np.ndarray:
    if model_name == "Mean baseline":
        return np.full(len(x_score), float(y_fit.mean()), dtype=float)
    if model_name == "Mean ensemble (XGBoost + RF)":
        xgb_pipe = make_pipeline("XGBoost", x_fit.columns.tolist())
        rf_pipe = make_pipeline("Random Forest", x_fit.columns.tolist())
        assert xgb_pipe is not None and rf_pipe is not None
        xgb_pipe.fit(x_fit, y_fit)
        rf_pipe.fit(x_fit, y_fit)
        return (
            np.asarray(xgb_pipe.predict(x_score), dtype=float)
            + np.asarray(rf_pipe.predict(x_score), dtype=float)
        ) / 2.0
    pipeline = make_pipeline(model_name, x_fit.columns.tolist())
    assert pipeline is not None
    pipeline.fit(x_fit, y_fit)
    return np.asarray(pipeline.predict(x_score), dtype=float)


def score(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    return (
        float(r2_score(y_true, y_pred)),
        float(mean_absolute_error(y_true, y_pred)),
    )


def evaluate_splits(
    model_name: str,
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, float]], np.ndarray]:
    scores = []
    oof_predictions = np.full(len(y), np.nan, dtype=float)
    for seed, (fit_indices, test_indices) in enumerate(splits):
        fit_groups = set(groups.iloc[fit_indices])
        test_groups = set(groups.iloc[test_indices])
        if fit_groups & test_groups:
            raise AssertionError(
                f"Composition leakage for {model_name}, repeated split {seed}."
            )
        prediction = fit_predict(
            model_name,
            x.iloc[fit_indices],
            y.iloc[fit_indices],
            x.iloc[test_indices],
        )
        r2, mae = score(y.iloc[test_indices], prediction)
        scores.append({"r2": r2, "mae": mae})

    folds = GroupKFold(n_splits=5)
    for fit_indices, validation_indices in folds.split(x, y, groups=groups):
        if set(groups.iloc[fit_indices]) & set(groups.iloc[validation_indices]):
            raise AssertionError(f"Composition leakage in {model_name} GroupKFold.")
        oof_predictions[validation_indices] = fit_predict(
            model_name,
            x.iloc[fit_indices],
            y.iloc[fit_indices],
            x.iloc[validation_indices],
        )
    if not np.isfinite(oof_predictions).all():
        raise AssertionError(f"Missing out-of-fold predictions for {model_name}.")
    return scores, oof_predictions


def summary(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    return (
        float(array.mean()),
        float(array.std(ddof=1)),
        float(array.min()),
        float(array.max()),
    )


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if TARGET not in data:
        raise ValueError("dataset_v2.csv is missing E_GPa.")
    composition_columns = [
        column for column in data.columns if column.endswith("_wt")
    ]
    x = build_feature_frame(data, composition_columns)
    y = pd.to_numeric(data[TARGET], errors="raise")
    groups = (
        data[composition_columns]
        .apply(pd.to_numeric, errors="raise")
        .round(1)
        .astype(str)
        .agg("|".join, axis=1)
    )

    forbidden = {TARGET, "Ti_at", "ti_based", "phase_note"}
    if set(x.columns) & forbidden:
        raise AssertionError("A forbidden feature is present.")

    split_indices = [
        next(
            GroupShuffleSplit(
                n_splits=1,
                test_size=0.2,
                random_state=seed,
            ).split(x, y, groups=groups)
        )
        for seed in range(30)
    ]
    model_names = [
        "XGBoost",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "LightGBM",
        "CatBoost",
        "SVR (RBF)",
        "Ridge",
        "Mean baseline",
        "Mean ensemble (XGBoost + RF)",
    ]
    all_scores: dict[str, list[dict[str, float]]] = {}
    all_oof_predictions: dict[str, np.ndarray] = {}
    for model_name in model_names:
        print(f"Evaluating {model_name}...", flush=True)
        scores, oof = evaluate_splits(
            model_name,
            x,
            y,
            groups,
            split_indices,
        )
        all_scores[model_name] = scores
        all_oof_predictions[model_name] = oof

    baseline = all_scores["XGBoost"]
    lines = [
        "REGRESSION MODEL COMPARISON — FAIR GROUPED PROTOCOL",
        "====================================================",
        f"Dataset: dataset_v2.csv ({len(data)} rows).",
        f"Feature matrix: {len(composition_columns)} *_wt columns + 7 numeric engineered features + phase_est.",
        "Excluded features: E_GPa, Ti_at, ti_based, phase_note, and identifiers.",
        "phase_est was one-hot encoded within each training fold/split.",
        "Repeated evaluation: GroupShuffleSplit(test_size=0.2, random_state=0..29), groups are composition columns rounded to one decimal.",
        "Pooled OOF evaluation: 5-fold GroupKFold over all rows; each row received exactly one prediction from a fold model that did not see its composition group.",
        f"Library versions: pandas={pd.__version__}, numpy={np.__version__}, scikit-learn={sklearn.__version__}, xgboost={xgboost.__version__}, lightgbm={lightgbm.__version__}, catboost={catboost.__version__}.",
        "Numeric median imputation was fitted on each training partition only for Gradient Boosting, SVR, and Ridge, which require finite numeric input. XGBoost, Random Forest, Extra Trees, LightGBM, and CatBoost received NaNs natively. The mean baseline used only the training target mean.",
        "SVR and Ridge standardized the transformed numeric and one-hot features within each training partition.",
        "Random seeds were fixed to 42 for stochastic estimators where supported; all other model parameters used defaults except the requested tree counts and CatBoost verbose=0.",
        "",
        "Results (R2 and MAE mean +/- sample SD across 30 splits; pooled OOF scores from concatenated held-out predictions):",
        "Model | split R2 mean +/- SD [min, max] | split MAE mean +/- SD [min, max] GPa | pooled OOF R2 | pooled OOF MAE GPa | paired ΔR2 vs XGB mean +/- SD | wins / 30 | within one paired split-SD?",
        "------|-------------------------------|-----------------------------------------------|---------------|---------------------|--------------------------|-----------|---------------------------",
    ]

    paired_differences: dict[str, list[float]] = {}
    for model_name in model_names:
        paired = [
            candidate["r2"] - baseline[index]["r2"]
            for index, candidate in enumerate(all_scores[model_name])
        ]
        paired_differences[model_name] = paired

    within_one_sd: dict[str, bool] = {}
    for model_name in model_names:
        split_r2 = [row["r2"] for row in all_scores[model_name]]
        split_mae = [row["mae"] for row in all_scores[model_name]]
        r2_mean, r2_std, r2_min, r2_max = summary(split_r2)
        mae_mean, mae_std, mae_min, mae_max = summary(split_mae)
        pooled_r2, pooled_mae = score(y, all_oof_predictions[model_name])
        paired = np.asarray(paired_differences[model_name], dtype=float)
        paired_mean = float(paired.mean())
        paired_std = float(paired.std(ddof=1))
        wins = int(np.sum(paired > 0))
        within = abs(paired_mean) <= paired_std
        within_one_sd[model_name] = within
        marker = "yes" if within else "no"
        lines.append(
            f"{model_name} | {r2_mean:.4f} +/- {r2_std:.4f} [{r2_min:.4f}, {r2_max:.4f}] | "
            f"{mae_mean:.3f} +/- {mae_std:.3f} [{mae_min:.3f}, {mae_max:.3f}] | "
            f"{pooled_r2:.4f} | {pooled_mae:.3f} | "
            f"{paired_mean:+.4f} +/- {paired_std:.4f} | {wins}/30 | {marker}"
        )

    lines.extend(
        [
            "",
            "Per-split paired R2 differences versus XGBoost (candidate minus baseline; positive means candidate won):",
        ]
    )
    for model_name in model_names:
        paired = paired_differences[model_name]
        lines.append(
            f"- {model_name}: "
            + ", ".join(f"{value:+.4f}" for value in paired)
        )

    lines.extend(
        [
            "",
            "Interpretation of one split-standard-deviation comparison:",
            "For each model, the absolute mean paired R2 difference from XGBoost was compared with the sample SD of its 30 paired differences. 'Yes' means the mean difference was within one paired split-SD and is small relative to observed split variability; it is not a significance test.",
        ]
    )
    for model_name in model_names:
        if model_name == "XGBoost":
            continue
        paired = np.asarray(paired_differences[model_name], dtype=float)
        delta_mean = float(paired.mean())
        delta_std = float(paired.std(ddof=1))
        relation = "within" if abs(delta_mean) <= delta_std else "outside"
        lines.append(
            f"- {model_name}: mean paired ΔR2={delta_mean:+.4f}, paired SD={delta_std:.4f}; {relation} one paired split-SD."
        )

    lines.extend(
        [
            "",
            "Choices and caveats:",
            "- Randomness was made reproducible with random_state/random_seed=42 for estimators that expose such a parameter, without changing their other defaults.",
            "- Median imputation was limited to Gradient Boosting, SVR, and Ridge; imputation statistics were learned separately within each training split/fold.",
            "- Random Forest and Extra Trees in the installed scikit-learn version accept NaNs natively; they were not imputed.",
            "- LightGBM and CatBoost natively accept missing numeric values; their numeric NaNs were not imputed.",
            "- For pooled OOF R2 and MAE, metrics were calculated once on all concatenated out-of-fold predictions, not averaged across five fold metrics.",
            "- This is a fixed-model comparison with no tuning or error-based row filtering.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
