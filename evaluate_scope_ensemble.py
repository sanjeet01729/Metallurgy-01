from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from features import build_features


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_v2.csv"
REPORT_PATH = BASE_DIR / "scope_ensemble_report.txt"
TARGET = "E_GPa"
ENGINEERED_NUMERIC = [
    "VEC", "delta_r", "delta_S_mix", "Mo_eq", "Al_eq", "Bo", "Md",
]
CATEGORICAL_FEATURES = ["phase_est"]


def build_features_frame(
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


def build_pipeline(
    feature_columns: list[str], model_kind: str
) -> Pipeline | tuple[Pipeline, Pipeline]:
    categorical = [
        column for column in feature_columns if column in CATEGORICAL_FEATURES
    ]
    numeric = [
        column for column in feature_columns if column not in CATEGORICAL_FEATURES
    ]
    preprocessor = ColumnTransformer(
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
    if model_kind == "xgb":
        return Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:absoluteerror",
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_kind == "rf":
        return Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_kind == "ensemble":
        return (
            build_pipeline(feature_columns, "xgb"),
            build_pipeline(feature_columns, "rf"),
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def fit_predict(
    model_or_models, x_fit: pd.DataFrame, y_fit: pd.Series, x_score: pd.DataFrame
) -> np.ndarray:
    if isinstance(model_or_models, tuple):
        xgb_model, rf_model = model_or_models
        xgb_model.fit(x_fit, y_fit)
        rf_model.fit(x_fit, y_fit)
        return (
            np.asarray(xgb_model.predict(x_score), dtype=float)
            + np.asarray(rf_model.predict(x_score), dtype=float)
        ) / 2.0
    model_or_models.fit(x_fit, y_fit)
    return np.asarray(model_or_models.predict(x_score), dtype=float)


def metrics(y_true: pd.Series, predictions: np.ndarray) -> tuple[float, float]:
    return (
        float(r2_score(y_true, predictions)),
        float(mean_absolute_error(y_true, predictions)),
    )


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate_candidate(
    *,
    label: str,
    scope_description: str,
    eligible: pd.Series,
    data: pd.DataFrame,
    x_all: pd.DataFrame,
    groups_all: pd.Series,
    feature_columns: list[str],
    model_kind: str,
) -> dict:
    positions = np.flatnonzero(eligible.to_numpy())
    candidate_data = data.iloc[positions].reset_index(drop=True)
    candidate_x = x_all.iloc[positions].reset_index(drop=True)
    candidate_y = pd.to_numeric(
        candidate_data[TARGET], errors="raise"
    ).reset_index(drop=True)
    candidate_groups = groups_all.iloc[positions].reset_index(drop=True)
    n_groups = int(candidate_groups.nunique())

    split_results = []
    for seed in range(30):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=seed
        )
        fit_indices, test_indices = next(
            splitter.split(candidate_x, candidate_y, groups=candidate_groups)
        )
        if set(candidate_groups.iloc[fit_indices]) & set(
            candidate_groups.iloc[test_indices]
        ):
            raise AssertionError(
                f"Composition leakage in {label}, split seed {seed}."
            )
        fitted = build_pipeline(feature_columns, model_kind)
        predicted = fit_predict(
            fitted,
            candidate_x.iloc[fit_indices],
            candidate_y.iloc[fit_indices],
            candidate_x.iloc[test_indices],
        )
        r2, mae = metrics(candidate_y.iloc[test_indices], predicted)
        split_results.append(
            {
                "seed": seed,
                "n_test": int(len(test_indices)),
                "r2": r2,
                "mae": mae,
            }
        )

    # Each eligible row receives exactly one prediction from a model that did
    # not train on that row's rounded composition group.
    pooled_predictions = np.full(len(candidate_data), np.nan, dtype=float)
    group_kfold = GroupKFold(n_splits=5)
    for fit_indices, validation_indices in group_kfold.split(
        candidate_x, candidate_y, groups=candidate_groups
    ):
        if set(candidate_groups.iloc[fit_indices]) & set(
            candidate_groups.iloc[validation_indices]
        ):
            raise AssertionError(f"Composition leakage in {label} GroupKFold.")
        fitted = build_pipeline(feature_columns, model_kind)
        pooled_predictions[validation_indices] = fit_predict(
            fitted,
            candidate_x.iloc[fit_indices],
            candidate_y.iloc[fit_indices],
            candidate_x.iloc[validation_indices],
        )
    if not np.isfinite(pooled_predictions).all():
        raise AssertionError(f"Some rows have no pooled OOF prediction for {label}.")
    pooled_r2, pooled_mae = metrics(candidate_y, pooled_predictions)

    return {
        "label": label,
        "scope": scope_description,
        "n_rows": int(len(candidate_data)),
        "n_groups": n_groups,
        "split_results": split_results,
        "r2_summary": summarize([row["r2"] for row in split_results]),
        "mae_summary": summarize([row["mae"] for row in split_results]),
        "pooled_oof_r2": pooled_r2,
        "pooled_oof_mae": pooled_mae,
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if TARGET not in data:
        raise ValueError(f"{DATA_PATH.name} must contain {TARGET}.")
    composition_columns = [
        column for column in data.columns if column.endswith("_wt")
    ]
    required_composition = {"Ti_wt", "Nb_wt", "Ta_wt", "Mo_wt", "V_wt"}
    if not required_composition.issubset(composition_columns):
        raise ValueError(
            "Missing required composition columns: "
            f"{sorted(required_composition - set(composition_columns))}"
        )

    x = build_features_frame(data, composition_columns)
    target = pd.to_numeric(data[TARGET], errors="raise")
    groups = (
        data[composition_columns]
        .apply(pd.to_numeric, errors="raise")
        .round(1)
        .astype(str)
        .agg("|".join, axis=1)
    )
    feature_columns = x.columns.tolist()
    forbidden = {TARGET, "Ti_at", "ti_based", "phase_note"}
    if set(feature_columns) & forbidden:
        raise AssertionError(
            f"Forbidden features entered candidate matrix: {set(feature_columns) & forbidden}"
        )

    beta_scope = (
        data["Mo_eq"].gt(10.0)
        | (
            data["Nb_wt"]
            + data["Ta_wt"]
            + data["Mo_wt"]
            + data["V_wt"]
        ).ge(15.0)
    )
    heavy_scope = data["Ti_wt"].lt(70.0)
    candidates = [
        {
            "label": "Candidate 1 — baseline XGBoost",
            "scope": "All rows in dataset_v2.csv.",
            "eligible": pd.Series(True, index=data.index),
            "kind": "xgb",
        },
        {
            "label": "Candidate 2 — XGBoost + Random Forest mean ensemble",
            "scope": "All rows in dataset_v2.csv; prediction is the arithmetic mean of XGBoost and Random Forest predictions.",
            "eligible": pd.Series(True, index=data.index),
            "kind": "ensemble",
        },
        {
            "label": "Candidate 3 — beta-type scope-restricted XGBoost",
            "scope": "Rows with Mo_eq > 10 wt% OR (Nb_wt + Ta_wt + Mo_wt + V_wt >= 15 wt%). This model only covers this defined beta-type scope.",
            "eligible": beta_scope,
            "kind": "xgb",
        },
        {
            "label": "Candidate 4 — Ti-matrix heavy-element XGBoost",
            "scope": "Rows in dataset_v2.csv with Ti_wt < 70 wt%; dataset_v2.csv is already restricted to Ti_at >= 60 at%. This model only covers these heavy-element Ti-matrix rows.",
            "eligible": heavy_scope,
            "kind": "xgb",
        },
    ]

    results = []
    for candidate in candidates:
        result = evaluate_candidate(
            label=candidate["label"],
            scope_description=candidate["scope"],
            eligible=candidate["eligible"],
            data=data,
            x_all=x,
            groups_all=groups,
            feature_columns=feature_columns,
            model_kind=candidate["kind"],
        )
        results.append(result)

    lines = [
        "SCOPE RESTRICTION AND ENSEMBLE EVALUATION",
        "=========================================",
        "Dataset: dataset_v2.csv; rows are composition groups after dataset preparation.",
        "All candidates use the same features: all *_wt columns plus VEC, delta_r, delta_S_mix, Mo_eq, Al_eq, Bo, Md, and one-hot phase_est.",
        "The target E_GPa, Ti_at, ti_based, phase_note, and any identifier columns were excluded.",
        "Composition features were normalized to 100 wt% before build_features; engineered feature NaNs were left for XGBoost.",
        "RandomForest was one-hot encoded with an in-fold OneHotEncoder and fits only its training partition.",
        f"Library versions: pandas={pd.__version__}, numpy={np.__version__}, scikit-learn={sklearn.__version__}, xgboost={xgboost.__version__}.",
        "Repeated split protocol: 30 GroupShuffleSplit runs, test_size=0.2, seeds 0 through 29, group ID from all *_wt columns rounded to one decimal.",
        "Pooled OOF protocol: 5-fold GroupKFold, each eligible row predicted once by a model that did not see its composition group.",
        "Summary standard deviation is the sample SD (ddof=1) across the 30 split scores.",
        "",
    ]
    for result in results:
        r2_stats = result["r2_summary"]
        mae_stats = result["mae_summary"]
        lines.extend(
            [
                result["label"],
                "-" * len(result["label"]),
                f"Scope: {result['scope']}",
                f"Eligible rows: {result['n_rows']} across {result['n_groups']} composition groups.",
                f"Repeated-split test R2: mean={r2_stats['mean']:.6f}, std={r2_stats['std']:.6f}, min={r2_stats['min']:.6f}, max={r2_stats['max']:.6f}.",
                f"Repeated-split test MAE (GPa): mean={mae_stats['mean']:.4f}, std={mae_stats['std']:.4f}, min={mae_stats['min']:.4f}, max={mae_stats['max']:.4f}.",
                f"Pooled 5-fold GroupKFold OOF: R2={result['pooled_oof_r2']:.6f}, MAE={result['pooled_oof_mae']:.4f} GPa.",
                "Per-split scores (seed | test rows | R2 | MAE GPa):",
            ]
        )
        lines.extend(
            f"  {row['seed']:>2} | {row['n_test']:>3} | {row['r2']:.6f} | {row['mae']:.4f}"
            for row in result["split_results"]
        )
        lines.append("")

    lines.extend(
        [
            "Like-for-like interpretation",
            "-----------------------------",
            "Candidates 1 and 2 use the same full-dataset scope and identical split seeds, so their repeated-split results are like-for-like.",
            "Candidates 3 and 4 are evaluated only within their respective restricted scopes. Their test rows differ from Candidate 1's test rows; their R2 and MAE are not directly comparable to Candidate 1 as evidence of a better general-purpose model.",
            "A scope-restricted model only covers rows satisfying its scope definition; predictions outside that scope are not evaluated or claimed.",
            "No rows or splits were selected or removed based on model errors or scores.",
            "",
            "Choices:",
            "- Candidate 3 uses the user-defined OR condition exactly and applies it before each split.",
            "- Candidate 4 uses Ti_wt < 70 within dataset_v2.csv, whose existing population is Ti_at >= 60 at%.",
            "- The arithmetic ensemble gives equal weight to the XGBoost and Random Forest predictions.",
            "- The pooled OOF R2 is computed once from the concatenated held-out predictions, rather than averaging fold R2 values.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
