import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


BASE = Path(__file__).parent
TRAIN_FILE = BASE / "train.csv"
TEST_FILE = BASE / "test.csv"
MODEL_FILE = BASE / "ti_ym_model.joblib"
FEATURE_COLUMNS_FILE = BASE / "feature_columns.json"
IMPORTANCE_FILE = BASE / "feature_importance_new.csv"
RESULTS_JSON = BASE / "results.json"
RESULTS_TEXT = BASE / "results.txt"
TARGET = "E_GPa"
SEED = 42
N_SPLITS = 5
SEARCH_ITERATIONS = 60
PERMUTATION_REPEATS = 20

COMPOSITION_EXCLUSIONS = {"E_GPa", "ti_based"}
ENGINEERED_NUMERIC = [
    "VEC", "delta_r", "delta_S_mix", "Mo_eq", "Al_eq", "Bo", "Md",
]
ENGINEERED_CATEGORICAL = ["phase_est"]
ENGINEERED_FEATURES = ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL


def make_preprocessor(
    feature_columns: list[str],
    categorical_columns: list[str],
    *,
    impute: bool,
    scale: bool = False,
) -> ColumnTransformer:
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    numeric_steps = []
    if impute:
        numeric_steps.append(
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
        )
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = (
        Pipeline(numeric_steps) if numeric_steps else "passthrough"
    )
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def make_pipeline(
    model,
    feature_columns: list[str],
    categorical_columns: list[str],
    *,
    impute: bool,
    scale: bool = False,
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    feature_columns,
                    categorical_columns,
                    impute=impute,
                    scale=scale,
                ),
            ),
            ("model", model),
        ]
    )


def make_model(model_name: str):
    if model_name == "XGBRegressor":
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=SEED,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=SEED,
            n_jobs=1,
        )
    if model_name == "Ridge":
        return Ridge(alpha=1.0)
    if model_name == "DummyRegressor":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown model: {model_name}")


def grouped_cv_scores(
    model_name: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    x: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    params: dict | None = None,
) -> dict:
    fold_metrics = []
    for train_index, validation_index in folds:
        model = make_model(model_name)
        if params:
            model.set_params(**params)
        baseline = model_name != "XGBRegressor"
        pipeline = make_pipeline(
            model,
            feature_columns,
            categorical_columns,
            impute=baseline,
            scale=model_name == "Ridge",
        )
        pipeline.fit(x.iloc[train_index], y.iloc[train_index])
        prediction = pipeline.predict(x.iloc[validation_index])
        actual = y.iloc[validation_index]
        fold_metrics.append(
            {
                "mae": float(mean_absolute_error(actual, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
                "r2": float(r2_score(actual, prediction)),
            }
        )
    result = {}
    for metric in ("mae", "rmse", "r2"):
        values = np.array([fold[metric] for fold in fold_metrics], dtype=float)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=1))
        result[f"{metric}_folds"] = values.tolist()
    return result


def distribution(y: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
    }


def format_stats(stats: dict[str, float]) -> str:
    return (
        f"MAE={stats['mae']:.4f}, RMSE={stats['rmse']:.4f}, "
        f"R2={stats['r2']:.4f}"
    )


def composition_string(row: pd.Series, composition_columns: list[str]) -> str:
    parts = []
    for column in composition_columns:
        value = row[column]
        if pd.notna(value) and float(value) != 0:
            parts.append(f"{column[:-3]}={float(value):g}")
    return "; ".join(parts) if parts else "(no nonzero composition)"


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    train_frame = pd.read_csv(TRAIN_FILE)
    test_frame = pd.read_csv(TEST_FILE)
    if train_frame.columns.tolist() != test_frame.columns.tolist():
        raise ValueError("train.csv and test.csv columns/order must match.")
    if TARGET not in train_frame or TARGET not in test_frame:
        raise ValueError(f"Both files must include {TARGET}.")
    if train_frame[TARGET].isna().any() or test_frame[TARGET].isna().any():
        raise ValueError("Target values are missing; rows will not be dropped.")

    all_columns = train_frame.columns.tolist()
    composition_columns = [
        column for column in all_columns
        if column.endswith("_wt") and column not in COMPOSITION_EXCLUSIONS
    ]
    if not composition_columns:
        raise ValueError("No composition columns ending in '_wt' were found.")
    missing_engineered = sorted(set(ENGINEERED_FEATURES) - set(all_columns))
    if missing_engineered:
        raise ValueError(f"Missing engineered columns: {missing_engineered}")

    feature_sets = {
        "A": composition_columns.copy(),
        "B": composition_columns.copy() + ENGINEERED_FEATURES,
        "C": ENGINEERED_FEATURES.copy(),
    }
    for name, columns in feature_sets.items():
        if TARGET in columns or "ti_based" in columns or "phase_note" in columns:
            raise AssertionError(f"Forbidden feature in set {name}.")
        if any("id" in column.lower() or "source" in column.lower() for column in columns):
            raise AssertionError(f"Identifier/source feature found in set {name}.")

    x_train = train_frame.drop(columns=[TARGET])
    x_test = test_frame.drop(columns=[TARGET])
    y_train = train_frame[TARGET].astype(float)
    y_test = test_frame[TARGET].astype(float)
    group_ids = (
        train_frame[composition_columns]
        .round(1)
        .astype("string")
        .fillna("<NA>")
        .agg("|".join, axis=1)
    )
    splitter = GroupKFold(n_splits=N_SPLITS)
    folds = list(splitter.split(x_train, y_train, groups=group_ids))
    for train_index, validation_index in folds:
        train_groups = set(group_ids.iloc[train_index])
        validation_groups = set(group_ids.iloc[validation_index])
        if train_groups & validation_groups:
            raise AssertionError("Grouped CV leakage detected.")

    print(
        "Library versions: "
        f"pandas={pd.__version__}, numpy={np.__version__}, "
        f"scikit-learn={sklearn.__version__}, xgboost={xgboost.__version__}, "
        f"joblib={joblib.__version__}"
    )
    print(
        f"Rows: train={len(train_frame)}, test={len(test_frame)}; "
        f"unique train composition groups={group_ids.nunique()}"
    )

    models = [
        "XGBRegressor", "RandomForestRegressor", "Ridge", "DummyRegressor",
    ]
    cv_results = {}
    for set_name, feature_columns in feature_sets.items():
        category_columns = [
            column for column in ENGINEERED_CATEGORICAL
            if column in feature_columns
        ]
        for model_name in models:
            key = f"{set_name}/{model_name}/default"
            print(f"Cross-validating {key}...", flush=True)
            cv_results[key] = grouped_cv_scores(
                model_name,
                feature_columns,
                category_columns,
                x_train,
                y_train,
                folds,
            )

    parameter_distributions = {
        "model__n_estimators": randint(200, 1501),
        "model__learning_rate": uniform(0.01, 0.19),
        "model__max_depth": randint(2, 9),
        "model__min_child_weight": randint(1, 11),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_alpha": uniform(0.0, 5.0),
        "model__reg_lambda": uniform(0.0, 10.0),
    }
    tuned_results = {}
    tuned_params = {}
    for set_name in ("A", "B"):
        feature_columns = feature_sets[set_name]
        category_columns = [
            column for column in ENGINEERED_CATEGORICAL
            if column in feature_columns
        ]
        search_pipeline = make_pipeline(
            make_model("XGBRegressor"),
            feature_columns,
            category_columns,
            impute=False,
        )
        search = RandomizedSearchCV(
            estimator=search_pipeline,
            param_distributions=parameter_distributions,
            n_iter=SEARCH_ITERATIONS,
            scoring="neg_mean_absolute_error",
            cv=splitter,
            random_state=SEED,
            n_jobs=1,
            refit=False,
            error_score="raise",
            return_train_score=False,
        )
        print(
            f"Tuning XGBoost on feature set {set_name} "
            f"({SEARCH_ITERATIONS} randomized settings)...",
            flush=True,
        )
        search.fit(x_train[feature_columns], y_train, groups=group_ids)
        best_params = {
            key.removeprefix("model__"): value
            for key, value in search.best_params_.items()
        }
        tuned_params[set_name] = best_params
        tuned_results[set_name] = grouped_cv_scores(
            "XGBRegressor",
            feature_columns,
            category_columns,
            x_train,
            y_train,
            folds,
            best_params,
        )
        tuned_results[set_name]["search_best_cv_mae"] = float(-search.best_score_)
        cv_results[f"{set_name}/XGBRegressor/tuned"] = tuned_results[set_name]

    candidates = []
    for key, metrics in cv_results.items():
        set_name, model_name, parameterization = key.split("/")
        candidates.append(
            {
                "feature_set": set_name,
                "model": model_name,
                "parameterization": parameterization,
                "cv_mae": metrics["mae_mean"],
                "cv_mae_std": metrics["mae_std"],
            }
        )
    winner = min(candidates, key=lambda item: item["cv_mae"])
    final_set = winner["feature_set"]
    final_model_name = winner["model"]
    final_parameterization = winner["parameterization"]
    final_feature_columns = feature_sets[final_set]
    final_category_columns = [
        column for column in ENGINEERED_CATEGORICAL
        if column in final_feature_columns
    ]
    final_params = (
        tuned_params[final_set]
        if final_model_name == "XGBRegressor" and final_parameterization == "tuned"
        else None
    )
    final_model = make_model(final_model_name)
    if final_params:
        final_model.set_params(**final_params)
    final_pipeline = make_pipeline(
        final_model,
        final_feature_columns,
        final_category_columns,
        impute=final_model_name != "XGBRegressor",
        scale=final_model_name == "Ridge",
    )
    final_pipeline.fit(x_train[final_feature_columns], y_train)
    final_predictions = final_pipeline.predict(x_test[final_feature_columns])
    final_test_metrics = distribution(y_test, final_predictions)

    # Evaluate each requested feature set with XGBoost: tuned for A/B, default for C.
    set_test_metrics = {}
    set_pipelines = {}
    set_predictions = {}
    for set_name, feature_columns in feature_sets.items():
        category_columns = [
            column for column in ENGINEERED_CATEGORICAL
            if column in feature_columns
        ]
        model = make_model("XGBRegressor")
        params = tuned_params.get(set_name) if set_name in ("A", "B") else None
        if params:
            model.set_params(**params)
        pipeline = make_pipeline(
            model,
            feature_columns,
            category_columns,
            impute=False,
        )
        pipeline.fit(x_train[feature_columns], y_train)
        prediction = pipeline.predict(x_test[feature_columns])
        set_pipelines[set_name] = pipeline
        set_predictions[set_name] = prediction
        set_test_metrics[set_name] = {
            "model": "XGBRegressor",
            "parameterization": "tuned" if params else "default",
            **distribution(y_test, prediction),
        }

    beta_mask = test_frame["phase_est"].eq("beta")
    low_e_mask = y_test < 60.0
    subset_metrics = {}
    for subset_name, mask in (
        ("phase_est == beta", beta_mask),
        ("E_GPa < 60", low_e_mask),
    ):
        subset_metrics[subset_name] = {
            "rows": int(mask.sum()),
            "mae": (
                float(mean_absolute_error(y_test.loc[mask], final_predictions[mask.to_numpy()]))
                if mask.any() else None
            ),
        }

    absolute_errors = np.abs(y_test.to_numpy() - final_predictions)
    top_error_indices = np.argsort(absolute_errors)[::-1][:10]
    error_rows = []
    for test_position in top_error_indices:
        row = test_frame.iloc[test_position]
        error_rows.append(
            {
                "test_row": int(test_position + 2),
                "composition": composition_string(row, composition_columns),
                "true_E_GPa": float(y_test.iloc[test_position]),
                "predicted_E_GPa": float(final_predictions[test_position]),
                "absolute_error_GPa": float(absolute_errors[test_position]),
            }
        )

    # Feature importance uses the best-CV XGBoost configuration, independent of
    # the final model type; permutation importance is measured for the final model.
    xgb_candidates = [
        item for item in candidates if item["model"] == "XGBRegressor"
    ]
    best_xgb = min(xgb_candidates, key=lambda item: item["cv_mae"])
    importance_set = best_xgb["feature_set"]
    importance_columns = feature_sets[importance_set]
    importance_categories = [
        column for column in ENGINEERED_CATEGORICAL
        if column in importance_columns
    ]
    importance_model = make_model("XGBRegressor")
    importance_params = (
        tuned_params[importance_set]
        if importance_set in ("A", "B")
        and best_xgb["parameterization"] == "tuned"
        else None
    )
    if importance_params:
        importance_model.set_params(**importance_params)
    importance_pipeline = make_pipeline(
        importance_model,
        importance_columns,
        importance_categories,
        impute=False,
    )
    importance_pipeline.fit(x_train[importance_columns], y_train)
    transformed_names = importance_pipeline.named_steps[
        "preprocess"
    ].get_feature_names_out()
    gain_by_name = {name: 0.0 for name in transformed_names}
    booster_scores = importance_pipeline.named_steps["model"].get_booster().get_score(
        importance_type="gain"
    )
    for key, gain in booster_scores.items():
        if key.startswith("f") and key[1:].isdigit():
            feature_position = int(key[1:])
            if feature_position < len(transformed_names):
                gain_by_name[transformed_names[feature_position]] = float(gain)
        elif key in gain_by_name:
            gain_by_name[key] = float(gain)

    permutation = permutation_importance(
        final_pipeline,
        x_test[final_feature_columns],
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=SEED,
        n_jobs=1,
    )
    permutation_by_name = {
        column: {
            "mean": float(permutation.importances_mean[position]),
            "std": float(permutation.importances_std[position]),
        }
        for position, column in enumerate(final_feature_columns)
    }
    importance_rows = []
    all_importance_names = sorted(set(gain_by_name) | set(permutation_by_name))
    for name in all_importance_names:
        permutation_data = permutation_by_name.get(name, {})
        importance_rows.append(
            {
                "feature": name,
                "xgb_gain": gain_by_name.get(name, np.nan),
                "permutation_importance_mean": permutation_data.get("mean", np.nan),
                "permutation_importance_std": permutation_data.get("std", np.nan),
                "gain_feature_set": importance_set,
                "gain_model": (
                    "XGBRegressor tuned"
                    if importance_params else "XGBRegressor default"
                ),
                "permutation_model": (
                    f"{final_model_name} {final_parameterization}, set {final_set}"
                ),
            }
        )
    pd.DataFrame(importance_rows).to_csv(IMPORTANCE_FILE, index=False)

    # Compare the selected tuned-XGB A/B CV MAEs against the observed fold spread.
    a_maes = np.array(tuned_results["A"]["mae_folds"])
    b_maes = np.array(tuned_results["B"]["mae_folds"])
    paired_improvement = float(np.mean(a_maes - b_maes))
    b_fold_std = float(np.std(b_maes, ddof=1))
    engineered_beat_threshold = paired_improvement > b_fold_std
    comparison = {
        "feature_set_A_tuned_xgb_cv_mae_mean": float(a_maes.mean()),
        "feature_set_A_tuned_xgb_cv_mae_std": float(a_maes.std(ddof=1)),
        "feature_set_B_tuned_xgb_cv_mae_mean": float(b_maes.mean()),
        "feature_set_B_tuned_xgb_cv_mae_std": b_fold_std,
        "paired_mean_mae_improvement_A_minus_B": paired_improvement,
        "threshold_one_std_of_B_fold_MAE": b_fold_std,
        "engineered_features_beat_A_by_more_than_one_std": bool(
            engineered_beat_threshold
        ),
    }

    results = {
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
            "scipy": scipy.__version__,
        },
        "random_state": SEED,
        "rows": {
            "train": int(len(train_frame)),
            "test": int(len(test_frame)),
            "train_groups_rounded_1_decimal": int(group_ids.nunique()),
            "cv_splits": N_SPLITS,
        },
        "feature_sets": {
            name: columns for name, columns in feature_sets.items()
        },
        "cv_results": cv_results,
        "tuning": {
            "search": "RandomizedSearchCV",
            "n_iter": SEARCH_ITERATIONS,
            "random_state": SEED,
            "scoring": "neg_mean_absolute_error",
            "feature_sets_tuned": ["A", "B"],
            "best_params": tuned_params,
            "best_cv_mae": {
                name: tuned_results[name]["search_best_cv_mae"]
                for name in ("A", "B")
            },
        },
        "selected_model": {
            **winner,
            "feature_columns": final_feature_columns,
            "parameters": tuned_params.get(final_set)
            if final_model_name == "XGBRegressor"
            and final_parameterization == "tuned"
            else "default parameters",
        },
        "test_metrics": {
            "final_model": final_test_metrics,
            "xgboost_by_feature_set": set_test_metrics,
            "final_model_subsets": subset_metrics,
        },
        "top_10_largest_absolute_test_errors": error_rows,
        "importance": {
            "gain_model": f"XGBRegressor, set {importance_set}, "
            f"{best_xgb['parameterization']} parameters; gain on fitted train model",
            "permutation_model": (
                f"final model ({final_model_name}, set {final_set}); "
                f"measured on test set with {PERMUTATION_REPEATS} repeats"
            ),
            "file": IMPORTANCE_FILE.name,
        },
        "engineered_feature_comparison": comparison,
        "choices": [
            "The supplied train.csv was used as-is, including rows previously appended from dropped_non_ti.csv; no rows were dropped.",
            "Group IDs use every *_wt column rounded to 1 decimal, with missing group components represented by <NA>.",
            "XGBoost receives native numeric NaNs; RandomForest, Ridge, and Dummy preprocessing imputes numeric columns using each training fold's median.",
            "Ridge numeric features are standardized after fold-local median imputation; tree model features are not scaled.",
            "For requested A/B/C test-set comparison, XGBoost is used; A and B use their tuned best parameters and C uses default XGBoost parameters. The selected model is additionally reported separately.",
            "Importance output combines gain from the best-CV XGBoost configuration and test permutation importance from the selected final model; unmatched feature names are left blank.",
        ],
    }
    RESULTS_JSON.write_text(
        json.dumps(json_safe(results), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "Titanium-alloy Young's modulus model results",
        "=" * 45,
        "Library versions:",
        *[
            f"- {name}: {version}"
            for name, version in results["library_versions"].items()
        ],
        f"- Random state: {SEED}",
        "",
        f"Rows: train={len(train_frame)}, test={len(test_frame)}; "
        f"unique rounded-composition train groups={group_ids.nunique()}",
        "",
        "Grouped 5-fold CV on train.csv (mean +/- standard deviation):",
    ]
    for key, metrics in cv_results.items():
        report_lines.append(
            f"- {key}: "
            f"MAE={metrics['mae_mean']:.4f} +/- {metrics['mae_std']:.4f}; "
            f"RMSE={metrics['rmse_mean']:.4f} +/- {metrics['rmse_std']:.4f}; "
            f"R2={metrics['r2_mean']:.4f} +/- {metrics['r2_std']:.4f}"
        )
    report_lines.extend(
        [
            "",
            "Randomized XGBoost tuning (A and B; 60 iterations each):",
        ]
    )
    for name in ("A", "B"):
        report_lines.append(
            f"- Set {name}: best CV MAE={tuned_results[name]['search_best_cv_mae']:.4f}; "
            f"best params={json.dumps(json_safe(tuned_params[name]), sort_keys=True)}"
        )
    report_lines.extend(
        [
            "",
            "Selected by lowest grouped-CV MAE:",
            f"- Feature set {final_set}, {final_model_name} ({final_parameterization}), "
            f"CV MAE={winner['cv_mae']:.4f} +/- {winner['cv_mae_std']:.4f}",
            "",
            "Held-out test metrics (evaluated once after all choices were fixed):",
            f"- Selected final model: {format_stats(final_test_metrics)}",
            "- Feature-set comparison with XGBoost:",
        ]
    )
    for set_name in ("A", "B", "C"):
        metrics = set_test_metrics[set_name]
        report_lines.append(
            f"  - {set_name} ({metrics['parameterization']} parameters): "
            f"{format_stats(metrics)}"
        )
    report_lines.extend(["", "Selected-model test subsets (MAE only):"])
    for name, metrics in subset_metrics.items():
        value = "not available" if metrics["mae"] is None else f"{metrics['mae']:.4f} GPa"
        report_lines.append(f"- {name}: {value} (n={metrics['rows']})")
    report_lines.extend(
        [
            "",
            "10 test rows with largest absolute error (composition wt%, E in GPa):",
        ]
    )
    for rank, row in enumerate(error_rows, start=1):
        report_lines.append(
            f"{rank}. row {row['test_row']}: {row['composition']}; "
            f"true={row['true_E_GPa']:.2f}; predicted={row['predicted_E_GPa']:.2f}; "
            f"|error|={row['absolute_error_GPa']:.2f}"
        )
    report_lines.extend(
        [
            "",
            "Engineered-feature comparison A vs B (tuned XGBoost, paired folds):",
            f"- Mean paired MAE improvement (A minus B): {paired_improvement:.4f} GPa",
            f"- One standard deviation of B fold MAEs: {b_fold_std:.4f} GPa",
            (
                "- Does B beat A by more than one standard deviation? yes"
                if engineered_beat_threshold
                else "- Does B beat A by more than one standard deviation? no"
            ),
        ]
    )
    if not engineered_beat_threshold:
        report_lines.append("- The engineered features aren't clearly helping under this criterion.")
    report_lines.extend(
        [
            "",
            "Saved artifacts:",
            f"- Model pipeline: {MODEL_FILE.resolve()}",
            f"- Exact selected feature columns: {FEATURE_COLUMNS_FILE.resolve()}",
            f"- Feature importance: {IMPORTANCE_FILE.resolve()}",
            f"- JSON results: {RESULTS_JSON.resolve()}",
            "",
            "Choices:",
            *[f"- {choice}" for choice in results["choices"]],
        ]
    )
    report = "\n".join(report_lines) + "\n"
    RESULTS_TEXT.write_text(report, encoding="utf-8")
    print("\n" + report)

    joblib.dump(final_pipeline, MODEL_FILE)
    FEATURE_COLUMNS_FILE.write_text(
        json.dumps(final_feature_columns, indent=2)
        + "\n",
        encoding="utf-8",
    )

    # Reload and validate that the persisted raw-DataFrame pipeline predicts.
    saved_pipeline = joblib.load(MODEL_FILE)
    saved_predictions = saved_pipeline.predict(x_test[final_feature_columns])
    if not np.allclose(saved_predictions, final_predictions):
        raise AssertionError("Saved pipeline predictions differ from fitted pipeline.")
    if len(saved_predictions) != len(test_frame):
        raise AssertionError("Saved pipeline did not predict every test row.")
    print(
        f"Saved and reloaded model validated on {len(saved_predictions)} raw test rows."
    )


if __name__ == "__main__":
    main()
