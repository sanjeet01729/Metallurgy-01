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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


BASE = Path(__file__).parent
BASE_FILE = BASE / "combined_ym_features.csv"
DFT_FILE = BASE / "ti_dft_materials_project.csv"
OUTPUT_BASE = BASE / "dataset_v2.csv"
REMOVED_DFT_FILE = BASE / "removed_dft_v2.csv"
REMOVED_NON_TI_FILE = BASE / "removed_non_ti_v2.csv"
GROUP_STATS_FILE = BASE / "e_spread_v2.csv"
TRAIN_FILE = BASE / "train_v2.csv"
TEST_FILE = BASE / "test_v2.csv"
REPORT_TEXT = BASE / "results_v2.txt"
REPORT_JSON = BASE / "results_v2.json"
MODEL_FILE = BASE / "ti_ym_model_v2.joblib"
FEATURE_COLUMNS_FILE = BASE / "feature_columns_v2.json"
IMPORTANCE_FILE = BASE / "feature_importance_v2.csv"

SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 5
N_ITER = 60
TARGET = "E_GPa"
MATCH_TOLERANCE = 0.05
OLD_TEST_MAE = 11.53
OLD_CV_MAE = 11.47

ATOMIC_MASS = {
    "Ti": 47.867, "Nb": 92.906, "Zr": 91.224, "Co": 58.933,
    "Mo": 95.95, "Sn": 118.71, "Ta": 180.948, "Al": 26.982,
    "V": 50.942, "Fe": 55.845, "Cr": 51.996, "Hf": 178.49,
    "Cu": 63.546, "Mn": 54.938044, "W": 183.84, "Sc": 44.955908,
    "Si": 28.085, "O": 15.999, "Pd": 106.42,
}

FIGURE_DFT_ROWS = [
    {"Ti": 88.55, "Nb": 11.45, "E_GPa": 39.0},
    {"Ti": 78.30, "Nb": 21.70, "E_GPa": 58.5},
    {"Ti": 69.07, "Nb": 30.93, "E_GPa": 60.5},
    {"Ti": 60.72, "Nb": 39.28, "E_GPa": 81.0},
    {"Ti": 53.13, "Nb": 46.87, "E_GPa": 70.0},
    {"Ti": 46.20, "Nb": 53.80, "E_GPa": 69.5},
    {"Ti": 34.01, "Nb": 65.99, "E_GPa": 70.0},
    {"Ti": 53.22, "Nb": 37.56, "Zr": 9.22, "E_GPa": 54.0},
    {"Ti": 46.35, "Nb": 35.98, "Zr": 17.67, "E_GPa": 63.0},
    {"Ti": 40.03, "Nb": 34.53, "Zr": 25.43, "E_GPa": 61.5},
    {"Ti": 34.21, "Nb": 33.20, "Zr": 32.60, "E_GPa": 60.5},
]

ENGINEERED_NUMERIC = [
    "VEC", "delta_r", "delta_S_mix", "Mo_eq", "Al_eq", "Bo", "Md",
]
ENGINEERED_CATEGORICAL = ["phase_est"]


def composition_key(frame: pd.DataFrame, composition_columns: list[str]) -> pd.Series:
    rounded = frame[composition_columns].round(1)

    def format_value(value):
        return "<NA>" if pd.isna(value) else f"{float(value):.1f}"

    return rounded.apply(
        lambda row: "|".join(format_value(value) for value in row),
        axis=1,
    )


def dft_rows_as_weight_percent(
    dft: pd.DataFrame, output_elements: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    at_columns = [column for column in dft if column.endswith("_at")]
    if not at_columns:
        wt_columns = [column for column in dft if column.endswith("_wt")]
        if not wt_columns:
            raise ValueError("DFT file has no recognized _at or _wt composition columns.")
        values = np.zeros((len(dft), len(output_elements)), dtype=float)
        for column in wt_columns:
            element = column[:-3]
            if element in output_elements:
                values[:, output_elements.index(element)] = pd.to_numeric(
                    dft[column], errors="coerce"
                ).fillna(0).to_numpy(dtype=float)
        row_sums = values.sum(axis=1)
        valid = np.isfinite(values).all(axis=1) & (row_sums > 0)
        values[valid] = values[valid] / row_sums[valid, None] * 100.0
        energy = pd.to_numeric(dft[TARGET], errors="coerce").to_numpy(dtype=float)
        return values, energy

    values = np.zeros((len(dft), len(output_elements)), dtype=float)
    for column in at_columns:
        element = column[:-3]
        if element not in ATOMIC_MASS:
            raise ValueError(
                f"Atomic mass not available for DFT composition element {element!r}."
            )
        if element in output_elements:
            amounts = pd.to_numeric(dft[column], errors="coerce").fillna(0).to_numpy(dtype=float)
            values[:, output_elements.index(element)] = amounts * ATOMIC_MASS[element]
    row_sums = values.sum(axis=1)
    valid = np.isfinite(values).all(axis=1) & (row_sums > 0)
    values[valid] = values[valid] / row_sums[valid, None] * 100.0
    values[~valid] = np.nan
    energy = pd.to_numeric(dft[TARGET], errors="coerce").to_numpy(dtype=float)
    return values, energy


def figure_rows_as_weight_percent(
    figure_rows: list[dict], output_elements: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(figure_rows), len(output_elements)), dtype=float)
    energies = np.empty(len(figure_rows), dtype=float)
    for row_index, row in enumerate(figure_rows):
        total = sum(value for element, value in row.items() if element != TARGET)
        if total <= 0:
            raise ValueError(f"Invalid figure composition: {row}")
        for element, value in row.items():
            if element == TARGET:
                continue
            if element not in output_elements:
                raise ValueError(f"Figure element {element!r} is absent from base dataset.")
            values[row_index, output_elements.index(element)] = value / total * 100.0
        energies[row_index] = row[TARGET]
    return values, energies


def mark_matches(
    data: pd.DataFrame,
    comp_values: np.ndarray,
    e_values: np.ndarray,
    composition_columns: list[str],
) -> np.ndarray:
    source_comp = data[composition_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    source_e = pd.to_numeric(data[TARGET], errors="coerce").to_numpy(dtype=float)
    matches = np.zeros(len(data), dtype=bool)
    for candidate_comp, candidate_e in zip(comp_values, e_values):
        if not np.isfinite(candidate_e) or not np.isfinite(candidate_comp).all():
            continue
        row_matches = (
            np.isfinite(source_e)
            & (np.abs(source_e - candidate_e) <= MATCH_TOLERANCE + 1e-12)
        )
        row_matches &= np.isfinite(source_comp).all(axis=1)
        row_matches &= (
            np.abs(source_comp - candidate_comp[None, :])
            <= MATCH_TOLERANCE + 1e-12
        ).all(axis=1)
        matches |= row_matches
    return matches


def make_preprocessor(
    feature_columns: list[str],
    categorical_columns: list[str],
    *,
    impute: bool,
) -> ColumnTransformer:
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    numeric_transformer = "passthrough"
    if impute and numeric_columns:
        numeric_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                )
            ]
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
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    feature_columns,
                    categorical_columns,
                    impute=impute,
                ),
            ),
            ("model", model),
        ]
    )


def make_xgb(objective: str) -> XGBRegressor:
    return XGBRegressor(
        objective=objective,
        eval_metric="mae",
        random_state=SEED,
        n_jobs=1,
        tree_method="hist",
        verbosity=0,
    )


def make_model(model_name: str, objective: str | None = None):
    if model_name == "XGBRegressor":
        if objective is None:
            raise ValueError("XGBoost objective is required.")
        return make_xgb(objective)
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=300,
            random_state=SEED,
            n_jobs=1,
        )
    if model_name == "DummyRegressor":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown model: {model_name}")


def cv_metrics(
    pipeline_factory,
    x: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    fold_results = []
    for train_index, validation_index in folds:
        pipeline = pipeline_factory()
        pipeline.fit(x.iloc[train_index], y.iloc[train_index])
        prediction = pipeline.predict(x.iloc[validation_index])
        actual = y.iloc[validation_index]
        fold_results.append(
            {
                "mae": float(mean_absolute_error(actual, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
                "r2": float(r2_score(actual, prediction)),
            }
        )
    metrics = {}
    for metric in ("mae", "rmse", "r2"):
        values = np.array([item[metric] for item in fold_results], dtype=float)
        metrics[f"{metric}_mean"] = float(values.mean())
        metrics[f"{metric}_std"] = float(values.std(ddof=1))
        metrics[f"{metric}_folds"] = values.tolist()
    return metrics


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    base_data = pd.read_csv(BASE_FILE)
    dft = pd.read_csv(DFT_FILE)
    if TARGET not in base_data or TARGET not in dft:
        raise ValueError(f"Both source files must contain {TARGET}.")
    composition_columns = [
        column for column in base_data.columns if column.endswith("_wt")
    ]
    elements = [column[:-3] for column in composition_columns]
    if "Ti" not in elements:
        raise ValueError("Base dataset must have a Ti_wt column.")

    dft_compositions, dft_energies = dft_rows_as_weight_percent(dft, elements)
    dft_matches = mark_matches(
        base_data, dft_compositions, dft_energies, composition_columns
    )
    removed_rule_a = base_data.loc[dft_matches].copy()
    remaining = base_data.loc[~dft_matches].copy()
    figure_comp, figure_energy = figure_rows_as_weight_percent(
        FIGURE_DFT_ROWS, elements
    )
    figure_matches = mark_matches(
        remaining, figure_comp, figure_energy, composition_columns
    )
    removed_rule_b = remaining.loc[figure_matches].copy()
    retained = remaining.loc[~figure_matches].copy()
    removed_dft = pd.concat(
        [
            removed_rule_a.assign(removal_rule="a: matched DFT source"),
            removed_rule_b.assign(removal_rule="b: matched figure DFT row"),
        ],
        ignore_index=True,
    )
    removed_dft.to_csv(REMOVED_DFT_FILE, index=False)

    composition = retained[composition_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    weights = composition.to_numpy(dtype=float)
    masses = np.array([ATOMIC_MASS[element] for element in elements], dtype=float)
    composition_nan = ~np.isfinite(weights).all(axis=1)
    if composition_nan.any():
        raise ValueError(
            f"{int(composition_nan.sum())} retained rows have missing/non-finite "
            "composition values; Ti_at cannot be calculated without changing/dropping them."
        )
    if (weights < 0).any():
        raise ValueError("Negative composition values found; refusing atomic-fraction filtering.")
    row_totals = weights.sum(axis=1)
    if (row_totals <= 0).any():
        raise ValueError("Zero-total composition found; refusing atomic-fraction filtering.")
    mole_amounts = weights / masses[None, :]
    atomic_fractions = mole_amounts / mole_amounts.sum(axis=1, keepdims=True)
    ti_at = atomic_fractions[:, elements.index("Ti")] * 100.0
    retained = retained.copy()
    retained["Ti_at"] = ti_at
    keep_ti = ti_at >= 60.0
    removed_non_ti = retained.loc[~keep_ti].copy()
    removed_non_ti.to_csv(REMOVED_NON_TI_FILE, index=False)
    ti_data = retained.loc[keep_ti].copy()

    keys = composition_key(ti_data, composition_columns)
    group_positions: dict[str, list[int]] = {}
    for position, group_id in enumerate(keys.tolist()):
        group_positions.setdefault(group_id, []).append(position)

    collapsed_rows = []
    group_stats = []
    for group_id, positions in group_positions.items():
        group = ti_data.iloc[positions]
        first_row = group.iloc[0].copy()
        energies = pd.to_numeric(group[TARGET], errors="coerce")
        first_row[TARGET] = float(energies.median())
        collapsed_rows.append(first_row)
        group_stats.append(
            {
                "group_id": group_id,
                "n_rows": int(len(group)),
                "E_min": float(energies.min()),
                "E_max": float(energies.max()),
                "E_std": float(energies.std(ddof=1)) if len(group) > 1 else np.nan,
            }
        )

    dataset = pd.DataFrame(collapsed_rows).reset_index(drop=True)
    group_stats_frame = pd.DataFrame(group_stats)
    group_stats_frame.to_csv(GROUP_STATS_FILE, index=False)
    dataset.to_csv(OUTPUT_BASE, index=False)
    multirow_stats = group_stats_frame.loc[group_stats_frame["n_rows"] > 1]
    mean_within_group_std = (
        float(multirow_stats["E_std"].mean()) if len(multirow_stats) else 0.0
    )

    dataset_keys = composition_key(dataset, composition_columns)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_index, test_index = next(
        splitter.split(dataset, groups=dataset_keys)
    )
    train = dataset.iloc[train_index].copy()
    test = dataset.iloc[test_index].copy()
    train_keys = dataset_keys.iloc[train_index]
    test_keys = dataset_keys.iloc[test_index]
    train_mean = float(train[TARGET].mean())
    test_mean = float(test[TARGET].mean())
    split_seed = SEED
    if abs(train_mean - test_mean) > 5.0:
        options = []
        for seed in range(43, 53):
            candidate_splitter = GroupShuffleSplit(
                n_splits=1, test_size=TEST_SIZE, random_state=seed
            )
            candidate_train_index, candidate_test_index = next(
                candidate_splitter.split(dataset, groups=dataset_keys)
            )
            candidate_train = dataset.iloc[candidate_train_index]
            candidate_test = dataset.iloc[candidate_test_index]
            delta = abs(
                float(candidate_train[TARGET].mean())
                - float(candidate_test[TARGET].mean())
            )
            options.append(
                (
                    delta,
                    seed,
                    candidate_train_index,
                    candidate_test_index,
                )
            )
        _, split_seed, train_index, test_index = min(
            options, key=lambda item: (item[0], item[1])
        )
        train = dataset.iloc[train_index].copy()
        test = dataset.iloc[test_index].copy()
        train_keys = dataset_keys.iloc[train_index]
        test_keys = dataset_keys.iloc[test_index]
    shared = set(train_keys) & set(test_keys)
    assert not shared, f"Composition leakage between train and test: {len(shared)} groups."
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    feature_a = composition_columns.copy()
    feature_b = feature_a + ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL
    for features in (feature_a, feature_b):
        if set(features) & {TARGET, "Ti_at", "ti_based", "phase_note"}:
            raise AssertionError("Forbidden feature included.")
    train_x = train.drop(columns=[TARGET])
    train_y = train[TARGET].astype(float)
    test_x = test.drop(columns=[TARGET])
    test_y = test[TARGET].astype(float)
    cv_groups = composition_key(train, composition_columns)
    group_kfold = GroupKFold(n_splits=N_SPLITS)
    folds = list(group_kfold.split(train_x, train_y, groups=cv_groups))
    for fold_train, fold_validation in folds:
        if set(cv_groups.iloc[fold_train]) & set(cv_groups.iloc[fold_validation]):
            raise AssertionError("Composition leakage in GroupKFold.")

    versions = {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "joblib": joblib.__version__,
        "scipy": scipy.__version__,
    }
    print("Library versions:", json.dumps(versions, sort_keys=True))

    feature_sets = {"A": feature_a, "B": feature_b}
    categorical_by_set = {
        name: [column for column in ENGINEERED_CATEGORICAL if column in columns]
        for name, columns in feature_sets.items()
    }
    cv_results = {}
    objective_cv_means = {objective: [] for objective in ("reg:absoluteerror", "reg:squarederror")}

    def evaluate_model(model_name, feature_name, objective=None):
        feature_columns = feature_sets[feature_name]
        categorical = categorical_by_set[feature_name]
        impute = model_name != "XGBRegressor"

        def factory():
            model = make_model(model_name, objective)
            return make_pipeline(
                model,
                feature_columns,
                categorical,
                impute=impute,
            )

        label = f"{feature_name}/{model_name}"
        if objective:
            label += f"/{objective}"
        print(f"Cross-validating {label}...", flush=True)
        scores = cv_metrics(
            factory,
            train_x[feature_columns],
            train_y,
            folds,
        )
        cv_results[label] = scores
        return scores

    for feature_name in ("A", "B"):
        for objective in ("reg:absoluteerror", "reg:squarederror"):
            scores = evaluate_model("XGBRegressor", feature_name, objective)
            objective_cv_means[objective].append(scores["mae_mean"])
        evaluate_model("RandomForestRegressor", feature_name)
        evaluate_model("DummyRegressor", feature_name)

    selected_objective = min(
        objective_cv_means,
        key=lambda objective: (
            float(np.mean(objective_cv_means[objective])),
            objective,
        ),
    )

    search_space = {
        "model__n_estimators": randint(200, 1501),
        "model__learning_rate": uniform(0.01, 0.19),
        "model__max_depth": randint(2, 9),
        "model__min_child_weight": randint(1, 11),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_alpha": uniform(0.0, 5.0),
        "model__reg_lambda": uniform(0.0, 10.0),
    }
    tuning = {}
    for feature_name in ("A", "B"):
        feature_columns = feature_sets[feature_name]
        pipeline = make_pipeline(
            make_xgb(selected_objective),
            feature_columns,
            categorical_by_set[feature_name],
            impute=False,
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=search_space,
            n_iter=N_ITER,
            scoring="neg_mean_absolute_error",
            cv=group_kfold,
            random_state=SEED,
            n_jobs=1,
            refit=False,
            error_score="raise",
            return_train_score=False,
        )
        print(
            f"Tuning {selected_objective} on feature set {feature_name} "
            f"({N_ITER} randomized settings)...",
            flush=True,
        )
        search.fit(
            train_x[feature_columns],
            train_y,
            groups=cv_groups,
        )
        parameters = {
            key.removeprefix("model__"): value
            for key, value in search.best_params_.items()
        }
        tuning[feature_name] = {
            "best_params": parameters,
            "best_search_cv_mae": float(-search.best_score_),
        }

        def tuned_factory():
            model = make_xgb(selected_objective)
            model.set_params(**parameters)
            return make_pipeline(
                model,
                feature_columns,
                categorical_by_set[feature_name],
                impute=False,
            )

        label = f"{feature_name}/XGBRegressor/{selected_objective}/tuned"
        cv_results[label] = cv_metrics(
            tuned_factory,
            train_x[feature_columns],
            train_y,
            folds,
        )
        cv_results[label]["search_best_cv_mae"] = float(-search.best_score_)

    candidates = []
    for label, scores in cv_results.items():
        parts = label.split("/")
        feature_name, model_name = parts[:2]
        objective = parts[2] if model_name == "XGBRegressor" else None
        tuned = len(parts) > 3 and parts[3] == "tuned"
        candidates.append(
            {
                "label": label,
                "feature_set": feature_name,
                "model": model_name,
                "objective": objective,
                "parameterization": "tuned" if tuned else "default",
                "cv_mae": scores["mae_mean"],
                "cv_mae_std": scores["mae_std"],
            }
        )
    selected = min(candidates, key=lambda item: item["cv_mae"])
    selected_set = selected["feature_set"]
    selected_features = feature_sets[selected_set]
    selected_categories = categorical_by_set[selected_set]
    selected_parameters = (
        tuning[selected_set]["best_params"]
        if selected["model"] == "XGBRegressor"
        and selected["parameterization"] == "tuned"
        else None
    )
    final_model = make_model(
        selected["model"],
        selected["objective"] if selected["model"] == "XGBRegressor" else None,
    )
    if selected_parameters:
        final_model.set_params(**selected_parameters)
    final_pipeline = make_pipeline(
        final_model,
        selected_features,
        selected_categories,
        impute=selected["model"] != "XGBRegressor",
    )
    final_pipeline.fit(train_x[selected_features], train_y)
    final_predictions = final_pipeline.predict(test_x[selected_features])
    test_metrics = metrics(test_y, final_predictions)

    subsets = {}
    subset_masks = {
        "phase_est == beta": test["phase_est"].eq("beta"),
        "E_GPa < 60": test_y < 60.0,
        "Ti_wt < 70": test["Ti_wt"] < 70.0,
    }
    for name, mask in subset_masks.items():
        subsets[name] = {
            "rows": int(mask.sum()),
            "mae": (
                float(mean_absolute_error(test_y.loc[mask], final_predictions[mask.to_numpy()]))
                if mask.any() else None
            ),
        }

    # Identify the same rounded-composition group stats for each collapsed test row.
    group_lookup = group_stats_frame.set_index("group_id").to_dict(orient="index")
    worst_positions = np.argsort(
        np.abs(test_y.to_numpy() - final_predictions)
    )[::-1][:10]
    worst_rows = []
    for position in worst_positions:
        row = test.iloc[position]
        group_id = composition_key(
            pd.DataFrame([row]), composition_columns
        ).iloc[0]
        group_stat = group_lookup[group_id]
        composition_description = ", ".join(
            f"{column[:-3]}={float(row[column]):g}"
            for column in composition_columns
            if pd.notna(row[column]) and float(row[column]) != 0
        )
        worst_rows.append(
            {
                "test_row": int(position + 2),
                "group_id": group_id,
                "composition": composition_description,
                "true_E_GPa": float(test_y.iloc[position]),
                "predicted_E_GPa": float(final_predictions[position]),
                "absolute_error_GPa": float(
                    abs(test_y.iloc[position] - final_predictions[position])
                ),
                "n_rows": int(group_stat["n_rows"]),
                "E_std": (
                    float(group_stat["E_std"])
                    if pd.notna(group_stat["E_std"])
                    else None
                ),
            }
        )

    # Compare tuned A/B folds for the selected XGBoost objective.
    tuned_a = cv_results[
        f"A/XGBRegressor/{selected_objective}/tuned"
    ]["mae_folds"]
    tuned_b = cv_results[
        f"B/XGBRegressor/{selected_objective}/tuned"
    ]["mae_folds"]
    paired_improvement = float(np.mean(np.array(tuned_a) - np.array(tuned_b)))
    fold_std = float(np.std(tuned_b, ddof=1))
    b_beats_a = paired_improvement > fold_std
    noise_comparison = {
        "mean_within_group_E_std_GPa": mean_within_group_std,
        "final_test_mae_GPa": test_metrics["mae"],
        "test_mae_minus_noise_floor_GPa": test_metrics["mae"] - mean_within_group_std,
        "test_mae_to_noise_floor_ratio": (
            test_metrics["mae"] / mean_within_group_std
            if mean_within_group_std > 0 else None
        ),
    }

    transformed_names = final_pipeline.named_steps[
        "preprocess"
    ].get_feature_names_out()
    estimator = final_pipeline.named_steps["model"]
    if selected["model"] == "XGBRegressor":
        scores = estimator.get_booster().get_score(importance_type="gain")
        importance_by_name = {name: 0.0 for name in transformed_names}
        for name, score in scores.items():
            if name.startswith("f") and name[1:].isdigit():
                index = int(name[1:])
                if index < len(transformed_names):
                    importance_by_name[transformed_names[index]] = float(score)
            elif name in importance_by_name:
                importance_by_name[name] = float(score)
        importance_type = "xgboost_gain"
    else:
        importance_by_name = {
            name: float(value)
            for name, value in zip(transformed_names, estimator.feature_importances_)
        }
        importance_type = "random_forest_impurity"
    pd.DataFrame(
        [
            {
                "feature": name,
                "importance": importance_by_name[name],
                "importance_type": importance_type,
                "feature_set": selected_set,
                "model": selected["model"],
                "objective": selected["objective"],
            }
            for name in transformed_names
        ]
    ).sort_values("importance", ascending=False).to_csv(
        IMPORTANCE_FILE, index=False
    )

    # Save the model pipeline and its raw-data feature contract.
    joblib.dump(final_pipeline, MODEL_FILE)
    FEATURE_COLUMNS_FILE.write_text(
        json.dumps(selected_features, indent=2) + "\n",
        encoding="utf-8",
    )

    phase_counts = {
        "train": {
            str(key): int(value)
            for key, value in train["phase_est"].value_counts(dropna=False).items()
        },
        "test": {
            str(key): int(value)
            for key, value in test["phase_est"].value_counts(dropna=False).items()
        },
    }
    split_statistics = {}
    for name, frame in (("train", train), ("test", test)):
        split_statistics[name] = {
            "rows": int(len(frame)),
            "percent": float(len(frame) / len(dataset) * 100.0),
            "E_GPa": {
                "min": float(frame[TARGET].min()),
                "max": float(frame[TARGET].max()),
                "mean": float(frame[TARGET].mean()),
                "std": float(frame[TARGET].std()),
            },
            "Ti_wt_below_70_rows": int((frame["Ti_wt"] < 70.0).sum()),
        }

    results = {
        "library_versions": versions,
        "random_state": SEED,
        "df_exclusion": {
            "base_rows": int(len(base_data)),
            "source_dft_rows": int(len(dft)),
            "figure_rows_specified": int(len(FIGURE_DFT_ROWS)),
            "removed_by_rule_a": int(len(removed_rule_a)),
            "removed_by_rule_b": int(len(removed_rule_b)),
            "removed_total_unique_rows": int(len(removed_dft)),
        },
        "ti_matrix_filter": {
            "rows_after_dft_removal": int(len(retained)),
            "retained_ti_at_ge_60": int(len(ti_data)),
            "removed_ti_at_lt_60": int(len(removed_non_ti)),
            "threshold": 60.0,
        },
        "collapse": {
            "rows_before_collapse": int(len(ti_data)),
            "rows_after_collapse": int(len(dataset)),
            "groups_with_n_rows_gt_1": int(len(multirow_stats)),
            "mean_within_group_E_std_GPa": mean_within_group_std,
            "std_definition": "Sample standard deviation (ddof=1), averaged across groups with n_rows > 1.",
        },
        "split": {
            "seed_used": split_seed,
            "group_shuffle_split_test_size": TEST_SIZE,
            "shared_groups": int(len(shared)),
            "statistics": split_statistics,
            "phase_counts": phase_counts,
        },
        "feature_sets": {"A": feature_a, "B": feature_b},
        "cross_validation": {
            "folds": N_SPLITS,
            "groups": "Composition key from all *_wt columns rounded to 1 decimal.",
            "metrics": cv_results,
        },
        "objective_selection": {
            "default_cv_mae_mean_by_objective": {
                objective: float(np.mean(values))
                for objective, values in objective_cv_means.items()
            },
            "selected_objective_for_tuning": selected_objective,
        },
        "tuning": {
            "method": "RandomizedSearchCV",
            "iterations_per_feature_set": N_ITER,
            "scoring": "neg_mean_absolute_error",
            "random_state": SEED,
            "results": tuning,
        },
        "selected_model": {
            **selected,
            "feature_columns": selected_features,
            "parameters": selected_parameters or "default parameters",
        },
        "test_metrics": test_metrics,
        "test_subset_mae": subsets,
        "label_noise_comparison": noise_comparison,
        "engineered_feature_comparison": {
            "comparison": "Tuned XGBoost on A vs B using the selected objective and paired GroupKFold folds.",
            "mean_paired_mae_improvement_A_minus_B_GPa": paired_improvement,
            "one_std_of_B_fold_MAE_GPa": fold_std,
            "B_beats_A_by_more_than_one_fold_std": bool(b_beats_a),
        },
        "worst_test_rows": worst_rows,
        "importance": {
            "file": IMPORTANCE_FILE.name,
            "importance_type": importance_type,
            "model": selected["model"],
            "feature_set": selected_set,
        },
        "before_after_previous_run": {
            "old_test_mae_GPa": OLD_TEST_MAE,
            "new_test_mae_GPa": test_metrics["mae"],
            "test_mae_change_GPa": test_metrics["mae"] - OLD_TEST_MAE,
            "old_cv_mae_GPa": OLD_CV_MAE,
            "new_selected_cv_mae_GPa": selected["cv_mae"],
            "cv_mae_change_GPa": selected["cv_mae"] - OLD_CV_MAE,
        },
        "choices": [
            "DFT row matching checks every base *_wt element plus E_GPa within inclusive absolute tolerance 0.05; missing elements are zero.",
            "DFT at% compositions are converted to weight percentages with standard atomic masses, then normalized to 100 wt%.",
            "The explicit figure rows are treated as wt% and normalized before matching; their listed values are already near 100.",
            "The same composition/evaluation matching is applied sequentially: rule b counts rows not already removed under rule a.",
            "Ti_at is calculated from each row's normalized wt% composition converted to atomic fractions; filtering keeps Ti_at >= 60.",
            "Composition groups use all *_wt columns rounded to one decimal and values are not rounded in the saved dataset.",
            "E_std uses sample standard deviation; singleton groups have NaN E_std and are excluded from the label-noise-floor mean.",
            "Default XGBoost objective selection is by lowest mean CV MAE averaged across feature sets A and B; that objective is tuned independently on both feature sets.",
            "The A/B engineered-feature comparison uses paired tuned-XGBoost fold MAEs and compares mean A-minus-B improvement with the sample standard deviation of B fold MAEs.",
            "Feature importance is training-fit model importance (XGBoost gain or RF impurity); test data is not used for feature importance.",
        ],
    }
    REPORT_JSON.write_text(
        json.dumps(json_safe(results), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "Titanium Young's modulus dataset v2 and model report",
        "=" * 54,
        "Library versions:",
        *[f"- {name}: {version}" for name, version in versions.items()],
        "",
        "DFT exclusion:",
        f"- Base rows: {len(base_data)}",
        f"- Removed by source DFT matches (rule a): {len(removed_rule_a)}",
        f"- Removed by figure DFT matches (rule b): {len(removed_rule_b)}",
        f"- Removed total: {len(removed_dft)}",
        "",
        "Ti-matrix filter:",
        f"- Rows after DFT removal: {len(retained)}",
        f"- Retained (Ti_at >= 60): {len(ti_data)}",
        f"- Removed (Ti_at < 60): {len(removed_non_ti)}",
        "",
        "Composition collapse and label-noise floor:",
        f"- Rows before collapse: {len(ti_data)}",
        f"- Groups after collapse: {len(dataset)}",
        f"- Groups with n_rows > 1: {len(multirow_stats)}",
        f"- Mean within-group E std (sample std, duplicated groups only): {mean_within_group_std:.4f} GPa",
        "- This mean within-group E standard deviation is the estimated label-noise floor.",
        "",
        f"Grouped split (seed {split_seed}; shared groups={len(shared)}):",
    ]
    for name in ("train", "test"):
        split = split_statistics[name]
        stats = split["E_GPa"]
        report_lines.append(
            f"- {name}: {split['rows']} rows ({split['percent']:.2f}%), "
            f"E min={stats['min']:.3f}, max={stats['max']:.3f}, "
            f"mean={stats['mean']:.3f}, std={stats['std']:.3f}; "
            f"Ti_wt < 70 rows={split['Ti_wt_below_70_rows']}"
        )
    report_lines.append("- phase_est counts:")
    for phase in sorted(set(phase_counts["train"]) | set(phase_counts["test"])):
        report_lines.append(
            f"  - {phase}: train={phase_counts['train'].get(phase, 0)}, "
            f"test={phase_counts['test'].get(phase, 0)}"
        )
    report_lines.extend(
        [
            "",
            "Grouped 5-fold CV on train_v2.csv only (mean +/- std):",
        ]
    )
    for label, values in cv_results.items():
        report_lines.append(
            f"- {label}: MAE={values['mae_mean']:.4f} +/- {values['mae_std']:.4f}; "
            f"RMSE={values['rmse_mean']:.4f} +/- {values['rmse_std']:.4f}; "
            f"R2={values['r2_mean']:.4f} +/- {values['r2_std']:.4f}"
        )
    report_lines.extend(
        [
            "",
            f"Selected objective for tuning: {selected_objective}",
            "RandomizedSearchCV (60 iterations per feature set):",
        ]
    )
    for name, data in tuning.items():
        report_lines.append(
            f"- {name}: best search CV MAE={data['best_search_cv_mae']:.4f}; "
            f"params={json.dumps(json_safe(data['best_params']), sort_keys=True)}"
        )
    report_lines.extend(
        [
            "",
            "Selected model (lowest CV MAE among all defaults and tuned XGBoost variants):",
            f"- {selected['label']}: CV MAE={selected['cv_mae']:.4f} +/- {selected['cv_mae_std']:.4f}",
            f"- Test MAE={test_metrics['mae']:.4f} GPa",
            f"- Test RMSE={test_metrics['rmse']:.4f} GPa",
            f"- Test R2={test_metrics['r2']:.4f}",
            "",
            "Test subset MAE:",
        ]
    )
    for name, values in subsets.items():
        report_lines.append(
            f"- {name}: "
            + (f"{values['mae']:.4f} GPa" if values["mae"] is not None else "not available")
            + f" (n={values['rows']})"
        )
    report_lines.extend(
        [
            "",
            "Engineered-feature comparison (tuned XGBoost, paired folds):",
            f"- Mean A-minus-B MAE improvement: {paired_improvement:.4f} GPa",
            f"- One fold-std of B MAE: {fold_std:.4f} GPa",
            f"- Does B beat A by more than one fold-std? {'yes' if b_beats_a else 'no'}",
            "",
            "Test MAE vs label-noise floor:",
            f"- Label-noise floor: {mean_within_group_std:.4f} GPa",
            f"- Test MAE: {test_metrics['mae']:.4f} GPa",
            f"- Test MAE minus floor: {test_metrics['mae'] - mean_within_group_std:.4f} GPa",
            f"- Test MAE / floor: {noise_comparison['test_mae_to_noise_floor_ratio']:.4f}x"
            if noise_comparison["test_mae_to_noise_floor_ratio"] is not None
            else "- Test MAE / floor: undefined (zero noise floor)",
            "",
            "10 worst test rows:",
        ]
    )
    for rank, row in enumerate(worst_rows, start=1):
        report_lines.append(
            f"{rank}. {row['composition']}; true E={row['true_E_GPa']:.2f}, "
            f"predicted E={row['predicted_E_GPa']:.2f}, "
            f"|error|={row['absolute_error_GPa']:.2f}; "
            f"n_rows={row['n_rows']}, E_std="
            + (f"{row['E_std']:.4f}" if row["E_std"] is not None else "NaN")
        )
    report_lines.extend(
        [
            "",
            "Comparison with earlier run:",
            f"- Old test MAE: {OLD_TEST_MAE:.2f}; new: {test_metrics['mae']:.4f}; "
            f"change (new-old): {test_metrics['mae'] - OLD_TEST_MAE:+.4f} GPa",
            f"- Old CV MAE: {OLD_CV_MAE:.2f}; new: {selected['cv_mae']:.4f}; "
            f"change (new-old): {selected['cv_mae'] - OLD_CV_MAE:+.4f} GPa",
            "",
            "Outputs:",
            *[
                f"- {path.name}"
                for path in (
                    REMOVED_DFT_FILE, REMOVED_NON_TI_FILE, GROUP_STATS_FILE,
                    OUTPUT_BASE, TRAIN_FILE, TEST_FILE, MODEL_FILE,
                    FEATURE_COLUMNS_FILE, IMPORTANCE_FILE, REPORT_JSON,
                )
            ],
            "",
            "Choices:",
            *[f"- {choice}" for choice in results["choices"]],
        ]
    )
    report = "\n".join(report_lines) + "\n"
    REPORT_TEXT.write_text(report, encoding="utf-8")
    print(report)

    # Reload outputs and the raw-DataFrame model to validate persisted artifacts.
    persisted_dataset = pd.read_csv(OUTPUT_BASE)
    persisted_train = pd.read_csv(TRAIN_FILE)
    persisted_test = pd.read_csv(TEST_FILE)
    persisted_removed = pd.read_csv(REMOVED_DFT_FILE)
    persisted_pipeline = joblib.load(MODEL_FILE)
    predictions_after_reload = persisted_pipeline.predict(
        persisted_test[selected_features]
    )
    if len(persisted_dataset) != len(dataset):
        raise AssertionError("Persisted dataset row count changed.")
    if len(persisted_removed) != len(removed_dft):
        raise AssertionError("Persisted DFT removal row count changed.")
    if set(composition_key(persisted_train, composition_columns)) & set(
        composition_key(persisted_test, composition_columns)
    ):
        raise AssertionError("Persisted train/test composition overlap.")
    if not np.allclose(predictions_after_reload, final_predictions):
        raise AssertionError("Persisted model predictions changed after reload.")
    print(
        f"Validated persisted v2 dataset ({len(persisted_dataset)} rows), "
        f"train/test ({len(persisted_train)}/{len(persisted_test)} rows), "
        f"and reloaded model predictions ({len(predictions_after_reload)} rows)."
    )


if __name__ == "__main__":
    main()
