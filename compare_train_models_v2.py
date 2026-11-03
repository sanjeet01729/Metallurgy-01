from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import joblib
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from features import make_ensemble_features


BASE = Path(__file__).resolve().parent
TRAIN_FILE = BASE / "train_v2.csv"
TEST_FILE = BASE / "test_v2.csv"
FEATURE_COLUMNS_FILE = BASE / "feature_columns_v2.json"
SAVED_MODEL_FILE = BASE / "ti_ym_model_v2.joblib"
ENSEMBLE_MODEL_FILE = BASE / "ti_ym_ensemble_v2.joblib"
ENSEMBLE_FEATURE_COLUMNS_FILE = BASE / "ensemble_feature_columns_v2.json"
REPORT_FILE = BASE / "model_training_comparison.md"


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def feature_frame(frame: pd.DataFrame, composition_columns: list[str]) -> pd.DataFrame:
    result = make_ensemble_features(frame, composition_columns)
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("Non-finite model feature values found.")
    return result


def make_model(name: str):
    xgb = XGBRegressor(
        objective="reg:absoluteerror",
        random_state=42,
        n_jobs=1,
    )
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=1,
    )

    if name == "XGBoost (absolute error)":
        return xgb
    if name == "Random Forest":
        return make_pipeline(SimpleImputer(strategy="median"), rf)
    if name == "Extra Trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=1,
            ),
        )
    if name == "Gradient Boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(random_state=42),
        )
    if name == "LightGBM":
        return LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1)
    if name == "CatBoost":
        return CatBoostRegressor(
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
    if name == "SVR (RBF)":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVR(kernel="rbf"),
        )
    if name == "Ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(),
        )
    if name == "Mean baseline":
        return DummyRegressor(strategy="mean")
    if name == "XGBoost + Random Forest mean":
        rf_pipeline = make_pipeline(SimpleImputer(strategy="median"), clone(rf))
        return VotingRegressor(
            estimators=[("xgb", clone(xgb)), ("rf", rf_pipeline)],
            n_jobs=1,
        )
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    dataset = pd.concat([train, test], ignore_index=True)
    feature_columns = json.loads(FEATURE_COLUMNS_FILE.read_text(encoding="utf-8"))
    composition_columns = [
        column for column in train.columns if column.endswith("_wt")
    ]
    excluded = {"E_GPa", "Ti_at", "ti_based", "phase_note"}
    if any(column in feature_columns for column in excluded):
        raise ValueError("A prohibited target/derived column appears in the feature contract.")

    if not set(composition_columns).issubset(feature_columns):
        raise ValueError("The saved v2 feature contract is missing composition columns.")
    train_groups = train[composition_columns].round(1).astype(str).agg("|".join, axis=1)
    test_groups = test[composition_columns].round(1).astype(str).agg("|".join, axis=1)
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise ValueError(f"Train/test composition leakage detected ({len(overlap)} groups).")
    if train["E_GPa"].isna().any() or test["E_GPa"].isna().any():
        raise ValueError("Missing target values found.")

    x_train = feature_frame(train, composition_columns)
    x_test = feature_frame(test, composition_columns).reindex(
        columns=x_train.columns
    )
    y_train = train["E_GPa"].to_numpy(dtype=float)
    y_test = test["E_GPa"].to_numpy(dtype=float)

    candidates = [
        "XGBoost (absolute error)",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "LightGBM",
        "CatBoost",
        "SVR (RBF)",
        "Ridge",
        "Mean baseline",
        "XGBoost + Random Forest mean",
    ]
    splitter = GroupKFold(n_splits=5)
    cv_results: list[dict[str, object]] = []
    for name in candidates:
        fold_scores = []
        for train_index, validation_index in splitter.split(
            x_train, y_train, groups=train_groups
        ):
            model = make_model(name)
            model.fit(x_train.iloc[train_index], y_train[train_index])
            predictions = model.predict(x_train.iloc[validation_index])
            fold_scores.append(
                metrics(y_train[validation_index], predictions)
            )
        cv_results.append(
            {
                "name": name,
                "folds": fold_scores,
                "mean": {
                    metric: float(np.mean([score[metric] for score in fold_scores]))
                    for metric in ("r2", "mae", "rmse")
                },
                "std": {
                    metric: float(np.std(
                        [score[metric] for score in fold_scores], ddof=1
                    ))
                    for metric in ("r2", "mae", "rmse")
                },
            }
        )

    best = min(cv_results, key=lambda row: row["mean"]["mae"])
    selected_model = make_model(str(best["name"]))
    selected_model.fit(x_train, y_train)
    selected_train_metrics = metrics(y_train, selected_model.predict(x_train))
    selected_test_metrics = metrics(y_test, selected_model.predict(x_test))
    joblib.dump(selected_model, ENSEMBLE_MODEL_FILE)
    ENSEMBLE_FEATURE_COLUMNS_FILE.write_text(
        json.dumps(x_train.columns.tolist(), indent=2) + "\n",
        encoding="utf-8",
    )

    saved_test_metrics = None
    if SAVED_MODEL_FILE.exists():
        saved_model = joblib.load(SAVED_MODEL_FILE)
        saved_test_metrics = metrics(y_test, saved_model.predict(test[feature_columns]))

    versions = {
        "Python": __import__("platform").python_version(),
        "NumPy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "XGBoost": importlib.metadata.version("xgboost"),
        "LightGBM": importlib.metadata.version("lightgbm"),
        "CatBoost": importlib.metadata.version("catboost"),
    }
    lines = [
        "# Titanium Young's Modulus: Model Training and Comparison",
        "",
        "## Summary",
        "",
        f"- Dataset: {len(dataset)} composition groups; training split: {len(train)} rows; held-out test split: {len(test)} rows.",
        f"- Best model by training-only 5-fold grouped CV MAE: **{best['name']}**.",
        f"- Held-out test: R² = **{selected_test_metrics['r2']:.3f}**, "
        f"MAE = **{selected_test_metrics['mae']:.2f} GPa**, "
        f"RMSE = **{selected_test_metrics['rmse']:.2f} GPa**.",
        "- The test split was not used for model selection. Only the CV-selected candidate was evaluated on it.",
        f"- App model artifact: `{ENSEMBLE_MODEL_FILE.name}`.",
        "",
        "## Model comparison",
        "",
        "Metrics are the mean and sample standard deviation across five GroupKFold validation folds on `train_v2.csv`. "
        "MAE was the selection metric; lower is better. R² is unitless; MAE and RMSE are in GPa.",
        "",
        "| Model | CV R² (mean ± SD) | CV MAE (GPa, mean ± SD) | CV RMSE (GPa, mean ± SD) |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted(cv_results, key=lambda value: value["mean"]["mae"]):
        lines.append(
            f"| {row['name']} | {row['mean']['r2']:.3f} ± {row['std']['r2']:.3f} "
            f"| {row['mean']['mae']:.2f} ± {row['std']['mae']:.2f} "
            f"| {row['mean']['rmse']:.2f} ± {row['std']['rmse']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Selected model scores",
            "",
            "| Split | Rows | R² | MAE (GPa) | RMSE (GPa) |",
            "|---|---:|---:|---:|---:|",
            f"| Training (in-sample) | {len(train)} | {selected_train_metrics['r2']:.3f} "
            f"| {selected_train_metrics['mae']:.2f} | {selected_train_metrics['rmse']:.2f} |",
            f"| Held-out test | {len(test)} | {selected_test_metrics['r2']:.3f} "
            f"| {selected_test_metrics['mae']:.2f} | {selected_test_metrics['rmse']:.2f} |",
        ]
    )
    if saved_test_metrics is not None:
        lines.extend(
            [
                "",
                "### Existing saved v2 model (test-set reference)",
                "",
                f"The pre-existing `ti_ym_model_v2.joblib` scored R² = "
                f"{saved_test_metrics['r2']:.3f}, MAE = {saved_test_metrics['mae']:.2f} GPa, "
                f"and RMSE = {saved_test_metrics['rmse']:.2f} GPa on the same held-out test split. "
                "This is a reference evaluation of the existing model, not a candidate used to choose the newly trained model.",
            ]
        )
    lines.extend(
        [
            "",
            "## Training method",
            "",
            "1. Used the existing `train_v2.csv` and `test_v2.csv` split; kept the test set separate during validation and selection.",
            "2. Used all `*_wt` columns, the stored engineered descriptors (`VEC`, `delta_r`, `delta_S_mix`, `Mo_eq`, `Al_eq`, `Bo`, `Md`), and one-hot indicators for `phase_est`.",
            "3. Excluded `E_GPa` from inputs, along with `Ti_at`, `ti_based`, and `phase_note`. No feature scaling or imputation was fitted before a fold.",
            "4. Compared ten regressors using 5-fold `GroupKFold` on the training set. Groups were all composition columns rounded to 1 decimal place.",
            "5. XGBoost, LightGBM, and CatBoost received missing values directly. Models that required complete numeric inputs used median imputation fitted separately inside each fold; SVR and Ridge additionally used fold-fitted standardization.",
            "6. Chose the candidate with the lowest mean validation MAE, refit it on all training rows, and evaluated it on the held-out test rows. The pre-existing saved model was also evaluated on test only as a historical reference.",
            f"7. Saved the selected fitted pipeline to `{ENSEMBLE_MODEL_FILE.name}` and its encoded feature order to `{ENSEMBLE_FEATURE_COLUMNS_FILE.name}` for app inference.",
            "",
            "The in-sample training scores are optimistic and are not the preferred estimate of generalization. The grouped CV results and held-out test metrics are more informative. The data are modest in size and combine measurements from different sources and methods, so results should not be treated as a guarantee for new alloys or processing conditions.",
            "",
            "## Data checks and software versions",
            "",
            f"- Train/test composition overlap at 0.1 wt% grouping tolerance: **{len(overlap)} groups**.",
            f"- Model input columns: {len(x_train.columns)} (including one-hot phase indicators).",
            "- Versions:",
        ]
    )
    lines.extend(f"  - {name}: {version}" for name, version in versions.items())
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Selected model: {best['name']}")
    print(f"Selected held-out test metrics: {selected_test_metrics}")
    print(f"Existing model held-out test metrics: {saved_test_metrics}")
    print(f"Saved app ensemble: {ENSEMBLE_MODEL_FILE}")
    print(f"Report written: {REPORT_FILE}")


if __name__ == "__main__":
    main()
