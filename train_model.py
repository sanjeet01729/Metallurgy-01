
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TRAIN_FILE = "titanium_train.csv"
TEST_FILE = "titanium_test.csv"

TARGET = "YM (GPa)"

# Composition-only features: must match your cleaned CSV.
FEATURES = [
    "Ti", "Nb", "Zr", "Fe", "Sn", "Ta", "Mo", "Al",
    "V", "Cr", "Cu", "Hf", "Mn", "W", "Sc", "Si"
]

MODEL_FILE = "titanium_ym_pipeline.pkl"
METRICS_FILE = "model_metrics.pkl"
IMPORTANCE_FILE = "shap_feature_importance.csv"
PLOT_FILE = "shap_summary.png"

RANDOM_STATE = 42
N_SPLITS = 5

# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run this script from the "
            "directory containing the CSV files."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert composition and target columns to numeric.
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # A target value is required for supervised training/evaluation.
    df = df.dropna(subset=[TARGET]).copy()

    # Remove invalid target values.
    df = df[np.isfinite(df[TARGET])]

    # Treat infinite feature values as missing; imputer handles them.
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)

    if df.empty:
        raise ValueError(f"No usable rows found in {path}.")

    return df


train_df = load_dataset(TRAIN_FILE)
test_df = load_dataset(TEST_FILE)

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("=" * 60)
print("TITANIUM ALLOY YOUNG'S MODULUS MODEL")
print("=" * 60)
print(f"Training rows: {len(train_df)}")
print(f"Testing rows:  {len(test_df)}")
print(f"Features:      {len(FEATURES)}")
print(f"Target:        {TARGET}")

# --------------------------------------------------
# DEFINE MODELS
# --------------------------------------------------

models = {
    "Ridge": Ridge(alpha=1.0),

    "RandomForest": RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        loss="huber",
        random_state=RANDOM_STATE,
    ),

    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}

# --------------------------------------------------
# CROSS-VALIDATION ON TRAINING DATA ONLY
# --------------------------------------------------

cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
}

cv_results = []

print("\nCross-validation results (training data only)")
print("-" * 60)

for name, estimator in models.items():
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clone(estimator)),
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    mean_r2 = scores["test_r2"].mean()
    std_r2 = scores["test_r2"].std()
    mean_mae = -scores["test_mae"].mean()
    mean_rmse = np.sqrt(-scores["test_mse"]).mean()

    cv_results.append({
        "Model": name,
        "CV_R2_mean": mean_r2,
        "CV_R2_std": std_r2,
        "CV_MAE_GPa": mean_mae,
        "CV_RMSE_GPa": mean_rmse,
    })

    print(
        f"{name:22s} | "
        f"R² = {mean_r2:.4f} ± {std_r2:.4f} | "
        f"MAE = {mean_mae:.3f} GPa | "
        f"RMSE = {mean_rmse:.3f} GPa"
    )

results_df = pd.DataFrame(cv_results).sort_values(
    "CV_RMSE_GPa", ascending=True
)

results_df.to_csv("model_comparison.csv", index=False)

# Select the best model based on training-set CV RMSE.
best_name = results_df.iloc[0]["Model"]
best_estimator = clone(models[best_name])

print("\nSelected model:", best_name)
print("Selection criterion: lowest mean CV RMSE")

# --------------------------------------------------
# FIT SELECTED MODEL ON TRAINING DATA
# --------------------------------------------------

best_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", best_estimator),
])

best_pipeline.fit(X_train, y_train)

# --------------------------------------------------
# FINAL HELD-OUT TEST EVALUATION
# --------------------------------------------------

test_predictions = best_pipeline.predict(X_test)

test_r2 = r2_score(y_test, test_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print("\nHeld-out test results")
print("-" * 60)
print(f"Model: {best_name}")
print(f"Test R²:   {test_r2:.4f}")
print(f"Test MAE:  {test_mae:.3f} GPa")
print(f"Test RMSE: {test_rmse:.3f} GPa")

# Save individual test predictions for inspection.
predictions_df = pd.DataFrame({
    "Actual_YM_GPa": y_test.to_numpy(),
    "Predicted_YM_GPa": test_predictions,
    "Absolute_Error_GPa": np.abs(
        y_test.to_numpy() - test_predictions
    ),
})

predictions_df.to_csv("test_predictions.csv", index=False)

# --------------------------------------------------
# SHAP FEATURE IMPORTANCE
# --------------------------------------------------

print("\nCalculating SHAP feature importance...")

fitted_model = best_pipeline.named_steps["model"]
imputer = best_pipeline.named_steps["imputer"]

X_train_imputed = pd.DataFrame(
    imputer.transform(X_train),
    columns=FEATURES,
    index=X_train.index,
)

# Tree models: use TreeExplainer.
# Ridge: use LinearExplainer.
try:
    if best_name == "Ridge":
        explainer = shap.LinearExplainer(
            fitted_model,
            X_train_imputed,
        )
    else:
        explainer = shap.TreeExplainer(fitted_model)

    # Bound computation time on larger datasets.
    X_shap = X_train_imputed.sample(
        n=min(300, len(X_train_imputed)),
        random_state=RANDOM_STATE,
    )

    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Mean_Absolute_SHAP": mean_abs_shap,
    }).sort_values("Mean_Absolute_SHAP", ascending=False)

    importance_df.to_csv(IMPORTANCE_FILE, index=False)

    shap.summary_plot(
        shap_values,
        X_shap,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nTop 10 SHAP features:")
    print(importance_df.head(10).to_string(index=False))

except Exception as exc:
    print(f"SHAP analysis failed: {exc}")
    print("Model training and test evaluation completed regardless.")

# --------------------------------------------------
# SAVE MODEL AND METRICS
# --------------------------------------------------

metrics = {
    "selected_model": best_name,
    "features": FEATURES,
    "target": TARGET,
    "training_rows": len(train_df),
    "test_rows": len(test_df),
    "cv_r2_mean": float(results_df.iloc[0]["CV_R2_mean"]),
    "cv_r2_std": float(results_df.iloc[0]["CV_R2_std"]),
    "cv_mae_gpa": float(results_df.iloc[0]["CV_MAE_GPa"]),
    "cv_rmse_gpa": float(results_df.iloc[0]["CV_RMSE_GPa"]),
    "test_r2": float(test_r2),
    "test_mae_gpa": float(test_mae),
    "test_rmse_gpa": float(test_rmse),
}

# Save the fitted sklearn Pipeline, not a wrapper, so an app
# that loads the pipeline with joblib.load can remain compatible.
joblib.dump(best_pipeline, MODEL_FILE)
joblib.dump(metrics, METRICS_FILE)

print("\nSaved files:")
print(f"- {MODEL_FILE}")
print(f"- {METRICS_FILE}")
print("- model_comparison.csv")
print("- test_predictions.csv")
print(f"- {IMPORTANCE_FILE} (if SHAP succeeded)")
print(f"- {PLOT_FILE} (if SHAP succeeded)")

print("\nTraining and evaluation complete.")
