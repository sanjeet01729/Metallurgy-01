import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.model_selection import KFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "titanium_alloys.csv"

# Output files — pipeline name matches what app.py loads
MODEL_FILE = "titanium_ym_pipeline.pkl"
METRICS_FILE = "model_metrics.pkl"
SHAP_FILE = "shap_feature_importance.csv"
SHAP_PLOT = "shap_summary.png"

df = pd.read_csv(DATA_FILE)

# Clean column names
df.columns = [str(c).strip() for c in df.columns]

TARGET = "YM (GPa)"

# Remove rows without target
df = df.dropna(subset=[TARGET]).copy()

print("=" * 70)
print("DATASET")
print("=" * 70)

print(f"Samples: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Target:  {TARGET}")


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

df["VEC_sq"] = df["VEC (-)"] ** 2

df["Mo_eq_x_VEC"] = (
    df["(Mo)eq"] * df["VEC (-)"]
)

df["Beta_Stabilizers"] = (
    df["Nb"]
    + df["Zr"]
    + df["Fe"]
    + df["Mo"]
)


# ============================================================
# 3. SELECT FEATURES
# ============================================================
#
# Column names contain special Unicode characters (Greek Ano
# Teleia vs Middle Dot, etc.) that can silently mismatch when
# the script is saved in a different editor.  To avoid this,
# we build the feature list from the actual DataFrame columns.
#

def _col(df, keyword):
    """Find a column whose name contains *keyword* (case-sensitive)."""
    matches = [c for c in df.columns if keyword in c]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Return shortest match to prefer exact hits
        return min(matches, key=len)
    raise KeyError(f"No column matching '{keyword}' found in dataset.")


feature_cols = [
    # Composition (exact names)
    "Ti", "Nb", "Zr", "Fe", "Sn", "Ta", "Mo",
    "Al", "V", "Cr", "Cu", "Hf", "Mn", "W", "Sc", "Si",

    # Physical properties — looked up by substring
    _col(df, "cm"),          # ρ (g·cm⁻³)
    _col(df, "Thc"),         # Thc (W·m⁻¹·K⁻¹)
    "r (pm)",

    # Electronic descriptors
    _col(df, "χᴾ (-)"),      # χᴾ (-)    — Pauling electronegativity
    _col(df, "χᴬ (-)"),      # χᴬ (-)    — Allen electronegativity
    "VEC (-)",
    "e/a (-)",

    # Atomic / chemical descriptors
    "δr (%)",
    _col(df, "δχᴾ"),         # δχᴾ (%)
    _col(df, "Δχᴾ"),         # Δχᴾ (-)
    _col(df, "δχᴬ"),         # δχᴬ (%)
    _col(df, "Δχᴬ"),         # Δχᴬ (-)
    _col(df, "ΔVEC"),         # ΔVEC (-)

    # Thermodynamic descriptors
    _col(df, "Sᵐ"),          # ΔSᵐ (J·mol⁻¹·K⁻¹)
    _col(df, "Hᶦ"),          # ΔHᶦᶜ (kJ·mol⁻¹)

    # Phase-related descriptors
    _col(df, "bcc) (J"),     # Sˣˢ (bcc) (J·mol⁻¹·K⁻¹)
    _col(df, "fcc) (J"),     # Sˣˢ (fcc) (J·mol⁻¹·K⁻¹)
    _col(df, "ɣ"),           # ɣ (-)
    _col(df, "φ (bcc)"),     # φ (bcc) (-)
    _col(df, "φ (fcc)"),     # φ (fcc) (-)

    # Beta stability
    "(Mo)eq",
    "Type",

    # Engineered features
    "VEC_sq",
    "Mo_eq_x_VEC",
    "Beta_Stabilizers"
]


# Final safety check
missing_features = [
    col for col in feature_cols
    if col not in df.columns
]

if missing_features:
    print("\nERROR: Missing feature columns:")
    for col in missing_features:
        print(" -", col)
    raise ValueError(
        "Some required feature columns are missing from the dataset."
    )


X = df[feature_cols].copy()
y = df[TARGET].copy()

# Convert predictors to numeric
X = X.apply(pd.to_numeric, errors="coerce")


print(f"\nNumber of predictor features: {X.shape[1]}")
print(f"Total training samples:       {X.shape[0]}  (entire dataset)")


# ============================================================
# 4. XGBOOST MODEL
# ============================================================

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 5. PREPROCESSING PIPELINE
# ============================================================

model_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "model",
        xgb_model
    )
])


# ============================================================
# 6. 5-FOLD CROSS VALIDATION  (for performance reporting)
# ============================================================

print()
print("=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "mape": "neg_mean_absolute_percentage_error"
}

cv_results = cross_validate(
    model_pipeline,
    X,
    y,
    cv=kf,
    scoring=scoring,
    n_jobs=-1
)


# Convert negative sklearn scoring back to positive
cv_r2 = cv_results["test_r2"]

cv_mae = -cv_results["test_mae"]

cv_rmse = -cv_results["test_rmse"]

cv_mape = -cv_results["test_mape"] * 100


print(
    f"R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}"
)

print(
    f"MAE  : {cv_mae.mean():.2f} ± "
    f"{cv_mae.std():.2f} GPa"
)

print(
    f"RMSE : {cv_rmse.mean():.2f} ± "
    f"{cv_rmse.std():.2f} GPa"
)

print(
    f"MAPE : {cv_mape.mean():.2f} ± "
    f"{cv_mape.std():.2f}%"
)


# ============================================================
# 7. TRAIN FINAL MODEL ON ENTIRE DATASET
# ============================================================

print()
print("=" * 70)
print("TRAINING FINAL MODEL (entire dataset)")
print("=" * 70)

model_pipeline.fit(X, y)

# Quick sanity check — resubstitution metrics
y_pred_train = model_pipeline.predict(X)
train_r2 = r2_score(y, y_pred_train)
train_mae = mean_absolute_error(y, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y, y_pred_train))

print(f"Resubstitution R²   : {train_r2:.4f}")
print(f"Resubstitution MAE  : {train_mae:.2f} GPa")
print(f"Resubstitution RMSE : {train_rmse:.2f} GPa")


# ============================================================
# 8. SHAP ANALYSIS
# ============================================================

print()
print("=" * 70)
print("SHAP ANALYSIS")
print("=" * 70)


# Extract fitted imputer
imputer = model_pipeline.named_steps["imputer"]

# Extract fitted XGBoost model
model = model_pipeline.named_steps["model"]


# Transform all data using the fitted imputer
X_imputed = imputer.transform(X)

# Convert back to DataFrame
X_imputed = pd.DataFrame(
    X_imputed,
    columns=X.columns,
    index=X.index
)


# Create SHAP explainer
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X_imputed
)


# ============================================================
# 9. GLOBAL SHAP IMPORTANCE
# ============================================================

shap_importance = pd.DataFrame({
    "Feature": X_imputed.columns,
    "Mean_Absolute_SHAP": np.abs(shap_values).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


print()
print("TOP 15 FEATURES")
print("-" * 50)

print(
    shap_importance
    .head(15)
    .to_string(index=False)
)


# Save SHAP importance
shap_importance.to_csv(
    SHAP_FILE,
    index=False
)


# ============================================================
# 10. SHAP SUMMARY PLOT
# ============================================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values,
    X_imputed,
    show=False
)

plt.tight_layout()

plt.savefig(
    SHAP_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. SAVE MODEL
# ============================================================

joblib.dump(
    model_pipeline,
    MODEL_FILE
)


# ============================================================
# 12. SAVE METRICS
# ============================================================

metrics = {

    # Dataset
    "total_samples": len(df),
    "training_samples": len(X),
    "num_features": len(feature_cols),

    # Cross validation
    "cv_r2_mean": float(cv_r2.mean()),
    "cv_r2_std": float(cv_r2.std()),

    "cv_mae_mean": float(cv_mae.mean()),
    "cv_mae_std": float(cv_mae.std()),

    "cv_rmse_mean": float(cv_rmse.mean()),
    "cv_rmse_std": float(cv_rmse.std()),

    "cv_mape_mean": float(cv_mape.mean()),
    "cv_mape_std": float(cv_mape.std()),

    # Resubstitution (train on all)
    "train_r2": float(train_r2),
    "train_mae": float(train_mae),
    "train_rmse": float(train_rmse),
}


joblib.dump(
    metrics,
    METRICS_FILE
)


# ============================================================
# 13. FINISHED
# ============================================================

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"Model:       {MODEL_FILE}")
print(f"Metrics:     {METRICS_FILE}")
print(f"SHAP CSV:    {SHAP_FILE}")
print(f"SHAP Plot:   {SHAP_PLOT}")

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)