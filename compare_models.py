import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = "titanium_alloys.csv"

df = pd.read_csv(DATA_FILE)
df.columns = [str(c).strip() for c in df.columns]

TARGET = "YM (GPa)"

# Only rows with known Young's modulus
df = df.dropna(subset=[TARGET]).copy()


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
# 3. FEATURES
# ============================================================

feature_cols = [

    # Composition
    "Ti",
    "Nb",
    "Zr",
    "Fe",
    "Sn",
    "Ta",
    "Mo",
    "Al",
    "V",
    "Cr",
    "Cu",
    "Hf",
    "Mn",
    "W",
    "Sc",
    "Si",

    # Physical
    "ρ (g·cm⁻³)",
    "Thc (W·m⁻¹·K⁻¹)",
    "r (pm)",

    # Electronic
    "χᴾ (-)",
    "χᴬ (-)",
    "VEC (-)",
    "e/a (-)",

    # Chemical
    "δr (%)",
    "δχᴾ (%)",
    "Δχᴾ (-)",
    "δχᴬ (%)",
    "Δχᴬ (-)",
    "ΔVEC (-)",

    # Thermodynamic
    "ΔSᵐ (J·mol⁻¹·K⁻¹)",
    "ΔHᶦᶜ (kJ·mol⁻¹)",

    # Phase
    "Sˣˢ (bcc) (J·mol⁻¹·K⁻¹)",
    "Sˣˢ (fcc) (J·mol⁻¹·K⁻¹)",
    "ɣ (-)",
    "φ (bcc) (-)",
    "φ (fcc) (-)",

    "(Mo)eq",
    "Type",

    # Engineered
    "VEC_sq",
    "Mo_eq_x_VEC",
    "Beta_Stabilizers"
]


X = df[feature_cols].apply(
    pd.to_numeric,
    errors="coerce"
)

y = df[TARGET]


print(f"Samples: {len(df)}")
print(f"Features: {len(feature_cols)}")


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Ridge": Ridge(
        alpha=1.0
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 5. CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "MAPE": "neg_mean_absolute_percentage_error"
}


# ============================================================
# 6. TEST EVERY MODEL
# ============================================================

results = []


for name, model in models.items():

    print()
    print("=" * 60)
    print(f"Testing: {name}")
    print("=" * 60)

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            model
        )
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=kf,
        scoring=scoring,
        n_jobs=-1
    )

    r2 = scores["test_R2"]

    mae = -scores["test_MAE"]

    rmse = -scores["test_RMSE"]

    mape = -scores["test_MAPE"] * 100


    results.append({

        "Model": name,

        "CV R² Mean": r2.mean(),
        "CV R² Std": r2.std(),

        "CV MAE Mean (GPa)": mae.mean(),
        "CV MAE Std": mae.std(),

        "CV RMSE Mean (GPa)": rmse.mean(),
        "CV RMSE Std": rmse.std(),

        "CV MAPE Mean (%)": mape.mean(),
        "CV MAPE Std": mape.std()
    })


# ============================================================
# 7. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    "CV R² Mean",
    ascending=False
)


print()
print()
print("=" * 90)
print("MODEL COMPARISON")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print()
print("Saved:")
print("model_comparison.csv")