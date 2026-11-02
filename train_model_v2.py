import unicodedata, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedKFold, KFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from features import COMP, UNITS, featurize

DATA_FILE  = "titanium_train.csv"
MODEL_FILE = "titanium_ym_pipeline.pkl"
TARGET     = "YM (GPa)"
USE_PHI    = False   # phi = diagnostic only unless app.py can compute it

norm = lambda s: unicodedata.normalize("NFC", str(s).strip())
df = pd.read_csv(DATA_FILE)
df.columns = [norm(c) for c in df.columns]
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET]
PHI = next(c for c in df.columns if c.startswith("φ (bcc)"))

print("Composition row sums (should be ~100):", df[COMP].sum(axis=1).describe()[["min","mean","max"]].round(2).to_dict())

# ---- feature sets ----
F_comp = featurize(df[COMP], UNITS, True, False)
F_phys = featurize(df[COMP], UNITS, False, True)
F_all  = pd.concat([F_comp, F_phys], axis=1)
sets = {
    "comp":            F_comp,
    "phys only":       F_phys,
    "comp+phys":       F_all,
    "comp+phys+phi*":  pd.concat([F_all, df[[PHI]]], axis=1),   # * not deployable
}

models = {
    "Ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
    "RF":    RandomForestRegressor(500, min_samples_leaf=2, random_state=42, n_jobs=1),
    "ET":    ExtraTreesRegressor(500, min_samples_leaf=2, random_state=42, n_jobs=1),
    "GB":    GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3,
                                       subsample=0.8, random_state=42),
    "XGB":   XGBRegressor(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=3,
                          subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
                          random_state=42, n_jobs=1),
}

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
sc = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
rows = []
for sname, X in sets.items():
    for mname, m in models.items():
        s = cross_validate(m, X, y, cv=cv, scoring=sc, n_jobs=-1)
        rows.append(dict(features=sname, model=mname, R2=s["test_r2"].mean(),
                         R2_std=s["test_r2"].std(), MAE=-s["test_mae"].mean(),
                         RMSE=-s["test_rmse"].mean()))
        print(rows[-1])
res = pd.DataFrame(rows).sort_values("R2", ascending=False)
print("\n", res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
res.to_csv("model_comparison_v2.csv", index=False)

# ---- pick best deployable ----
ok = res[res.features.str.contains(r"\*") == False] if not USE_PHI else res
best = ok.iloc[0]
print(f"\nDeploying: {best.model} on '{best.features}'  (CV R2={best.R2:.3f})")
X = sets[best.features]
cols = list(X.columns)
oof = cross_val_predict(models[best.model], X, y, cv=KFold(5, shuffle=True, random_state=42))
oof_r2, oof_mae = r2_score(y, oof), mean_absolute_error(y, oof)
oof_rmse = float(np.sqrt(mean_squared_error(y, oof)))
print(f"OOF R2={oof_r2:.3f}  MAE={oof_mae:.2f}  RMSE={oof_rmse:.2f} GPa")

final = models[best.model].fit(X, y)
with_comp = best.features in ("comp", "comp+phys")
with_phys = best.features in ("phys only", "comp+phys")
bundle = {
    "model": final, "columns": cols, "feature_set": best.features,
    "with_comp": with_comp, "with_phys": with_phys, "units": UNITS,
    "comp_bounds": {c: (float(df[c].min()), float(df[c].max())) for c in COMP},
    "metrics": dict(oof_r2=oof_r2, oof_mae=oof_mae, oof_rmse=oof_rmse, n=len(df)),
}
joblib.dump(bundle, MODEL_FILE)
print(f"Saved {MODEL_FILE}")