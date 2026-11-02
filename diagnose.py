import unicodedata
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import RepeatedKFold, KFold, cross_validate, cross_val_predict

FILE   = "titanium_alloys.csv"   # ORIGINAL file (with Type + descriptors), not the cleaned one
TARGET = "YM (GPa)"
COMP   = ["Ti","Nb","Zr","Fe","Sn","Ta","Mo","Al","V","Cr","Cu","Hf","Mn","W","Sc","Si"]

norm = lambda s: unicodedata.normalize("NFC", str(s).strip())
df = pd.read_csv(FILE)
df.columns = [norm(c) for c in df.columns]
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
y = df[TARGET]
print("Columns:", list(df.columns))

# ---------- 1. noise ceiling: identical compositions, different YM ----------
print("\n=== NOISE CEILING ===")
for dec in (0, 1):
    key = df[COMP].round(dec).astype(str).agg("|".join, axis=1)
    g = df.groupby(key)[TARGET]
    multi = g.filter(lambda s: len(s) > 1)
    if len(multi) == 0:
        print(f"round({dec}): no duplicate compositions"); continue
    within = (df.assign(k=key).groupby("k")[TARGET].transform(lambda s: s - s.mean()))
    dup_mask = key.map(key.value_counts()) > 1
    var_w = (within[dup_mask] ** 2).mean()
    print(f"round({dec}): {dup_mask.sum()} rows in dup groups | within-group std = {np.sqrt(var_w):.1f} GPa "
          f"| max R2 on those rows ~ {1 - var_w / y[dup_mask].var():.2f}")

# ---------- 2. does Type / descriptors help? ----------
print("\n=== FEATURE TEST (XGB, 5x2 CV) ===")
phi = [c for c in df.columns if c.startswith("φ")]
other = [c for c in df.columns if c not in COMP + phi + [TARGET, "Type"]]
Xc = df[COMP]
sets = {"comp": Xc}
if "Type" in df.columns:
    print("Type values:\n", df["Type"].value_counts(dropna=False).to_string())
    T = pd.get_dummies(df["Type"].astype(str), prefix="Type").astype(float)
    sets["comp+Type"] = pd.concat([Xc, T], axis=1)
D = df[other].apply(pd.to_numeric, errors="coerce")
sets["comp+descriptors"] = pd.concat([Xc, D], axis=1)
if "Type" in df.columns:
    sets["comp+Type+descriptors"] = pd.concat([Xc, T, D], axis=1)

xgb = XGBRegressor(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=3,
                   subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, random_state=42, n_jobs=1)
cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
for name, X in sets.items():
    s = cross_validate(xgb, X, y, cv=cv, scoring={"r2": "r2", "mae": "neg_mean_absolute_error"}, n_jobs=-1)
    print(f"{name:24s} R2={s['test_r2'].mean():.3f} ± {s['test_r2'].std():.3f}   MAE={-s['test_mae'].mean():.2f}")

# ---------- 3. worst residuals (comp only) ----------
print("\n=== TOP 15 WORST PREDICTIONS (comp only) ===")
oof = cross_val_predict(xgb, Xc, y, cv=KFold(5, shuffle=True, random_state=42))
err = (y - oof)
for i in err.abs().sort_values(ascending=False).index[:15]:
    comp = ", ".join(f"{e}{df.loc[i, e]:g}" for e in COMP if df.loc[i, e] > 0)
    print(f"row {i:4d} | true {y[i]:6.1f} | pred {oof[i]:6.1f} | {comp}")