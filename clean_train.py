import unicodedata, joblib
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

FILE       = "titanium_alloys.csv"      # original file (needs System column; only COMP is used as input)
MODEL_FILE = "titanium_ym_pipeline.pkl"
TARGET     = "YM (GPa)"
COMP       = ["Ti","Nb","Zr","Fe","Sn","Ta","Mo","Al","V","Cr","Cu","Hf","Mn","W","Sc","Si"]
UNITS      = "wt"
MAD_K      = 3.0     # flag if |OOF error| > MAD_K * robust sigma
LOW_YM     = 30.0    # also flag YM below this (implausible for most Ti alloys)

norm = lambda s: unicodedata.normalize("NFC", str(s).strip())
df = pd.read_csv(FILE)
df.columns = [norm(c) for c in df.columns]
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
X, y = df[COMP], df[TARGET]

def make(huber=False):
    kw = dict(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=3,
              subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, random_state=42, n_jobs=1)
    if huber:
        kw.update(objective="reg:pseudohubererror", huber_slope=10.0, base_score=float(y.median()))
    return XGBRegressor(**kw)

# ---- 1. flag suspicious labels ----
oof = cross_val_predict(make(), X, y, cv=KFold(5, shuffle=True, random_state=42))
err = y - oof
sigma = 1.4826 * np.median(np.abs(err - np.median(err)))
flag = (err.abs() > MAD_K * sigma) | (y < LOW_YM)
print(f"Flagged {flag.sum()} / {len(df)} rows (robust sigma = {sigma:.1f} GPa)")
fl = df.index[flag]
for i in fl[np.argsort(-err[fl].abs().values)]:
    comp = ", ".join(f"{e}{df.loc[i, e]:g}" for e in COMP if df.loc[i, e] > 0)
    src = f" | {df.loc[i, 'System']}" if "System" in df.columns else ""
    print(f"  row {i:4d} | true {y[i]:6.1f} | pred {oof[i]:6.1f} | {comp}{src}")
pd.DataFrame({"row": fl, "YM": y[fl].values, "pred": oof[fl], "comp": [
    ", ".join(f"{e}{df.loc[i, e]:g}" for e in COMP if df.loc[i, e] > 0) for i in fl]}
).to_csv("flagged_rows.csv", index=False)

# ---- 2. compare: L2 vs Huber, trained on all vs trained on clean ----
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=7)
configs = {"L2 / all": (False, False), "L2 / clean": (False, True),
           "Huber / all": (True, False), "Huber / clean": (True, True)}
rows = []
for name, (hub, clean) in configs.items():
    r2a, maea, r2c, maec = [], [], [], []
    for tr, te in cv.split(X):
        tr_fit = tr[~flag.values[tr]] if clean else tr
        m = make(hub).fit(X.iloc[tr_fit], y.iloc[tr_fit])
        p = m.predict(X.iloc[te])
        te_c = ~flag.values[te]
        r2a.append(r2_score(y.iloc[te], p)); maea.append(mean_absolute_error(y.iloc[te], p))
        r2c.append(r2_score(y.iloc[te][te_c], p[te_c])); maec.append(mean_absolute_error(y.iloc[te][te_c], p[te_c]))
    rows.append(dict(config=name, R2_all=np.mean(r2a), MAE_all=np.mean(maea),
                     R2_nonflagged=np.mean(r2c), MAE_nonflagged=np.mean(maec)))
res = pd.DataFrame(rows).sort_values("R2_nonflagged", ascending=False)
print("\n", res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---- 3. deploy best by non-flagged R2 ----
best = res.iloc[0]["config"]; hub, clean = configs[best]
print("\nDeploying:", best)
fit_idx = df.index[~flag] if clean else df.index
final = make(hub).fit(X.loc[fit_idx], y.loc[fit_idx])

if "System" in df.columns:
    g = df["System"].astype(str)
    gp = cross_val_predict(make(hub), X, y, cv=GroupKFold(5), groups=g)
    print(f"GroupKFold by System (unseen alloy systems): R2={r2_score(y, gp):.3f}  MAE={mean_absolute_error(y, gp):.2f}")

bundle = {
    "model": final, "columns": COMP, "feature_set": "comp",
    "with_comp": True, "with_phys": False, "units": UNITS,
    "comp_bounds": {c: (float(df[c].min()), float(df[c].max())) for c in COMP},
    "metrics": dict(r2_all=float(res.iloc[0]["R2_all"]), mae_all=float(res.iloc[0]["MAE_all"]),
                    r2_nonflagged=float(res.iloc[0]["R2_nonflagged"]),
                    mae_nonflagged=float(res.iloc[0]["MAE_nonflagged"]),
                    n=len(df), n_flagged=int(flag.sum())),
}
joblib.dump(bundle, MODEL_FILE)
print(f"Saved {MODEL_FILE} + flagged_rows.csv")