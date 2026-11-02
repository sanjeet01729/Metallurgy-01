import re, unicodedata, joblib
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

BASE_FILE  = "titanium_alloys.csv"          # existing data (only COMP + YM used)
LIT_FILE   = "literature_extra.csv"         # name,comp,YM  (comp like "Ti73 Nb20 Fe3 Sn4")
OUT_MODEL  = "titanium_ym_pipeline_merged.pkl"
OUT_CSV    = "titanium_merged.csv"
TARGET     = "YM (GPa)"
COMP       = ["Ti","Nb","Zr","Fe","Sn","Ta","Mo","Al","V","Cr","Cu","Hf","Mn","W","Sc","Si"]
UNITS      = "wt"
MAX_OTHER  = 0.3     # drop lit rows with > this wt% of elements not in COMP (O, Pd, Ni, In, B...)
SEEDS      = 3

norm = lambda s: unicodedata.normalize("NFC", str(s).strip())

def make():
    return XGBRegressor(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=3,
                        subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, random_state=42, n_jobs=-1)

def metrics(y, p):
    return r2_score(y, p), mean_absolute_error(y, p)

# ---------- load base ----------
base = pd.read_csv(BASE_FILE)
base.columns = [norm(c) for c in base.columns]
base = base.dropna(subset=[TARGET]).reset_index(drop=True)
Xb, yb = base[COMP].astype(float), base[TARGET].astype(float)

# ---------- parse literature ----------
raw = pd.read_csv(LIT_FILE)
rows = []
for _, r in raw.iterrows():
    d = {}
    for el, v in re.findall(r"([A-Z][a-z]?)\s*([\d.]+)", str(r["comp"])):
        d[el] = d.get(el, 0.0) + float(v)
    other = sum(v for k, v in d.items() if k not in COMP)
    if other > MAX_OTHER:
        continue
    c = np.array([d.get(e, 0.0) for e in COMP])
    c = c / c.sum() * 100
    rows.append(dict(zip(COMP, c), YM=float(r["YM"])))
lit = pd.DataFrame(rows)
n_raw = len(raw)
lit["key"] = lit[COMP].round(2).astype(str).agg("|".join, axis=1)
lit = lit.groupby("key", as_index=False).agg({**{e: "first" for e in COMP}, "YM": "mean"})
print(f"Literature: {n_raw} raw rows -> {len(lit)} unique compositions "
      f"(after dropping rows with >{MAX_OTHER}% other elements and averaging ST/STA/aged duplicates)")
Xl, yl = lit[COMP], lit["YM"]

key_b = Xb.round(1).astype(str).agg("|".join, axis=1)
key_l = Xl.round(1).astype(str).agg("|".join, axis=1)
print(f"Lit compositions already present in base data: {key_l.isin(set(key_b)).sum()}")

# ---------- 1. external test: train on base, predict lit ----------
m0 = make().fit(Xb, yb)
pl = m0.predict(Xl)
r2, mae = metrics(yl, pl)
print(f"\n[EXTERNAL] train on base -> predict lit: R2={r2:.3f}  MAE={mae:.2f} GPa")
err = (yl - pl)
worst = err.abs().sort_values(ascending=False).index[:8]
print("Worst lit rows (true vs pred):")
for i in worst:
    comp = " ".join(f"{e}{Xl.loc[i, e]:.4g}" for e in COMP if Xl.loc[i, e] > 0)
    print(f"  {comp:40s} true {yl[i]:6.1f}  pred {pl[i]:6.1f}")

# ---------- 2. grouped CV (same composition never split across folds) ----------
def cv_oof(X, y, keys, seed):
    uniq = keys.unique()
    rng = np.random.RandomState(seed)
    fold = dict(zip(uniq, rng.randint(0, 5, len(uniq))))
    f = keys.map(fold).values
    oof = np.zeros(len(y))
    for k in range(5):
        te = f == k
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        oof[te] = make().fit(X[~te], y[~te]).predict(X[te])
    return oof

Xm = pd.concat([Xb, Xl], ignore_index=True)
ym = pd.concat([yb, yl], ignore_index=True)
km = pd.concat([key_b, key_l], ignore_index=True)
is_lit = np.r_[np.zeros(len(Xb), bool), np.ones(len(Xl), bool)]

resA, resB_base, resB_lit, resB_all, abs_err = [], [], [], [], []
for s in range(SEEDS):
    oa = cv_oof(Xb, yb, key_b, s)
    ob = cv_oof(Xm, ym, km, s)
    resA.append(metrics(yb, oa))
    resB_base.append(metrics(ym[~is_lit], ob[~is_lit]))
    resB_lit.append(metrics(ym[is_lit], ob[is_lit]))
    resB_all.append(metrics(ym, ob))
    abs_err.append(np.abs(ym.values - ob))
avg = lambda L: tuple(np.mean(L, axis=0))
print("\n=== GROUPED CV ===")
print("base-only  -> base rows : R2=%.3f MAE=%.2f" % avg(resA))
print("merged     -> base rows : R2=%.3f MAE=%.2f" % avg(resB_base))
print("merged     -> lit rows  : R2=%.3f MAE=%.2f" % avg(resB_lit))
print("merged     -> all rows  : R2=%.3f MAE=%.2f" % avg(resB_all))

# ---------- 3. final model on merged ----------
final = make().fit(Xm, ym)
q90 = float(np.quantile(np.concatenate(abs_err), 0.90))
bundle = {
    "model": final, "columns": COMP, "feature_set": "comp",
    "with_comp": True, "with_phys": False, "units": UNITS,
    "comp_bounds": {c: (float(Xm[c].min()), float(Xm[c].max())) for c in COMP},
    "interval_q90": q90,
    "metrics": dict(r2_all=avg(resB_all)[0], mae_all=avg(resB_all)[1], n=len(Xm), n_lit=len(Xl)),
}
joblib.dump(bundle, OUT_MODEL)
out = Xm.copy(); out[TARGET] = ym; out["source"] = np.where(is_lit, "lit", "base")
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_MODEL} (90% interval ±{q90:.1f} GPa) and {OUT_CSV}")