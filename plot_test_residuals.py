import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from compare_train_models_v2 import feature_frame


ROOT = Path(__file__).resolve().parent
train = pd.read_csv(ROOT / "train_v2.csv")
test = pd.read_csv(ROOT / "test_v2.csv")
composition_columns = [
    column for column in train.columns if column.endswith("_wt")
]
feature_columns = json.loads(
    (ROOT / "ensemble_feature_columns_v2.json").read_text(encoding="utf-8")
)

X_test = feature_frame(test, composition_columns).reindex(
    columns=feature_columns
)
model = joblib.load(ROOT / "ti_ym_ensemble_v2.joblib")
actual = test["E_GPa"].to_numpy(dtype=float)
predicted = model.predict(X_test)
residual = actual - predicted

r2 = r2_score(actual, predicted)
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
residual_mean = float(np.mean(residual))

fig, (ax_residual, ax_distribution) = plt.subplots(
    1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.65, 1]}
)
ax_residual.scatter(
    predicted,
    residual,
    alpha=0.78,
    s=38,
    color="#2878B5",
    edgecolors="white",
    linewidths=0.45,
)
ax_residual.axhline(0, color="#333333", linestyle="--", linewidth=1.5)
ax_residual.set_xlabel("Predicted Young's modulus (GPa)")
ax_residual.set_ylabel("Residual: measured − predicted (GPa)")
ax_residual.set_title("Test residuals vs. prediction")
ax_residual.grid(alpha=0.22)

ax_distribution.hist(
    residual,
    bins="fd",
    color="#68A6D3",
    edgecolor="white",
    alpha=0.9,
)
ax_distribution.axvline(0, color="#333333", linestyle="--", linewidth=1.5, label="Zero error")
ax_distribution.axvline(
    residual_mean,
    color="#D95F59",
    linewidth=1.8,
    label=f"Mean residual = {residual_mean:.2f} GPa",
)
ax_distribution.set_xlabel("Residual: measured − predicted (GPa)")
ax_distribution.set_ylabel("Number of test alloys")
ax_distribution.set_title("Residual distribution")
ax_distribution.grid(axis="y", alpha=0.22)
ax_distribution.legend(frameon=False, fontsize=9)

fig.suptitle(
    f"Held-out test residual analysis (n = {len(test)})\n"
    f"R² = {r2:.3f}  |  MAE = {mae:.2f} GPa  |  RMSE = {rmse:.2f} GPa",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.90))
output_file = ROOT / "test_residuals_v2.png"
fig.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved graph: {output_file}")
print(
    f"Test metrics: R2={r2:.4f}, MAE={mae:.4f} GPa, "
    f"RMSE={rmse:.4f} GPa, mean residual={residual_mean:.4f} GPa"
)
