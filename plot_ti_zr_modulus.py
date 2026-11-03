import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from features import make_ensemble_features, prepare_model_input


ROOT = Path(__file__).resolve().parent
training_header = pd.read_csv(ROOT / "train_v2.csv", nrows=0)
composition_columns = [
    column for column in training_header if column.endswith("_wt")
]
raw_features = json.loads(
    (ROOT / "feature_columns_v2.json").read_text(encoding="utf-8")
)
model_features = json.loads(
    (ROOT / "ensemble_feature_columns_v2.json").read_text(encoding="utf-8")
)
model = joblib.load(ROOT / "ti_ym_ensemble_v2.joblib")

zirconium_percentages = [1.0 + 4.5 * step for step in range(11)]
zirconium_percentages.append(50.0)
compositions = pd.DataFrame(
    0.0, index=zirconium_percentages, columns=composition_columns
)
compositions["Zr_wt"] = zirconium_percentages
compositions["Ti_wt"] = 100.0 - compositions["Zr_wt"]

engineered, _, _ = prepare_model_input(
    compositions, raw_features, composition_columns
)
model_input = make_ensemble_features(
    engineered, composition_columns
).reindex(columns=model_features)
predictions = model.predict(model_input)

plt.figure(figsize=(8, 5))
plt.plot(
    compositions["Zr_wt"], predictions, marker="o", linewidth=1.8
)
plt.xlabel("Zirconium composition (wt%)")
plt.ylabel("Predicted Young's modulus (GPa)")
plt.title("Predicted Young's modulus of binary Ti–Zr alloys")
plt.xticks(zirconium_percentages, rotation=45)
plt.grid(alpha=0.25)
plt.tight_layout()
output_file = ROOT / "ti_zr_modulus_vs_zr.png"
plt.savefig(output_file, dpi=300)
print(f"Saved graph: {output_file}")
print(f"Zr compositions (wt%): {zirconium_percentages}")
