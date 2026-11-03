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

niobium_percentages = range(41)
tantalum_levels = (0, 5, 10, 15, 20)
fig, ax = plt.subplots(figsize=(9, 6))

for tantalum_wt in tantalum_levels:
    compositions = pd.DataFrame(
        0.0, index=niobium_percentages, columns=composition_columns
    )
    compositions["Nb_wt"] = niobium_percentages
    compositions["Ta_wt"] = tantalum_wt
    compositions["Ti_wt"] = 100.0 - compositions["Nb_wt"] - tantalum_wt

    engineered, ti_at, _ = prepare_model_input(
        compositions, raw_features, composition_columns
    )
    model_input = make_ensemble_features(
        engineered, composition_columns
    ).reindex(columns=model_features)
    predictions = model.predict(model_input)

    ax.plot(
        compositions["Nb_wt"],
        predictions,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label=f"Ta = {tantalum_wt} wt%",
    )

ax.set_xlabel("Niobium composition (wt%)")
ax.set_ylabel("Predicted Young's modulus (GPa)")
ax.set_title("Predicted Young's modulus of Ti–Nb–Ta alloys")
ax.set_xlim(0, 40)
ax.set_xticks(range(0, 41, 2))
ax.grid(alpha=0.25)
ax.legend(title="Fixed tantalum")
fig.tight_layout()
output_file = ROOT / "ti_nb_ta_modulus_vs_nb.png"
fig.savefig(output_file, dpi=300)
print(f"Saved graph: {output_file}")
print(
    "Plotted Nb = 0–40 wt% in 1 wt% steps; "
    "Ta = 0, 5, 10, 15, 20 wt%; Ti is the balance."
)
