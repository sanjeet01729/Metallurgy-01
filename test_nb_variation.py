
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Load model relative to this script's directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "titanium_ym_pipeline.pkl"

model = joblib.load(MODEL_PATH)

# Determine model and input features
if isinstance(model, dict):
    pipeline = model["model"]
    columns = model.get("columns")
else:
    pipeline = model
    columns = getattr(pipeline, "feature_names_in_", None)

# Recover feature names from pipeline steps if needed
if columns is None and hasattr(pipeline, "named_steps"):
    for estimator in reversed(list(pipeline.named_steps.values())):
        if hasattr(estimator, "feature_names_in_"):
            columns = estimator.feature_names_in_
            break

if columns is None:
    raise ValueError("Could not determine model input columns.")

columns = list(columns)

if "Ti" not in columns or "Nb" not in columns:
    raise ValueError(
        f"Expected Ti and Nb features. Available features: {columns}"
    )

# Generate binary Ti-Nb compositions: 0 to 50 wt.% Nb
nb_values = np.arange(0, 50.5, 0.5)
results = []

for nb in nb_values:
    sample = {col: 0.0 for col in columns}

    sample["Ti"] = 100.0 - nb
    sample["Nb"] = float(nb)

    X = pd.DataFrame([sample], columns=columns)
    ym = float(np.asarray(pipeline.predict(X)).ravel()[0])

    results.append({
        "Ti (wt.%)": 100.0 - nb,
        "Nb (wt.%)": float(nb),
        "Predicted YM (GPa)": round(ym, 2),
    })

df = pd.DataFrame(results)

print("\nTi-Nb Young's Modulus Predictions")
print(df.to_string(index=False))

# Save predictions
df.to_csv(BASE_DIR / "nb_variation_predictions.csv", index=False)
print("\nSaved: nb_variation_predictions.csv")

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(
    df["Nb (wt.%)"],
    df["Predicted YM (GPa)"],
    marker="o",
    linewidth=2,
    markersize=3,
    label="Predicted Young's modulus",
)

plt.xlabel("Niobium concentration, Nb (wt.%)")
plt.ylabel("Predicted Young's modulus (GPa)")
plt.title("Ti-Nb Alloy: Young's Modulus vs Nb Concentration")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save and display graph
plt.savefig(
    BASE_DIR / "nb_variation_graph.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("Saved: nb_variation_graph.png")
