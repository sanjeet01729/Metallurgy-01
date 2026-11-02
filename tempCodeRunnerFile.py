import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "titanium_ym_pipeline.pkl"

model = joblib.load(MODEL_PATH)

# Determine the input features expected by the saved model
if isinstance(model, dict):
    pipeline = model["model"]
    columns = model.get("columns")
else:
    pipeline = model
    columns = getattr(pipeline, "feature_names_in_", None)

# Try retrieving feature names from the final estimator
if columns is None and hasattr(pipeline, "named_steps"):
    for estimator in reversed(list(pipeline.named_steps.values())):
        if hasattr(estimator, "feature_names_in_"):
            columns = estimator.feature_names_in_
            break

if columns is None:
    raise ValueError(
        "Could not determine model input columns. "
        "Check how the model was trained and saved."
    )

columns = list(columns)

# Generate binary Ti-Nb compositions
nb_values = np.arange(30, 51, 0.5)
results = []

for nb in nb_values:
    sample = {col: 0.0 for col in columns}

    if "Ti" not in sample or "Nb" not in sample:
        raise ValueError(
            f"Expected Ti and Nb features. Model features: {columns}"
        )

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

df.to_csv("nb_variation_predictions.csv", index=False)
print("\nSaved: nb_variation_predictions.csv")


# Plot Nb concentration vs predicted Young's modulus
plt.figure(figsize=(10, 6))

plt.plot(
    df["Nb (wt.%)"],
    df["Predicted YM (GPa)"],
    marker="o",
    linewidth=2,
    markersize=4,
    label="Predicted Young's modulus"
)

plt.xlabel("Niobium concentration, Nb (wt.%)")
plt.ylabel("Predicted Young's modulus (GPa)")
plt.title("Ti-Nb Alloy: Young's Modulus vs Nb Concentration")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# Save the graph
plt.savefig("nb_variation_graph.png", dpi=300, bbox_inches="tight")

# Display the graph
plt.show()

print("Saved graph: nb_variation_graph.png")
