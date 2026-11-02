from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import prepare_model_input


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    train = pd.read_csv(BASE_DIR / "train_v2.csv")
    model = joblib.load(BASE_DIR / "ti_ym_model_v2.joblib")
    feature_columns = pd.read_json(
        BASE_DIR / "feature_columns_v2.json", typ="series"
    ).tolist()
    composition_columns = [
        column for column in train.columns if column.endswith("_wt")
    ]

    max_sn = float(train["Sn_wt"].max())
    sn_wt = np.arange(0.0, max_sn, 0.5)
    if not np.isclose(sn_wt[-1], max_sn):
        sn_wt = np.append(sn_wt, max_sn)

    compositions = pd.DataFrame(0.0, index=range(len(sn_wt)), columns=composition_columns)
    compositions["Sn_wt"] = sn_wt
    compositions["Ti_wt"] = 100.0 - sn_wt

    model_input, _, _ = prepare_model_input(
        compositions, feature_columns, composition_columns
    )
    predictions = np.asarray(model.predict(model_input), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(sn_wt, predictions, linewidth=2)
    ax.set(
        xlabel="Sn (wt.%)",
        ylabel="Predicted Young's modulus (GPa)",
        title="Binary Ti-Sn Young's modulus prediction",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output = BASE_DIR / "sn_variation_graph.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Sn range: 0 to {max_sn:.2f} wt.% (maximum in train_v2.csv)")
    print(f"Predicted E range: {predictions.min():.2f} to {predictions.max():.2f} GPa")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
