from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from features import build_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ti_ym_model_v2.joblib"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns_v2.json"
TRAIN_PATH = BASE_DIR / "train_v2.csv"
TEST_PATH = BASE_DIR / "test_v2.csv"

ATOMIC_MASS_PLOT = {"Ti": 47.867, "Nb": 92.906, "Zr": 91.224}
ZIRCONIUM_LEVELS = [0, 4, 8, 12, 18]

REAL_POINTS = {
    0: [
        (18, 61.2), (23, 58.3), (24, 57.9), (25, 50.5), (26, 52.6),
        (27, 57.2), (28, 55.6), (30, 61.3), (35, 69.8), (40, 70.3),
    ],
    4: [
        (15, 60.0), (20, 58.2), (21, 58.1), (22, 48.3), (23, 52.9),
        (24, 52.4), (25, 54.2), (30, 57.5), (35, 63.8), (40, 64.7),
    ],
    8: [
        (16, 61.6), (18, 58.8), (19, 47.5), (20, 48.5), (21, 50.8),
        (22, 53.2), (25, 55.3), (30, 57.5), (35, 58.9), (40, 62.5),
    ],
    12: [
        (15, 57.0), (16, 58.2), (17, 44.0), (18, 55.7), (25, 57.3),
        (30, 64.3), (35, 64.6), (40, 66.8),
    ],
    18: [
        (12, 50.1), (13, 39.4), (14, 38.5), (15, 45.5), (16, 46.9),
        (17, 50.8),
    ],
}


def atomic_percent_to_wt(
    nb_at: float | np.ndarray, zr_at: float, ti_at: float | np.ndarray
) -> pd.DataFrame:
    """Convert Ti-Nb-Zr atomic percentages to normalized weight percentages."""
    nb_values = np.asarray(nb_at, dtype=float)
    ti_values = np.asarray(ti_at, dtype=float)
    weights = {
        "Ti_wt": ti_values * ATOMIC_MASS_PLOT["Ti"],
        "Nb_wt": nb_values * ATOMIC_MASS_PLOT["Nb"],
        "Zr_wt": np.full(np.broadcast_shapes(nb_values.shape, ti_values.shape), zr_at)
        * ATOMIC_MASS_PLOT["Zr"],
    }
    weighted = pd.DataFrame(weights)
    totals = weighted.sum(axis=1)
    for column in weighted.columns:
        weighted[column] = weighted[column] / totals * 100.0
    return weighted


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    with FEATURE_COLUMNS_PATH.open(encoding="utf-8") as file:
        feature_columns = json.load(file)
    model = joblib.load(MODEL_PATH)

    composition_columns = [
        column for column in train.columns if column.endswith("_wt")
    ]
    if not composition_columns:
        raise ValueError("train_v2.csv has no composition columns ending in '_wt'.")
    if "E_GPa" not in test:
        raise ValueError("test_v2.csv does not contain the E_GPa target.")

    model_input_columns = list(getattr(model, "feature_names_in_", feature_columns))
    if model_input_columns != feature_columns:
        raise ValueError(
            "feature_columns_v2.json does not match the saved model input schema: "
            f"{feature_columns!r} != {model_input_columns!r}"
        )

    def make_X(df: pd.DataFrame) -> pd.DataFrame:
        """Build the exact saved-model feature schema from raw weight percentages."""
        composition = df.reindex(columns=composition_columns, fill_value=0.0).copy()
        composition = composition.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        totals = composition.sum(axis=1)
        if (totals <= 0).any() or not np.isfinite(totals.to_numpy()).all():
            raise ValueError("Every input composition must have a finite positive total.")
        normalized = composition.div(totals, axis=0) * 100.0
        engineered = build_features(normalized)

        # This saved pipeline expects raw phase_est and one-hot encodes it internally.
        if "phase_est" not in feature_columns:
            phase_columns = [
                column for column in feature_columns
                if column.startswith("phase_est_")
            ]
            if phase_columns:
                one_hot = pd.get_dummies(
                    engineered["phase_est"], prefix="phase_est", dtype=float
                ).reindex(columns=phase_columns, fill_value=0.0)
                engineered = pd.concat(
                    [engineered.drop(columns=["phase_est"]), one_hot], axis=1
                )

        missing = [column for column in feature_columns if column not in engineered]
        if missing:
            raise ValueError(f"Feature builder did not produce: {missing}")
        return engineered.loc[:, feature_columns]

    y_test = pd.to_numeric(test["E_GPa"], errors="raise")
    test_predictions = np.asarray(model.predict(make_X(test)), dtype=float)
    test_mae = float(mean_absolute_error(y_test, test_predictions))
    test_r2 = float(r2_score(y_test, test_predictions))
    print(f"Test MAE: {test_mae:.4f} GPa")
    print(f"Test R2: {test_r2:.4f}")

    # Guard against a broken or changed inference pipeline before generating figures.
    if abs(test_mae - 9.9) > 1.0 or abs(test_r2 - 0.68) > 0.05:
        raise SystemExit(
            "Feature-pipeline check failed: expected test MAE near 9.9 GPa and "
            "R2 near 0.68; no plots were written."
        )

    parity_fig, parity_ax = plt.subplots(figsize=(7, 7))
    parity_ax.scatter(y_test, test_predictions, alpha=0.8, edgecolors="none")
    limits = [
        min(float(y_test.min()), float(test_predictions.min())),
        max(float(y_test.max()), float(test_predictions.max())),
    ]
    parity_ax.plot(limits, limits, linestyle="--", color="black", linewidth=1.2)
    parity_ax.set(
        xlabel="Actual Young's modulus (GPa)",
        ylabel="Predicted Young's modulus (GPa)",
        title=f"Test parity — MAE {test_mae:.2f} GPa, R² {test_r2:.3f}",
        xlim=limits,
        ylim=limits,
    )
    parity_ax.grid(alpha=0.25)
    parity_fig.tight_layout()
    parity_fig.savefig(BASE_DIR / "parity_test.png", dpi=300, bbox_inches="tight")
    plt.close(parity_fig)

    nb_grid_at = np.arange(10.0, 40.0 + 0.25, 0.5)
    fig, axes = plt.subplots(
        1, len(ZIRCONIUM_LEVELS), figsize=(19, 4.8), sharey=True
    )
    all_real_true: list[float] = []
    all_real_predicted: list[float] = []
    train_composition = train.reindex(
        columns=composition_columns, fill_value=0.0
    ).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    train_e = pd.to_numeric(train["E_GPa"], errors="raise").to_numpy()
    in_train_matches = 0
    matched_points = []

    for ax, zr_at in zip(axes, ZIRCONIUM_LEVELS):
        curve = atomic_percent_to_wt(
            nb_grid_at, zr_at, 100.0 - nb_grid_at - zr_at
        )
        curve_predictions = np.asarray(model.predict(make_X(curve)), dtype=float)
        ax.plot(nb_grid_at, curve_predictions, linewidth=1.8, label="Model curve")

        points = REAL_POINTS[zr_at]
        real_nb_at = np.array([point[0] for point in points], dtype=float)
        real_e = np.array([point[1] for point in points], dtype=float)
        real_composition = atomic_percent_to_wt(
            real_nb_at, zr_at, 100.0 - real_nb_at - zr_at
        )
        real_predictions = np.asarray(
            model.predict(make_X(real_composition)), dtype=float
        )
        all_real_true.extend(real_e.tolist())
        all_real_predicted.extend(real_predictions.tolist())
        ax.scatter(
            real_nb_at, real_e, color="red", s=28, zorder=3,
            label="Real data" if zr_at == ZIRCONIUM_LEVELS[0] else None,
        )
        ax.set_title(f"Zr = {zr_at} at.%")
        ax.set_xlabel("Nb (at.%)")
        ax.grid(alpha=0.25)

        for index, (nb_at, observed_e) in enumerate(points):
            point_wt = real_composition.iloc[index]
            close_composition = (
                train_composition[composition_columns]
                .sub(point_wt.reindex(composition_columns, fill_value=0.0), axis=1)
                .abs()
                .le(0.5)
                .all(axis=1)
            )
            close_e = np.abs(train_e - observed_e) <= 0.5
            match_indices = np.flatnonzero(close_composition.to_numpy() & close_e)
            if match_indices.size:
                in_train_matches += 1
                matched_points.append((zr_at, nb_at, observed_e))

    axes[0].set_ylabel("Young's modulus (GPa)")
    axes[-1].legend(loc="best")
    fig.suptitle("Ti-Nb-Zr: model predictions and reported data")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "nb_zr_vs_real.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    overall_real_mae = float(
        mean_absolute_error(all_real_true, all_real_predicted)
    )
    print(
        "Real-point model MAE overall: "
        f"{overall_real_mae:.4f} GPa ({len(all_real_true)} points)"
    )
    for zr_at in ZIRCONIUM_LEVELS:
        points = REAL_POINTS[zr_at]
        truth = np.array([point[1] for point in points], dtype=float)
        compositions = atomic_percent_to_wt(
            np.array([point[0] for point in points], dtype=float),
            zr_at,
            100.0 - np.array([point[0] for point in points], dtype=float) - zr_at,
        )
        predictions = np.asarray(model.predict(make_X(compositions)), dtype=float)
        print(
            f"Real-point model MAE at Zr {zr_at} at.%: "
            f"{mean_absolute_error(truth, predictions):.4f} GPa "
            f"({len(truth)} points)"
        )

    print(
        f"Real points matching train_v2.csv within 0.5 wt% composition "
        f"and 0.5 GPa E: {in_train_matches}/{len(all_real_true)} "
        f"({'yes' if in_train_matches else 'no'})"
    )
    if matched_points:
        print("Matched (Zr at.%, Nb at.%, E GPa):", matched_points)

    nb_wt_grid = np.arange(0.0, 50.0 + 0.25, 0.5)
    binary_composition = pd.DataFrame(
        {
            "Ti_wt": 100.0 - nb_wt_grid,
            "Nb_wt": nb_wt_grid,
        }
    )
    binary_predictions = np.asarray(
        model.predict(make_X(binary_composition)), dtype=float
    )
    variation_fig, variation_ax = plt.subplots(figsize=(8, 5.5))
    variation_ax.plot(nb_wt_grid, binary_predictions, linewidth=2)
    variation_ax.set(
        xlabel="Nb (wt.%)",
        ylabel="Predicted Young's modulus (GPa)",
        title="Binary Ti-Nb Young's modulus prediction",
    )
    variation_ax.grid(alpha=0.3)
    variation_fig.tight_layout()
    variation_fig.savefig(
        BASE_DIR / "nb_variation_graph.png", dpi=300, bbox_inches="tight"
    )
    plt.close(variation_fig)

    print("Saved plots: parity_test.png, nb_zr_vs_real.png, nb_variation_graph.png")


if __name__ == "__main__":
    main()
