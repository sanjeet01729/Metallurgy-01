"""Shared by train_model_v2.py and app.py. One source of truth for features."""
import numpy as np
import pandas as pd

COMP = ["Ti","Nb","Zr","Fe","Sn","Ta","Mo","Al","V","Cr","Cu","Hf","Mn","W","Sc","Si"]

# Set to "wt" or "at" depending on how your dataset composition is given
UNITS = "wt"

#           VEC  r(pm)  chi(P)  mass     rho(g/cc)  Moeq(wt)
P = {
 "Ti": (4,  146, 1.54,  47.867,  4.506,  0.0),
 "Nb": (5,  146, 1.60,  92.906,  8.570,  0.28),
 "Zr": (4,  160, 1.33,  91.224,  6.520,  0.0),
 "Fe": (8,  126, 1.83,  55.845,  7.874,  2.9),
 "Sn": (4,  145, 1.96, 118.710,  7.310,  0.0),
 "Ta": (5,  146, 1.50, 180.948, 16.690,  0.22),
 "Mo": (6,  139, 2.16,  95.950, 10.280,  1.0),
 "Al": (3,  143, 1.61,  26.982,  2.700, -1.0),
 "V":  (5,  134, 1.63,  50.942,  6.110,  0.67),
 "Cr": (6,  128, 1.66,  51.996,  7.190,  1.6),
 "Cu": (11, 128, 1.90,  63.546,  8.960,  1.3),
 "Hf": (4,  159, 1.30, 178.490, 13.310,  0.0),
 "Mn": (7,  127, 1.55,  54.938,  7.210,  1.6),
 "W":  (6,  139, 2.36, 183.840, 19.250,  0.44),
 "Sc": (3,  162, 1.36,  44.956,  2.985,  0.0),
 "Si": (4,  117, 1.90,  28.086,  2.330,  0.0),
}
VEC  = np.array([P[e][0] for e in COMP], float)
R    = np.array([P[e][1] for e in COMP], float)
CHI  = np.array([P[e][2] for e in COMP], float)
MASS = np.array([P[e][3] for e in COMP], float)
RHO  = np.array([P[e][4] for e in COMP], float)
MOEQ = np.array([P[e][5] for e in COMP], float)

PHYS = ["VEC", "r_mean", "delta_r", "chi_mean", "delta_chi", "dS_mix",
        "rho_mix", "Mo_eq", "Mo_eq_x_VEC", "Ti_frac"]


def featurize(comp: pd.DataFrame, units: str = UNITS, with_comp: bool = True,
              with_phys: bool = True) -> pd.DataFrame:
    c = comp.reindex(columns=COMP).fillna(0.0).astype(float).values
    c = c / c.sum(axis=1, keepdims=True)          # normalize to fractions
    if units == "wt":
        w = c
        a = (w / MASS) / (w / MASS).sum(axis=1, keepdims=True)
    else:
        a = c
        w = (a * MASS) / (a * MASS).sum(axis=1, keepdims=True)

    out = pd.DataFrame(index=comp.index)
    if with_comp:
        for i, e in enumerate(COMP):
            out[e] = comp.reindex(columns=COMP).fillna(0.0).iloc[:, i].values
    if with_phys:
        vec = a @ VEC
        rm = a @ R
        chim = a @ CHI
        out["VEC"] = vec
        out["r_mean"] = rm
        out["delta_r"] = 100 * np.sqrt((a * (1 - R / rm[:, None]) ** 2).sum(axis=1))
        out["chi_mean"] = chim
        out["delta_chi"] = np.sqrt((a * (CHI - chim[:, None]) ** 2).sum(axis=1))
        out["dS_mix"] = -8.314 * np.where(a > 0, a * np.log(np.where(a > 0, a, 1)), 0).sum(axis=1)
        out["rho_mix"] = 1.0 / (w / RHO).sum(axis=1)
        moeq = w * 100 @ MOEQ
        out["Mo_eq"] = moeq
        out["Mo_eq_x_VEC"] = moeq * vec
        out["Ti_frac"] = a[:, 0]
    return out


ATOMIC_MASS = {
    "Ti": 47.867, "Nb": 92.906, "Zr": 91.224, "Al": 26.9815385,
    "Co": 58.933194, "Cr": 51.9961, "Cu": 63.546, "Fe": 55.845,
    "Hf": 178.49, "Mn": 54.938044, "Mo": 95.95, "O": 15.999,
    "Pd": 106.42, "Sc": 44.955908, "Si": 28.085, "Sn": 118.710,
    "Ta": 180.94788, "V": 50.9415, "W": 183.84,
}

VEC_TABLE = {
    "Ti": 4, "Zr": 4, "Hf": 4, "Al": 3, "Sn": 4, "Si": 4,
    "V": 5, "Nb": 5, "Ta": 5, "Cr": 6, "Mo": 6, "W": 6,
    "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Pd": 10,
    "O": 6, "N": 5,
}

METALLIC_RADIUS_PM = {
    "Ti": 146, "Nb": 146, "Zr": 160, "Al": 143, "Co": 126,
    "Cr": 128, "Cu": 128, "Fe": 126, "Hf": 159, "Mn": 127,
    "Mo": 139, "Pd": 137, "Sc": 162, "Si": 117, "Sn": 145,
    "Ta": 146, "V": 134, "W": 139,
}

MORINAGA = {
    "Ti": (2.790, 2.447), "Zr": (3.086, 2.934), "Hf": (3.110, 3.020),
    "V": (2.805, 1.543), "Nb": (3.099, 2.424), "Ta": (3.144, 2.531),
    "Cr": (2.779, 1.478), "Mo": (3.063, 1.961), "W": (3.125, 2.072),
    "Mn": (2.723, 1.461), "Fe": (2.651, 0.969), "Co": (2.529, 0.807),
    "Ni": (2.412, 0.724), "Cu": (2.114, 0.567), "Al": (2.426, 2.200),
    "Sn": (2.283, 2.100), "Si": (2.561, 2.200),
}

MO_EQ_COEFFICIENT = {
    "Mo": 1.0, "Ta": 0.2, "Nb": 0.28, "W": 0.4, "V": 0.67,
    "Cr": 1.25, "Ni": 1.25, "Mn": 1.7, "Co": 1.7, "Fe": 2.5,
    "Al": -1.0,
}
AL_EQ_COEFFICIENT = {
    "Al": 1.0, "Zr": 1.0 / 6.0, "Sn": 1.0 / 3.0, "O": 10.0,
}
GAS_CONSTANT = 8.314462618
ALPHA_EQ_UNCERTAIN_CUTOFF = 8.0
ALPHA_BETA_MO_EQ_CUTOFF = 2.5
BETA_MO_EQ_CUTOFF = 10.0
BUILD_FEATURE_COLUMNS = [
    "VEC", "delta_r", "delta_S_mix", "Mo_eq", "Al_eq", "Bo", "Md",
    "phase_est", "phase_note", "ti_based",
]
ENSEMBLE_PHASES = ("alpha", "alpha+beta", "beta", "uncertain")
ENSEMBLE_NUMERIC_FEATURES = (
    "VEC", "delta_r", "delta_S_mix", "Mo_eq", "Al_eq", "Bo", "Md",
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the v2 composition-derived feature columns to a wt% dataframe."""
    element_columns = [
        column for column in df.columns if column.endswith("_wt")
    ]
    if not element_columns:
        raise ValueError("No composition columns ending in '_wt' were found.")
    elements = [column[:-3] for column in element_columns]
    missing_masses = sorted(set(elements) - set(ATOMIC_MASS))
    if missing_masses:
        raise ValueError(f"Atomic masses unavailable for elements: {missing_masses}")

    composition = df[element_columns].apply(pd.to_numeric, errors="coerce")
    any_input_nan = df.isna().any(axis=1) | composition.isna().any(axis=1)
    raw_values = composition.to_numpy(dtype=float)
    finite_composition = np.isfinite(raw_values).all(axis=1)
    negative_composition = (composition < 0).any(axis=1).to_numpy()
    raw_sums = np.nansum(raw_values, axis=1)
    valid = (
        ~any_input_nan.to_numpy()
        & finite_composition
        & ~negative_composition
        & (raw_sums > 0)
    )
    normalized = np.full_like(raw_values, np.nan, dtype=float)
    normalized[valid] = raw_values[valid] / raw_sums[valid, None] * 100.0
    mass_array = np.array([ATOMIC_MASS[element] for element in elements], dtype=float)
    weighted_moles = normalized / mass_array
    atomic_fraction = weighted_moles / np.nansum(
        weighted_moles, axis=1, keepdims=True
    )
    weight_fraction = normalized / 100.0
    element_index = {element: index for index, element in enumerate(elements)}

    def atomic_linear(values):
        known = [element for element in elements if element in values]
        indices = [element_index[element] for element in known]
        coefficients = np.array([values[element] for element in known], dtype=float)
        return atomic_fraction[:, indices] @ coefficients

    def weight_linear(values):
        known = [element for element in elements if element in values]
        indices = [element_index[element] for element in known]
        coefficients = np.array([values[element] for element in known], dtype=float)
        return (weight_fraction[:, indices] * 100.0) @ coefficients

    features = pd.DataFrame(index=df.index, columns=BUILD_FEATURE_COLUMNS)
    features["VEC"] = atomic_linear(VEC_TABLE)
    vec_missing = [
        element for element in elements
        if element not in VEC_TABLE
        and np.any(valid & (normalized[:, element_index[element]] > 0))
    ]
    if vec_missing:
        vec_missing_mask = np.zeros(len(df), dtype=bool)
        for element in vec_missing:
            vec_missing_mask |= normalized[:, element_index[element]] > 0
        features.loc[vec_missing_mask, "VEC"] = np.nan

    radius_missing = [
        element for element in elements
        if element not in METALLIC_RADIUS_PM
        and np.any(valid & (normalized[:, element_index[element]] > 0))
    ]
    radius_available = [
        element for element in elements if element in METALLIC_RADIUS_PM
    ]
    radius_indices = [element_index[element] for element in radius_available]
    radius_array = np.array(
        [METALLIC_RADIUS_PM[element] for element in radius_available], dtype=float
    )
    radius_atomic = atomic_fraction[:, radius_indices]
    radius_mean = radius_atomic @ radius_array
    features["delta_r"] = 100.0 * np.sqrt(
        np.sum(
            radius_atomic
            * (1.0 - radius_array[None, :] / radius_mean[:, None]) ** 2,
            axis=1,
        )
    )
    if radius_missing:
        radius_missing_mask = np.zeros(len(df), dtype=bool)
        for element in radius_missing:
            radius_missing_mask |= normalized[:, element_index[element]] > 0
        features.loc[radius_missing_mask, "delta_r"] = np.nan

    entropy_terms = np.zeros_like(atomic_fraction)
    positive = atomic_fraction > 0
    entropy_terms[positive] = (
        atomic_fraction[positive] * np.log(atomic_fraction[positive])
    )
    features["delta_S_mix"] = np.maximum(
        -GAS_CONSTANT * np.sum(entropy_terms, axis=1), 0.0
    )
    features["Mo_eq"] = weight_linear(MO_EQ_COEFFICIENT)
    features["Al_eq"] = weight_linear(AL_EQ_COEFFICIENT)

    morinaga_missing = [
        element for element in elements
        if element not in MORINAGA
        and np.any(valid & (normalized[:, element_index[element]] > 0))
    ]
    morinaga_available = [
        element for element in elements if element in MORINAGA
    ]
    morinaga_indices = [element_index[element] for element in morinaga_available]
    bo_values = np.array(
        [MORINAGA[element][0] for element in morinaga_available]
    )
    md_values = np.array(
        [MORINAGA[element][1] for element in morinaga_available]
    )
    features["Bo"] = atomic_fraction[:, morinaga_indices] @ bo_values
    features["Md"] = atomic_fraction[:, morinaga_indices] @ md_values
    if morinaga_missing:
        morinaga_missing_mask = np.zeros(len(df), dtype=bool)
        for element in morinaga_missing:
            morinaga_missing_mask |= normalized[:, element_index[element]] > 0
        features.loc[morinaga_missing_mask, ["Bo", "Md"]] = np.nan

    ti_weight = (
        weight_fraction[:, element_index["Ti"]] * 100.0
        if "Ti" in element_index
        else np.zeros(len(df), dtype=float)
    )
    ti_based = ti_weight >= 70.0
    features["ti_based"] = pd.Series(
        ti_based, index=df.index, dtype="boolean"
    )
    phase = np.full(len(df), "uncertain", dtype=object)
    phase_notes = np.full(len(df), "", dtype=object)
    for row_index in range(len(df)):
        if not valid[row_index]:
            phase_notes[row_index] = (
                "Input row has NaN, non-finite, negative, or zero-total composition."
            )
        elif not ti_based[row_index]:
            phase_notes[row_index] = (
                "Ti < 70 wt%; not treated as a Ti-based alloy for phase screening."
            )
        elif features["Al_eq"].iloc[row_index] >= ALPHA_EQ_UNCERTAIN_CUTOFF:
            phase_notes[row_index] = (
                f"Al_eq >= {ALPHA_EQ_UNCERTAIN_CUTOFF:g} wt%; strong alpha stabilization "
                "makes Mo_eq-only classification uncertain."
            )
        else:
            mo_eq = features["Mo_eq"].iloc[row_index]
            if mo_eq <= BETA_MO_EQ_CUTOFF:
                if mo_eq < ALPHA_BETA_MO_EQ_CUTOFF:
                    phase[row_index] = "alpha"
                    phase_notes[row_index] = (
                        f"Mo_eq < {ALPHA_BETA_MO_EQ_CUTOFF:g} wt%; conventional "
                        "Mo_eq screening range, with near-alpha alloys grouped as alpha."
                    )
                else:
                    phase[row_index] = "alpha+beta"
                    phase_notes[row_index] = (
                        f"{ALPHA_BETA_MO_EQ_CUTOFF:g} <= Mo_eq <= "
                        f"{BETA_MO_EQ_CUTOFF:g} wt%; conventional alpha+beta screening range."
                    )
            else:
                phase[row_index] = "beta"
                phase_notes[row_index] = (
                    f"Mo_eq > {BETA_MO_EQ_CUTOFF:g} wt%; conventional beta-alloy screening range."
                )
    features["phase_est"] = phase
    features["phase_note"] = phase_notes
    features.loc[~valid, BUILD_FEATURE_COLUMNS] = np.nan
    return pd.concat([df.copy(), features[BUILD_FEATURE_COLUMNS]], axis=1)


def prepare_model_input(
    composition: pd.DataFrame,
    feature_columns: list[str],
    trained_composition_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Normalize raw wt% inputs, engineer features, and select model columns."""
    values = composition.copy()
    unknown_elements = [
        column for column in values.columns
        if column.endswith("_wt") and column not in trained_composition_columns
        and pd.to_numeric(values[column], errors="coerce").fillna(0).ne(0).any()
    ]
    for column in trained_composition_columns:
        if column not in values:
            values[column] = 0.0
    values = values[trained_composition_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    totals = values.sum(axis=1, min_count=1)
    if totals.isna().any() or (totals <= 0).any():
        raise ValueError("Composition must have a positive, non-missing total.")
    normalized = values.div(totals, axis=0) * 100.0
    engineered = build_features(normalized)
    if "Ti_wt" not in normalized:
        raise ValueError("The model composition columns do not include Ti_wt.")
    masses = np.array(
        [ATOMIC_MASS[column[:-3]] for column in trained_composition_columns],
        dtype=float,
    )
    moles = normalized.to_numpy(dtype=float) / masses[None, :]
    atomic_fractions = moles / moles.sum(axis=1, keepdims=True)
    ti_at = pd.Series(
        atomic_fractions[:, trained_composition_columns.index("Ti_wt")] * 100.0,
        index=composition.index,
        name="Ti_at",
    )
    missing_feature_columns = [
        column for column in feature_columns if column not in engineered
    ]
    if missing_feature_columns:
        raise ValueError(
            f"Engineered model features are missing: {missing_feature_columns}"
        )
    model_input = engineered[feature_columns].copy()
    return model_input, ti_at, unknown_elements


def make_ensemble_features(
    engineered: pd.DataFrame,
    composition_columns: list[str],
) -> pd.DataFrame:
    """Create the numeric, fixed-category input matrix used by the v2 ensemble."""
    required = [
        *composition_columns,
        *ENSEMBLE_NUMERIC_FEATURES,
        "phase_est",
    ]
    missing = [column for column in required if column not in engineered.columns]
    if missing:
        raise ValueError(f"Ensemble input is missing required features: {missing}")

    numeric = engineered[
        [*composition_columns, *ENSEMBLE_NUMERIC_FEATURES]
    ].apply(pd.to_numeric, errors="coerce")
    phases = pd.Categorical(
        engineered["phase_est"],
        categories=ENSEMBLE_PHASES,
    )
    one_hot = pd.get_dummies(
        pd.Series(phases, index=engineered.index, name="phase_est"),
        prefix="phase_est",
        dtype=float,
    )
    phase_columns = [
        f"phase_est_{phase}" for phase in ENSEMBLE_PHASES
    ]
    return pd.concat(
        [numeric, one_hot.reindex(columns=phase_columns, fill_value=0.0)],
        axis=1,
    )