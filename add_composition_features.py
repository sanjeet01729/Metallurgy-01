from pathlib import Path

import numpy as np
import pandas as pd


# Configuration
UNIT = "wt"  # Change to "at" when the element columns end in _at.
INPUT_FILE = Path(__file__).with_name("combined_ym_clean.csv")
OUTPUT_FILE = Path(__file__).with_name("combined_ym_features.csv")
NOTES_FILE = Path(__file__).with_name("feature_notes.txt")
GAS_CONSTANT = 8.314462618  # J mol^-1 K^-1
ALPHA_EQ_UNCERTAIN_CUTOFF = 8.0  # wt%; a conservative alpha-stabilization screen.
ALPHA_BETA_MO_EQ_CUTOFF = 2.5  # wt%.
BETA_MO_EQ_CUTOFF = 10.0  # wt%.

ATOMIC_MASS = {
    "Ti": 47.867, "Nb": 92.906, "Zr": 91.224, "Al": 26.9815385,
    "Co": 58.933194, "Cr": 51.9961, "Cu": 63.546, "Fe": 55.845,
    "Hf": 178.49, "Mn": 54.938044, "Mo": 95.95, "O": 15.999,
    "Pd": 106.42, "Sc": 44.955908, "Si": 28.085, "Sn": 118.710,
    "Ta": 180.94788, "V": 50.9415, "W": 183.84,
}

VEC = {
    "Ti": 4, "Zr": 4, "Hf": 4, "Al": 3, "Sn": 4, "Si": 4,
    "V": 5, "Nb": 5, "Ta": 5, "Cr": 6, "Mo": 6, "W": 6,
    "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11, "Pd": 10,
    "O": 6, "N": 5,
}

# Metallic radii in pm. Existing project values are retained where present;
# Pd uses the standard empirical metallic radius. Oxygen has no metallic radius.
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
AL_EQ_COEFFICIENT = {"Al": 1.0, "Zr": 1.0 / 6.0, "Sn": 1.0 / 3.0, "O": 10.0}

DERIVED_COLUMNS = [
    "VEC", "delta_r", "delta_S_mix", "Mo_eq", "Al_eq", "Bo", "Md",
    "phase_est", "phase_note", "ti_based",
]


def format_table(mapping, columns):
    return "\n".join(
        f"- {symbol}: " + ", ".join(f"{name}={mapping[symbol][i]}" for i, name in enumerate(columns))
        for symbol in sorted(mapping)
    )


def main():
    if UNIT not in {"wt", "at"}:
        raise ValueError(f"UNIT must be 'wt' or 'at', got {UNIT!r}")

    data = pd.read_csv(INPUT_FILE)
    original_columns = data.columns.tolist()
    suffix = f"_{UNIT}"
    element_columns = [column for column in original_columns if column.endswith(suffix)]
    if not element_columns:
        raise ValueError(f"No element columns ending in {suffix!r} in {INPUT_FILE}")
    elements = [column[: -len(suffix)] for column in element_columns]
    if any(not element for element in elements):
        raise ValueError("An element composition column has an empty element symbol.")
    if len(set(elements)) != len(elements):
        raise ValueError("Duplicate element symbols were found in composition columns.")

    missing_masses = sorted(set(elements) - set(ATOMIC_MASS))
    if missing_masses:
        raise ValueError(f"Atomic masses are unavailable for composition elements: {missing_masses}")
    composition = data[element_columns].apply(pd.to_numeric, errors="coerce")
    any_input_nan = data.isna().any(axis=1) | composition.isna().any(axis=1)
    finite_composition = np.isfinite(composition.to_numpy(dtype=float)).all(axis=1)
    negative_composition = (composition < 0).any(axis=1).to_numpy()
    raw_values = composition.to_numpy(dtype=float)
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
    if UNIT == "wt":
        weighted_moles = normalized / mass_array
        atomic_fraction = weighted_moles / np.nansum(weighted_moles, axis=1, keepdims=True)
        weight_fraction = normalized / 100.0
    else:
        atomic_fraction = normalized / 100.0
        unnormalized_weight = atomic_fraction * mass_array
        weight_fraction = unnormalized_weight / np.nansum(
            unnormalized_weight, axis=1, keepdims=True
        )

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

    result = data.copy()
    features = pd.DataFrame(index=data.index, columns=DERIVED_COLUMNS)
    features["VEC"] = atomic_linear(VEC)
    vec_missing = [
        element for element in elements
        if element not in VEC
        and np.any(valid & (normalized[:, element_index[element]] > 0))
    ]
    if vec_missing:
        vec_missing_mask = np.zeros(len(data), dtype=bool)
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
    radius_delta = 100.0 * np.sqrt(
        np.sum(
            radius_atomic
            * (1.0 - radius_array[None, :] / radius_mean[:, None]) ** 2,
            axis=1,
        )
    )
    features["delta_r"] = radius_delta
    if radius_missing:
        radius_missing_mask = np.zeros(len(data), dtype=bool)
        for element in radius_missing:
            radius_missing_mask |= normalized[:, element_index[element]] > 0
        features.loc[radius_missing_mask, "delta_r"] = np.nan

    entropy_terms = np.zeros_like(atomic_fraction)
    positive = atomic_fraction > 0
    entropy_terms[positive] = atomic_fraction[positive] * np.log(atomic_fraction[positive])
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
    morinaga_available = [element for element in elements if element in MORINAGA]
    morinaga_indices = [element_index[element] for element in morinaga_available]
    bo_values = np.array([MORINAGA[element][0] for element in morinaga_available])
    md_values = np.array([MORINAGA[element][1] for element in morinaga_available])
    features["Bo"] = atomic_fraction[:, morinaga_indices] @ bo_values
    features["Md"] = atomic_fraction[:, morinaga_indices] @ md_values
    if morinaga_missing:
        morinaga_missing_mask = np.zeros(len(data), dtype=bool)
        for element in morinaga_missing:
            morinaga_missing_mask |= normalized[:, element_index[element]] > 0
        features.loc[morinaga_missing_mask, ["Bo", "Md"]] = np.nan

    ti_weight = (
        weight_fraction[:, element_index["Ti"]] * 100.0
        if "Ti" in element_index
        else np.zeros(len(data), dtype=float)
    )
    ti_based = ti_weight >= 70.0
    features["ti_based"] = pd.Series(ti_based, index=data.index, dtype="boolean")
    phase = np.full(len(data), "uncertain", dtype=object)
    phase_notes = np.full(len(data), "", dtype=object)
    for row_index in range(len(data)):
        if not valid[row_index]:
            phase_notes[row_index] = "Input row has NaN, non-finite, negative, or zero-total composition."
        elif not ti_based[row_index]:
            phase_notes[row_index] = "Ti < 70 wt%; not treated as a Ti-based alloy for phase screening."
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
    features.loc[~valid, DERIVED_COLUMNS] = np.nan

    result = pd.concat([result, features[DERIVED_COLUMNS]], axis=1)
    result.to_csv(OUTPUT_FILE, index=False)

    reloaded = pd.read_csv(OUTPUT_FILE)
    duplicate_count = int(reloaded.duplicated().sum())
    if reloaded.columns.tolist() != original_columns + DERIVED_COLUMNS:
        raise AssertionError("Output columns or column order do not match specification.")
    if len(reloaded) != len(data):
        raise AssertionError("Output row count changed.")
    if not reloaded[original_columns].equals(data[original_columns]):
        raise AssertionError("An original input column changed during feature generation.")

    notes = [
        "Titanium alloy composition-feature generation",
        "=" * 44,
        f"Input: {INPUT_FILE.resolve()}",
        f"Output: {OUTPUT_FILE.resolve()}",
        f"Rows before: {len(data)}",
        f"Rows after: {len(result)}",
        f"Exact duplicate rows after feature generation (counted, not removed): {duplicate_count}",
        f"Configured UNIT: {UNIT!r}; row normalization is applied before feature calculations.",
        "Input columns are copied unchanged and remain in their original order; derived columns follow.",
        "Rows with any NaN in any input column receive NaN for every derived column. Invalid negative or zero-total compositions are also marked NaN.",
        "Absent elements (zero concentration) contribute zero. Derived features are not rounded.",
        "",
        "Feature definitions, units, and references:",
        "- VEC (electron/atom): sum(x_i * VEC_i), x_i atomic fraction. VEC assignments were specified in the task; the electron/atom design concept is described by Williams (1975), 'Using electron/atom ratio in titanium alloy design,' Journal of Materials Science 10, 1239-1241, doi:10.1007/BF00541408.",
        "- delta_r (%): 100*sqrt(sum(x_i*(1-r_i/r_mean)^2)); r_mean=sum(x_i*r_i). Formula follows the atomic-size mismatch descriptor; see Zhang et al. (2008), 'Solid-Solution Phase Formation Rules for Multi-Component Alloys,' Advanced Engineering Materials 10, 534-538, doi:10.1002/adem.200700240. Metallic-radius reference context: Villars and Calvert (1991), Pearson's Handbook of Crystallographic Data for Intermetallic Phases, 2nd ed. The exact table below reuses existing project values where available and states the Pd choice; the project table does not record primary provenance for each radius.",
        "- delta_S_mix (J mol^-1 K^-1): -R*sum(x_i*ln(x_i)), zero terms omitted; ideal configurational mixing entropy (Gibbs, 'On the Equilibrium of Heterogeneous Substances,' 1878-1879).",
        "- Mo_eq (wt%): task-specified linear weight-percent equivalent. Coefficients and formula are the supplied definition; conventional titanium alloy stability/equivalent context: Kolli and Devaraj (2018), 'A Review of Metastable Beta Titanium Alloys,' Metals 8(7), 506, doi:10.3390/met8070506.",
        "- Al_eq (wt%): Al + Zr/6 + Sn/3 + 10*O using normalized wt%; task-specified formula. Alpha/beta alloy context: Donachie (2000), Titanium: A Technical Guide, 2nd ed., ASM International.",
        "- Bo (bond order, dimensionless) and Md (eV): atomic-fraction-weighted Morinaga parameters. Morinaga et al. (1986), 'Electronic Structure and Phase Stability of Titanium Alloys,' Tetsu-to-Hagane 72(6), 555-562, doi:10.2355/tetsutohagane1955.72.6_555. Supplied table values are used without extrapolation.",
        "- phase_est (categorical screening label) and phase_note: Mo_eq < 2.5 -> alpha (near-alpha grouped here); 2.5 <= Mo_eq <= 10 -> alpha+beta; Mo_eq > 10 -> beta. This is a coarse engineering screen, not an equilibrium phase prediction; Mo_eq context: Kolli and Devaraj (2018) and Donachie (2000).",
        "- ti_based (boolean): normalized Ti_wt >= 70. This is the task-specified applicability flag, not a literature-derived threshold.",
        "",
        "Assumptions / interpretation:",
        "- Atomic masses are standard atomic weights (IUPAC/CIAAW, Standard Atomic Weights 2021; Pure and Applied Chemistry, 2022). The values actually used are listed below.",
        "- The composition is normalized to 100 in its configured input unit before conversion; atomic fractions use normalized atomic amounts and weight equivalents use normalized weight fractions.",
        f"- phase_est uses the conventional coarse Mo_eq screen: below 2.5 wt% -> alpha, 2.5 through 10 wt% -> alpha+beta, above 10 wt% -> beta. Near-alpha alloys are grouped under 'alpha' because no near-alpha output category was requested.",
        f"- Al_eq >= {ALPHA_EQ_UNCERTAIN_CUTOFF:g} wt% is treated as large and causes phase_est='uncertain'. This is a conservative screening cutoff, not a universal phase boundary.",
        "- Pd uses an empirical metallic radius of 137 pm for delta_r but has no supplied Morinaga parameter; rows containing Pd therefore have Bo/Md=NaN.",
        "- Sc has no supplied Morinaga parameter; rows containing Sc therefore have Bo/Md=NaN.",
        "- Oxygen has no metallic radius and no supplied Morinaga parameter; rows containing O therefore have delta_r, Bo, and Md as NaN. O is present as a column, so Al_eq is not marked as a lower bound.",
        "",
        "Atomic masses used (g mol^-1):",
    ]
    notes.extend(f"- {element}: {ATOMIC_MASS[element]}" for element in elements)
    notes.extend(["", "Metallic radii used (pm; table only includes radii available/needed):"])
    notes.extend(
        f"- {element}: {METALLIC_RADIUS_PM[element]}"
        for element in elements if element in METALLIC_RADIUS_PM
    )
    notes.append("- O has no metallic radius; it is not assigned an invented metallic radius.")
    notes.extend([
        "",
        "VEC values used (electrons/atom):",
    ])
    notes.extend(f"- {element}: {VEC[element]}" for element in elements if element in VEC)
    notes.extend(f"- {element}: not defined in supplied VEC table" for element in elements if element not in VEC)
    notes.extend(["", "Morinaga parameters used (Bo dimensionless / Md eV):"])
    notes.extend(
        f"- {element}: Bo={MORINAGA[element][0]:.3f}, Md={MORINAGA[element][1]:.3f}"
        for element in elements if element in MORINAGA
    )
    notes.extend(["", "Rows with NaN features or uncertain phase:"])
    flagged_rows = []
    for position in range(len(data)):
        index_label = data.index[position]
        reasons = []
        if any_input_nan.iloc[position]:
            nan_columns = data.columns[data.iloc[position].isna()].tolist()
            reasons.append(f"input NaN in {', '.join(nan_columns)}; all derived features set to NaN")
        if not valid[position] and not any_input_nan.iloc[position]:
            reasons.append("invalid composition: non-finite, negative, or zero-total values")
        if valid[position]:
            present = [
                element for element in elements
                if normalized[position, element_index[element]] > 0
            ]
            missing_morinaga = [element for element in present if element not in MORINAGA]
            missing_radius = [element for element in present if element not in METALLIC_RADIUS_PM]
            if missing_morinaga:
                reasons.append(f"Bo/Md unavailable for {', '.join(missing_morinaga)}")
            if missing_radius:
                reasons.append(f"delta_r unavailable for {', '.join(missing_radius)}")
            missing_vec = [element for element in present if element not in VEC]
            if missing_vec:
                reasons.append(f"VEC unavailable for {', '.join(missing_vec)}")
            if features["phase_est"].iloc[position] == "uncertain":
                reasons.append(str(features["phase_note"].iloc[position]))
        if reasons:
            flagged_rows.append(f"- CSV data row {position + 2} (index {index_label}): " + "; ".join(reasons))
    notes.extend(flagged_rows if flagged_rows else ["- None."])
    notes.extend(["", "Per-feature min, max, and mean (NaN values omitted):"])
    for feature in ["VEC", "delta_r", "delta_S_mix", "Mo_eq", "Al_eq", "Bo", "Md"]:
        values = pd.to_numeric(features[feature], errors="coerce")
        notes.append(
            f"- {feature}: min={values.min():.8g}, max={values.max():.8g}, "
            f"mean={values.mean():.8g}; non-NaN={int(values.notna().sum())}, "
            f"NaN={int(values.isna().sum())}."
        )
    ti_series = features["ti_based"].astype("boolean")
    notes.append(
        f"- ti_based: min={int(ti_series.min())}, max={int(ti_series.max())}, "
        f"mean={ti_series.astype(float).mean():.8g}; true={int(ti_series.sum())}, "
        f"false={int((~ti_series).sum())}."
    )
    notes.append("- phase_est category counts (categorical feature):")
    notes.extend(
        f"  - {category}: {int(count)}"
        for category, count in features["phase_est"].value_counts(dropna=False).items()
    )
    notes.append("- phase_note category counts (categorical explanation):")
    notes.extend(
        f"  - {reason}: {int(count)}"
        for reason, count in features["phase_note"].value_counts(dropna=False).items()
    )
    notes.extend(["", "Physically implausible values:"])
    implausible = []
    ranges = {
        "VEC": (min(VEC.values()), max(VEC.values())),
        "delta_r": (0.0, 100.0),
        "delta_S_mix": (0.0, GAS_CONSTANT * np.log(max(2, len(elements)))),
        "Mo_eq": (
            100.0 * min(MO_EQ_COEFFICIENT.values()),
            100.0 * max(MO_EQ_COEFFICIENT.values()),
        ),
        "Al_eq": (
            100.0 * min(0.0, *AL_EQ_COEFFICIENT.values()),
            100.0 * max(0.0, *AL_EQ_COEFFICIENT.values()),
        ),
        "Bo": (min(v[0] for v in MORINAGA.values()), max(v[0] for v in MORINAGA.values())),
        "Md": (min(v[1] for v in MORINAGA.values()), max(v[1] for v in MORINAGA.values())),
    }
    for feature, (low, high) in ranges.items():
        values = pd.to_numeric(features[feature], errors="coerce")
        mask = values.notna() & ((values < low - 1e-10) | (values > high + 1e-10))
        rows = np.flatnonzero(mask.to_numpy())
        if len(rows):
            implausible.append(
                f"- {feature}: {len(rows)} value(s) outside screening range [{low:.6g}, {high:.6g}]: "
                + ", ".join(f"row {row + 2}={values.iloc[row]:.6g}" for row in rows)
            )
    notes.extend(implausible if implausible else ["- None under the broad descriptor sanity ranges stated in the script."])
    notes.extend([
        "",
        f"Rows before: {len(data)}",
        f"Rows after: {len(result)}",
        f"Exact duplicate rows after adding features (not dropped): {duplicate_count}",
        f"Rows with any input NaN: {int(any_input_nan.sum())}",
        f"Rows with phase_est='uncertain': {int(features['phase_est'].eq('uncertain').sum())}",
        f"Rows with missing Bo/Md due to missing Morinaga table entries: {int(features['Bo'].isna().sum() - any_input_nan.sum())}",
        f"Rows with missing delta_r due to missing metallic radii: {int(features['delta_r'].isna().sum() - any_input_nan.sum())}",
        f"Rows with missing VEC due to missing VEC table entries: {int(features['VEC'].isna().sum() - any_input_nan.sum())}",
    ])
    NOTES_FILE.write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"Rows before: {len(data)}")
    print(f"Rows after: {len(result)}")
    print(f"Exact duplicate rows after feature generation (not dropped): {duplicate_count}")
    print(f"Rows with any input NaN: {int(any_input_nan.sum())}")
    print(f"Rows with phase_est='uncertain': {int(features['phase_est'].eq('uncertain').sum())}")
    print(f"Output: {OUTPUT_FILE.resolve()}")
    print(f"Notes: {NOTES_FILE.resolve()}")


if __name__ == "__main__":
    main()
