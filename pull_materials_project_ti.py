"""
Pull computed (DFT) elastic data for Ti-containing metallic materials from the
Materials Project and write it in the same column layout as
ti_modulus_master_clean.csv.

Setup (run locally, not in Claude):
    pip install mp-api pandas
    get a free API key at https://next-gen.materialsproject.org/api
    export MP_API_KEY="your_key"        (Windows PowerShell: $env:MP_API_KEY="your_key")
    python pull_materials_project_ti.py

Output: ti_dft_materials_project.csv
    - source = "DFT (Materials Project)"; these are 0 K, ideal-crystal values.
    - Keep them SEPARATE from the experimental rows. Do not merge blindly.
    - E is computed from the Voigt-Reuss-Hill K and G:  E = 9KG / (3K + G)

Not tested against the live API (no network in the sandbox where this was written).
Call signatures were checked against mp-api's installed client. If a field name has
changed in your version, the error message will name it.
"""
import os
import sys

import pandas as pd
from mp_api.client import MPRester

API_KEY = os.environ.get("MP_API_KEY")
if not API_KEY:
    sys.exit("Set MP_API_KEY first (see the top of this file).")

# Elements allowed in a row. Matches the elements in the experimental master file.
ALLOWED = {"Ti", "Nb", "Zr", "Ta", "Sn", "Mo", "Fe", "Al", "V", "Cr", "Cu",
           "Hf", "Mn", "W", "Sc", "Si"}
# Excluded at query time to drop oxides, nitrides, hydrides, etc.
EXCLUDE = ["O", "N", "C", "H", "F", "Cl", "Br", "I", "S", "Se", "Te", "P", "As",
           "B", "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba"]
COLS = ["Ti", "Nb", "Zr", "Ta", "Sn", "Mo", "Fe", "Al", "V", "Cr", "Cu",
        "Hf", "Mn", "W", "Sc", "Si"]
FIELDS = ["material_id", "formula_pretty", "composition", "nelements", "nsites",
          "energy_above_hull", "is_stable", "symmetry", "bulk_modulus", "shear_modulus"]


def vrh(x):
    """bulk_modulus / shear_modulus may come back as a dict or an object."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x.get("vrh")
    return getattr(x, "vrh", None)


def to_row(d):
    k, g = vrh(d.bulk_modulus), vrh(d.shear_modulus)
    if k is None or g is None or (3 * k + g) <= 0:
        return None
    comp = d.composition.fractional_composition
    amt = {el.symbol: comp[el] * 100.0 for el in comp.elements}
    if not set(amt) <= ALLOWED:
        return None
    sym = d.symmetry
    row = {c: round(amt.get(c, 0.0), 4) for c in COLS}
    row.update({
        "alloy": d.formula_pretty,
        "E_GPa": round(9 * k * g / (3 * k + g), 2),
        "K_vrh_GPa": round(k, 2),
        "G_vrh_GPa": round(g, 2),
        "poisson": round((3 * k - 2 * g) / (2 * (3 * k + g)), 3),
        "energy_above_hull_eV": d.energy_above_hull,
        "is_stable": d.is_stable,
        "spacegroup_number": getattr(sym, "number", None),
        "spacegroup_symbol": getattr(sym, "symbol", None),
        "nsites": d.nsites,
        "nelements": d.nelements,
        "src_file": "Materials Project API",
        "source": "DFT (Materials Project)",
        "ref": d.material_id,
    })
    return row


def main():
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            elements=["Ti"],
            has_props=["elasticity"],
            fields=FIELDS,
        )

    print(
        f"Materials Project returned {len(docs)} "
        "Ti entries with elasticity data"
    )

    rows = [row for row in (to_row(doc) for doc in docs) if row]

    if not rows:
        sys.exit(
            "No valid rows found. Check the returned fields "
            "and elasticity data."
        )

    df = pd.DataFrame(rows)

    df = df[
        (df["E_GPa"] > 0) & (df["E_GPa"] < 600)
    ].reset_index(drop=True)

    df = df.rename(columns={c: f"{c}_at" for c in COLS})

    df.to_csv("ti_dft_materials_project.csv", index=False)

    print(f"Wrote ti_dft_materials_project.csv: {len(df)} rows")
    print(df["E_GPa"].describe().round(1))

    print(
        "Rows with energy_above_hull <= 0.05 eV:",
        int((df["energy_above_hull_eV"] <= 0.05).sum()),
    )


if __name__ == "__main__":
    main()
