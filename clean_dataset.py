
import pandas as pd
from pathlib import Path

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
INPUT_FILE = Path("titanium_alloys.csv")
OUTPUT_FILE = Path("titanium_alloys_clean.csv")

TARGET = "YM (GPa)"

# Columns to remove
DROP_COLUMNS = [
    "VEC (-)",
    "e/a (-)",
    "δr (%)",
    "System",
    "ρ (g·cm⁻³)",
    "Thc (W·m⁻¹·K⁻¹)",
    "r (pm)",
    "χᴾ (-)",
    "χᴬ (-)",
    "δχᴾ (%)",
    "Δχᴾ (-)",
    "δχᴬ (%)",
    "Δχᴬ (-)",
    "ΔVEC (-)",
    "ΔSᵐ (J·mol⁻¹·K⁻¹)",
    "ΔHᶦᶜ (kJ·mol⁻¹)",
    "Sˣˢ (bcc) (J·mol⁻¹·K⁻¹)",
    "Sˣˢ (fcc) (J·mol⁻¹·K⁻¹)",
    "ɣ (-)",
    "(Mo)eq",
    "Type",
]


def main():
    # Check input file
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE.resolve()}"
        )

    # Read dataset and normalize column names
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    print("=" * 60)
    print("DATASET CLEANING")
    print("=" * 60)
    print(f"Original dataset: {len(df)} rows x {len(df.columns)} columns")

    # Verify target exists
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Remove unnecessary columns
    columns_to_drop = [
        col for col in DROP_COLUMNS if col in df.columns
    ]

    df = df.drop(columns=columns_to_drop)

    print(f"\nColumns removed: {len(columns_to_drop)}")
    for col in columns_to_drop:
        print(f"  - {col}")

    # Report configured columns that were not found
    missing_columns = [
        col for col in DROP_COLUMNS if col not in columns_to_drop
    ]

    if missing_columns:
        print("\nConfigured columns not found (skipped):")
        for col in missing_columns:
            print(f"  - {col}")

    # Remove rows with missing target values
    rows_before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    rows_removed = rows_before - len(df)

    print(f"\nRows removed due to missing {TARGET}: {rows_removed}")

    # Report missing values in remaining features
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        print("\nMissing values in remaining columns:")
        print(missing_values.to_string())
    else:
        print("No missing values in remaining columns.")

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"Final dataset: {len(df)} rows x {len(df.columns)} columns")
    print(f"Output file: {OUTPUT_FILE.resolve()}")

    print("\nRemaining columns:")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i:2}. {col}")

    print("\nOriginal file was not modified.")


if __name__ == "__main__":
    main()
