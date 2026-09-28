import pandas as pd

# Load dataset
file = "titanium_alloys.csv"
df = pd.read_csv(file)

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

# Rows and columns
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:2}. {col}")

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes.to_string())

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

for col, count in missing.items():
    print(f"{col}: {count}")

print("\n" + "=" * 70)
print("UNIQUE VALUES")
print("=" * 70)

for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head().to_string())

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

print(df.describe().T.to_string())