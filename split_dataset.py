import pandas as pd
from pathlib import Path

INPUT_FILE = Path("titanium_alloys_clean.csv")
TEST_FILE = Path("titanium_test.csv")
TRAIN_FILE = Path("titanium_train.csv")

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

if len(df) < 70:
    raise ValueError("Dataset must contain at least 70 rows.")

if df["YM (GPa)"].isna().any():
    raise ValueError("Missing YM values found. Clean them before splitting.")

# Reproducible random shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# First 70 rows for testing; remaining rows for training
test_df = df.iloc[:70].copy()
train_df = df.iloc[70:].copy()

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print(f"Training rows: {len(train_df)}")
print(f"Testing rows:  {len(test_df)}")
print(f"Columns:       {len(df.columns)}")
print(f"Training file: {TRAIN_FILE.resolve()}")
print(f"Testing file:  {TEST_FILE.resolve()}")