from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


INPUT_FILE = Path(__file__).with_name("combined_ym_features.csv")
TRAIN_FILE = Path(__file__).with_name("train.csv")
TEST_FILE = Path(__file__).with_name("test.csv")
DROPPED_FILE = Path(__file__).with_name("dropped_non_ti.csv")
REPORT_FILE = Path(__file__).with_name("split_report.txt")
TEST_SIZE = 0.2
INITIAL_SEED = 42
ALTERNATE_SEEDS = range(43, 53)


def build_group_ids(frame: pd.DataFrame, composition_columns: list[str]) -> pd.Series:
    rounded = frame[composition_columns].round(1)
    return rounded.astype("string").fillna("<NA>").agg("|".join, axis=1)


def split_once(
    frame: pd.DataFrame, group_ids: pd.Series, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=seed,
    )
    train_index, test_index = next(
        splitter.split(frame, groups=group_ids)
    )
    return frame.iloc[train_index].copy(), frame.iloc[test_index].copy()


def target_statistics(frame: pd.DataFrame) -> dict[str, float]:
    target = pd.to_numeric(frame["E_GPa"], errors="coerce")
    return {
        "min": float(target.min()),
        "max": float(target.max()),
        "mean": float(target.mean()),
        "std": float(target.std()),
    }


def main() -> None:
    data = pd.read_csv(INPUT_FILE)
    original_columns = data.columns.tolist()
    required = {"ti_based", "E_GPa", "phase_est"}
    missing = required - set(original_columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    composition_columns = [
        column for column in original_columns if column.endswith("_wt")
    ]
    if not composition_columns:
        raise ValueError("No composition columns ending in '_wt' were found.")

    ti_mask = data["ti_based"].eq(True)
    dropped = data.loc[~ti_mask].copy()
    filtered = data.loc[ti_mask].copy()
    filtered_groups = build_group_ids(filtered, composition_columns)

    selected_seed = INITIAL_SEED
    train, test = split_once(filtered, filtered_groups, selected_seed)
    train_stats = target_statistics(train)
    test_stats = target_statistics(test)

    if abs(train_stats["mean"] - test_stats["mean"]) > 5.0:
        candidates = []
        for seed in ALTERNATE_SEEDS:
            candidate_train, candidate_test = split_once(
                filtered, filtered_groups, seed
            )
            candidate_train_stats = target_statistics(candidate_train)
            candidate_test_stats = target_statistics(candidate_test)
            difference = abs(
                candidate_train_stats["mean"] - candidate_test_stats["mean"]
            )
            candidates.append(
                (
                    difference,
                    seed,
                    candidate_train,
                    candidate_test,
                    candidate_train_stats,
                    candidate_test_stats,
                )
            )
        (
            _,
            selected_seed,
            train,
            test,
            train_stats,
            test_stats,
        ) = min(candidates, key=lambda item: (item[0], item[1]))

    train_groups = build_group_ids(train, composition_columns)
    test_groups = build_group_ids(test, composition_columns)
    shared_groups = set(train_groups) & set(test_groups)
    assert not shared_groups, f"Composition leakage: {len(shared_groups)} shared groups"
    assert train.columns.tolist() == original_columns
    assert test.columns.tolist() == original_columns
    assert len(train) + len(test) == len(filtered)

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    dropped.to_csv(DROPPED_FILE, index=False)

    train_stats = target_statistics(train)
    test_stats = target_statistics(test)
    total_kept = len(filtered)
    train_percent = 100.0 * len(train) / total_kept if total_kept else 0.0
    test_percent = 100.0 * len(test) / total_kept if total_kept else 0.0

    report_lines = [
        "Titanium-alloy Young's modulus grouped split report",
        "=" * 50,
        f"Input: {INPUT_FILE.resolve()}",
        f"Output train: {TRAIN_FILE.resolve()}",
        f"Output test: {TEST_FILE.resolve()}",
        f"Dropped non-Ti rows: {DROPPED_FILE.resolve()}",
        f"Input rows: {len(data)}",
        f"Rows retained (ti_based == True): {len(filtered)}",
        f"Rows dropped (ti_based != True): {len(dropped)}",
        f"Group columns: {', '.join(composition_columns)}",
        "Group IDs use composition values rounded to 1 decimal place.",
        f"Group-aware splitter: GroupShuffleSplit(n_splits=1, test_size={TEST_SIZE}, random_state={selected_seed})",
        f"Seed used: {selected_seed}",
        "",
        "Split sizes:",
        f"- Train: {len(train)} rows ({train_percent:.2f}%)",
        f"- Test: {len(test)} rows ({test_percent:.2f}%)",
        f"- Unique composition groups: {filtered_groups.nunique()}",
        f"- Shared train/test group IDs: {len(shared_groups)}",
        "",
        "E_GPa statistics (GPa):",
        f"- Train: min={train_stats['min']:.4f}, max={train_stats['max']:.4f}, "
        f"mean={train_stats['mean']:.4f}, std={train_stats['std']:.4f}",
        f"- Test: min={test_stats['min']:.4f}, max={test_stats['max']:.4f}, "
        f"mean={test_stats['mean']:.4f}, std={test_stats['std']:.4f}",
        f"- Absolute difference in means: {abs(train_stats['mean'] - test_stats['mean']):.4f} GPa",
        "",
        "Rows per phase_est value:",
    ]
    phase_values = sorted(
        set(train["phase_est"].dropna().astype(str))
        | set(test["phase_est"].dropna().astype(str))
    )
    for phase in phase_values:
        train_count = int(train["phase_est"].astype("string").eq(phase).sum())
        test_count = int(test["phase_est"].astype("string").eq(phase).sum())
        report_lines.append(f"- {phase}: train={train_count}, test={test_count}")
    report_lines.extend(
        [
            "",
            "Handling notes:",
            "- Rows with ti_based not equal to True were saved to dropped_non_ti.csv.",
            "- All columns and original values were preserved; no scaling, imputation, or encoding was applied.",
            "- NaNs in the retained dataset are preserved. For composition grouping only, missing composition values use a literal <NA> token after 1-decimal rounding.",
            "- GroupShuffleSplit keeps every rounded-composition group in exactly one split; duplicate/near-duplicate compositions with differing E_GPa remain together.",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report, end="")

    train_check = pd.read_csv(TRAIN_FILE)
    test_check = pd.read_csv(TEST_FILE)
    dropped_check = pd.read_csv(DROPPED_FILE)
    if train_check.columns.tolist() != original_columns:
        raise AssertionError("train.csv columns/order differ from input.")
    if test_check.columns.tolist() != original_columns:
        raise AssertionError("test.csv columns/order differ from input.")
    if len(dropped_check) != len(dropped):
        raise AssertionError("Dropped-row output count does not match.")


if __name__ == "__main__":
    main()
