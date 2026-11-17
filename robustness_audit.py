"""Robustness checks for the Hotel Booking Cancellation Audit.

Examples:
    python robustness_audit.py --fast
    python robustness_audit.py

The script repeats the notebook's conservative XGBoost experiment across
different random split seeds, ablates questionable features one at a time,
and evaluates several chronological cutoffs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Save figures without requiring a local desktop/Tk installation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


SEED = 42
TARGET = "is_canceled"
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
DEFINITE_LEAK = ["reservation_status", "reservation_status_date"]
RISKY_FEATURES = ["country", "deposit_type", "assigned_room_type", "booking_changes"]
DROP_ALWAYS = [TARGET, "arrival_date", "arrival_date_year"]


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Apply the same deterministic cleaning and feature engineering as the notebook."""
    df = pd.read_csv(csv_path)
    d = df.copy()
    d["children"] = d["children"].fillna(0)
    d["agent"] = d["agent"].fillna(0).astype(int).astype(str)
    d["country"] = d["country"].fillna("UNK")
    d = d.drop(columns=["company"])
    d = d[(d["adults"] + d["children"] + d["babies"]) > 0]
    d = d[(d["adr"] >= 0) & (d["adr"] < 1000)]

    month_num = {month: index + 1 for index, month in enumerate(MONTHS)}
    d["arrival_date"] = pd.to_datetime(
        dict(
            year=d["arrival_date_year"],
            month=d["arrival_date_month"].map(month_num),
            day=d["arrival_date_day_of_month"],
        )
    )
    d["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    d["total_guests"] = d["adults"] + d["children"] + d["babies"]
    d["reservation_status_date"] = (
        pd.to_datetime(d["reservation_status_date"]) - pd.Timestamp("2014-01-01")
    ).dt.days
    return d.drop_duplicates().sort_values("arrival_date").reset_index(drop=True)


def feature_columns(data: pd.DataFrame, added_risky: list[str] | None = None) -> list[str]:
    """Return the conservative feature set, optionally restoring named risky columns."""
    excluded = set(DROP_ALWAYS + DEFINITE_LEAK + RISKY_FEATURES)
    if added_risky:
        excluded.difference_update(added_risky)
    return [column for column in data.columns if column not in excluded]


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric = x_train.select_dtypes("number").columns.tolist()
    categorical = [column for column in x_train.columns if column not in numeric]
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def score_split(
    data: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    columns: list[str],
    fast: bool,
) -> float:
    x_train, x_test = data.loc[train_index, columns], data.loc[test_index, columns]
    y_train, y_test = data.loc[train_index, TARGET], data.loc[test_index, TARGET]
    model = XGBClassifier(
        n_estimators=75 if fast else 300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=SEED,
    )
    pipeline = Pipeline([("prep", build_preprocessor(x_train)), ("model", model)])
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    return average_precision_score(y_test, probabilities)


def chronological_indices(data: pd.DataFrame, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    index = data.index.to_numpy()
    cutoff = int(len(index) * fraction)
    return index[:cutoff], index[cutoff:]


def random_indices(data: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    index = data.index.to_numpy()
    return train_test_split(index, test_size=0.30, stratify=data[TARGET], random_state=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cancellation-model robustness checks.")
    parser.add_argument("--csv", type=Path, default=Path("data/raw/hotel_bookings.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/robustness"))
    parser.add_argument("--fast", action="store_true", help="Use two seeds and 75 trees for a smoke test.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random split seeds for the full run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_clean(args.csv)
    clean_columns = feature_columns(data)
    seeds = [42, 1] if args.fast else [42, *range(1, args.seeds)]
    print(f"Cleaned rows: {len(data):,}; random seeds: {seeds}")

    chrono_train, chrono_test = chronological_indices(data, 0.70)
    chronological_pr_auc = score_split(data, chrono_train, chrono_test, clean_columns, args.fast)
    random_rows = []
    for seed in seeds:
        train_index, test_index = random_indices(data, seed)
        pr_auc = score_split(data, train_index, test_index, clean_columns, args.fast)
        random_rows.append({"seed": seed, "feature_set": "clean", "pr_auc": pr_auc})
        print(f"Random seed {seed}: PR-AUC {pr_auc:.4f}")
    random_results = pd.DataFrame(random_rows)
    random_results.to_csv(args.output_dir / "seed_stability_results.csv", index=False)

    random_mean = random_results.pr_auc.mean()
    random_std = random_results.pr_auc.std(ddof=1) if len(random_results) > 1 else np.nan
    gap = random_mean - chronological_pr_auc
    gap_vs_1std = gap / random_std if pd.notna(random_std) and random_std > 0 else np.nan
    summary = pd.DataFrame(
        [
            {
                "feature_set": "clean",
                "chronological_pr_auc": chronological_pr_auc,
                "random_pr_auc_mean": random_mean,
                "random_pr_auc_std": random_std,
                "random_minus_chronological": gap,
                "gap_vs_1std": gap_vs_1std,
                "n_random_seeds": len(seeds),
            }
        ]
    )
    summary.to_csv(args.output_dir / "robustness_summary.csv", index=False)

    ablation_rows = []
    for feature in RISKY_FEATURES:
        columns = feature_columns(data, [feature])
        chrono_score = score_split(data, chrono_train, chrono_test, columns, args.fast)
        ablation_rows.append({"split": "chronological", "feature_added": feature, "seed": np.nan, "pr_auc": chrono_score})
        print(f"Chronological + {feature}: PR-AUC {chrono_score:.4f}")
        for seed in seeds:
            train_index, test_index = random_indices(data, seed)
            score = score_split(data, train_index, test_index, columns, args.fast)
            ablation_rows.append({"split": "random", "feature_added": feature, "seed": seed, "pr_auc": score})
    ablations = pd.DataFrame(ablation_rows)
    ablations.to_csv(args.output_dir / "feature_ablation_results.csv", index=False)
    ablation_summary = (
        ablations.groupby(["split", "feature_added"], as_index=False).pr_auc.agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "pr_auc_mean", "std": "pr_auc_std"})
    )
    ablation_summary.to_csv(args.output_dir / "feature_ablation_summary.csv", index=False)

    cutoffs = [0.50, 0.60, 0.70, 0.80, 0.90] if not args.fast else [0.50, 0.70, 0.90]
    rolling_rows = []
    for cutoff in cutoffs:
        train_index, test_index = chronological_indices(data, cutoff)
        score = score_split(data, train_index, test_index, clean_columns, args.fast)
        rolling_rows.append({"train_fraction": cutoff, "test_fraction": 1 - cutoff, "pr_auc": score})
        print(f"Chronological {cutoff:.0%}/{1 - cutoff:.0%}: PR-AUC {score:.4f}")
    rolling = pd.DataFrame(rolling_rows)
    rolling.to_csv(args.output_dir / "rolling_cutoff_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(random_results.pr_auc, bins=min(8, len(random_results)), edgecolor="white")
    axes[0].axvline(chronological_pr_auc, color="crimson", linestyle="--", label="Chronological (70/30)")
    axes[0].set(title="Random-split PR-AUC across seeds", xlabel="PR-AUC", ylabel="Count")
    axes[0].legend()
    axes[1].plot(rolling.train_fraction, rolling.pr_auc, marker="o")
    axes[1].set(title="Chronological cutoff sensitivity", xlabel="Training fraction", ylabel="PR-AUC")
    axes[1].set_xticks(rolling.train_fraction)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(args.output_dir / "robustness_plots.png", dpi=180)
    print(f"\nSaved results and plot to: {args.output_dir}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
