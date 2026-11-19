
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

# =========================
# 1. Configuration
# =========================

DATA_PATH = Path("processed_data/train_features.parquet")
OUTPUT_DIR = Path("processed_data/simple_baselines")
CUTOFF_DATE = pd.Timestamp("2026-10-26", tz="UTC")

ACTIVITIES = [
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Load validation data
# =========================

df = pd.read_parquet(DATA_PATH)

df["snapshot_date"] = pd.to_datetime(
    df["snapshot_date"], utc=True
)

# Use exactly the same validation period as RF and GNN
val_df = df[df["snapshot_date"] >= CUTOFF_DATE].copy()
val_df = val_df.sort_values(
    ["snapshot_date", "github_user_id"]
).reset_index(drop=True)

print("Validation data shape:", val_df.shape)
print("Validation snapshots:")
print(sorted(val_df["snapshot_date"].dt.strftime("%Y-%m-%d").unique()))


# =========================
# 3. Copy Last Week
# =========================

# Predict next week's activity using activity in the previous 7 days.
# If count_7d > 0, predict 1; otherwise predict 0.

results = {}

for activity in ACTIVITIES:
    feature_col = f"{activity}_count_7d"
    label_col = f"y_{activity}"

    y_true = val_df[label_col].fillna(0).astype(int).values
    y_pred = (
        val_df[feature_col].fillna(0) > 0
    ).astype(int).values

    results[activity] = {
        "y_true": y_true,
        "y_pred": y_pred,
    }


# =========================
# 4. Overall Evaluation
# =========================

y_true_all = np.column_stack([
    results[a]["y_true"] for a in ACTIVITIES
])

y_pred_all = np.column_stack([
    results[a]["y_pred"] for a in ACTIVITIES
])

macro_f1 = f1_score(
    y_true_all, y_pred_all,
    average="macro",
    zero_division=0
)

micro_f1 = f1_score(
    y_true_all, y_pred_all,
    average="micro",
    zero_division=0
)

combined_f1 = (macro_f1 + micro_f1) / 2

print("\n========== Copy Last Week: Overall Results ==========")
print(f"Macro-F1:    {macro_f1:.4f}")
print(f"Micro-F1:    {micro_f1:.4f}")
print(f"Combined F1: {combined_f1:.4f}")


# =========================
# 5. Per-Activity Evaluation
# =========================

print("\n========== Per-Activity F1 ==========")

per_activity = []

for activity in ACTIVITIES:
    y_true = results[activity]["y_true"]
    y_pred = results[activity]["y_pred"]

    score = f1_score(
        y_true, y_pred,
        zero_division=0
    )

    per_activity.append({
        "activity": activity,
        "f1": score
    })

    print(f"{activity:20s}: {score:.4f}")

pd.DataFrame(per_activity).to_csv(
    OUTPUT_DIR / "per_activity_f1.csv",
    index=False
)


# =========================
# 6. Weekly Evaluation
# =========================

print("\n========== Weekly Combined F1 ==========")

weekly_results = []

for snapshot_date, group in val_df.groupby("snapshot_date"):
    weekly_true = np.column_stack([
        group[f"y_{a}"].fillna(0).astype(int).values
        for a in ACTIVITIES
    ])

    weekly_pred = np.column_stack([
        (group[f"{a}_count_7d"].fillna(0) > 0).astype(int).values
        for a in ACTIVITIES
    ])

    weekly_macro = f1_score(
        weekly_true, weekly_pred,
        average="macro",
        zero_division=0
    )

    weekly_micro = f1_score(
        weekly_true, weekly_pred,
        average="micro",
        zero_division=0
    )

    weekly_combined = (weekly_macro + weekly_micro) / 2

    date_str = snapshot_date.strftime("%Y-%m-%d")

    print(f"{date_str}: {weekly_combined:.4f}")

    weekly_results.append({
        "snapshot_date": date_str,
        "macro_f1": weekly_macro,
        "micro_f1": weekly_micro,
        "combined_f1": weekly_combined
    })

pd.DataFrame(weekly_results).to_csv(
    OUTPUT_DIR / "weekly_f1.csv",
    index=False
)


# =========================
# 7. Save Predictions
# =========================

pred_df = val_df[
    ["github_user_id", "snapshot_date"]
].copy()

for activity in ACTIVITIES:
    pred_df[f"true_{activity}"] = results[activity]["y_true"]
    pred_df[f"pred_{activity}"] = results[activity]["y_pred"]

pred_df.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False
)

print("\nPredictions and evaluation results saved to:")
print(OUTPUT_DIR)
