
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

# =========================
# 1. Load data
# =========================

df = pd.read_parquet("processed_data/train_features.parquet")

label_cols = [
    "y_open_pr",
    "y_open_issue",
    "y_issue_comment",
    "y_pr_review",
    "y_pr_review_comment",
    "y_push"
]

feature_cols = [
    c for c in df.columns
    if c not in ["github_user_id", "snapshot_date"] + label_cols
]

# =========================
# 2. Time-based split
# =========================

cutoff = pd.Timestamp("2026-10-26", tz="UTC")

train = df[df["snapshot_date"] < cutoff].copy()
valid = df[df["snapshot_date"] >= cutoff].copy()

X_train = train[feature_cols]
X_valid = valid[feature_cols]

print("Training shape:", X_train.shape)
print("Validation shape:", X_valid.shape)

# =========================
# 3. Train and evaluate models
# =========================

model_names = ["Logistic Regression", "Random Forest"]

results = []
predictions = {}

for model_name in model_names:

    predictions[model_name] = {}

    for label in label_cols:

        y_train = train[label].astype(int)
        y_valid = valid[label].astype(int)

        if model_name == "Logistic Regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                ))
            ])

        else:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            )

        model.fit(X_train, y_train)
        pred = model.predict(X_valid)

        # Save predictions for later evaluation.
        predictions[model_name][label] = pred

        results.append({
            "Model": model_name,
            "Activity": label,
            "F1": f1_score(y_valid, pred, zero_division=0),
            "Precision": precision_score(
                y_valid, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_valid, pred, zero_division=0
            )
        })

result_df = pd.DataFrame(results)

# =========================
# 4. Overall per-label results
# =========================

print("\n=== Overall Per-label Results ===")
print(
    result_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# =========================
# 5. Overall combined scores
# =========================

print("\n=== Overall Results ===")

for model_name in model_names:

    subset = result_df[result_df["Model"] == model_name]

    macro_f1 = subset["F1"].mean()

    all_true = np.concatenate([
        valid[label].astype(int).to_numpy()
        for label in label_cols
    ])

    all_pred = np.concatenate([
        predictions[model_name][label]
        for label in label_cols
    ])

    micro_f1 = f1_score(
        all_true, all_pred, zero_division=0
    )

    combined = (macro_f1 + micro_f1) / 2

    print(
        f"{model_name}: "
        f"Macro-F1={macro_f1:.4f}, "
        f"Micro-F1={micro_f1:.4f}, "
        f"Combined={combined:.4f}"
    )

# =========================
# 6. Weekly evaluation
# =========================

print("\n=== Weekly Overall Results ===")

weekly_results = []

for snapshot_date in sorted(valid["snapshot_date"].unique()):

    week_data = valid[
        valid["snapshot_date"] == snapshot_date
    ]

    for model_name in model_names:

        weekly_f1 = []
        all_true = []
        all_pred = []

        for label in label_cols:

            y_true = week_data[label].astype(int).to_numpy()

            # Get predictions corresponding to this week.
            week_indices = (
                valid["snapshot_date"] == snapshot_date
            ).to_numpy()

            y_pred = predictions[model_name][label][week_indices]

            f1 = f1_score(
                y_true, y_pred, zero_division=0
            )

            weekly_f1.append(f1)

            all_true.extend(y_true)
            all_pred.extend(y_pred)

        macro_f1 = np.mean(weekly_f1)

        micro_f1 = f1_score(
            all_true, all_pred, zero_division=0
        )

        combined = (macro_f1 + micro_f1) / 2

        weekly_results.append({
            "Week": pd.Timestamp(snapshot_date).strftime("%Y-%m-%d"),
            "Model": model_name,
            "Macro-F1": macro_f1,
            "Micro-F1": micro_f1,
            "Combined": combined
        })

weekly_df = pd.DataFrame(weekly_results)

print(
    weekly_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# =========================
# 7. Weekly per-label F1
# =========================

print("\n=== Weekly Per-label F1 ===")

weekly_label_results = []

for snapshot_date in sorted(valid["snapshot_date"].unique()):

    week_data = valid[
        valid["snapshot_date"] == snapshot_date
    ]

    week_indices = (
        valid["snapshot_date"] == snapshot_date
    ).to_numpy()

    for model_name in model_names:

        for label in label_cols:

            y_true = week_data[label].astype(int).to_numpy()
            y_pred = predictions[model_name][label][week_indices]

            weekly_label_results.append({
                "Week": pd.Timestamp(snapshot_date).strftime("%Y-%m-%d"),
                "Model": model_name,
                "Activity": label,
                "F1": f1_score(
                    y_true, y_pred, zero_division=0
                )
            })

weekly_label_df = pd.DataFrame(weekly_label_results)

print(
    weekly_label_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)
