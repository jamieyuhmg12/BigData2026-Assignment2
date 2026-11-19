from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


# 1. Settings

DB_DIR = Path("processed_data/db")
CHECKPOINT_DIR = Path("processed_data/gnn_first_seen_h64_d02_e30")
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_gnn_model.pt"

TEST_PATH = Path("submission/test_input.csv")
OUTPUT_PATH = Path("submission/output_112705020.csv")

# Temporary prediction cutoff:
# 2026-11-16 00:00 UTC = 2026-11-16 08:00 Taiwan time
CUTOFF = pd.Timestamp("2026-11-16 00:00:00", tz="UTC")

LOOKBACK_DAYS = 30

ACTIVITY_CONFIG = {
    "open_pr": {
        "table": "PullRequests",
        "actor_col": "author_id",
        "time_col": "first_seen_at",
    },
    "open_issue": {
        "table": "Issues",
        "actor_col": "author_id",
        "time_col": "first_seen_at",
    },
    "issue_comment": {
        "table": "IssueComments",
        "actor_col": "author_id",
        "time_col": "created_at",
    },
    "pr_review": {
        "table": "PRReviews",
        "actor_col": "reviewer_id",
        "time_col": "first_seen_at",
    },
    "pr_review_comment": {
        "table": "PRReviewComments",
        "actor_col": "author_id",
        "time_col": "created_at",
    },
    "push": {
        "table": "Pushes",
        "actor_col": "user_id",
        "time_col": "created_at",
    },
}

LABEL_NAMES = [
    "open_pr",
    "open_issue",
    "issue_comment",
    "pr_review",
    "pr_review_comment",
    "push",
]

DEVICE = torch.device("cpu")


# 2. Feature columns

FEATURE_COLS = [
    "unique_repos_7d",
    "unique_repos_30d",
    "avg_repo_events_30d",
    "is_bot",

    "open_pr_count_7d",
    "open_pr_count_14d",
    "open_pr_count_30d",
    "open_pr_recency",

    "open_issue_count_7d",
    "open_issue_count_14d",
    "open_issue_count_30d",
    "open_issue_recency",

    "issue_comment_count_7d",
    "issue_comment_count_14d",
    "issue_comment_count_30d",
    "issue_comment_recency",

    "pr_review_count_7d",
    "pr_review_count_14d",
    "pr_review_count_30d",
    "pr_review_recency",

    "pr_review_comment_count_7d",
    "pr_review_comment_count_14d",
    "pr_review_comment_count_30d",
    "pr_review_comment_recency",

    "push_count_7d",
    "push_count_14d",
    "push_count_30d",
    "push_recency",
]


# 3. Model architecture

RELATIONS = [
    ("user", "open_pr", "repo"),
    ("user", "open_issue", "repo"),
    ("user", "issue_comment", "repo"),
    ("user", "pr_review", "repo"),
    ("user", "pr_review_comment", "repo"),
    ("user", "push", "repo"),

    ("repo", "rev_open_pr", "user"),
    ("repo", "rev_open_issue", "user"),
    ("repo", "rev_issue_comment", "user"),
    ("repo", "rev_pr_review", "user"),
    ("repo", "rev_pr_review_comment", "user"),
    ("repo", "rev_push", "user"),
]


class HeteroGraphSAGE(nn.Module):

    def __init__(self, hidden_channels=64, dropout=0.2):
        super().__init__()

        self.conv1 = HeteroConv(
            {
                rel: SAGEConv(
                    (-1, -1),
                    hidden_channels,
                )
                for rel in RELATIONS
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                rel: SAGEConv(
                    (-1, -1),
                    hidden_channels,
                )
                for rel in RELATIONS
            },
            aggr="sum",
        )

        self.user_norm1 = nn.LayerNorm(hidden_channels)
        self.repo_norm1 = nn.LayerNorm(hidden_channels)

        self.user_norm2 = nn.LayerNorm(hidden_channels)
        self.repo_norm2 = nn.LayerNorm(hidden_channels)

        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 6),
        )

    def forward(self, data):

        x_dict = {
            "user": data["user"].x,
            "repo": data["repo"].x,
        }

        x_dict = self.conv1(
            x_dict,
            data.edge_index_dict,
        )

        x_dict["user"] = F.relu(
            self.user_norm1(x_dict["user"])
        )

        x_dict["repo"] = F.relu(
            self.repo_norm1(x_dict["repo"])
        )

        x_dict = {
            key: F.dropout(
                value,
                p=self.dropout,
                training=self.training,
            )
            for key, value in x_dict.items()
        }

        x_dict = self.conv2(
            x_dict,
            data.edge_index_dict,
        )

        x_dict["user"] = F.relu(
            self.user_norm2(x_dict["user"])
        )

        x_dict["repo"] = F.relu(
            self.repo_norm2(x_dict["repo"])
        )

        logits = self.classifier(x_dict["user"])

        return logits


# =========================================================
# 4. Helper functions
# =========================================================

def normalize_github_ids(series):
    """Normalize GitHub IDs to nullable integer strings."""

    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
    )


def load_checkpoint():

    print("\nLoading trained GNN checkpoint...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
        )

    return checkpoint


def prepare_users():

    print("\nLoading users...")

    users = pd.read_parquet(DB_DIR / "Users.parquet")

    cohort_users = users[
        users["is_cohort"] == True
    ].copy()

    cohort_users["github_user_id"] = normalize_github_ids(
        cohort_users["github_user_id"]
    )

    cohort_users = cohort_users.dropna(
        subset=["github_user_id"]
    )

    cohort_users = (
        cohort_users
        .drop_duplicates("github_user_id")
        .sort_values("github_user_id")
        .reset_index(drop=True)
    )

    user_ids = cohort_users["github_user_id"].tolist()

    # All users are used for activity-to-GitHub-ID mapping.
    user_map = users.set_index(
        users["user_id"].astype(str)
    )["github_user_id"]

    user_map = normalize_github_ids(user_map)

    # Only cohort users are graph nodes.
    cohort_map = cohort_users.set_index(
        cohort_users["user_id"].astype(str)
    )["github_user_id"]

    cohort_map = normalize_github_ids(cohort_map)

    print("Cohort users:", len(user_ids))

    return users, cohort_users, user_ids, user_map, cohort_map


# 5. Load and prepare activity events

def prepare_activities(user_ids, user_map):

    print("\nPreparing activity events...")

    activities = {}

    cutoff_start = CUTOFF - pd.Timedelta(
        days=LOOKBACK_DAYS
    )

    for name, config in ACTIVITY_CONFIG.items():

        print("Preparing:", name)

        df = pd.read_parquet(
            DB_DIR / f'{config["table"]}.parquet'
        )

        actor_col = config["actor_col"]
        time_col = config["time_col"]

        if name == "open_issue":
            df = df[
                df["is_pull_request"] == False
            ].copy()

        df = df.dropna(
            subset=[actor_col, time_col]
        ).copy()

        df["_actor_key"] = df[actor_col].astype(str)

        df["github_user_id"] = df[
            "_actor_key"
        ].map(user_map)

        df = df.dropna(
            subset=["github_user_id"]
        ).copy()

        df["github_user_id"] = normalize_github_ids(
            df["github_user_id"]
        )

        df["event_time"] = pd.to_datetime(
            df[time_col],
            utc=True,
            errors="coerce",
        )

        df = df.dropna(
            subset=["github_user_id", "event_time"]
        )

        # Keep only cohort users and records before cutoff.
        df = df[
            df["github_user_id"].isin(user_ids)
            & (df["event_time"] >= cutoff_start)
            & (df["event_time"] < CUTOFF)
        ].copy()

        activities[name] = df[
            ["github_user_id", "event_time", "repo_id"]
        ].copy()

        print(
            f"  Events in lookback window: "
            f"{len(activities[name]):,}"
        )

    return activities


# =========================================================
# 6. Build user features
# =========================================================

def build_user_features(
    cohort_users,
    user_ids,
    activities,
):

    print("\nBuilding user features...")

    snapshot = pd.DataFrame({
        "github_user_id": user_ids
    })

    repo_events = pd.concat(
        activities.values(),
        ignore_index=True,
    )

    repo_events = repo_events.dropna(
        subset=["repo_id"]
    ).copy()

    repo_events["repo_id"] = (
        repo_events["repo_id"].astype(str)
    )

    # Repository relationship features.
    history_7d = repo_events[
        repo_events["event_time"]
        >= CUTOFF - pd.Timedelta(days=7)
    ]

    unique_repos_7d = (
        history_7d
        .groupby("github_user_id")["repo_id"]
        .nunique()
    )

    snapshot["unique_repos_7d"] = (
        snapshot["github_user_id"]
        .map(unique_repos_7d)
        .fillna(0)
        .astype(int)
    )

    unique_repos_30d = (
        repo_events
        .groupby("github_user_id")["repo_id"]
        .nunique()
    )

    snapshot["unique_repos_30d"] = (
        snapshot["github_user_id"]
        .map(unique_repos_30d)
        .fillna(0)
        .astype(int)
    )

    repo_counts = (
        repo_events
        .groupby(["github_user_id", "repo_id"])
        .size()
        .reset_index(name="event_count")
    )

    avg_repo_events = (
        repo_counts
        .groupby("github_user_id")["event_count"]
        .mean()
    )

    snapshot["avg_repo_events_30d"] = (
        snapshot["github_user_id"]
        .map(avg_repo_events)
        .fillna(0)
    )

    # Static bot feature.
    bot_map = (
        cohort_users
        .drop_duplicates("github_user_id")
        .set_index("github_user_id")["is_bot"]
    )

    snapshot["is_bot"] = (
        snapshot["github_user_id"]
        .map(bot_map)
        .fillna(False)
        .astype(int)
    )

    # Activity count and recency features.
    for name, df in activities.items():

        for days in [7, 14, 30]:

            window = df[
                df["event_time"]
                >= CUTOFF - pd.Timedelta(days=days)
            ]

            counts = (
                window
                .groupby("github_user_id")
                .size()
            )

            feature_name = f"{name}_count_{days}d"

            snapshot[feature_name] = (
                snapshot["github_user_id"]
                .map(counts)
                .fillna(0)
                .astype(int)
            )

        last_event = (
            df.groupby("github_user_id")["event_time"]
            .max()
        )

        recency = (
            CUTOFF
            - snapshot["github_user_id"].map(last_event)
        ).dt.total_seconds() / 86400

        snapshot[f"{name}_recency"] = (
            recency.fillna(999)
        )

    # Match the exact feature order used during training.
    snapshot = snapshot[
        ["github_user_id"] + FEATURE_COLS
    ]

    print("User feature shape:", snapshot[FEATURE_COLS].shape)

    return snapshot, repo_events


# 7. Build inference graph

def build_inference_graph(
    user_features,
    user_ids,
    cohort_map,
    activities,
    repo_events,
    checkpoint,
):

    print("\nBuilding inference graph...")

    user_to_idx = {
        uid: idx
        for idx, uid in enumerate(user_ids)
    }

    user_x = (
        user_features[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    # Build graph activity events using cohort mapping,
    # matching build_graph_data.py.
    graph_events = {}

    for name, config in ACTIVITY_CONFIG.items():

        df = pd.read_parquet(
            DB_DIR / f'{config["table"]}.parquet'
        )

        actor_col = config["actor_col"]
        time_col = config["time_col"]

        if name == "open_issue":
            df = df[
                df["is_pull_request"] == False
            ].copy()

        df = df.dropna(
            subset=[actor_col, time_col, "repo_id"]
        ).copy()

        df[time_col] = pd.to_datetime(
            df[time_col],
            utc=True,
            errors="coerce",
        )

        df = df.dropna(subset=[time_col])

        df = df[
            (df[time_col] >= CUTOFF - pd.Timedelta(days=30))
            & (df[time_col] < CUTOFF)
        ].copy()

        df["_actor_key"] = df[actor_col].astype(str)

        df["github_user_id"] = df[
            "_actor_key"
        ].map(cohort_map)

        df = df.dropna(
            subset=["github_user_id"]
        ).copy()

        df["github_user_id"] = normalize_github_ids(
            df["github_user_id"]
        )

        df = df.dropna(
            subset=["github_user_id"]
        )

        df["repo_id"] = df["repo_id"].astype(str)

        graph_events[name] = df[
            ["github_user_id", "repo_id"]
        ].copy()

        print(
            f"[{name}] Graph events: {len(df):,}"
        )

    # Collect repositories appearing in the graph.
    all_repo_ids = set()

    for df in graph_events.values():
        all_repo_ids.update(
            df["repo_id"].unique().tolist()
        )

    repo_ids = sorted(all_repo_ids)

    if len(repo_ids) == 0:
        raise RuntimeError(
            "No repositories found in the 30-day graph window."
        )

    repo_to_idx = {
        rid: idx
        for idx, rid in enumerate(repo_ids)
    }

    # 12 repository features.
    repo_x = np.zeros(
        (len(repo_ids), 12),
        dtype=np.float32,
    )

    relation_edges = {}

    activity_names = list(ACTIVITY_CONFIG.keys())

    for i, activity in enumerate(activity_names):

        df = graph_events[activity].copy()

        df = df[
            df["github_user_id"].isin(user_to_idx)
        ].copy()

        if len(df) > 0:

            event_counts = (
                df.groupby("repo_id").size()
            )

            user_counts = (
                df.groupby("repo_id")["github_user_id"]
                .nunique()
            )

            for repo_id, count in event_counts.items():
                repo_x[repo_to_idx[repo_id], i] = np.log1p(count)

            for repo_id, count in user_counts.items():
                repo_x[repo_to_idx[repo_id], i + 6] = np.log1p(count)

            user_indices = (
                df["github_user_id"]
                .map(user_to_idx)
                .to_numpy(dtype=np.int64)
            )

            repo_indices = (
                df["repo_id"]
                .map(repo_to_idx)
                .to_numpy(dtype=np.int64)
            )

            edge_index = torch.tensor(
                np.vstack([user_indices, repo_indices]),
                dtype=torch.long,
            )

        else:

            edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
            )

        relation_edges[activity] = edge_index

    # Create heterogeneous graph.
    data = HeteroData()

    data["user"].x = torch.tensor(
        user_x,
        dtype=torch.float32,
    )

    data["repo"].x = torch.tensor(
        repo_x,
        dtype=torch.float32,
    )

    for activity, edge_index in relation_edges.items():

        forward = ("user", activity, "repo")
        reverse = ("repo", f"rev_{activity}", "user")

        data[forward].edge_index = edge_index
        data[reverse].edge_index = edge_index.flip(0)

    # Apply training-set normalization statistics.
    user_mean = checkpoint["user_mean"].float()
    user_std = checkpoint["user_std"].float()

    repo_mean = checkpoint["repo_mean"].float()
    repo_std = checkpoint["repo_std"].float()

    data["user"].x = (
        data["user"].x - user_mean
    ) / user_std

    data["repo"].x = (
        data["repo"].x - repo_mean
    ) / repo_std

    print("User nodes:", data["user"].num_nodes)
    print("Repository nodes:", data["repo"].num_nodes)

    for activity in activity_names:
        edge = data["user", activity, "repo"].edge_index
        print(f"{activity} edges: {edge.shape[1]:,}")

    return data


# 8. Load model and predict

def predict(data, checkpoint):

    print("\nLoading model...")

    model = HeteroGraphSAGE(
        hidden_channels=64,
        dropout=0.2,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Model loaded successfully.")

    thresholds = np.asarray(
        checkpoint["calibrated_thresholds"],
        dtype=np.float32,
    )

    print("\nCalibrated thresholds:")

    for name, threshold in zip(
        LABEL_NAMES,
        thresholds,
    ):
        print(f"{name:20s}: {threshold:.2f}")

    print("\nGenerating predictions...")

    with torch.no_grad():

        logits = model(data)

        probabilities = torch.sigmoid(logits).cpu().numpy()

    predictions = (
        probabilities >= thresholds
    ).astype(int)

    return predictions


# 9. Main

def main():

    print("=" * 60)
    print("GitHub User Activity Prediction")
    print("=" * 60)

    print("Cutoff:", CUTOFF)
    print("Lookback:", LOOKBACK_DAYS, "days")

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test input not found: {TEST_PATH}"
        )

    checkpoint = load_checkpoint()

    
    # Read test users first.
    test_df = pd.read_csv(TEST_PATH)

    if test_df.shape[1] < 1:
        raise RuntimeError("test_input.csv has no columns.")

    test_id_col = test_df.columns[0]

    test_ids = normalize_github_ids(
        test_df[test_id_col]
    )

    if test_ids.isna().any():
        raise RuntimeError(
            "Some test user IDs could not be parsed as GitHub IDs."
        )

    if test_ids.duplicated().any():
        raise RuntimeError(
            "Duplicate user IDs found in test_input.csv."
        )

    # Load cohort users and ID mappings.
    (
        users,
        cohort_users,
        user_ids,
        user_map,
        cohort_map,
    ) = prepare_users()

    # Add test users that do not exist in the observed cohort.
    existing_ids = set(user_ids)

    missing_ids = [
        uid for uid in test_ids.tolist()
        if uid not in existing_ids
    ]

    if missing_ids:
        print(
            f"\nAdding {len(missing_ids)} test users "
            "with no observed history..."
        )

        # Create synthetic user rows with no historical activity.
        synthetic_users = pd.DataFrame(
            index=range(len(missing_ids)),
            columns=cohort_users.columns,
        )

        synthetic_users["github_user_id"] = missing_ids
        synthetic_users["user_id"] = [
            -(i + 1) for i in range(len(missing_ids))
        ]
        synthetic_users["is_cohort"] = True

        if "is_bot" in synthetic_users.columns:
            synthetic_users["is_bot"] = False

        cohort_users = pd.concat(
            [cohort_users, synthetic_users],
            ignore_index=True,
        )

        cohort_users = (
            cohort_users
            .sort_values("github_user_id")
            .reset_index(drop=True)
        )

        user_ids = cohort_users["github_user_id"].tolist()

        print("Synthetic users added:", len(missing_ids))
        print("Total inference users:", len(user_ids))

    # Load events.
    activities = prepare_activities(
        user_ids,
        user_map,
    )

    # Build user features.
    user_features, repo_events = build_user_features(
        cohort_users,
        user_ids,
        activities,
    )

    # Build graph and normalize with saved training statistics.
    data = build_inference_graph(
        user_features,
        user_ids,
        cohort_map,
        activities,
        repo_events,
        checkpoint,
    )

    # Predict all users, including synthetic users.
    all_predictions = predict(
        data,
        checkpoint,
    )


    
    # Map test users to inference graph indices.
    user_to_idx = {
        uid: idx
        for idx, uid in enumerate(user_ids)
    }

    test_indices = [
        user_to_idx[uid]
        for uid in test_ids.tolist()
    ]

    test_predictions = all_predictions[test_indices]


    # Use sample_submission header if available.
    template_candidates = [
        Path("submission/sample_submission.csv"),
        Path("sample_submission.csv"),
    ]

    template_path = next(
        (
            p for p in template_candidates
            if p.exists()
        ),
        None,
    )

    if template_path is not None:

        template = pd.read_csv(template_path)
        output_columns = template.columns.tolist()

        if len(output_columns) != 7:
            raise RuntimeError(
                "sample_submission.csv should have "
                "one ID column and six label columns."
            )

    else:

        output_columns = [
            test_id_col
        ] + LABEL_NAMES

    # Preserve test_input row order.
    output = pd.DataFrame({
        output_columns[0]: test_df[test_id_col].values
    })

    for i, label in enumerate(output_columns[1:]):
        output[label] = test_predictions[:, i].astype(int)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # Final checks.
    saved = pd.read_csv(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("Prediction completed!")
    print("=" * 60)

    print("Output:", OUTPUT_PATH)
    print("Rows:", len(saved))
    print("Columns:", saved.columns.tolist())

    print("\nPrediction positive counts:")

    for col in output_columns[1:]:
        print(
            f"{col:25s}: "
            f"{int(saved[col].sum()):,}"
        )

    if len(saved) != len(test_df):
        raise RuntimeError(
            "Output row count does not match test_input."
        )

    if saved.iloc[:, 1:].isna().any().any():
        raise RuntimeError(
            "Output contains missing predictions."
        )

    if not saved.iloc[:, 1:].isin([0, 1]).all().all():
        raise RuntimeError(
            "Output contains values other than 0 and 1."
        )

    print("\nAll output checks passed.")


if __name__ == "__main__":
    main()