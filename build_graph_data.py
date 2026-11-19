from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


# 1. Paths and settings

DB_DIR = Path("processed_data/db")
FEATURE_PATH = Path("processed_data/train_features.parquet")
OUTPUT_DIR = Path("processed_data/graphs_weighted")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

LABEL_COLS = [
    "y_open_pr",
    "y_open_issue",
    "y_issue_comment",
    "y_pr_review",
    "y_pr_review_comment",
    "y_push",
]


# 2. Load source tables

print("Loading Parquet tables...")

users = pd.read_parquet(DB_DIR / "Users.parquet")
repositories = pd.read_parquet(DB_DIR / "Repositories.parquet")
train = pd.read_parquet(FEATURE_PATH)

# Only cohort users are prediction targets.
cohort_users = users[users["is_cohort"] == True].copy()

# Map internal user_id -> stable github_user_id.
internal_to_github = dict(
    zip(cohort_users["user_id"], cohort_users["github_user_id"])
)

# Load activity tables.
activity_tables = {}

for activity, config in ACTIVITY_CONFIG.items():
    df = pd.read_parquet(DB_DIR / f'{config["table"]}.parquet')

    # open_issue should exclude PR conversations.
    if activity == "open_issue":
        df = df[df["is_pull_request"] == False]

    activity_tables[activity] = df

print("Tables loaded.")


# 3. Helper: get events within one snapshot's lookback window

def get_snapshot_events(snapshot_date):
    """
    Return activity events in:
    [snapshot_date - LOOKBACK_DAYS, snapshot_date)
    """

    start_date = snapshot_date - pd.Timedelta(days=LOOKBACK_DAYS)
    snapshot_events = {}

    for activity, config in ACTIVITY_CONFIG.items():
        df = activity_tables[activity].copy()

        actor_col = config["actor_col"]
        time_col = config["time_col"]

        df[time_col] = pd.to_datetime(df[time_col], utc=True)

        # Keep only historical events before snapshot_date.
        df = df[
            (df[time_col] >= start_date)
            & (df[time_col] < snapshot_date)
        ].copy()

        # Remove events without actor or repository.
        df = df.dropna(subset=[actor_col, "repo_id"])

        # Map internal user IDs to stable GitHub user IDs.
        df["github_user_id"] = df[actor_col].map(internal_to_github)

        # Remove unmapped users.
        df = df.dropna(subset=["github_user_id"])

        # Normalize GitHub IDs to consistent string format.
        df["github_user_id"] = (
            pd.to_numeric(df["github_user_id"], errors="coerce")
            .astype("Int64")
            .astype("string")
        )

        df = df.dropna(subset=["github_user_id"])

        print(
            f"[{activity}] "
            f"After mapping: {len(df)}, "
            f"Actor dtype: {df[actor_col].dtype}, "
            f"GitHub ID dtype: {df['github_user_id'].dtype}"
        )

        snapshot_events[activity] = df[
            ["github_user_id", "repo_id"]
        ].copy()

    return snapshot_events


# 4. Build one heterogeneous graph

def build_one_graph(snapshot_date, snapshot_train):
    print(f"\nBuilding graph for {snapshot_date.date()}...")

    # Keep a consistent user ordering.
    snapshot_train["github_user_id"] = (
        pd.to_numeric(
            snapshot_train["github_user_id"],
            errors="coerce"
        )
        .astype("Int64")
        .astype("string")
    )

    snapshot_train = (
        snapshot_train
        .dropna(subset=["github_user_id"])
        .sort_values("github_user_id")
        .reset_index(drop=True)
    )

    user_ids = snapshot_train["github_user_id"].tolist()
    print("User ID dtype in train:", snapshot_train["github_user_id"].dtype)
    print("User ID dtype in graph:", type(user_ids[0]) if user_ids else None)
    print("Number of graph users:", len(user_ids))

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    # User node features.
    user_x = (
        snapshot_train[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    # User labels.
    user_y = (
        snapshot_train[LABEL_COLS]
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    # Get historical events for this snapshot.
    events = get_snapshot_events(snapshot_date)

    # Collect repositories that appeared in these historical events.
    all_repo_ids = set()

    for df in events.values():
        all_repo_ids.update(df["repo_id"].unique().tolist())

    repo_ids = sorted(all_repo_ids)
    repo_to_idx = {rid: idx for idx, rid in enumerate(repo_ids)}

    # Repository features:
    # 6 event counts + 6 distinct participating-user counts.
    repo_x = np.zeros((len(repo_ids), 12), dtype=np.float32)

    relation_edges = {}

    activity_names = list(ACTIVITY_CONFIG.keys())

    for i, activity in enumerate(activity_names):
        df = events[activity].copy()

        # Keep only users in this snapshot's user list.
        before_filter = len(df)

        df = df[df["github_user_id"].isin(user_to_idx)]

        print(
            f"[{activity}] "
            f"Before cohort filter: {before_filter}, "
            f"After cohort filter: {len(df)}"
        )

        if len(df) > 0:
            # Count events per repository.
            event_counts = df.groupby("repo_id").size()

            # Count distinct participating users per repository.
            user_counts = df.groupby("repo_id")[
                "github_user_id"
            ].nunique()

            for repo_id, count in event_counts.items():
                repo_x[repo_to_idx[repo_id], i] = np.log1p(count)

            for repo_id, count in user_counts.items():
                repo_x[repo_to_idx[repo_id], i + 6] = np.log1p(count)

        # Preserve every activity event as an edge.
        # Repeated User-Repository interactions are retained
        # to reflect interaction frequency.

        if len(df) > 0:
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
            edge_index = torch.empty((2, 0), dtype=torch.long)

        relation_edges[activity] = edge_index

    # Create PyG heterogeneous graph.
    data = HeteroData()

    data["user"].x = torch.tensor(user_x, dtype=torch.float32)
    data["user"].y = torch.tensor(user_y, dtype=torch.float32)

    data["repo"].x = torch.tensor(repo_x, dtype=torch.float32)

    # Add six forward relations and six reverse relations.
    for activity, edge_index in relation_edges.items():
        forward = ("user", activity, "repo")
        reverse = ("repo", f"rev_{activity}", "user")

        data[forward].edge_index = edge_index
        data[reverse].edge_index = edge_index.flip(0)

    # Save metadata separately from the graph tensors.
    result = {
        "data": data,
        "snapshot_date": str(snapshot_date),
        "github_user_ids": user_ids,
        "feature_cols": FEATURE_COLS,
        "label_cols": LABEL_COLS,
    }

    return result


# 5. Build all snapshots and save

train["snapshot_date"] = pd.to_datetime(
    train["snapshot_date"], utc=True
)

snapshot_dates = sorted(train["snapshot_date"].unique())

for snapshot_date in snapshot_dates:
    snapshot_train = train[
        train["snapshot_date"] == snapshot_date
    ].copy()

    graph_result = build_one_graph(
        snapshot_date,
        snapshot_train,
    )

    output_path = OUTPUT_DIR / (
        f"graph_{snapshot_date.strftime('%Y-%m-%d')}.pt"
    )

    torch.save(graph_result, output_path)

    data = graph_result["data"]

    print("  Users:", data["user"].num_nodes)
    print("  Repositories:", data["repo"].num_nodes)
    print("  User features:", data["user"].x.shape)
    print("  Repo features:", data["repo"].x.shape)
    print("  Labels:", data["user"].y.shape)
    print("  Saved:", output_path)

print("\nAll snapshot graphs built successfully!")