
import pandas as pd
import numpy as np
from pathlib import Path


# 1. Paths and settings


DB_DIR = Path("processed_data/db")
OUTPUT_DIR = Path("processed_data")

# All timestamps in the tables are UTC.
# 30 days of history are available from 2026-08-15.
SNAPSHOT_DATES = pd.date_range(
    start="2026-09-14",
    end="2026-11-09",
    freq="7D",
    tz="UTC"
)

LOOKBACK_DAYS = [7, 14, 30]
PREDICTION_DAYS = 7


# 2. Load tables


print("Loading Parquet tables...")

users = pd.read_parquet(DB_DIR / "Users.parquet")

print("Users:", len(users))

# Only use cohort users as the training population.
cohort_users = users[users["is_cohort"] == True].copy()

cohort_users["github_user_id"] = (
    cohort_users["github_user_id"].astype(str)
)

user_ids = cohort_users["github_user_id"].drop_duplicates()

print("Cohort users:", len(user_ids))

# Map internal user_id to stable github_user_id.
user_map = users.set_index("user_id")["github_user_id"]
user_map.index = user_map.index.astype(str)
user_map = user_map.astype(str)


# 3. Prepare activity data


activity_config = {
    "open_pr": (
        "PullRequests", "author_id", "first_seen_at"
    ),
    "open_issue": (
        "Issues", "author_id", "first_seen_at"
    ),
    "issue_comment": (
        "IssueComments", "author_id", "created_at"
    ),
    "pr_review": (
        "PRReviews", "reviewer_id", "first_seen_at"
    ),
    "pr_review_comment": (
        "PRReviewComments", "author_id", "created_at"
    ),
    "push": (
        "Pushes", "user_id", "created_at"
    ),
}

activities = {}

for name, (table_name, actor_col, date_col) in activity_config.items():

    print("Preparing:", name)

    df = pd.read_parquet(DB_DIR / f"{table_name}.parquet")

    # Issues must exclude pull request conversations.
    if name == "open_issue":
        df = df[df["is_pull_request"] == False]

    # Keep records with a valid actor and timestamp.
    df = df.dropna(subset=[actor_col, date_col]).copy()

    # Map internal actor ID to stable GitHub user ID.
    df["_actor_key"] = df[actor_col].astype(str)
    df["github_user_id"] = df["_actor_key"].map(user_map)

    df = df.dropna(subset=["github_user_id"])

    # Convert timestamps to UTC.
    df["event_time"] = pd.to_datetime(
        df[date_col], utc=True, errors="coerce"
    )

    df = df.dropna(subset=["event_time"])

    # Keep only cohort users.
    df["github_user_id"] = df["github_user_id"].astype(str)
    df = df[df["github_user_id"].isin(user_ids)]

    activities[name] = df[
        ["github_user_id", "event_time", "repo_id"]
    ].copy()

    print(f"  Events: {len(activities[name]):,}")

# Prepare repository activity data


print("\nPreparing repository activity data...")

repo_events = pd.concat(
    activities.values(),
    ignore_index=True
)

# Only records with a valid repository ID.
repo_events = repo_events.dropna(subset=["repo_id"]).copy()

# Normalize repository IDs for consistent grouping.
repo_events["repo_id"] = repo_events["repo_id"].astype(str)

print("Repository events:", len(repo_events))
print("Unique repositories:", repo_events["repo_id"].nunique())


# 4. Build snapshots


all_snapshots = []

for snapshot_date in SNAPSHOT_DATES:

    print("\nBuilding snapshot:", snapshot_date)

    snapshot = pd.DataFrame({
        "github_user_id": user_ids.values
    })

    snapshot["snapshot_date"] = snapshot_date

    
    # Repository relationship features
    

    # Historical repository events: [cutoff - 30 days, cutoff)
    repo_history = repo_events[
        (repo_events["event_time"] >=
         snapshot_date - pd.Timedelta(days=30)) &
        (repo_events["event_time"] < snapshot_date)
    ].copy()

    # Number of unique repositories per user in the last 7 days.
    repo_history_7d = repo_history[
        repo_history["event_time"] >=
        snapshot_date - pd.Timedelta(days=7)
    ]

    unique_repos_7d = (
        repo_history_7d
        .groupby("github_user_id")["repo_id"]
        .nunique()
    )

    snapshot["unique_repos_7d"] = (
        snapshot["github_user_id"]
        .map(unique_repos_7d)
        .fillna(0)
        .astype(int)
    )

    # Number of unique repositories per user in the last 30 days.
    unique_repos_30d = (
        repo_history
        .groupby("github_user_id")["repo_id"]
        .nunique()
    )

    snapshot["unique_repos_30d"] = (
        snapshot["github_user_id"]
        .map(unique_repos_30d)
        .fillna(0)
        .astype(int)
    )

    # Average number of observed events per repository for each user.
    repo_counts = (
        repo_history
        .groupby(["github_user_id", "repo_id"])
        .size()
        .reset_index(name="event_count")
    )

    avg_repo_events_30d = (
        repo_counts
        .groupby("github_user_id")["event_count"]
        .mean()
    )

    snapshot["avg_repo_events_30d"] = (
        snapshot["github_user_id"]
        .map(avg_repo_events_30d)
        .fillna(0)
    )


    # User-level static feature.
    bot_map = (
        cohort_users
        .drop_duplicates("github_user_id")
        .set_index("github_user_id")["is_bot"]
    )

    snapshot["is_bot"] = (
        snapshot["github_user_id"].map(bot_map).fillna(False).astype(int)
    )

    for name, df in activities.items():

        # Historical data: [cutoff - 30 days, cutoff)
        history = df[
            (df["event_time"] >= snapshot_date - pd.Timedelta(days=30)) &
            (df["event_time"] < snapshot_date)
        ]

        # Count features: 7, 14, and 30 days.
        for days in LOOKBACK_DAYS:

            window = history[
                history["event_time"] >=
                snapshot_date - pd.Timedelta(days=days)
            ]

            counts = window.groupby("github_user_id").size()

            feature_name = f"{name}_count_{days}d"

            snapshot[feature_name] = (
                snapshot["github_user_id"]
                .map(counts)
                .fillna(0)
                .astype(int)
            )


        # Recency: days since the most recent activity.
        last_event = history.groupby("github_user_id")[
            "event_time"
        ].max()

        recency = (
            snapshot_date - snapshot["github_user_id"].map(last_event)
        ).dt.total_seconds() / 86400

        snapshot[f"{name}_recency"] = recency.fillna(999)

        # Future labels: [cutoff, cutoff + 7 days)
        future = df[
            (df["event_time"] >= snapshot_date) &
            (df["event_time"] <
             snapshot_date + pd.Timedelta(days=PREDICTION_DAYS))
        ]

        future_users = set(future["github_user_id"].unique())

        snapshot[f"y_{name}"] = (
            snapshot["github_user_id"].isin(future_users).astype(int)
        )

    all_snapshots.append(snapshot)


# 5. Combine and save


train_data = pd.concat(all_snapshots, ignore_index=True)

output_path = OUTPUT_DIR / "train_features.parquet"
train_data.to_parquet(output_path, index=False)

print("\n" + "=" * 50)
print("Training data created!")
print("=" * 50)

print("Shape:", train_data.shape)
print("Snapshots:", train_data["snapshot_date"].nunique())
print("Unique users:", train_data["github_user_id"].nunique())

print("\nLabel positive rates:")
for name in activity_config:
    print(
        f"{name:20s}: "
        f"{train_data[f'y_{name}'].mean():.4f}"
    )

print("\nSaved to:", output_path)
