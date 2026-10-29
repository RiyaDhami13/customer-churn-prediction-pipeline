"""Train and evaluate the churn models end to end.

Usage:
    python train.py --csv Churn_Modelling.csv
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Let this script import the package in src/ without installing it
sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn.data import build_database, load_target_segment  # noqa: E402
from churn.evaluation import (  # noqa: E402
    bootstrap_interval,
    cv_pr_auc,
    threshold_sweep,
    tier_table,
)
from churn.modeling import (  # noqa: E402
    build_lr_pipeline,
    build_rf_pipeline,
    get_column_groups,
    split_features_target,
)

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Train churn models")
    p.add_argument("--csv", default="Churn_Modelling.csv", help="path to the raw CSV")
    p.add_argument("--db", default="churn_banking.db", help="SQLite file to create")
    p.add_argument("--models-dir", default="models", help="where to save the model")
    p.add_argument("--reports-dir", default="reports", help="where to save metrics")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Data: CSV -> SQLite -> target segment
    n_rows = build_database(args.csv, args.db)
    df = load_target_segment(args.db)
    print(f"Staged {n_rows} records; target segment: {len(df)} customers")

    # 2. Split FIRST, on raw features (this is what prevents leakage)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    cat_cols, num_cols = get_column_groups(X)
    print(f"Train {len(X_train)} / test {len(X_test)}; base churn rate {y_test.mean():.1%}")

    # 3. Fit both pipelines on training data, score on the test set, cross-validate
    pipes = {
        "LogReg": build_lr_pipeline(cat_cols, num_cols, SEED),
        "RandomForest": build_rf_pipeline(cat_cols, seed=SEED),
    }
    probs, metrics = {}, {}
    for name, pipe in pipes.items():
        pipe.fit(X_train, y_train)
        probs[name] = pipe.predict_proba(X_test)[:, 1]
        cv_mean, cv_std = cv_pr_auc(pipe, X, y, seed=SEED)
        metrics[name] = {
            "pr_auc_test": float(average_precision_score(y_test, probs[name])),
            "roc_auc_test": float(roc_auc_score(y_test, probs[name])),
            "pr_auc_cv_mean": cv_mean,
            "pr_auc_cv_std": cv_std,
        }
        m = metrics[name]
        print(f"{name}: test PR-AUC {m['pr_auc_test']:.3f}, ROC-AUC {m['roc_auc_test']:.3f}, "
              f"5-fold PR-AUC {m['pr_auc_cv_mean']:.3f} +/- {m['pr_auc_cv_std']:.3f}")

    # 4. Risk tiers, uncertainty and thresholds for the random forest
    rf_probs = probs["RandomForest"]
    tiers = tier_table(y_test, rf_probs)
    print("\nRisk tiers:\n", tiers.round(3))

    high_outcomes = y_test.values[rf_probs >= 0.70]
    if len(high_outcomes) > 0:
        low, high = bootstrap_interval(high_outcomes, seed=SEED)
        metrics["high_tier"] = {
            "customers": int(len(high_outcomes)),
            "precision": float(high_outcomes.mean()),
            "ci_95": [low, high],
        }
        print(f"\nHigh tier: {len(high_outcomes)} customers, "
              f"precision {high_outcomes.mean():.3f}, 95% interval [{low:.3f}, {high:.3f}]")

    sweep = threshold_sweep(y_test, rf_probs)
    print("\nThreshold sweep:\n", sweep.round(2).to_string(index=False))

    # 5. Save the trained model and the metrics
    Path(args.models_dir).mkdir(exist_ok=True)
    Path(args.reports_dir).mkdir(exist_ok=True)
    joblib.dump(pipes["RandomForest"], Path(args.models_dir) / "rf_pipeline.joblib")
    with open(Path(args.reports_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model to {args.models_dir}/ and metrics to {args.reports_dir}/")


if __name__ == "__main__":
    main()
