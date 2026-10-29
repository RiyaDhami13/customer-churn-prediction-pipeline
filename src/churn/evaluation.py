"""Evaluation layer: risk tiers, cross-validation, bootstrap interval, threshold sweep."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

MEDIUM_CUTOFF = 0.40  # probability at or above this is at least Medium risk
HIGH_CUTOFF = 0.70    # probability at or above this is High risk


def assign_tiers(probs):
    """Low < 0.40, Medium 0.40 to <0.70, High >= 0.70 (left edge included)."""
    return pd.cut(
        np.asarray(probs),
        bins=[-0.001, MEDIUM_CUTOFF, HIGH_CUTOFF, 1.001],
        labels=["Low", "Medium", "High"],
        right=False,
    )


def tier_table(y_true, probs):
    """Per tier: customers, churners, precision, share of all churners caught, lift."""
    y_true = np.asarray(y_true)
    d = pd.DataFrame({"y": y_true, "tier": assign_tiers(probs)})
    tbl = d.groupby("tier", observed=True).agg(
        customers=("y", "size"), churners=("y", "sum"), precision=("y", "mean")
    )
    tbl["recall_share"] = tbl["churners"] / y_true.sum()
    tbl["lift"] = tbl["precision"] / y_true.mean()
    return tbl.reindex(["High", "Medium", "Low"])


def bootstrap_interval(outcomes, n_boot=5000, seed=42, level=95):
    """Bootstrap interval for the mean of 0/1 outcomes (e.g. High-tier precision)."""
    outcomes = np.asarray(outcomes)
    rng = np.random.default_rng(seed)
    means = [
        rng.choice(outcomes, size=len(outcomes), replace=True).mean()
        for _ in range(n_boot)
    ]
    tail = (100 - level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def threshold_sweep(y_true, probs, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Precision and recall at several cutoffs: the precision/recall tradeoff."""
    probs = np.asarray(probs)
    rows = []
    for t in thresholds:
        flags = (probs >= t).astype(int)
        rows.append(
            {
                "threshold": t,
                "flagged": int(flags.sum()),
                "precision": precision_score(y_true, flags, zero_division=0),
                "recall": recall_score(y_true, flags, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def cv_pr_auc(pipe, X, y, n_splits=5, seed=42):
    """5-fold cross-validated PR-AUC. Returns (mean, std)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")
    return float(scores.mean()), float(scores.std())
