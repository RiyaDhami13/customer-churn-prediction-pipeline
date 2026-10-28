"""Tests for the churn package. Run with: pytest"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.data import build_database, load_target_segment
from churn.evaluation import assign_tiers, bootstrap_interval, tier_table
from churn.modeling import (
    build_lr_pipeline,
    build_rf_pipeline,
    get_column_groups,
    split_features_target,
)


def make_fake_df(n=400, seed=0):
    """Small made-up dataset with the same columns as Churn_Modelling.csv."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    exited = (rng.random(n) < (age - 18) / 100 + 0.1).astype(int)  # older = more churn
    return pd.DataFrame(
        {
            "RowNumber": np.arange(1, n + 1),
            "CustomerId": np.arange(10_000, 10_000 + n),
            "Surname": ["Test"] * n,
            "CreditScore": rng.integers(350, 850, n),
            "Geography": rng.choice(["France", "Germany", "Spain"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "Age": age,
            "Tenure": rng.integers(0, 11, n),
            "Balance": rng.uniform(0, 200_000, n).round(2),
            "NumOfProducts": rng.integers(1, 5, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.uniform(10_000, 150_000, n).round(2),
            "Exited": exited,
        }
    )


def test_assign_tiers_boundaries():
    # 0.40 is Medium and 0.70 is High (left edge included); 0.39 is still Low
    tiers = list(assign_tiers([0.39, 0.40, 0.69, 0.70, 0.99]))
    assert tiers == ["Low", "Medium", "Medium", "High", "High"]


def test_tier_table_counts_and_precision():
    y = [1, 1, 0, 1, 0, 0]
    probs = [0.9, 0.8, 0.75, 0.5, 0.2, 0.1]
    tbl = tier_table(y, probs)
    assert tbl.loc["High", "customers"] == 3
    assert tbl.loc["High", "churners"] == 2
    assert abs(tbl.loc["High", "precision"] - 2 / 3) < 1e-9
    assert abs(tbl.loc["High", "recall_share"] - 2 / 3) < 1e-9  # caught 2 of 3 churners


def test_bootstrap_interval_behaviour():
    low, high = bootstrap_interval([1] * 20, n_boot=200)
    assert low == high == 1.0  # no variation, no uncertainty
    low, high = bootstrap_interval([1, 0] * 20, n_boot=500)
    assert low <= 0.5 <= high


def test_database_and_segment(tmp_path):
    df = make_fake_df()
    csv_path, db_path = tmp_path / "data.csv", tmp_path / "test.db"
    df.to_csv(csv_path, index=False)

    n_rows = build_database(csv_path, db_path)
    seg = load_target_segment(db_path)

    expected = ((df["IsActiveMember"] == 0) & (df["Balance"] > 50_000)).sum()
    assert n_rows == len(df)
    assert len(seg) == expected
    assert (seg["Balance"] > 50_000).all()


def test_pipeline_outputs_valid_probabilities():
    X, y = split_features_target(make_fake_df())
    cat_cols, _ = get_column_groups(X)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    pipe = build_rf_pipeline(cat_cols, n_estimators=20, max_depth=4)
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]

    assert len(probs) == len(X_test)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_scaler_learns_from_training_rows_only():
    # This is the leakage test: the scaler's mean must equal the TRAIN mean, not the full-data mean
    X, y = split_features_target(make_fake_df())
    cat_cols, num_cols = get_column_groups(X)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    pipe = build_lr_pipeline(cat_cols, num_cols)
    pipe.fit(X_train, y_train)
    scaler = pipe.named_steps["prep"].named_transformers_["num"]

    np.testing.assert_allclose(scaler.mean_, X_train[num_cols].mean().values)
    assert not np.allclose(scaler.mean_, X[num_cols].mean().values)
