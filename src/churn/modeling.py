"""Modeling layer: leakage-safe preprocessing + models, built as scikit-learn Pipelines.

Every preprocessing step lives INSIDE a Pipeline, so it is fitted on training rows only.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Exited"
ID_COLS = ["CustomerId", "Surname"]
CATEGORICAL_COLS = ["Geography", "Gender"]


def split_features_target(df):
    """Separate the clues (X) from the answer (y). No scaling or encoding yet."""
    X = df.drop(columns=ID_COLS + [TARGET])
    y = df[TARGET]
    return X, y


def get_column_groups(X):
    """Return (categorical columns, numeric columns)."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


def build_rf_pipeline(cat_cols, n_estimators=100, max_depth=8, seed=42):
    """Random forest: one-hot encode categories only (trees do not need scaling)."""
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed
    )
    return Pipeline([("prep", prep), ("model", model)])


def build_lr_pipeline(cat_cols, num_cols, seed=42):
    """Logistic regression baseline: one-hot encode + scale (scaler learns from train only)."""
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    model = LogisticRegression(max_iter=1000, random_state=seed)
    return Pipeline([("prep", prep), ("model", model)])
