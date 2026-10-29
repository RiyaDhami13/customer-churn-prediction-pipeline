"""Data layer: CSV -> SQLite staging -> three tables -> target segment (via SQL JOINs)."""
import sqlite3

import pandas as pd

# Inactive customers (IsActiveMember = 0) with balance above a minimum.
TARGET_QUERY = """
SELECT
    c.CustomerId, c.Surname, c.Geography, c.Gender, c.Age,
    cr.CreditScore, cr.Tenure,
    a.Balance, a.NumOfProducts, a.HasCrCard, a.EstimatedSalary, a.Exited
FROM db_account a
JOIN db_customer c ON a.CustomerId = c.CustomerId
JOIN db_credit cr ON a.CustomerId = cr.CustomerId
WHERE a.IsActiveMember = 0
  AND a.Balance > ?;
"""


def build_database(csv_path, db_path):
    """Load the raw CSV into SQLite and split it into customer / credit / account tables.

    Safe to run many times: old tables are dropped and rebuilt each run.
    Returns the number of rows staged.
    """
    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("raw_churn_data", conn, if_exists="replace", index=False)
        cur = conn.cursor()
        for table in ("db_customer", "db_credit", "db_account"):
            cur.execute(f"DROP TABLE IF EXISTS {table}")

        cur.execute(
            "CREATE TABLE db_customer AS "
            "SELECT CustomerId, Surname, Geography, Gender, Age FROM raw_churn_data"
        )
        cur.execute(
            "CREATE TABLE db_credit AS "
            "SELECT CustomerId, CreditScore, Tenure FROM raw_churn_data"
        )
        cur.execute(
            "CREATE TABLE db_account AS "
            "SELECT CustomerId, Balance, NumOfProducts, HasCrCard, IsActiveMember, "
            "EstimatedSalary, Exited FROM raw_churn_data"
        )
        cur.execute("DROP TABLE IF EXISTS raw_churn_data")
        conn.commit()
    finally:
        conn.close()
    return len(df)


def load_target_segment(db_path, min_balance=50_000):
    """Return inactive customers whose balance is above `min_balance`."""
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(TARGET_QUERY, conn, params=(min_balance,))
    finally:
        conn.close()
