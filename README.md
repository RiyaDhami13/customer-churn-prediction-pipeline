## Results

**Scope:** 3,070 inactive accounts with balance > $50K (SQL-extracted). 80/20 stratified split; test set = 614 customers, 193 churners (31.4% base rate).

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic regression (baseline) | 0.727 | 0.809 |
| Random forest (depth 8, 100 trees) | 0.788 | 0.842 |

**Threshold tradeoff (random forest):**

| Threshold | Flagged | Precision | Recall |
|---|---|---|---|
| 0.3 | 206 | 0.67 | 0.72 |
| 0.4 | 156 | 0.81 | 0.65 |
| 0.5 | 129 | 0.84 | 0.56 |
| 0.7 | 95 | 0.89 | 0.44 |

**Risk tiers (≥0.70 / 0.40–0.69 / <0.40):**

| Tier | Customers | Churn rate | Share of all churners | Lift |
|---|---|---|---|---|
| High | 95 | 89.5% | 44% | 2.8x |
| Medium | 61 | 67.2% | 21% | 2.1x |
| Low | 458 | 14.6% | 35% | 0.5x |

**Takeaway:** The High tier is highly reliable but captures under half of churners. I recommend a 0.40 cutoff as the best precision/recall tradeoff; the final cutoff should come from the retention offer's cost vs the value of a retained customer.

**Limitations:** Single train/test split; High-tier estimate rests on 95 customers (roughly ±6 points); feature importances show what the model uses, not what causes churn.
