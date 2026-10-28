## Results

**Scope:** 3,070 inactive accounts with balance > $50K (SQL-extracted). 80/20 stratified split; test set = 614 customers, 193 churners (31.4% base rate). All preprocessing runs inside scikit-learn Pipelines, fitted on training data only.

| Model | PR-AUC (test split) | PR-AUC (5-fold CV) | ROC-AUC (test split) |
|---|---|---|---|
| Logistic regression (baseline) | 0.727 | 0.685 ± 0.031 | 0.809 |
| Random forest (depth 8, 100 trees) | 0.785 | 0.769 ± 0.027 | 0.845 |

**Risk tiers (Low < 0.40, Medium 0.40-0.70, High ≥ 0.70):**

| Tier | Customers | Churn rate | Share of all churners | Lift |
|---|---|---|---|---|
| High | 86 | 91.9% | 41% | 2.9x |
| Medium | 72 | 63.9% | 24% | 2.0x |
| Low | 456 | 14.9% | 35% | 0.5x |

High-tier precision: 91.9% (95% bootstrap interval 86.0-96.5%, n = 86).

**Threshold tradeoff:**

| Threshold | Flagged | Precision | Recall |
|---|---|---|---|
| 0.3 | 204 | 0.68 | 0.72 |
| 0.4 | 158 | 0.79 | 0.65 |
| 0.5 | 130 | 0.85 | 0.57 |
| 0.6 | 104 | 0.88 | 0.47 |
| 0.7 | 86 | 0.92 | 0.41 |

**Takeaway:** The High tier is very reliable but captures only 41% of churners; 35% fall in the Low tier and would receive no offer. A 0.40 cutoff gives a reasonable balance (79% precision, 65% recall); the final cutoff should come from the cost of a retention offer versus the value of a retained customer.

**Limitations:** One dataset, one 80/20 split plus 5-fold CV; High-tier estimate rests on 86 customers; the model's feature usage does not imply causes of churn.
