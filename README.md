# Customer Churn Prediction & Risk Tier Pipeline

An end-to-end Machine Learning pipeline designed to predict customer churn, quantify departure probability, and categorize accounts into operational risk tiers for targeted retention.

## 📌 Features & Workflow
* **SQL Staging**: Extracted and filtered relational banking records into a priority target dataset.
* **Preprocessing**: Applied One-Hot Encoding and `MinMaxScaler` standardization while avoiding data leakage.
* **Predictive Modeling**: Trained a Random Forest Classifier (`n_estimators=100`, `max_depth=8`) optimized for high precision.
* **Risk Deployment**: Segmented predicted probabilities into High ($\ge 70\%$), Medium ($40\%-69\%$), and Low ($< 40\%$) risk tiers.

---

## 📊 Performance & Key Findings
* **Precision Score**: `0.84`
* **ROC-AUC Score**: `0.84`
* **High-Risk Tier Accuracy**: Identified accounts with an **89.5% actual churn rate**.
* **Top Churn Drivers**: `Age` (48.0%) and `NumOfProducts` (15.9%).

---

## 🛠️ Stack & Libraries
* **Language**: Python
* **Database**: SQLite
* **Data Manipulation**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn
* **Visualization**: Matplotlib

---

## 🚀 Quickstart
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/customer-churn-prediction-pipeline.git](https://github.com/YOUR_USERNAME/customer-churn-prediction-pipeline.git)
