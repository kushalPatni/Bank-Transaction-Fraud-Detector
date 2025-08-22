# Bank Transaction Fraud Detector

**Project overview**

A Streamlit-based application that detects suspicious or fraudulent bank transactions using a combination of supervised and unsupervised machine learning models (Logistic Regression, Random Forest, and Isolation Forest). The app includes data ingestion, preprocessing, model training, evaluation, single-transaction prediction, and an interactive dashboard for analysis and monitoring.

---

## Problem statement

Banks process millions of transactions daily. Detecting fraudulent transactions in (near) real-time helps prevent financial loss for customers and the institution. This project builds and compares models that automatically flag suspicious transactions for review.

---

##  Dataset

A sample dataset (`AIML Dataset.csv`) is used for training and testing. Expected columns (rename to match your data if needed):

* `step` — time step of the transaction
* `type` — transaction type (CASH\_OUT, PAYMENT, CASH\_IN, TRANSFER, DEBIT)
* `amount` — transaction amount
* `nameOrig` — origin account identifier (dropped for modeling)
* `oldbalanceOrg` — origin account balance before transaction
* `newbalanceOrig` — origin account balance after transaction
* `nameDest` — destination account identifier (dropped for modeling)
* `oldbalanceDest` — destination account balance before transaction
* `newbalanceDest` — destination account balance after transaction
* `isFraud` — target label (0 = normal, 1 = fraud)
* `isFlaggedFraud` — optional flagged indicator

> The repository also contains code to generate sample data if a CSV is not provided.

---

## Features

* Data ingestion with optional chunked reading and stratified sampling for large files
* Memory-efficient preprocessing and feature engineering (balance changes, hour, round-amount indicator)
* Multiple models: Logistic Regression, Random Forest, Isolation Forest
* Class imbalance handling (class weights, sampling options)
* Model comparison with precision, recall, F1-score, ROC-AUC
* Single-transaction prediction UI and a real-time transaction simulator
* Save/load trained models and preprocessing objects (scaler, encoders)
* Interactive visualizations (transaction distribution, fraud rate by hour/type, ROC/Confusion matrix)

---

## Requirements

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## How to run

### 1) Run full Streamlit app (train + dashboard)

```bash
streamlit run streamlit_app.py
```

This will open a browser UI where you can:

* Load a CSV dataset (or use generated sample data)
* Train models (Logistic Regression, Isolation Forest, Random Forest)
* View model comparison metrics and visualizations
* Predict a single transaction or generate random transactions for real-time testing
* Save trained models (`.pkl` files)

### 2) Run the model-picker / predictor UI (load saved models)

```bash
streamlit run streamlit_predict.py
```

This app loads saved models and preprocessing objects, lets users pick a model, and predict on manual or random inputs.

---

## Save / Load Models

The app uses `joblib` to save/load models and preprocessing objects.

* Saved files:

  * `logistic_regression_model.pkl`
  * `random_forest_model.pkl`
  * `isolation_forest_model.pkl`
  * `scaler.pkl` (StandardScaler)
  * `feature_columns.pkl` (saved list/Index of feature column names)

Load a model in code:

```py
import joblib
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
```

---

##  Tips to improve model performance

* **Feature engineering:** add aggregate user features (transaction counts, avg amount per account, time since last txn).
* **Temporal validation:** split data by time to avoid leakage and simulate real-world performance.
* **Imbalance handling:** try SMOTE, SMOTETomek, or focal loss for better minority class learning.
* **Model tuning:** grid/randomized search or Bayesian optimization (optuna) for hyperparameter tuning.
* **Use gradient-boosted trees:** LightGBM or XGBoost often improve tabular classification.
* **Threshold tuning:** choose a probability threshold based on business trade-offs (recall vs precision).

---


## Notes

* The provided app is for **educational and demonstration purposes**. For production use, include secure data handling, logging, monitoring, model drift detection, and stricter validation.
* Ensure you have the right to use any real transaction data and follow data privacy regulations.

---

## Contributing

Contributions, improvements, and bug reports are welcome. Create an issue or open a pull request with a clear description and tests where appropriate.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

---

*Generated README — feel free to edit for tone, company branding, or to add screenshots and badges.*
