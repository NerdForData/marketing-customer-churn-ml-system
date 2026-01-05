Perfect choice. A **strong README** is what turns this into a **recruiter-ready, interview-ready flagship project**.

Below is a **complete, professional `README.md`** you can copy–paste directly.
It is written in a **balanced analytics + ML tone**, exactly aligned with how hiring managers read GitHub projects.

---

# 📊 Marketing Customer Churn Analytics & ML System

## 📌 Overview

Customer churn is one of the most critical challenges faced by subscription-based businesses. Acquiring new customers is significantly more expensive than retaining existing ones, making **early identification of high-risk customers** a key marketing priority.

This project builds an **end-to-end marketing analytics and machine learning system** to:

* Understand *why* customers churn
* Predict *which* customers are likely to churn
* Enable *actionable, capacity-aware retention campaigns*

The solution combines **marketing-driven data analysis**, **machine learning**, and **production-ready deployment**.

---

## 🎯 Business Problem

Marketing teams typically ask:

* What percentage of customers are churning?
* Which customer segments are most at risk?
* If we can only target a limited number of customers, *who should we prioritize?*

This project answers those questions by:

* Performing **marketing-focused exploratory analysis**
* Training a churn prediction model
* Translating model outputs into **campaign decisions**, not just probabilities

---

## 📂 Dataset

* **Source:** IBM Telco Customer Churn Dataset (Kaggle)
* **Size:** 7,043 customers
* **Target Variable:** `Churn` (Yes / No)
* **Features:** Customer demographics, services, billing, contract details

The dataset represents a realistic telecom / subscription business scenario and is widely used in industry churn modeling.

---

## 🔍 Phase 1 — Marketing Analytics (EDA)

The exploratory analysis focuses on **business-relevant insights**, not generic plots.

### Key Findings

* **Overall churn rate:** ~26.5%
  → Roughly 1 in 4 customers churn, indicating a significant retention opportunity.

* **Tenure effect:**

  * Churned customers leave much earlier in their lifecycle.
  * Churn is primarily an **early-stage customer problem**.

* **Contract type impact:**

  * Month-to-month customers churn ~15× more than two-year contract customers.
  * Contract length is the **strongest churn driver**.

* **Payment behavior:**

  * Certain payment methods (e.g. electronic check) show higher churn risk.

These insights directly inform feature engineering and campaign strategy.

---

## 🧠 Phase 2 — Feature Engineering

Production-grade preprocessing was implemented in `src/data_prep.py`:

* Target encoding (`Churn → 0/1`)
* Safe handling of missing values
* Separation of numeric and categorical features
* One-hot encoding with `handle_unknown="ignore"`
* Reusable preprocessing pipeline (shared across training, evaluation, and API)

This design ensures **consistency and reproducibility** across the system.

---

## 🤖 Phase 3 — Machine Learning Models

Two models were trained and compared:

| Model                          | ROC-AUC   | PR-AUC    |
| ------------------------------ | --------- | --------- |
| Logistic Regression (baseline) | 0.841     | 0.633     |
| Random Forest (final)          | **0.843** | **0.648** |

### Model Selection Rationale

* Logistic Regression provides interpretability and a strong baseline.
* Random Forest improves **precision–recall performance**, which is more relevant for churn targeting under class imbalance.
* Random Forest was selected as the **final production model**.

---

## 🎯 Phase 4 — Campaign-Oriented Thresholding

Instead of using a fixed probability threshold (e.g. 0.5), the model was aligned with **marketing capacity constraints**.

### Campaign Simulation

* **Target capacity:** Top 20% highest-risk customers
* **Derived threshold:** 0.6977

### Results

* **Precision:** ~66%
  → 2 out of 3 contacted customers are actual churners
* **Recall:** ~50%
  → Captures half of all churners with limited budget
* **Customers targeted:** 282
* **Actual churners captured:** 186

This demonstrates **real operational value**, not just model accuracy.

---

## 🚀 Phase 5 — Production API (FastAPI)

A production-ready API exposes the final model:

### Endpoints

* `GET /` — Health check
* `POST /predict` — Score a single customer
* `POST /predict-batch` — Score customers for marketing campaigns

### API Output

Each prediction returns:

* Churn probability
* Binary churn flag
* Campaign recommendation (target / no action)

This mirrors how churn models are consumed by CRM or marketing automation systems.

---

## 🗂️ Project Structure

```
marketing-customer-churn-ml-system/
├── data/
│   └── raw/
├── models/
│   ├── final_churn_pipeline.joblib
│   └── threshold.json
├── notebooks/
│   └── 01_marketing_eda.ipynb
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate.py
│   ├── finalize_model.py
│   └── app.py
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### 1️⃣ Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train models

```bash
# Train Logistic Regression (baseline)
python src/train.py --data-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# Train Random Forest (final model)
python src/train.py --data-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --model-type rf
```

**Note:** Make sure your virtual environment is activated before running these commands.

### 4️⃣ Evaluate model performance

```bash
# Evaluate with default settings (top 20% targeting)
python src/evaluate.py

# Or with custom model and targeting
python src/evaluate.py --model-path models/churn_pipeline_rf.joblib --target-fraction 0.15
```

### 5️⃣ Finalize model & threshold

```bash
python src/finalize_model.py --data-path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 6️⃣ Start API

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

Open your browser:

```
http://127.0.0.1:8000/docs
```

### 7️⃣ Test the API

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 846.0
  }'
```

---

## 🧩 Key Skills Demonstrated

* Marketing analytics & KPI interpretation
* Feature engineering for tabular data
* ML model comparison & selection
* Business-driven thresholding
* Production ML pipelines
* API deployment with FastAPI

---

## 👤 Author

**Aishwarya**
* M.Sc. Software Engineering & Management
* Focus: Data Science, ML Engineering, Analytics Engineering



