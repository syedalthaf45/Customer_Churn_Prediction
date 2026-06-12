# Customer Churn Prediction — End-to-End ML System

A production-grade machine learning system that predicts customer churn using the IBM Telco dataset.
Built with a fully modular pipeline (ingestion → validation → preprocessing → training → evaluation)
and deployed as a Flask web application on Render.

---

## Live Demo
> Deployed on Render: `https://your-app.onrender.com`

---

## Project Structure

```
churn_prediction/
├── src/
│   ├── data_ingestion.py      # Downloads & loads dataset
│   ├── data_validation.py     # Schema, null, type & duplicate checks
│   ├── data_preprocessing.py  # Encoding, scaling, train/test split
│   ├── model_trainer.py       # Multi-model GridSearchCV training
│   ├── model_evaluator.py     # Metrics, confusion matrix, feature importance
│   └── utils.py               # Logging, config helpers
├── templates/
│   ├── index.html             # Prediction form UI
│   ├── result.html            # Prediction result page
│   └── metrics.html           # Model metrics dashboard
├── data/                      # Raw & processed CSV files
├── artifacts/                 # Saved model, scaler, encoders, report
├── logs/                      # Pipeline logs
├── app.py                     # Flask application
├── run_pipeline.py            # Full pipeline orchestrator
├── config.yaml                # Central configuration
├── Dockerfile                 # Production Docker container
└── requirements.txt
```

---

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `data_ingestion.py` | Downloads IBM Telco CSV (7,043 rows) from GitHub |
| 2 | `data_validation.py` | Validates schema, nulls, types, duplicates |
| 3 | `data_preprocessing.py` | Encodes 15 categoricals, scales 19 numerics, stratified split |
| 4 | `model_trainer.py` | Compares RF, GradientBoosting, LR via GridSearchCV (5-fold CV) |
| 5 | `model_evaluator.py` | Accuracy, F1, ROC-AUC, confusion matrix, feature importance |

---

## Model Results

| Model | CV ROC-AUC |
|-------|-----------|
| **GradientBoosting ★** | **0.849** |
| Random Forest | 0.846 |
| Logistic Regression | 0.845 |

**Test Set Performance (GradientBoosting):**
- Accuracy: **80.48%**
- ROC-AUC: **84.57%**
- F1 Score: **0.584**







## Tech Stack
- **Python 3.11**
- **Scikit-learn** — ML models, GridSearchCV
- **Pandas / NumPy** — Data processing
- **Flask + Gunicorn** — Web framework & production server
- **PyYAML** — Configuration management
- **Docker** — Containerization
- **Render** — Cloud deployment

---

## Author
** Syed Althaf** 
