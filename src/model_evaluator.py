import json
import pickle
import sys
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config, ensure_dirs


class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("ModelEvaluator", config)
        ensure_dirs("artifacts")

    def compute_metrics(self, y_true, y_pred, y_proba) -> dict:
        metrics = {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
        }
        return metrics

    def get_confusion_matrix(self, y_true, y_pred) -> dict:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }

    def get_feature_importance(self, model, feature_names: list) -> dict:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            fi = dict(sorted(
                zip(feature_names, importances.tolist()),
                key=lambda x: x[1], reverse=True
            ))
            return fi
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0])
            fi = dict(sorted(
                zip(feature_names, coefs.tolist()),
                key=lambda x: x[1], reverse=True
            ))
            return fi
        return {}

    def save_report(self, report: dict):
        path = self.config["model"]["report_path"]
        with open(path, "w") as f:
            json.dump(report, f, indent=4)
        self.logger.info(f"Evaluation report saved to {path}")

    def run(self, model, X_test, y_test, model_name: str = "Model", all_results: dict = None):
        self.logger.info("=== Model Evaluation Started ===")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = self.compute_metrics(y_test, y_pred, y_proba)
        cm = self.get_confusion_matrix(y_test, y_pred)
        feature_importance = self.get_feature_importance(model, list(X_test.columns))
        clf_report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"  MODEL: {model_name}")
        self.logger.info(f"{'='*50}")
        for k, v in metrics.items():
            self.logger.info(f"  {k.upper():<15}: {v}")
        self.logger.info(f"\nConfusion Matrix:\n  {cm}")
        self.logger.info(f"\nClassification Report:\n{clf_report}")

        if feature_importance:
            top5 = list(feature_importance.items())[:5]
            self.logger.info(f"Top 5 Features: {top5}")

        # Model comparison summary
        comparison = {}
        if all_results:
            for name, res in all_results.items():
                comparison[name] = {"cv_roc_auc": round(res["cv_roc_auc"], 4)}

        report = {
            "model_name": model_name,
            "test_metrics": metrics,
            "confusion_matrix": cm,
            "top_features": dict(list(feature_importance.items())[:10]),
            "model_comparison": comparison,
            "classification_report": clf_report
        }

        self.save_report(report)
        self.logger.info("=== Evaluation Complete ===")
        return report


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    from src.data_validation import DataValidation
    from src.data_preprocessing import DataPreprocessing
    from src.model_trainer import ModelTrainer
    config = load_config()
    df = DataIngestion(config).run()
    DataValidation(config).run(df)
    X_train, X_test, y_train, y_test = DataPreprocessing(config).run(df)
    model, name, all_results = ModelTrainer(config).run(X_train, y_train)
    ModelEvaluator(config).run(model, X_test, y_test, name, all_results)
