import pickle
import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config, ensure_dirs


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("ModelTrainer", config)
        self.best_model = None
        ensure_dirs("artifacts")

    def get_candidates(self):
        """Return multiple candidate models for comparison."""
        return {
            "RandomForest": {
                "model": RandomForestClassifier(random_state=self.config["model"]["random_state"]),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                }
            },
            "GradientBoosting": {
                "model": HistGradientBoostingClassifier(random_state=self.config["model"]["random_state"]),
                "params": {
                    "max_iter": [100, 150],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                }
            },
            "LogisticRegression": {
                "model": LogisticRegression(max_iter=1000, random_state=self.config["model"]["random_state"]),
                "params": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs", "liblinear"],
                }
            }
        }

    def train_with_grid_search(self, model, params, X_train, y_train, model_name: str):
        cv = StratifiedKFold(
            n_splits=self.config["model"]["cv_folds"],
            shuffle=True,
            random_state=self.config["model"]["random_state"]
        )
        self.logger.info(f"Running GridSearchCV for {model_name} with {cv.n_splits}-fold CV...")
        gs = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        gs.fit(X_train, y_train)
        self.logger.info(f"  {model_name} best params: {gs.best_params_}")
        self.logger.info(f"  {model_name} best CV ROC-AUC: {gs.best_score_:.4f}")
        return gs.best_estimator_, gs.best_score_

    def select_best_model(self, X_train, y_train) -> tuple:
        candidates = self.get_candidates()
        results = {}

        for name, cfg in candidates.items():
            best_est, best_score = self.train_with_grid_search(
                cfg["model"], cfg["params"], X_train, y_train, name
            )
            results[name] = {"estimator": best_est, "cv_roc_auc": best_score}

        # Pick best by CV ROC-AUC
        best_name = max(results, key=lambda k: results[k]["cv_roc_auc"])
        best_model = results[best_name]["estimator"]
        best_score = results[best_name]["cv_roc_auc"]

        self.logger.info(f"\n>> Best model selected: {best_name} (CV ROC-AUC = {best_score:.4f})")

        # Log feature importances if available
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            self.logger.info(f"Top feature importances computed for {best_name}.")

        return best_model, best_name, results

    def save_model(self, model):
        model_path = self.config["model"]["model_path"]
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        self.logger.info(f"Model saved to {model_path}")

    def run(self, X_train, y_train):
        self.logger.info("=== Model Training Started ===")
        self.logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

        best_model, best_name, all_results = self.select_best_model(X_train, y_train)
        self.best_model = best_model
        self.save_model(best_model)

        self.logger.info("=== Model Training Complete ===")
        return best_model, best_name, all_results


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    from src.data_validation import DataValidation
    from src.data_preprocessing import DataPreprocessing
    config = load_config()
    df = DataIngestion(config).run()
    DataValidation(config).run(df)
    X_train, X_test, y_train, y_test = DataPreprocessing(config).run(df)
    trainer = ModelTrainer(config)
    model, name, results = trainer.run(X_train, y_train)
    print(f"Best model: {name}")
