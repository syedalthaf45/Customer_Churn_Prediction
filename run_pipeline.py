"""
run_pipeline.py — Orchestrates the full ML pipeline end-to-end.
Run this once to train, validate, and save the model.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, get_logger
from src.data_ingestion import DataIngestion
from src.data_validation import DataValidation
from src.data_preprocessing import DataPreprocessing
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator


def run_pipeline():
    config = load_config("config.yaml")
    logger = get_logger("Pipeline", config)

    logger.info("=" * 60)
    logger.info("   CUSTOMER CHURN PREDICTION — ML PIPELINE STARTED")
    logger.info("=" * 60)

    # Stage 1: Data Ingestion
    logger.info("\n[STAGE 1] Data Ingestion")
    ingestion = DataIngestion(config)
    df = ingestion.run()

    # Stage 2: Data Validation
    logger.info("\n[STAGE 2] Data Validation")
    validator = DataValidation(config)
    validator.run(df)

    # Stage 3: Preprocessing
    logger.info("\n[STAGE 3] Data Preprocessing")
    preprocessor = DataPreprocessing(config)
    X_train, X_test, y_train, y_test = preprocessor.run(df)

    # Stage 4: Model Training
    logger.info("\n[STAGE 4] Model Training")
    trainer = ModelTrainer(config)
    model, model_name, all_results = trainer.run(X_train, y_train)

    # Stage 5: Model Evaluation
    logger.info("\n[STAGE 5] Model Evaluation")
    evaluator = ModelEvaluator(config)
    report = evaluator.run(model, X_test, y_test, model_name, all_results)

    logger.info("\n" + "=" * 60)
    logger.info("   PIPELINE COMPLETE")
    logger.info(f"   Best Model  : {model_name}")
    logger.info(f"   Accuracy    : {report['test_metrics']['accuracy']}")
    logger.info(f"   F1 Score    : {report['test_metrics']['f1_score']}")
    logger.info(f"   ROC-AUC     : {report['test_metrics']['roc_auc']}")
    logger.info("=" * 60)
    return report


if __name__ == "__main__":
    run_pipeline()
