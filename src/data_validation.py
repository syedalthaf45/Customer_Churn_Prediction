import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config


class DataValidation:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("DataValidation", config)
        self.validation_cfg = config["validation"]
        self.errors = []
        self.warnings = []

    def check_required_columns(self, df: pd.DataFrame):
        required = self.validation_cfg["required_columns"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.errors.append(f"Missing required columns: {missing}")
            self.logger.error(f"Missing columns: {missing}")
        else:
            self.logger.info(f"[PASS] All {len(required)} required columns present.")

    def check_null_percentage(self, df: pd.DataFrame):
        max_null = self.validation_cfg["max_null_percent"]
        null_pct = (df.isnull().sum() / len(df)) * 100
        high_null_cols = null_pct[null_pct > max_null]
        if not high_null_cols.empty:
            self.warnings.append(f"High null % columns: {high_null_cols.to_dict()}")
            self.logger.warning(f"Columns with >{ max_null}% nulls: {high_null_cols.to_dict()}")
        else:
            self.logger.info(f"[PASS] No column exceeds {max_null}% null threshold.")

    def check_target_column(self, df: pd.DataFrame):
        target = self.config["target_column"]
        if target not in df.columns:
            self.errors.append(f"Target column '{target}' not found.")
            self.logger.error(f"Target column '{target}' missing.")
        else:
            dist = df[target].value_counts(normalize=True).round(3).to_dict()
            self.logger.info(f"[PASS] Target column '{target}' found. Distribution: {dist}")
            if min(dist.values()) < 0.10:
                self.warnings.append(f"Severe class imbalance detected in '{target}': {dist}")
                self.logger.warning(f"Class imbalance detected: {dist}")

    def check_data_types(self, df: pd.DataFrame):
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        for col in numeric_cols:
            if col in df.columns:
                try:
                    pd.to_numeric(df[col], errors="raise")
                    self.logger.info(f"[PASS] Column '{col}' is numeric.")
                except Exception:
                    self.warnings.append(f"Column '{col}' has non-numeric values.")
                    self.logger.warning(f"Column '{col}' contains non-numeric values.")

    def check_duplicates(self, df: pd.DataFrame):
        dupes = df.duplicated().sum()
        if dupes > 0:
            self.warnings.append(f"{dupes} duplicate rows found.")
            self.logger.warning(f"{dupes} duplicate rows detected.")
        else:
            self.logger.info("[PASS] No duplicate rows found.")

    def run(self, df: pd.DataFrame) -> bool:
        self.logger.info("=== Data Validation Started ===")
        self.check_required_columns(df)
        self.check_null_percentage(df)
        self.check_target_column(df)
        self.check_data_types(df)
        self.check_duplicates(df)

        self.logger.info(f"Validation Summary — Errors: {len(self.errors)}, Warnings: {len(self.warnings)}")
        if self.errors:
            for e in self.errors:
                self.logger.error(f"  ERROR: {e}")
            raise ValueError(f"Data validation failed with {len(self.errors)} error(s).")

        if self.warnings:
            for w in self.warnings:
                self.logger.warning(f"  WARNING: {w}")

        self.logger.info("=== Data Validation Passed ===")
        return True


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    config = load_config()
    df = DataIngestion(config).run()
    DataValidation(config).run(df)
