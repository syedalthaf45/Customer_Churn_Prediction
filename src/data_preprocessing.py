import pandas as pd
import numpy as np
import pickle
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config, ensure_dirs


class DataPreprocessing:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("DataPreprocessing", config)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        ensure_dirs("artifacts")

    def drop_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = ["customerID"]
        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)
        self.logger.info(f"Dropped columns: {existing}")
        return df

    def fix_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        # TotalCharges has some spaces — convert to numeric
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        before = int(df["TotalCharges"].isnull().sum())
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
        self.logger.info(f"Fixed TotalCharges: {before} null(s) replaced with median ({median_val:.2f}).")
        return df

    def drop_remaining_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped > 0:
            self.logger.warning(f"Dropped {dropped} rows with remaining null values.")
        else:
            self.logger.info("[PASS] No remaining null values after preprocessing.")
        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        target = self.config["target_column"]
        df[target] = df[target].map({"Yes": 1, "No": 0})
        self.logger.info(f"Target column '{target}' encoded: Yes=1, No=0.")
        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        target = self.config["target_column"]
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if target in cat_cols:
            cat_cols.remove(target)

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        self.logger.info(f"Label encoded {len(cat_cols)} categorical columns: {cat_cols}")
        return df

    def scale_numerics(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        target = self.config["target_column"]
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)

        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            self.logger.info(f"Fitted and scaled {len(numeric_cols)} numeric columns.")
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            self.logger.info(f"Scaled {len(numeric_cols)} numeric columns using existing scaler.")
        return df

    def save_artifacts(self):
        scaler_path = self.config["model"]["scaler_path"]
        le_path = self.config["model"]["label_encoder_path"]
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(le_path, "wb") as f:
            pickle.dump(self.label_encoders, f)
        self.logger.info(f"Saved scaler to {scaler_path}")
        self.logger.info(f"Saved label encoders to {le_path}")

    def split_data(self, df: pd.DataFrame):
        target = self.config["target_column"]
        test_size = self.config["model"]["test_size"]
        random_state = self.config["model"]["random_state"]

        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        self.logger.info(
            f"Train/Test split — Train: {X_train.shape[0]}, Test: {X_test.shape[0]} | stratified"
        )
        return X_train, X_test, y_train, y_test

    def run(self, df: pd.DataFrame):
        self.logger.info("=== Data Preprocessing Started ===")
        df = self.drop_irrelevant_columns(df)
        df = self.fix_total_charges(df)
        df = self.drop_remaining_nulls(df)
        df = self.encode_target(df)
        df = self.encode_categoricals(df)
        df = self.scale_numerics(df, fit=True)

        processed_path = self.config["data"]["processed_path"]
        df.to_csv(processed_path, index=False)
        self.logger.info(f"Processed data saved to {processed_path}")

        self.save_artifacts()
        X_train, X_test, y_train, y_test = self.split_data(df)
        self.logger.info("=== Preprocessing Complete ===")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    from src.data_validation import DataValidation
    config = load_config()
    df = DataIngestion(config).run()
    DataValidation(config).run(df)
    X_train, X_test, y_train, y_test = DataPreprocessing(config).run(df)
    print("X_train shape:", X_train.shape)
