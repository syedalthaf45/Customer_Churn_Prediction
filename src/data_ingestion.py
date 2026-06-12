import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger, load_config, ensure_dirs


class DataIngestion:
    def __init__(self, config: dict):
        self.config = config
        self.logger = get_logger("DataIngestion", config)
        ensure_dirs("data", "artifacts", "logs")

    def download_data(self) -> pd.DataFrame:
        url = self.config["data"]["url"]
        raw_path = self.config["data"]["raw_path"]

        self.logger.info(f"Downloading dataset from: {url}")
        try:
            df = pd.read_csv(url)
            df.to_csv(raw_path, index=False)
            self.logger.info(f"Data saved to {raw_path} | Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Data download failed: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        raw_path = self.config["data"]["raw_path"]
        if os.path.exists(raw_path):
            self.logger.info(f"Loading existing data from {raw_path}")
            return pd.read_csv(raw_path)
        else:
            self.logger.info("No local data found. Downloading...")
            return self.download_data()

    def run(self) -> pd.DataFrame:
        self.logger.info("=== Data Ingestion Started ===")
        df = self.load_data()
        self.logger.info(f"Ingestion complete. Rows: {df.shape[0]}, Cols: {df.shape[1]}")
        self.logger.info(f"Columns: {list(df.columns)}")
        return df


if __name__ == "__main__":
    config = load_config()
    ingestion = DataIngestion(config)
    df = ingestion.run()
    print(df.head())
