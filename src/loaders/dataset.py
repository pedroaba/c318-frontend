import pandas as pd

from pathlib import Path


class DatasetLoader:
    @staticmethod
    def load(dataset_path: Path):
        return pd.read_csv(dataset_path)
