import joblib

from os import PathLike
from pathlib import Path


class ModelLoader:
    def __init__(self, model_path: PathLike | str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._path = path
        self._model_cache = None

    def load(self):
        self._model_cache = joblib.load(self._path.resolve())
        return self._model_cache
