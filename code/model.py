"""model interface"""
from typing import List, Any
import pandas as pd

class Model:
    """Encapsulate models for a common interface."""

    def __init__(self, model_name: str) -> None:
        self._name: str = model_name
        self.is_trained: bool = False

    def name(self) -> str:
        """returns the name"""
        return self._name

    def format_inputs(self, inputs: List[Any]) -> Any:
        """format wav inputs accordingly"""
        raise NotImplementedError("Format method must be implemented by subclasses.")

    def load(self, weight_path: str) -> None:
        """load exisitng weights"""
        raise NotImplementedError("Load method must be implemented by subclasses.")

    def train(self, config_name: str, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """train with specific input"""
        raise NotImplementedError("Train method must be implemented by subclasses.")

    def predict(self, inputs: List[Any]) -> List[str]:
        """give model ouptuts"""
        raise NotImplementedError("Predict method must be implemented by subclasses.")
