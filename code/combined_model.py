from typing import Any, List
from model import Model
from language_model import NgramLanguageModel


class LMRescoredModel(Model):
    """Wraps any N-best-capable Model with an NgramLanguageModel rescorer."""

    def __init__(self, base_model: Model, lm: NgramLanguageModel, n_best: int = 5) -> None:
        super().__init__(model_name=f"{base_model.name()}_lm")
        self.base_model = base_model
        self.lm = lm
        self.n_best = n_best

    def format_inputs(self, inputs: Any) -> Any:
        return self.base_model.format_inputs(inputs)

    def predict(self, inputs: Any) -> List[str]:
        hypotheses, acoustic_scores = self.base_model.predict_nbest(inputs, n_best=self.n_best)
        return self.lm.rescore_batch(hypotheses, acoustic_scores)
