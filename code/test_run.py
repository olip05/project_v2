"""tester"""
from model import Model
from evaluator import Evaluator
from whisper_model import WhisperModel
import pandas as pd

data_frame = pd.read_csv("dataset/geo/dev.csv")

# m1: Model = Model("test")
# m2: Model = Model("test2")
m3: Model = WhisperModel("openai/whisper-large")

e: Evaluator = Evaluator(data_frame)

e.compare_models([m3], True)
# e.multiple_runs([m1, m2])
