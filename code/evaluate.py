"""run evaluation"""
import os
from typing import List
import argparse as argp
from model import Model
from whisper_model import WhisperModel
from vosk_model import VoskModel
from hmm_dnn_model import HMMDNNModel
from evaluator import Evaluator
import pandas as pd
import base_info as info

parser = argp.ArgumentParser(description="Evaluate ASR models and optionally run statistical inference.")
parser.add_argument("--base_model", nargs="*", default=[], help="List of HMM-DNN model names")
parser.add_argument("--vosk_model", nargs="*", default=[], help="List of Vosk model names")
parser.add_argument("--whisper_model", nargs="*", default=[], help="List of Whisper names")
# FIXME Noor if you adapt the wav2vec model to th emodel interface you can simply include a new option here to give it as a parameter
parser.add_argument("--avg", default=-1, type=int, help="Number of runs for statistical aggregation")
parser.add_argument("--save", action="store_true", help="Save evaluation results to file")
args = parser.parse_args()

data_frame=pd.read_csv(os.path.join(info.DATASET_PATH, "geo", "dev.csv"))

models: List[Model] = []

for name in args.base_model:
    models.append(HMMDNNModel(name))
for name in args.vosk_model:
    models.append(VoskModel(name))
for name in args.whisper_model:
    models.append(WhisperModel(name))
# FIXME Noor same here 

if len(models) == 0:
    raise ValueError("No model names given. Please specifiy at least one.")

e: Evaluator = Evaluator(data_frame)

if args.avg > 0:
    e.multiple_runs(models, args.avg)
else:
    e.compare_models(models, args.save)
