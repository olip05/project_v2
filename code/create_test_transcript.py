"""run test evaluation"""
import os
from typing import List, Any, Optional
import argparse as argp
from model import Model
from whisper_model import WhisperModel
from vosk_model import VoskModel
from hmm_dnn_model import HMMDNNModel
import pandas as pd
import base_info as info
import librosa
import datetime

BATCH_SIZE = 20

parser = argp.ArgumentParser(description="Create transcripts for test dataset.")
parser.add_argument("--base_model", nargs=1, required=False, help="List of HMM-DNN model names")
parser.add_argument("--vosk_model", type=str, required=False, help="List of Vosk model names")
parser.add_argument("--whisper_model", nargs=1, required=False, help="List of Whisper names")
args = parser.parse_args()

selected_models = [args.base_model, args.vosk_model, args.whisper_model]
if sum(x is not None for x in selected_models) != 1:
    raise ValueError("Only provide one model for testing.")

data_frame=pd.read_csv(os.path.join(info.DATASET_PATH, "geo", "dev.csv"))

model: Optional[Model] = None

if args.base_model:
    model = HMMDNNModel(args.base_model)
if args.vosk_model:
    model = VoskModel(args.vosk_model)
if args.whisper_model:
    model = WhisperModel(args.whisper_model)

if model is None:
    raise ValueError("No model names given.")

test_data: pd.DataFrame = pd.read_csv(os.path.join(info.DATASET_PATH, "geo", "test.csv"))
test_data["file"] = test_data["file"].astype(str)

inputs: List[Any] = []
for entry in test_data["file"].to_list():
    wav, _ = librosa.load(os.path.join(info.DATASET_PATH, entry), sr=info.SR)
    inputs.append(wav)

predictions: List[str] = []
for batch in range(len(inputs)//BATCH_SIZE):
    predictions += model.predict(model.format_inputs(inputs[batch*BATCH_SIZE:min(len(inputs), (batch+1)*BATCH_SIZE)]))
    print(f"{batch*BATCH_SIZE}/{len(inputs)} ({batch*BATCH_SIZE/len(inputs)*100:>.2f}%)")

test_data["transcript"] = predictions

out_path: str = os.path.join(info.RESULT_PATH, f"{str(datetime.datetime.now()).replace(" ", "_")}-{model.name()}.csv")
test_data.to_csv(out_path)
print("Results saved to ", out_path)
