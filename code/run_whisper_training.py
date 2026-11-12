"""run training for whisper config"""
import os
import argparse as argp
from whisper_model import WhisperModel
import pandas as pd
import base_info as info

parser = argp.ArgumentParser(description="Train base whipser with new data")
parser.add_argument("--model", type=str, required=True, help="Base model version")
parser.add_argument("--config", type=str, required=True, help="Config to use")
args = parser.parse_args()

model: WhisperModel = WhisperModel(model_name=str(args.model))

train_data: pd.DataFrame = pd.read_csv(os.path.join(info.DATASET_PATH, "geo", "train.csv"))
train_data["file"] = train_data["file"].astype(str)

val_data: pd.DataFrame = pd.read_csv(os.path.join(info.DATASET_PATH, "geo", "dev.csv"))
val_data["file"] = val_data["file"].astype(str)

print("Setup finished. Start training...")
model.train(train_data=train_data, val_data=val_data, config_name=args.config)
print("Training finished.")
