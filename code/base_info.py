"""base config info"""

import os

DATASET_PATH = os.path.join(os.curdir, "dataset")
SR = 24000

MODEL_PATH = os.path.join(os.curdir, "models")
CONFIG_PATH = os.path.join(os.curdir, "train_configs")

EVALUATION_SAMPLES = 200
EVALUATION_PATH = os.path.join(os.curdir, "evaluation_results")

RESULT_PATH = os.path.join(os.curdir, "test_results")
