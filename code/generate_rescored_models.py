import os
import pandas as pd
import librosa

import base_info as info
from whisper_model import WhisperModel  # adjust module name to match your file
from language_model import NgramLanguageModel
from combined_model import LMRescoredModel
from evaluator import Evaluator  # adjust module name to match your Evaluator class


def main() -> None:
    # 1. Load fine-tuned Whisper model
    whisper_model = WhisperModel(model_name="openai_whisper-small_whisper-eo-v6")

    # 2. Load the KenLM model trained strictly on the train split
    word_lm = NgramLanguageModel(
        klm_path="models/lm/eo_geo_word_3g.klm",
        alpha=0.0,
        beta=0.0,
        char_level=False,
    )

    # 3. Load labeled dev split (never tune on test)
    dev_df = pd.read_csv(os.path.join(info.DATASET_PATH, "geo", "dev.csv"))

    # 4. Initialize your standard Evaluator
    evaluator = Evaluator(dev_df)  # pass any required arguments your Evaluator constructor expects

    # 5. Sweep alpha values through the Evaluator
    alphas = [0.0, 0.05, 0.1, 0.2, 0.3]

    for alpha in alphas:
        word_lm.alpha = alpha
        rescored_model = LMRescoredModel(base_model=whisper_model, lm=word_lm, n_best=5)
        rescored_model._name = f"{whisper_model.name()}_lm_alpha_{alpha}"  # optional unique label if your Evaluator logs by model name

        print(f"\n--- Evaluating alpha={alpha} ---")
        evaluator.compare_models([rescored_model])

if __name__ == "__main__":
    main()