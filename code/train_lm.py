import pandas as pd
from language_model import NgramLanguageModel

train_df = pd.read_csv("dataset/geo/train.csv")
train_transcripts = train_df["transcript"].dropna().tolist()

# 1. Word-level 3-gram LM
word_lm = NgramLanguageModel.train_from_transcripts(
    transcripts=train_transcripts,
    output_dir="models/lm",
    model_name="eo_geo",
    order=3,
    char_level=False,
    alpha=0.15,
)

# 2. Optional character-level 4-gram LM (often safer on altered/non-standard words)
char_lm = NgramLanguageModel.train_from_transcripts(
    transcripts=train_transcripts,
    output_dir="models/lm",
    model_name="eo_geo",
    order=4,
    char_level=True,
    alpha=0.10,
)