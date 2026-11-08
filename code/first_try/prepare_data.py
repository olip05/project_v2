# mostly taken from https://medium.com/@ozdadirri/a-guide-to-fine-tune-whisper-model-with-hyper-parameter-tuning-c13645ba2dba
# use ssh kolibri when in kosh to connect to PC with GPU
from datasets import Dataset, DatasetDict
import pandas as pd
import librosa
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(MODEL_ID, task="transcribe")

def process_audio_batch(batch):
    input_features_list = []
    labels_list = []
    
    for file_path, transcript in zip(batch["file"], batch["transcript"]):
        wav, _ = librosa.load(f"dataset/{file_path}", sr=SR)
        audio_inputs = processor.feature_extractor(wav, sampling_rate=SR, return_tensors="np")
        input_features_list.append(audio_inputs.input_features[0])

        labels = processor.tokenizer(text_target=(transcript if len(transcript) > 0 else ""), return_tensors="np")
        labels_list.append(labels.input_ids[0])
    return {
        "input_features": input_features_list,
        "labels": labels_list
    }

def load(name):
    print(f"Loading {name}.csv...")
    df = pd.read_csv(f'dataset/geo/{name}.csv')
    df["transcript"] = df["transcript"].fillna("")
    return Dataset.from_pandas(df)

if __name__ == "__main__":
    dataset = DatasetDict({
        'train': load("train"),
        'test': load("test"),
        'dev': load("dev")
    })
    print("Loaded all partitions of dataset")

    dataset = dataset.map(
        process_audio_batch, 
        batched=True, 
        batch_size=100, 
        writer_batch_size=100,
        remove_columns=["file", "transcript"]
    )
    print("Speech data and labels successfully converted to features and tokens")

    print(dataset)

    dataset.save_to_disk(DATASET_PATH)
    print("Saved dataset to disk")