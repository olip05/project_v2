"""Whisper model implementation"""
from typing import List, Any, Dict, Optional
from dataclasses import dataclass
import os
import json
import torch
import pandas as pd
import numpy as np
import librosa
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import Dataset # type: ignore
from model import Model
import base_info as info

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Padding speech sequences and labels."""
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features: List[Dict[str, torch.Tensor]] = [{"input_features": feature["input_features"][0]} for feature in features]
        batch: Dict[str, torch.Tensor] = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features: List[Dict[str, List[int]]] = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch: Dict[str, torch.Tensor] = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels: torch.Tensor = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch

class WhisperModel(Model):
    """Whisper model interface"""

    def __init__(self, model_name: str, language: Optional[str] = None) -> None:
        super().__init__(model_name=model_name.replace("/", "_"))
        self.is_trained: bool = False
        self.language: Optional[str] = language
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe") # type: ignore
        self.model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device) # type: ignore

        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe") # type: ignore

    def format_inputs(self, inputs: List[Any]) -> torch.Tensor:
        audio_features: torch.Tensor = self.processor(
            inputs,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length", 
            max_length=480000,
            truncation=True
        ).input_features

        return audio_features.to(self.device)

    def load(self, weight_path: str) -> None:
        self.model = WhisperForConditionalGeneration.from_pretrained(os.path.join(info.MODEL_PATH, weight_path)).to(self.device) # type: ignore
        self.is_trained = True

    def predict(self, inputs: Any) -> List[str]:
        """Run prediction in small batches to prevent CUDA Out of Memory errors."""
        self.model.eval()
        transcriptions: List[str] = []
        batch_size = 4
        
        total_samples = inputs.shape[0]
        
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                batch_inputs = inputs[i : i + batch_size]
                predicted_ids: torch.Tensor = self.model.generate(batch_inputs)  # type: ignore
                batch_texts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True) # type: ignore
                transcriptions.extend(batch_texts)
                
        return transcriptions

    def train(self, config_name: str, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Train or fine-tune the whisper model using a configuration file."""
        def prepare_dataset(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            audio_arrays: List[np.ndarray] = []
            for path in batch["file"]:
                wav, _ = librosa.load(os.path.join(info.DATASET_PATH, path), sr=info.SR)
                audio_arrays.append(wav)

            batch["input_features"] = self.processor(
                audio_arrays, sampling_rate=16000
            ).input_features
            batch["labels"] = self.processor.tokenizer(batch["transcript"]).input_ids
            return batch

        config_file_path: str = os.path.join(info.CONFIG_PATH, config_name)
        with open(config_file_path, "r", encoding="utf8") as f:
            config: Dict[str, Any] = json.load(f)

        train_ds: Dataset = Dataset.from_pandas(train_data)
        train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, batched=True) # type: ignore

        val_ds: Any = Dataset.from_pandas(val_data)
        val_ds = val_ds.map(prepare_dataset, remove_columns=val_ds.column_names, batched=True) # type: ignore

        output_directory: str = os.path.join(info.MODEL_PATH, self.name(), config["name"])
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_directory,
            per_device_train_batch_size=int(config["batch_size"]),
            gradient_accumulation_steps=2,
            learning_rate=float(config["lr"]),
            warmup_steps=50,
            num_train_epochs=int(config["epochs"]),
            eval_strategy="epoch" if val_ds else "no",
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            predict_with_generate=True,
            logging_steps=25,
            report_to=["none"]
        )

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            processing_class=self.processor,
        )

        trainer.train() # type: ignore
        self.is_trained = True

        save_path: str = os.path.join(info.MODEL_PATH, self.name() + config["name"])
        self.model.save_pretrained(save_path) # type: ignore
        self.processor.save_pretrained(save_path) # type: ignore
