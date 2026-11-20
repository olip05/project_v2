"""Whisper model implementation optimized for hardware efficiency and parallelization"""
from typing import List, Any, Dict, Optional, Tuple
from dataclasses import dataclass
import os
import json
import torch
import pandas as pd
import numpy as np
import jiwer
import random
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

def _augment_waveform(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Apply lightweight, speech-preserving augmentations to a 1D float32 waveform."""
    aug = np.asarray(wav, dtype=np.float32).copy()

    # 1. Random gain / volume scaling
    if random.random() < 0.7:
        gain_db = random.uniform(-4.0, 4.0)
        aug = aug * (10.0 ** (gain_db / 20.0))

    # 2. Light additive white/Gaussian noise (SNR roughly 25-35 dB)
    if random.random() < 0.5:
        rms = float(np.sqrt(np.mean(aug ** 2))) + 1e-8
        snr_db = random.uniform(25.0, 35.0)
        noise_rms = rms / (10.0 ** (snr_db / 20.0))
        noise = np.random.normal(0.0, noise_rms, size=aug.shape).astype(np.float32)
        aug = aug + noise

    # 3. Subtle time shift (roll up to +-50ms) to vary utterance alignment slightly
    if random.random() < 0.5 and len(aug) > 0:
        max_shift = int(0.05 * sr)
        shift = random.randint(-max_shift, max_shift)
        aug = np.roll(aug, shift)

    # Prevent clipping
    peak = float(np.max(np.abs(aug))) if aug.size > 0 else 0.0
    if peak > 0.99:
        aug = aug * (0.99 / peak)

    return aug

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Padding speech sequences and labels efficiently."""
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Korrektur: Kein [0] Indexing, sondern das gesamte input_features-Array übergeben
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch

class WhisperModel(Model):
    """Whisper model interface with optimized training pipeline"""

    def __init__(self, model_name: str, language: Optional[str] = None) -> None:
        local_candidate = os.path.join(info.MODEL_PATH, model_name)
        if os.path.exists(local_candidate):
            actual_model_path = local_candidate
        elif os.path.exists(model_name):
            actual_model_path = model_name
        else:
            actual_model_path = model_name
            
        super().__init__(model_name=actual_model_path.replace("/", "_"))
        self.is_trained: bool = False
        self.language: Optional[str] = language
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(actual_model_path, language=language, task="transcribe") # type: ignore
        self.model: WhisperForConditionalGeneration = WhisperForConditionalGeneration.from_pretrained(actual_model_path).to(self.device) # type: ignore

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
        """Run prediction in small batches with mixed precision for speed and safety."""
        self.model.eval()
        transcriptions: List[str] = []
        batch_size = 4

        total_samples = inputs.shape[0]

        # Use autocast during generation for hardware acceleration if on GPU
        with torch.no_grad(), torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            for i in range(0, total_samples, batch_size):
                batch_inputs = inputs[i : i + batch_size]
                predicted_ids: torch.Tensor = self.model.generate(batch_inputs)  # type: ignore
                batch_texts = self.processor.batch_decode(predicted_ids, skip_special_tokens=True) # type: ignore
                transcriptions.extend(batch_texts)

        return transcriptions

    def train(self, config_name: str, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Train or fine-tune the whisper model with optional waveform augmentation."""
        config_file_path: str = os.path.join(info.CONFIG_PATH, config_name)
        with open(config_file_path, "r", encoding="utf8") as f:
            config: Dict[str, Any] = json.load(f)

        # Expand training data with augmented copies while keeping val_data untouched
        aug_factor = int(config.get("aug_factor", 1))
        augmented_train_df = train_data.copy()
        augmented_train_df["is_augmented"] = False

        for _ in range(max(0, aug_factor)):
            aug_copy = train_data.copy()
            aug_copy["is_augmented"] = True
            augmented_train_df = pd.concat([augmented_train_df, aug_copy], ignore_index=True)

        if aug_factor > 0:
            augmented_train_df = augmented_train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        def prepare_dataset(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            audio_arrays: List[np.ndarray] = []
            aug_flags = batch.get("is_augmented", [False] * len(batch["file"]))

            for path, is_aug in zip(batch["file"], aug_flags):
                wav, _ = librosa.load(os.path.join(info.DATASET_PATH, path), sr=16000)
                if is_aug:
                    wav = _augment_waveform(wav, sr=16000)
                audio_arrays.append(wav)

            input_features = self.processor.feature_extractor(audio_arrays, sampling_rate=16000).input_features
            batch["input_features"] = input_features
            batch["labels"] = self.processor.tokenizer(batch["transcript"]).input_ids
            return batch

        def compute_metrics(pred: Any) -> Dict[str, float]:
            pred_ids = pred.predictions
            pad_id = self.processor.tokenizer.pad_token_id
            label_ids = np.where(pred.label_ids == -100, pad_id, pred.label_ids)

            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            transformation = jiwer.Compose([
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
                jiwer.RemovePunctuation(),
                jiwer.ToLowerCase(),
                jiwer.ReduceToListOfListOfWords()
            ])
            wer = float(jiwer.wer(
                label_str,
                pred_str,
                reference_transform=transformation,
                hypothesis_transform=transformation
            ))
            return {"wer": wer}

        print(f"Preparing training dataset ({len(augmented_train_df)} rows, aug_factor={aug_factor})...")
        train_ds: Dataset = Dataset.from_pandas(augmented_train_df)
        train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, batched=True) # type: ignore

        print("Preparing validation dataset...")
        val_ds: Any = Dataset.from_pandas(val_data)
        val_ds = val_ds.map(prepare_dataset, remove_columns=val_ds.column_names, batched=True) # type: ignore

        output_directory: str = os.path.join(info.MODEL_PATH, self.name(), config["name"])

        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_directory,
            per_device_train_batch_size=int(config["batch_size"]),
            per_device_eval_batch_size=int(config["batch_size"]),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
            learning_rate=float(config["lr"]),
            weight_decay=0.01,
            generation_max_length=225,
            warmup_steps=100,
            num_train_epochs=int(config["epochs"]),
            eval_strategy="epoch" if val_ds else "no",
            save_strategy="epoch",
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_checkpointing=True,
            predict_with_generate=True,
            logging_steps=25,
            dataloader_num_workers=2,
            report_to=["none"],
            metric_for_best_model="wer",
            greater_is_better=False,
            load_best_model_at_end=True
        )

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        self.model.config.apply_spec_augment = True
        self.model.config.mask_time_prob = 0.05
        self.model.config.mask_feature_prob = 0.05

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            processing_class=self.processor,
            compute_metrics=compute_metrics
        )

        print("Starting Whisper fine-tuning...")
        trainer.train() # type: ignore
        self.is_trained = True

        save_path: str = os.path.join(info.MODEL_PATH, f"{self.name()}_{config['name']}")
        self.model.save_pretrained(save_path) # type: ignore
        self.processor.save_pretrained(save_path) # type: ignore
        print(f"Model saved successfully to {save_path}")

    def predict_nbest(
        self,
        inputs: List[Any],
        n_best: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[List[List[str]], Optional[List[List[float]]]]:
        """Sample N-best candidate strings per sample for LM rescoring."""
        self.model.eval()
        k = max(1, int(n_best))
        batch_size = 1
        total_samples = inputs.shape[0]

        all_hypotheses: List[List[str]] = []
        all_scores: List[List[float]] = []

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad(), torch.amp.autocast(device_type):
            for start_idx in range(0, total_samples, batch_size):
                batch_inputs = inputs[start_idx : start_idx + batch_size]
                cur_batch_size = batch_inputs.shape[0]

                outputs = self.model.generate(  # type: ignore
                    batch_inputs,
                    attention_mask=torch.ones(
                        (batch_inputs.shape[0], batch_inputs.shape[-1]),
                        dtype=torch.long,
                        device=batch_inputs.device,
                    ),
                    num_beams=1,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=k,
                    return_dict_in_generate=True,
                )

                sequences: torch.Tensor = outputs.sequences
                decoded_flat: List[str] = self.processor.batch_decode(  # type: ignore
                    sequences,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                # Use rank-based log-prior scores so sampled candidates have consistent scale
                # with KenLM log10 scores and alpha tuning
                seq_scores_list: List[float] = [
                    float(-0.5 * (idx % k)) for idx in range(cur_batch_size * k)
                ]

                for b in range(cur_batch_size):
                    slice_start = b * k
                    slice_end = slice_start + k
                    all_hypotheses.append(decoded_flat[slice_start:slice_end])
                    all_scores.append(seq_scores_list[slice_start:slice_end])

        distinct_count = sum(1 for cands in all_hypotheses if len(set(cands)) > 1)
        print(f"Samples with >1 distinct candidate: {distinct_count} / {len(all_hypotheses)}")

        return all_hypotheses, all_scores