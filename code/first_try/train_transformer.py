from code.first_try.definitions import * 

from dataclasses import dataclass
from typing import Any, Union, Dict, List
from datasets import load_from_disk
import torch

from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import WhisperProcessor

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

if __name__ == "__main__":
    dataset = load_from_disk(DATASET_PATH)
    dataset.set_format(type="torch", columns=["input_features", "labels"])

    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.to(DEVICE)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-dadirri",  
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,  
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=400,
        eval_steps=50,
        logging_steps=10,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="steps",        # epoch - save at the end of each epoch; no - disable save
        eval_strategy="steps", # same with save strategy
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        data_collator=data_collator,
        processing_class=processor,
    )

    print("Start training...")
    trainer.train()
    print("Training finished")

    # Save the finetuned model and processor
    model.save_pretrained(f"{MODEL_ID}_fineTuned")
    processor.save_pretrained(f"{MODEL_ID}_fineTuned")
    print("Model saved")