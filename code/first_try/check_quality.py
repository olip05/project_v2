from code.first_try.definitions import * 
from code.first_try.train_transformer import DataCollatorSpeechSeq2SeqWithPadding

from datasets import load_from_disk

from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import WhisperProcessor

model_path = f"{MODEL_ID}_fineTuned"

if __name__ == "__main__":
    print("Load model...")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(DEVICE)

    dataset = load_from_disk(DATASET_PATH)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=dataset["dev"][:50],
        data_collator=data_collator,
        processing_class=processor,
    )

    test_results = trainer.predict(dataset["dev"][:50])
    print("Test metrics:", test_results.metrics)

    predictions_ids = test_results.predictions
    label_ids = test_results.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(predictions_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    print("\n--- Samples ---")
    for i in range(min(5, len(pred_str))):
        print(f"Reference:     {label_str[i]}")
        print(f"Prediction:     {pred_str[i]}")
        print("-" * 40)