import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import soundfile as sf
import evaluate

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def prepare_dataset(batch, feature_extractor, tokenizer):
    audio = batch["wav_filename"]["array"]
    sampling_rate = batch["wav_filename"]["sampling_rate"]
    batch["input_features"] = feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

def main():
    train_file_path = '/content/train.csv'

    dataset = load_dataset('csv', data_files={'train': train_file_path, 'test': train_file_path})
    dataset = dataset.cast_column("wav_filename", Audio(sampling_rate=16000))

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Spanish", task="transcribe")
    processor = WhisperProcessor(feature_extractor, tokenizer)

    dataset = dataset.map(prepare_dataset, fn_kwargs={'feature_extractor': feature_extractor, 'tokenizer': tokenizer}, num_proc=1)

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.generation_config.language = "spanish"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-hi",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
         hub_token="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",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train() 

if __name__ == '__main__':
    main()
