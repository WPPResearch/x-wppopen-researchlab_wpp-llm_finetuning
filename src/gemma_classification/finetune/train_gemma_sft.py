import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
from datasets import DatasetDict, load_dataset
from google.cloud import storage
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    TrainingArguments,
)
from transformers.models.gemma3 import Gemma3ForCausalLM
from trl import SFTTrainer


# -------------------------
# GCS Helper Functions
# -------------------------
def download_gcs_file(gcs_path: str) -> str:
    """
    Download a file from Google Cloud Storage to /tmp if needed.
    Returns the local path.
    """
    if not gcs_path.startswith("gs://"):
        return gcs_path

    client = storage.Client()
    bucket_name, *path_parts = gcs_path.replace("gs://", "").split("/")
    blob_path = "/".join(path_parts)
    local_path = f"/tmp/{os.path.basename(gcs_path)}"

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    print(f"Downloaded {gcs_path} to {local_path}")
    return local_path


def upload_file(local_file: str, gcs_uri: str) -> None:
    """
    Upload a local file to Google Cloud Storage.
    """
    client = storage.Client()
    bucket_name, *path_parts = gcs_uri.replace("gs://", "").split("/")
    blob_path = "/".join(path_parts)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_file)
    print(f"Uploaded {local_file} to {gcs_uri}")


def upload_checkpoints(local_dir: str, gcs_uri: str) -> None:
    """
    Upload all files under a directory to Google Cloud Storage.
    """
    client = storage.Client()
    bucket_name, *path_parts = gcs_uri.replace("gs://", "").split("/")
    prefix = "/".join(path_parts)
    bucket = client.bucket(bucket_name)

    for file_path in Path(local_dir).rglob("*"):
        if file_path.is_file():
            blob_path = f"{prefix}/{file_path.relative_to(local_dir)}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(file_path)
            print(f"Uploaded {file_path} -> gs://{bucket_name}/{blob_path}")


# -------------------------
# Custom Logits Processor
# -------------------------
class LabelConstraint(LogitsProcessor):
    """Constrains generation to allowed token IDs."""

    def __init__(self, allowed_token_ids: List[int]) -> None:
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = 0
        return scores + mask


# -------------------------
# Inference
# -------------------------
def process_batch_for_inference(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    model: Gemma3ForCausalLM,
    logits_processor: LogitsProcessorList,
) -> List[str]:
    """
    Run one-token constrained inference for a batch of prompts.
    """
    tokenizer.padding_side = "left"
    formatted_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                      tokenize=False,
                                      add_generation_prompt=True)
        for p in prompts
    ]

    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            logits_processor=logits_processor,
            pad_token_id=tokenizer.pad_token_id,
        )

    return [tokenizer.decode([tid], skip_special_tokens=True).strip() for tid in outputs[:, -1]]


# -------------------------
# Dataset Formatting
# -------------------------
def format_dataset(ds: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """
    Apply Gemma3 chat template formatting to a dataset.
    """
    def formatting_func(examples: dict) -> dict:
        texts = []
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt.strip()},
                    {"role": "assistant", "content": completion.strip()},
                ],
                tokenize=False,
            )
            texts.append(text)
        return {"text": texts}

    return ds.map(formatting_func, batched=True)


def format_test_dataset(ds: DatasetDict, tokenizer: AutoTokenizer) -> DatasetDict:
    """Apply chat template formatting for test data (prompts only)."""
    def formatting_func(examples: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False,
                                          add_generation_prompt=True)
            for p in examples["prompt"]
        ]
        return {"text": texts}

    return ds.map(formatting_func, batched=True)


# -------------------------
# Argument Parsing
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--base_model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--freeze-bottom-layers", type=int, default=14)
    parser.add_argument("--test-data", type=str, default=None)
    return parser.parse_args()


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = parse_args()

    if os.getenv("HUGGING_FACE_HUB_TOKEN"):
        login(token=os.getenv("HUGGING_FACE_HUB_TOKEN"))

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Gemma3ForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Ensure vocab is synced
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Freeze bottom layers if specified
    if args.freeze_bottom_layers > 0:
        for name, param in model.named_parameters():
            if any(f"model.layers.{i}." in name for i in range(args.freeze_bottom_layers)):
                param.requires_grad = False

    # Load dataset
  
    ds = load_dataset("json", data_files={
        "train": download_gcs_file(args.train_data),
        "validation": download_gcs_file(args.val_data),
    })
    ds = ds.rename_column("response", "completion")
    ds = format_dataset(ds, tokenizer)
    if args.test_data:
        test_ds = load_dataset("json", data_files={"test": download_gcs_file(args.test_data)})
        test_ds = format_test_dataset(test_ds, tokenizer)

    # Setup Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=TrainingArguments(
            warmup_steps=10,
            num_train_epochs=args.epochs,
            bf16=True,
            report_to="none",
            output_dir="checkpoints",
            save_strategy="epoch",
            save_total_limit=1,
            learning_rate=args.lr,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            logging_strategy="steps",
            logging_steps=10,
            logging_first_step=True,
            eval_accumulation_steps=1,
            per_device_eval_batch_size=4,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
        ),
    )
    trainer.train()

    trainer.save_model("final_model")
    tokenizer.save_pretrained("final_model")
    upload_checkpoints("final_model", f"{args.output_dir}/final_model")

    # -------------------------
    # Test Inference
    # -------------------------
    if args.test_data:
        model.eval()
        batch_size = 64
        prompts = test_ds["test"]["prompt"]
        # Note that this only works for prompts with encoded integer labels of length 1 i.e. 0, 1, 2
        labels = ["0", "1", "2"]
        allowed_ids = list({tokenizer.encode(l, add_special_tokens=False)[0] for l in labels})
        logits_processor = LogitsProcessorList([LabelConstraint(allowed_ids)])

        predictions: List[str] = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_preds = process_batch_for_inference(batch_prompts, tokenizer, model, logits_processor)
            predictions.extend(batch_preds)
        
        print("Unique predicted values: ", set(predictions))
        output_file = "test_predictions.txt"
        with open(output_file, "w") as f:
            f.writelines(f"{p}\n" for p in predictions)

        upload_file(output_file, f"{args.output_dir}/test_predictions.txt")
        print(f"Saved {len(predictions)} predictions to {output_file}")


if __name__ == "__main__":
    main()