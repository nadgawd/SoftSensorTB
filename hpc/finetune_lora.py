#!/usr/bin/env python3
"""LoRA fine-tune on tool-calling conversations (TRL SFTTrainer + PEFT)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--use_4bit", action="store_true", help="QLoRA 4-bit base weights")
    p.add_argument(
        "--target_modules",
        default="all-linear",
        help=(
            "Comma-separated module names, or 'all-linear' (default). Projection "
            "names differ across architectures — Qwen3.5's gated delta networks "
            "do not use the Llama-style q_proj/k_proj naming."
        ),
    )
    return p.parse_args()


def format_example(row: dict) -> str:
    """Expect JSONL rows with a `messages` list (OpenAI chat format)."""
    messages = row.get("messages")
    if not messages:
        raise ValueError("Each row must have a 'messages' field")
    # TRL applies chat template via dataset formatting — store as JSON string of messages
    import json

    return json.dumps(messages, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = (
        "all-linear"
        if args.target_modules == "all-linear"
        else [m.strip() for m in args.target_modules.split(",") if m.strip()]
    )

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)

    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    def to_text(batch):
        texts = []
        for messages in batch["messages"]:
            texts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )
        return {"text": texts}

    dataset = dataset.map(to_text, batched=True, remove_columns=dataset.column_names)

    training_args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
