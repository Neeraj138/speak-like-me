#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the private casual-chat v1 LoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-file", type=Path, default=Path("data/v1/casual_chat/train.jsonl"))
    parser.add_argument("--validation-file", type=Path, default=Path("data/v1/casual_chat/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/casual-chat-qwen3-1.7b-lora"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def validate_inputs(train_file: Path, validation_file: Path) -> tuple[int, int]:
    def validate(path: Path) -> int:
        count = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            if not isinstance(row.get("prompt"), list) or not isinstance(row.get("completion"), list):
                raise ValueError(f"{path}:{line_number} is not conversational prompt-completion data")
            if not row["prompt"] or row["prompt"][-1].get("role") != "user":
                raise ValueError(f"{path}:{line_number} does not end its prompt with a user turn")
            if len(row["completion"]) != 1 or row["completion"][0].get("role") != "assistant":
                raise ValueError(f"{path}:{line_number} does not have one assistant completion")
            count += 1
        if not count:
            raise ValueError(f"{path} is empty")
        return count

    return validate(train_file), validate(validation_file)


def main() -> None:
    args = parse_args()
    train_count, validation_count = validate_inputs(args.train_file, args.validation_file)

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install training/requirements.txt in a GPU runtime."
        ) from error

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this QLoRA configuration.")

    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map="auto",
        dtype=compute_dtype,
    )
    model.config.use_cache = False

    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_length=args.max_length,
        completion_only_loss=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        logging_steps=5,
        optim="paged_adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        report_to="none",
        seed=42,
        data_seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "train_examples": train_count,
                "validation_examples": validation_count,
                "gpu": torch.cuda.get_device_name(0),
                "bf16": bf16,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    tokenizer.save_pretrained(args.output_dir / "final_adapter")


if __name__ == "__main__":
    main()
