#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate base and SFT candidates for preference review")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/casual-chat-qwen3-1.7b-lora/final_adapter"))
    parser.add_argument("--prompts", type=Path, default=Path("data/v1/preference/prompts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/preference/candidates.partial.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def clean_generation(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text or "[EMPTY_GENERATION]"


def generate(model, tokenizer, prompts, *, batch_size: int, seed: int, label: str) -> list[str]:
    import torch

    torch.manual_seed(seed)
    results: list[str] = []
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        results.extend(clean_generation(text) for text in tokenizer.batch_decode(output[:, prompt_width:], skip_special_tokens=True))
        print(f"{label}: generated {len(results)}/{len(prompts)} candidates", flush=True)
    return results


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for candidate generation.")
    rows = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line]
    prompts = [row["prompt"] for row in rows]
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map="auto",
        dtype=dtype,
    )
    base.eval()
    base_outputs = generate(base, tokenizer, prompts, batch_size=args.batch_size, seed=101, label="base")
    adapted = PeftModel.from_pretrained(base, args.adapter)
    adapted.eval()
    sft_outputs = generate(adapted, tokenizer, prompts, batch_size=args.batch_size, seed=202, label="sft")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row, base_text, sft_text in zip(rows, base_outputs, sft_outputs, strict=True):
            candidates = [
                {"source": "historical_self_reply", "text": row["reference_completion"]},
                {"source": "qwen3_1.7b_base", "text": base_text},
                {"source": "qwen3_1.7b_sft", "text": sft_text},
            ]
            candidates.sort(key=lambda item: hashlib.sha256(f"{row['preference_id']}\0{item['source']}".encode()).hexdigest())
            record = {
                "preference_id": row["preference_id"],
                "prompt": row["prompt"],
                "candidates": [
                    {"candidate_id": f"candidate_{index + 1}", **candidate}
                    for index, candidate in enumerate(candidates)
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    args.output.chmod(0o600)
    print(json.dumps({"prompts": len(rows), "candidates_per_prompt": 3, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
