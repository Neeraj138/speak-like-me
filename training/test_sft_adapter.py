from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = "Reply naturally in casual English and match the conversation's tone and length."


def load_models(
    model_name: str = "Qwen/Qwen3-1.7B",
    adapter_path: str | Path = "artifacts/casual-chat-qwen3-1.7b-lora/final_adapter",
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU runtime is required for this test configuration.")
    capability = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map="auto",
        dtype=dtype,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    print(f"Loaded base model and SFT adapter on {torch.cuda.get_device_name(0)}")
    return model, tokenizer


def normalize_history(history: list[dict[str, str]] | None, message: str) -> list[dict[str, str]]:
    messages = list(history or [])
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": message})
    return messages


def generate_reply(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    adapter_enabled: bool,
    seed: int = 42,
    max_new_tokens: int = 128,
) -> str:
    import torch

    torch.manual_seed(seed)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    context = nullcontext() if adapter_enabled else model.disable_adapter()
    with context, torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
    return reply.strip()


def compare(
    model: Any,
    tokenizer: Any,
    message: str,
    history: list[dict[str, str]] | None = None,
    *,
    seed: int = 42,
) -> dict[str, str]:
    messages = normalize_history(history, message)
    base = generate_reply(model, tokenizer, messages, adapter_enabled=False, seed=seed)
    sft = generate_reply(model, tokenizer, messages, adapter_enabled=True, seed=seed)
    print("\nBASE MODEL\n----------")
    print(base)
    print("\nYOUR SFT MODEL\n--------------")
    print(sft)
    return {"base": base, "sft": sft}
