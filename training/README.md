# Casual-chat v1 training

The prepared run uses `Qwen/Qwen3-1.7B` with 4-bit QLoRA. It trains only a LoRA adapter and
calculates loss on the assistant completion, which is the self-authored message in each
record. The default effective batch size is 16 and the default run is two epochs.

This laptop is suitable for llama.cpp inference but does not have a discrete CUDA GPU for
this training configuration. Use a temporary CUDA runtime with at least 16 GB VRAM. Keep
the session private and do not publish the dataset, notebook, checkpoints, or adapter.

From the repository root in the GPU runtime:

```bash
python -m pip install -r training/requirements.txt
python training/train_qlora.py
```

The script downloads the public base model, reads the local train and validation JSONL
files, evaluates every 25 steps, retains the two newest checkpoints, restores the checkpoint
with the lowest validation loss, and writes the final adapter under `artifacts/`.

Do not start with more epochs. Personal chat data contains many repeated patterns, and a
small dataset can overfit quickly. Compare the base model and adapter on held-out prompts
before changing rank, learning rate, epochs, or the dataset.

## Test the completed adapter locally

The Colab adapter is converted to a llama.cpp GGUF adapter and paired with the matching
Qwen3-1.7B Q4_K_M base model under `artifacts/local-llamacpp/`. Thinking is disabled for
ordinary casual-chat testing.

Start an interactive terminal chat:

```bash
./scripts/run_sft_local.sh
```

Or test one prompt and exit:

```bash
./scripts/run_sft_local.sh "are you free this evening?"
```

For a local browser interface, start the server and open `http://127.0.0.1:8080`:

```bash
./scripts/run_sft_server.sh
```

The server binds only to the local computer. Stop either program with `Ctrl+C`.
