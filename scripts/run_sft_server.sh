#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-$project_dir/../../local-llm/llama.cpp}"
llama_server="$llama_cpp_dir/build-cpu/bin/llama-server"
base_model="$project_dir/artifacts/local-llamacpp/Qwen3-1.7B-Q4_K_M.gguf"
adapter="$project_dir/artifacts/local-llamacpp/casual-chat-sft-lora-f16.gguf"

for required_file in "$llama_server" "$base_model" "$adapter"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

exec "$llama_server" \
  -m "$base_model" \
  --lora "$adapter" \
  --jinja \
  --reasoning off \
  --ctx-size 8192 \
  --host 127.0.0.1 \
  --port "${PORT:-8080}" \
  --webui
