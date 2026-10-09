#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_cpp_dir="${LLAMA_CPP_DIR:-$project_dir/../../local-llm/llama.cpp}"
llama_cli="$llama_cpp_dir/build-cpu/bin/llama-cli"
base_model="$project_dir/artifacts/local-llamacpp/Qwen3-1.7B-Q4_K_M.gguf"
adapter="$project_dir/artifacts/local-llamacpp/casual-chat-sft-lora-f16.gguf"

for required_file in "$llama_cli" "$base_model" "$adapter"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

common_args=(
  -m "$base_model"
  --lora "$adapter"
  --jinja
  --reasoning off
  --ctx-size 4096
  --temp 0.75
  --top-p 0.9
  --no-display-prompt
  --conversation
)

if (( $# > 0 )); then
  exec "$llama_cli" "${common_args[@]}" --single-turn --prompt "$*"
fi

exec "$llama_cli" "${common_args[@]}"
