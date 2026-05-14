#!/bin/bash
# start_vllm_server.sh — Launch the shared vLLM inference server.
#
# Memory budget:
#   --gpu-memory-utilization 0.35 on 128 GB = ~44.8 GB cap for vLLM
#   Leaves ~45 GB headroom for the nightly unsloth fine-tune (runs concurrently)
#
# Model: merged BF16 Qwen2.5-32B + LoRA, quantized to 4-bit on load by vLLM.
# The finance_qwen_32b_merged_latest symlink always points to the most recent merge.
# model_promoter.py updates the symlink and restarts this service after each fine-tune.
#
# Managed by ai-inference-server.service (systemd). Do not call directly.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERGED_MODEL="${PROJECT_ROOT}/finetune/finance_qwen_32b_merged_latest"

if [ ! -d "${MERGED_MODEL}" ]; then
    echo "ERROR: merged model not found at ${MERGED_MODEL}" >&2
    echo "Run the one-time LoRA merge first — see CLAUDE.md (vLLM migration, Step 2)." >&2
    exit 1
fi

exec "/home/zgx/miniforge3/envs/vllm-server/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MERGED_MODEL}" \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.35 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --host 127.0.0.1 \
    --port 8000 \
    --served-model-name "Qwen/Qwen2.5-32B-Instruct" \
    --no-enable-log-requests
