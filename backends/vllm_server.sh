#!/bin/bash
# Start a vLLM server exposing OpenAI-compatible API
# Usage: ./vllm_server.sh mistralai/Mistral-7B-Instruct-v0.2

MODEL_ID=${1:-"microsoft/phi-3-mini-4k-instruct"}
PORT=${2:-8000}

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_ID \
  --dtype half \
  --max-model-len 4096 \
  --port $PORT \
  --gpu-memory-utilization 0.92

