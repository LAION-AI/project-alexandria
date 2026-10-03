#!/usr/bin/env bash
set -euo pipefail
# Same BF16 weights/scoring as the prior comparison, one GPU alongside an existing job.
# Override these three variables for another installation; never pass credentials as arguments.
judge_python=${ALEXANDRIA_JUDGE_PYTHON:-/home/rvv/llm-inference/.venv27/bin/python}
judge_weights=${ALEXANDRIA_JUDGE_WEIGHTS:-/home/c4r33u19/models/Qwen2.5-7B-Instruct}
judge_gpu=${ALEXANDRIA_JUDGE_GPU:-1}
judge_context=${ALEXANDRIA_JUDGE_MAX_LEN:-16384}
judge_utilization=${ALEXANDRIA_JUDGE_GPU_UTIL:-0.68}
judge_port=${ALEXANDRIA_JUDGE_PORT:-8010}
CUDA_VISIBLE_DEVICES="$judge_gpu" VLLM_USE_FLASHINFER_SAMPLER=0 exec "$judge_python" \
  -m vllm.entrypoints.openai.api_server \
  --model "$judge_weights" --served-model-name qwen25 --host 127.0.0.1 --port "$judge_port" \
  --tensor-parallel-size 1 --attention-backend TRITON_ATTN --enable-prefix-caching \
  --max-model-len "$judge_context" --max-num-batched-tokens 4096 --max-num-seqs 4 \
  --gpu-memory-utilization "$judge_utilization" --enforce-eager --no-enable-log-requests --generation-config vllm
