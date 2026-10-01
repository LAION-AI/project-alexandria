#!/usr/bin/env bash
set -euo pipefail
extractor_python=${ALEXANDRIA_EXTRACTOR_PYTHON:-/home/rvv/llm-inference/.venv27/bin/python}
extractor_weights=${ALEXANDRIA_EXTRACTOR_WEIGHTS:-Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound}
CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_FLASHINFER_SAMPLER=0 exec "$extractor_python" \
  -m vllm.entrypoints.openai.api_server --model "$extractor_weights" \
  --revision 4756e3e4871aefd8d7cd5b0f6155ae5490451c1e \
  --served-model-name qwen38 --host 127.0.0.1 --port 8010 --tensor-parallel-size 2 \
  --attention-backend TRITON_ATTN --reasoning-parser qwen3 --enable-prefix-caching \
  --max-model-len 32768 --max-num-batched-tokens 8192 --max-num-seqs 10 \
  --gpu-memory-utilization 0.90 --enforce-eager --no-enable-log-requests --generation-config vllm
