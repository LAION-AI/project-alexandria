#!/bin/bash
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
RUN=/e/fscratch/reformo/schuhmann1/scientific-distillation-865-20261004
PY=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PATH="${PY%/*}:$PATH" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export QA_GPU_INDICES=0,3
exec "$PY" "$RUN/code/live_qa.py"
