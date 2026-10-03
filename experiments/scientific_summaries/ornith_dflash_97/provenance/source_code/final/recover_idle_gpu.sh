#!/usr/bin/env bash
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
RUN=/e/fscratch/reformo/schuhmann1/scientific-ornith-dflash-eval-97
PY=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PATH="$(dirname "$PY"):$PATH" PYTHONUNBUFFERED=1
srun --jobid=2160940 --overlap --nodes=1 --ntasks=1 --cpus-per-task=32 "$PY" -u "$RUN/code/recover_idle_gpu.py" > "$RUN/logs/idle-gpu-recovery.log" 2>&1
