#!/bin/bash
set -euo pipefail
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=0
RUN=/e/fscratch/reformo/schuhmann1/scientific-distillation-865-20261004
exec /e/fscratch/reformo/schuhmann1/scientific-distillation-lora/env/bin/python "$RUN/code/train_generator.py" --lora-rank "$1"
