#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=gemma-r128-no-thinking-97
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/e/fscratch/reformo/schuhmann1/scientific-gemma-r128-no-thinking-20261004/logs/benchmark-%j.out
#SBATCH --error=/e/fscratch/reformo/schuhmann1/scientific-gemma-r128-no-thinking-20261004/logs/benchmark-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
RUN=/e/fscratch/reformo/schuhmann1/scientific-gemma-r128-no-thinking-20261004
PY=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --ntasks=1 "$PY" "$RUN/code/benchmark.py"
