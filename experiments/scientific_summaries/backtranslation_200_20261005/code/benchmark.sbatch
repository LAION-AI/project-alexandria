#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=backtrans-200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=/e/fscratch/reformo/schuhmann1/scientific-backtranslation-200-20261005/logs/benchmark-%j.out
#SBATCH --error=/e/fscratch/reformo/schuhmann1/scientific-backtranslation-200-20261005/logs/benchmark-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
RUN=/e/fscratch/reformo/schuhmann1/scientific-backtranslation-200-20261005
PY=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MAX_JOBS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
srun --ntasks=1 "$PY" "$RUN/code/benchmark.py"
