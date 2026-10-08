#!/usr/bin/env bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=qwen27-ku1000-97
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/e/fscratch/reformo/schuhmann1/scientific-qwen27-ku1000-97-20261008/logs/slurm-%j.out
#SBATCH --error=/e/fscratch/reformo/schuhmann1/scientific-qwen27-ku1000-97-20261008/logs/slurm-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
KU_RUN_ROOT=/e/fscratch/reformo/schuhmann1/scientific-qwen27-ku1000-97-20261008
KU_PYTHON=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PATH="$(dirname "$KU_PYTHON"):$PATH" PYTHONUNBUFFERED=1
export ALEXANDRIA_KU_RUN="$KU_RUN_ROOT"
srun --ntasks=1 "$KU_PYTHON" -u "$KU_RUN_ROOT/code/orchestrate.py"
