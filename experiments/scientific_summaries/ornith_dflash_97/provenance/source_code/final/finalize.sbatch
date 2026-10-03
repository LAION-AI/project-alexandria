#!/usr/bin/env bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=ornith-eval97-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/e/fscratch/reformo/schuhmann1/scientific-ornith-dflash-eval-97/logs/finalize-%j.out
#SBATCH --error=/e/fscratch/reformo/schuhmann1/scientific-ornith-dflash-eval-97/logs/finalize-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
RUN=/e/fscratch/reformo/schuhmann1/scientific-ornith-dflash-eval-97
PY=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PATH="$(dirname "$PY"):$PATH" PYTHONUNBUFFERED=1
srun --ntasks=1 "$PY" -u "$RUN/code/finalize.py"
