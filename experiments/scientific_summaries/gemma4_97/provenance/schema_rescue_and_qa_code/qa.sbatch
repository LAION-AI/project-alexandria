#!/usr/bin/env bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=gemma4-qa97-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=/e/fscratch/reformo/schuhmann1/scientific-gemma4-eval-97/logs/qa-slurm-%j.out
#SBATCH --error=/e/fscratch/reformo/schuhmann1/scientific-gemma4-eval-97/logs/qa-slurm-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
PY=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PATH="$(dirname "$PY"):$PATH" PYTHONUNBUFFERED=1
srun --ntasks=1 "$PY" -u /e/fscratch/reformo/schuhmann1/scientific-gemma4-eval-97/rescue_code/qa_resume.py
