#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=ornith9-ar-qwen865
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=/e/fscratch/reformo/schuhmann1/scientific-distillation-865-20261004/logs/ornith-%j.out
#SBATCH --error=/e/fscratch/reformo/schuhmann1/scientific-distillation-865-20261004/logs/ornith-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
RUN=/e/fscratch/reformo/schuhmann1/scientific-distillation-865-20261004
PY=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
"$PY" "$RUN/code/prepare_release.py" --archive-only &
archive_pid=$!
srun --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 --kill-on-bad-exit=1 "$PY" "$RUN/code/ornith_worker.py"
wait "$archive_pid"
"$PY" "$RUN/code/finish_ornith.py"
