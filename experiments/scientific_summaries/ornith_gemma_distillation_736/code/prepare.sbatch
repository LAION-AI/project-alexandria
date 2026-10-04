#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=ornith736-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --output=/e/fscratch/reformo/schuhmann1/ornith-distillation-736-20261004/logs/prepare-%j.out
#SBATCH --error=/e/fscratch/reformo/schuhmann1/ornith-distillation-736-20261004/logs/prepare-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
RUN=/e/fscratch/reformo/schuhmann1/ornith-distillation-736-20261004
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
srun /e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python "$RUN/code/prepare_ornith_release.py"
DEST=/e/data1/datasets/playground/mmlaion/schuhmann1/ornith-distillation-736-20261004
mkdir -p "$DEST"
rsync -a --exclude=__pycache__/ --exclude=*.tmp --exclude=.cache/ "$RUN/release" "$RUN/training" "$RUN/inputs" "$RUN/code" "$DEST/"
