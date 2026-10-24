#!/usr/bin/env bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --time-min=00:08:00
#SBATCH --job-name=number-style97
#SBATCH --output=/e/scratch/reformo/schuhmann1/scientific-post-gemma-number-style-97-20261024/logs/slurm-%j.out
#SBATCH --error=/e/scratch/reformo/schuhmann1/scientific-post-gemma-number-style-97-20261024/logs/slurm-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
CASEGUARD_ROOT=/e/scratch/reformo/schuhmann1/scientific-post-gemma-number-style-97-20261024
CASEGUARD_PYTHON=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$(dirname "$CASEGUARD_PYTHON"):$PATH"
cd "$CASEGUARD_ROOT"
srun --ntasks=1 /usr/bin/python3 -u "$CASEGUARD_ROOT/code/node_cache_launch.py" "$CASEGUARD_PYTHON"
