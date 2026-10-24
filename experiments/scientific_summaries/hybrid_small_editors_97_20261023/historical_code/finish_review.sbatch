#!/usr/bin/env bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=hybrid-remaining-meaning-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=00:15:00
#SBATCH --time-min=00:15:00
#SBATCH --output=/e/scratch/reformo/schuhmann1/scientific-hybrid-qwen08-gemma-e2b-97-20261023/logs/finish-review-%j.out
#SBATCH --error=/e/scratch/reformo/schuhmann1/scientific-hybrid-qwen08-gemma-e2b-97-20261023/logs/finish-review-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
HYBRID_ROOT=/e/scratch/reformo/schuhmann1/scientific-hybrid-qwen08-gemma-e2b-97-20261023
HYBRID_PYTHON=/e/fscratch/reformo/schuhmann1/scientific-distillation-1000/env/bin/python
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
# Offline engines run in each lane's process, avoiding a second round of Python
# imports in spawned engine processes. No editor or QA model is loaded here.
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$(dirname "$HYBRID_PYTHON"):$PATH"
cd "$HYBRID_ROOT"
srun --ntasks=1 /usr/bin/python3 -u "$HYBRID_ROOT/code/node_cache_launch.py" "$HYBRID_PYTHON"
