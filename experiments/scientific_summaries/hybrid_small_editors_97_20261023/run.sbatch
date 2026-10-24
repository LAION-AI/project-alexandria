#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --job-name=alexandria-hybrid-repair
set -euo pipefail
# Supply your account/partition with sbatch; activate the pinned environment first.
# Prepare/download with bootstrap.py --inputs-only --download-models BEFORE sbatch.
HYBRID_ROOT=${ALEXANDRIA_HYBRID_ROOT:-${SLURM_SUBMIT_DIR}}
HYBRID_PYTHON=${HYBRID_PYTHON:-python}
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$HYBRID_ROOT"
srun --ntasks=1 "$HYBRID_PYTHON" -u "$HYBRID_ROOT/code/orchestrate.py"
