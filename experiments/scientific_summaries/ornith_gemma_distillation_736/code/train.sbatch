#!/bin/bash
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --job-name=ornith-gemma12-lora
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:30:00
#SBATCH --output=/e/fscratch/reformo/schuhmann1/ornith-distillation-736-20261004/logs/train-%j.out
#SBATCH --error=/e/fscratch/reformo/schuhmann1/ornith-distillation-736-20261004/logs/train-%j.err
set -euo pipefail
module load Stages/2026 Python/3.13.5 CUDA/13
RUN=/e/fscratch/reformo/schuhmann1/ornith-distillation-736-20261004
ENVROOT=/e/fscratch/reformo/schuhmann1/scientific-distillation-lora
export PATH="$ENVROOT/env/bin:$PATH" PYTHONPATH="$ENVROOT/python_packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29000+SLURM_JOB_ID%1000))
srun --ntasks=8 --ntasks-per-node=4 --gpus-per-task=1 --kill-on-bad-exit=1 bash "$RUN/code/train_worker.sh" "${SCIENCE_LORA_RANK:?}"
