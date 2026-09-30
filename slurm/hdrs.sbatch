#!/bin/bash
# Step 4 (HDRs) on a Slurm cluster, as a job array: task k handles sources
# [k*CHUNK, (k+1)*CHUNK) of the sorted posterior list. Sources that already have an HDR file are
# skipped, so resubmitting the same array resumes after a timeout or failure.
#
# Submit from the copu_cat root (after `uv sync`), with one array task per CHUNK sources:
#
#   export DATA_DIR=/path/to/erebor_0626 CHUNK=50
#   N=$(ls $DATA_DIR/posterior_chains/*_posterior.feather | wc -l)
#   mkdir -p logs && sbatch --array=0-$(( (N + CHUNK - 1) / CHUNK - 1 ))%50 slurm/hdrs.sbatch
#
# (%50 = at most 50 tasks at once; adjust to your cluster.) When all tasks are done:
#
#   .venv/bin/copu-cat-combine-hdrs --data-dir $DATA_DIR
#
# Adjust the #SBATCH lines (partition/account, time, memory) to your cluster.

#SBATCH --job-name=copu-cat-hdrs
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hdrs_%A_%a.out
##SBATCH --partition=...
##SBATCH --account=...

set -euo pipefail

: "${DATA_DIR:?set DATA_DIR to the data directory (with posterior_chains/ and injection_matches/)}"
CHUNK=${CHUNK:-50}
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
STOP=$(( START + CHUNK ))

# module load ...   # if your cluster needs modules for Python

# keep numpy/jax to the CPUs Slurm gave us
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

echo "task $SLURM_ARRAY_TASK_ID: sources [$START, $STOP) of $DATA_DIR on $(hostname)"
.venv/bin/copu-cat-hdrs --data-dir "$DATA_DIR" --start "$START" --stop "$STOP"
