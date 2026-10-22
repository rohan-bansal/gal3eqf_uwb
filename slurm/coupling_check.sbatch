#!/bin/bash
# Submit from the repo root:  sbatch slurm/coupling_check.sbatch [extra args]
# Some nodes run far past their allocation (load >> cores); check with
#   sinfo -p rl2-lab -N -o "%N %O %c"
# and add those to --exclude on the sbatch command line.
#SBATCH --job-name=coupling_check
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=sky1,sky2
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname), ${SLURM_CPUS_PER_TASK} cpus, $(nproc --all) cores, load $(cut -d' ' -f1-3 /proc/loadavg)"
.venv/bin/python experiments/coupling_check.py --workers "$SLURM_CPUS_PER_TASK" "$@"
