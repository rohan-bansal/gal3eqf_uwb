#!/bin/bash
# Submit from the repo root:  sbatch slurm/experiment.sbatch experiments/<NN_name>/<script>.py [extra args]
# Some nodes run far past their allocation (load >> cores); check with
#   sinfo -p rl2-lab -N -o "%N %O %c"
# and add those to --exclude on the sbatch command line.
#SBATCH --job-name=gal3uwb
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=sky1,sky2
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname), ${SLURM_CPUS_PER_TASK} cpus, $(nproc --all) cores, load $(cut -d' ' -f1-3 /proc/loadavg)"
script="$1"
shift
.venv/bin/python "$script" --workers "$SLURM_CPUS_PER_TASK" "$@"
