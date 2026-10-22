#!/bin/bash
# Submit from the repo root:  sbatch slurm/download_miluv.sbatch [extra args]
#SBATCH --job-name=download_miluv
#SBATCH --partition=rl2-lab
#SBATCH --account=rl2-lab
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --exclude=sky1,sky2
#SBATCH --output=results/slurm/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname)"
.venv/bin/python scripts/download_miluv.py "$@"
