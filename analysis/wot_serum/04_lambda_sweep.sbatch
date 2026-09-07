#!/bin/bash
#SBATCH --job-name=csa-lam
#SBATCH --account=qiulab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=shizhao.yang.gr@dartmouth.edu
#SBATCH --output=logs/lambda_sweep_%j.out
#SBATCH --error=logs/lambda_sweep_%j.err

# Stage D: lambda_pm sweep at fixed epsilon and K (build-order step 4).
#   sbatch 04_lambda_sweep.sbatch --epsilon 0.05 --K 12
# Six fits against one shared chain, so budget ~6x a single Stage C fit.
source ./_common.sh

"$PY" -u 04_lambda_sweep.py "$@"
