#!/bin/bash
#SBATCH --job-name=csa-K
#SBATCH --account=qiulab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=shizhao.yang.gr@dartmouth.edu
#SBATCH --output=logs/k_sweep_%A_%a.out
#SBATCH --error=logs/k_sweep_%A_%a.err

# Stage B: held-out K selection, protocol (b).  One K per array task.
#   sbatch 02_k_sweep.sbatch --epsilon 0.05
#
# --array=0-7 indexes 02_k_sweep.py's DEFAULT_KS = [6, 8, 10, 12, 16, 20, 24, 30].
# Change both together if you change the grid.
#
# --mail-type without ARRAY_TASKS sends ONE mail for the array as a whole.
# Add ARRAY_TASKS if you want one per K (8 emails).
source ./_common.sh

"$PY" -u 02_k_sweep.py --array-index "$SLURM_ARRAY_TASK_ID" "$@"
