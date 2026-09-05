#!/bin/bash
#SBATCH --job-name=csa-fit
#SBATCH --account=qiulab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=shizhao.yang.gr@dartmouth.edu
#SBATCH --output=logs/fit_%j.out
#SBATCH --error=logs/fit_%j.err

# Stage C: the fit.  Both lambda_pm = 0 and lambda_pm > 0 against one frozen chain.
#   sbatch 03_fit.sbatch --epsilon 0.05 --K 12
#
# For a GPU run, replace the partition line above with e.g.
#   #SBATCH --partition=gpuq
#   #SBATCH --gres=gpu:1
#   #SBATCH --time=5-00:00:00          # gpuq caps at 5 days
# and leave --device auto (the default): csa_wot.resolve_device sets BOTH
# cfg.coupling.device and cfg.model.device, which run_fit.py does not.
# CPU is sufficient, so this is optional.
source ./_common.sh

"$PY" -u 03_fit.py "$@"
