#!/bin/bash
#SBATCH --job-name=csa-eps
#SBATCH --account=qiulab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=shizhao.yang.gr@dartmouth.edu
#SBATCH --output=logs/eps_scan_%j.out
#SBATCH --error=logs/eps_scan_%j.err

# Stage A: the epsilon-informativeness scan (build-order step 1).
#   sbatch 01_eps_scan.sbatch
# Extra arguments are passed straight through to 01_eps_scan.py, e.g.
#   sbatch 01_eps_scan.sbatch --epsilons 0.05 0.1 0.2 0.5 1.0
source ./_common.sh

"$PY" -u 01_eps_scan.py --stride 1 --stride 2 "$@"
