#!/bin/bash
#SBATCH --job-name=csa-cmi
#SBATCH --account=qiulab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=shizhao.yang.gr@dartmouth.edu
#SBATCH --output=logs/07_cmi_compare_%j.out
#SBATCH --error=logs/07_cmi_compare_%j.err

# 16 cores, not 32: the qiulab account has a hard 80-core GrpTRES cap shared
# across all its users, so Workstreams A and B (24 + 24) plus headroom fit
# together and can run in parallel as the plan's HPC order step 3 requires.
source ./_common.sh

"$PY" -u 07_cmi_compare.py "$@"
