#!/bin/bash
#SBATCH --job-name=csa-pca
#SBATCH --account=qiulab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=shizhao.yang.gr@dartmouth.edu
#SBATCH --output=logs/serum_only_pca_%j.out
#SBATCH --error=logs/serum_only_pca_%j.err

# Step 1: build the SERUM-ONLY representation (Stage-0 requirement 1).
#
# 192G because the raw AnnData is 236,285 x 19,089 with 604M nonzeros: the CSR
# alone is ~5 GB in memory, sc.pp.scale then densifies the 2,000-HVG matrix, and
# scanpy holds intermediates. 16 cores is enough -- this is dominated by IO and a
# single randomized SVD, not by parallel work, and it keeps room under the
# qiulab 80-core account cap for the epsilon scan queued behind it.
source ./_common.sh

"$PY" -u 00_serum_only_pca.py "$@"
