# Sourced by every *.sbatch in this directory. Not meant to be run on its own.
set -euo pipefail

PY=/dartfs-hpc/rc/home/8/f007zb8/.conda/envs/LinContraLearn/bin/python

# Slurm starts the batch script in the submit directory, and the #SBATCH
# --output paths above are relative to it, so the log files already landed here.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f csa_wot.py ]]; then
    echo "ERROR: csa_wot.py not found in $(pwd)." >&2
    echo "Submit from CellStateAdj/analysis/wot_serum -- the --output paths and" >&2
    echo "the python imports are both relative to the submit directory." >&2
    exit 2
fi

# Slurm opens the --output file BEFORE this script runs, so logs/ has to exist
# at submit time; this is only a backstop for a stray manual run.
mkdir -p logs

NCPU="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NCPU"
export MKL_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU"
export NUMEXPR_NUM_THREADS="$NCPU"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=$NCPU started=$(date -Is)"
