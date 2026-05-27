#!/bin/bash
#SBATCH --job-name=WSWQ
#SBATCH --partition=cpu3_q
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

vasp="mpirun /opt/ohpc/pub/apps/vasp5.4.4/bin/vasp_std"
WSWQ_ROOT="$SLURM_SUBMIT_DIR"

run_dir () {
    d="$1"
    cd "$d"
	
    $vasp

    cd "$SLURM_SUBMIT_DIR"
}

cd "$SLURM_SUBMIT_DIR"

for branch in ground; do
    for i in 0 1 2 3 4 5 6 7 8; do
        run_dir "${WSWQ_ROOT}/${branch}/${i}"
    done
done