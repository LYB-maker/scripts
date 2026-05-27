#!/bin/bash
#SBATCH --job-name=relax
#SBATCH --partition=cpu2_q
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

set -euo pipefail

VASP="mpirun /opt/ohpc/pub/apps/vasp5.4.4/bin/vasp_std"
CCD_ROOT="$SLURM_SUBMIT_DIR"

run_dir () {
    d="$1"
    cd "$d"
	
    $VASP

    cd "$SLURM_SUBMIT_DIR"
}

cd "$SLURM_SUBMIT_DIR"

for branch in ground excited; do
    for i in 0 1 2 3 4 5 6 7 8; do
        run_dir "${CCD_ROOT}/${branch}/${i}"
    done
done