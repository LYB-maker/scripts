#!/bin/bash
#SBATCH --job-name=cc_dir
#SBATCH --partition=cpu1_q
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36

set -euo pipefail

VASP="mpirun /opt/ohpc/pub/apps/vasp5.4.4/bin/vasp_std"
CCD_ROOT="$SLURM_SUBMIT_DIR"

python 0_relax_input.py &&