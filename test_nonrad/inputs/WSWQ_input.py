import os
import numpy as np
from pathlib import Path
from shutil import copytree, copyfile
from pymatgen.core import Structure

work_dir = Path.cwd()
ground_files = Path(work_dir/'relax/ground_dir')
cc_dir = Path(work_dir/'cc_dir')
WSWQ_dir = Path(work_dir/'WSWQ_dir')
os.mkdir(str(WSWQ_dir))
copytree(cc_dir / 'ground', WSWQ_dir / 'ground')
copyfile(work_dir / 'job_WSWQ.sh', WSWQ_dir / 'job.sh')
for d in (WSWQ_dir / 'ground').glob('*'):
    if not d.is_dir():
        continue
    copyfile(work_dir / 'INCAR_WSWQ', d / 'INCAR')
    # copyfile(d / 'WAVECAR', d / 'WAVECAR.qqq')
    # copyfile(ground_files / 'WAVECAR', d / 'WAVECAR')
    copyfile(ground_files / 'POSCAR', d / 'POSCAR')
    copyfile(ground_files / 'POTCAR', d / 'POTCAR')
