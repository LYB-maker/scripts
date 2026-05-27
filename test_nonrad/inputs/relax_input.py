import os
import numpy as np
from pathlib import Path
from shutil import copytree, copyfile
from pymatgen.core import Structure

work_dir = Path.cwd()
relax = Path(work_dir/'relax')
ground_dir = Path(work_dir/'relax/ground_dir')
excited_dir = Path(work_dir/'relax/excited_dir')
os.mkdir(str(relax))
os.mkdir(str(relax/ground_dir))
os.mkdir(str(relax/excited_dir))
copyfile(work_dir / 'job_relax.sh', ground_dir / 'job.sh')
copyfile(work_dir / 'KPOINTS', ground_dir / 'KPOINTS')
copyfile(work_dir / 'INCAR_relax_ground', ground_dir / 'INCAR')
copyfile(work_dir / 'POSCAR_ground', ground_dir / 'POSCAR')
copyfile(work_dir / 'POTCAR', ground_dir / 'POTCAR')
copyfile(work_dir / 'job_relax.sh', excited_dir / 'job.sh')
copyfile(work_dir / 'KPOINTS', excited_dir / 'KPOINTS')
copyfile(work_dir / 'INCAR_relax_excited', excited_dir / 'INCAR')
copyfile(work_dir / 'POSCAR_excited', excited_dir / 'POSCAR')
copyfile(work_dir / 'POTCAR', excited_dir / 'POTCAR')

