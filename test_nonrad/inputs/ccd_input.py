import os
import numpy as np
from pathlib import Path
from shutil import copyfile
from pymatgen.core import Structure
from nonrad.ccd import get_cc_structures

# equilibrium structures from your first-principles calculation
work_dir = Path.cwd()
ground_files = Path(work_dir/'relax/ground_dir')
ground_struct = Structure.from_file(str(ground_files / 'CONTCAR'))
excited_files = Path(work_dir/'relax/excited_dir')
excited_struct = Structure.from_file(str(excited_files / 'CONTCAR'))

# output directory that will contain the input files for the CC diagram
cc_dir = Path(work_dir/'cc_dir')
os.mkdir(str(cc_dir))
os.mkdir(str(cc_dir / 'ground'))
os.mkdir(str(cc_dir / 'excited'))
copyfile(str(work_dir / 'job_ccd.sh'), str(cc_dir / 'job.sh'))

# displacements as a percentage, this will generate the displacements
# -50%, -37.5%, -25%, -12.5%, 0%, 12.5%, 25%, 37.5%, 50%
displacements = np.linspace(-0.5, 0.5, 9)

# note: the returned structures won't include the 0% displacement, this is intended
# it can be included by specifying remove_zero=False
ground, excited = get_cc_structures(ground_struct, excited_struct, displacements, remove_zero=False)

for i, struct in enumerate(ground):
    working_dir = cc_dir / 'ground' / str(i)
    os.mkdir(str(working_dir))
    
    # write structure and copy necessary input files
    struct.to(filename=str(working_dir / 'POSCAR'), fmt='poscar')
    copyfile(str(ground_files / 'POTCAR'), str(working_dir / 'POTCAR'))
    copyfile(str(work_dir / 'INCAR_ccd_ground'),str(working_dir / 'INCAR'))
    copyfile(str(work_dir / 'KPOINTS'),str(working_dir / 'KPOINTS'))
        
for i, struct in enumerate(excited):
    working_dir = cc_dir / 'excited' / str(i)
    os.mkdir(str(working_dir))
    
    # write structure and copy necessary input files
    struct.to(filename=str(working_dir / 'POSCAR'), fmt='poscar')
    copyfile(str(excited_files / 'POTCAR'), str(working_dir / 'POTCAR'))
    copyfile(str(work_dir / 'INCAR_ccd_excited'),str(working_dir / 'INCAR'))
    copyfile(str(work_dir / 'KPOINTS'),str(working_dir / 'KPOINTS'))
    