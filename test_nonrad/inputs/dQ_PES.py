from pathlib import Path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from nonrad.ccd import get_dQ, get_PES_from_vaspruns, get_omega_from_PES
import json

work_dir = Path.cwd()
ground_files = Path(work_dir/'relax/ground_dir')
ground_struct = Structure.from_file(str(ground_files / 'CONTCAR'))
excited_files = Path(work_dir/'relax/excited_dir')
excited_struct = Structure.from_file(str(excited_files / 'CONTCAR'))

cc_dir = work_dir / 'cc_dir'

with open(work_dir / "dE.json", "r") as f:
    data = json.load(f)

dE = data["dE"]

print(f"dE = {dE}")

dQ = get_dQ(ground_struct, excited_struct)
print(f"Calculated dQ: {dQ:.4f} amu^{1/2} Angstrom")

ground_vaspruns = sorted(glob(str(cc_dir / 'ground' / '*' / 'vasprun.xml')))
excited_vaspruns = sorted(glob(str(cc_dir / 'excited' / '*' / 'vasprun.xml')))

# ground_vasprun_0 = str(ground_files / 'vasprun.xml')
# excited_vasprun_0 = str(excited_files / 'vasprun.xml')

# if ground_vasprun_0 not in ground_vaspruns:
    # ground_vaspruns.append(ground_vasprun_0)
# if excited_vasprun_0 not in excited_vaspruns:
    # excited_vaspruns.append(excited_vasprun_0)

Q_ground, E_ground = get_PES_from_vaspruns(ground_struct, excited_struct, ground_vaspruns)
Q_excited, E_excited = get_PES_from_vaspruns(ground_struct, excited_struct, excited_vaspruns)

E_excited = E_excited + dE

fig, ax = plt.subplots(figsize=(6, 5))

ax.scatter(Q_ground, E_ground, s=15, label='Ground State', color='blue')
ax.scatter(Q_excited, E_excited, s=15, label='Excited State', color='red')

# q_min = min(np.min(Q_ground), np.min(Q_excited)) - 0.5
# q_max = max(np.max(Q_ground), np.max(Q_excited)) + 0.5
# q = np.linspace(q_min, q_max, 100)
q = np.linspace(-1.0, 3.5, 100)
ground_omega = get_omega_from_PES(Q_ground, E_ground, ax=ax, q=q)
excited_omega = get_omega_from_PES(Q_excited, E_excited, ax=ax, q=q)

print(f"Ground state omega: {ground_omega:.4f} eV")
print(f"Excited state omega: {excited_omega:.4f} eV")

ax.set_xlabel(r'$Q$ [amu$^{1/2}$ $\AA$]')
ax.set_ylabel(r'$E$ [eV]')
ax.legend()
plt.tight_layout()
plt.savefig(cc_dir/"dQ_PES.png", dpi=600)
plt.show()

params = {
    "dQ": float(dQ),
    "ground_omega": float(ground_omega),
    "excited_omega": float(excited_omega)
}

with open(cc_dir/"ccd_parameters.json", "w") as f:
    json.dump(params, f, indent=4)


np.savetxt(cc_dir/"ground_PES.dat",np.column_stack((Q_ground, E_ground)),header="Q_ground E_ground(eV)")

np.savetxt(cc_dir/"excited_PES.dat",np.column_stack((Q_excited, E_excited)),header="Q_excited  E_excited(eV)")

print("Data saved successfully.")
