import json
from pathlib import Path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from nonrad.ccd import get_Q_from_struct
from nonrad.elphon import get_Wif_from_WSWQ
from pymatgen.io.vasp import Vasprun
work_dir = Path.cwd()
cc_dir = work_dir / "cc_dir"
WSWQ_dir = work_dir / "WSWQ_dir"
with open(cc_dir / "ccd_parameters.json", "r") as f:
    data = json.load(f)

dQ = data["dQ"]

print(f"dQ = {dQ}")
factors = np.linspace(-0.5, 0.5, 9)
WSWQs = [
    (
        dQ * f,
        str(work_dir / 'WSWQ_dir' / 'ground' / str(i) / 'WSWQ')
    )
    for i, f in enumerate(factors)
]
ground_files=Path(work_dir/'relax/ground_dir')
fig = plt.figure(figsize=(12, 5))
Wifs = get_Wif_from_WSWQ(WSWQs, str(ground_files / 'vasprun.xml'), 192, [189, 190, 191], spin=1, fig=fig)
Wif = np.sqrt(np.mean([x[1]**2 for x in Wifs]))
print(Wifs, Wif)
plt.tight_layout()
plt.savefig(WSWQ_dir /'VBM_Wif.png', dpi=600)
plt.show()

wif_data = {
    "dQ": float(dQ),
    "Wif": float(Wif),
    "bands": [189, 190, 191],
    "spin": 1
}
with open(WSWQ_dir / "Wif.json", "w") as f:
    json.dump(wif_data, f, indent=4)
print("Wif.json saved successfully.")

