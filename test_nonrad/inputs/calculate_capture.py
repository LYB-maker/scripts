from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from nonrad import get_C
from nonrad.scaling import thermal_velocity

# ========================
# Read parameters
# ========================

work_dir = Path.cwd()

with open("dE.json") as f:
    dE = json.load(f)["dE"]

with open(work_dir / "cc_dir/ccd_parameters.json") as f:
    params = json.load(f)

dQ = params["dQ"]
ground_omega = params["ground_omega"]
excited_omega = params["excited_omega"]

with open(work_dir / "WSWQ_dir/Wif.json") as f:
    Wif = json.load(f)["Wif"]

# ========================
# Structure volume
# ========================

ground_struct = Structure.from_file(
    work_dir / "relax/ground_dir/CONTCAR"
)

volume = ground_struct.volume

# ========================
# Capture coefficient Cp(T)
# ========================

g = 4
T = np.linspace(25, 800, 1000)

Cp = get_C(
    dQ=dQ,
    dE=dE,
    wi=excited_omega,
    wf=ground_omega,
    Wif=Wif,
    volume=volume,
    g=g,
    T=T
)

# ========================
# Plot Cp(T)
# ========================

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].semilogy(T, Cp)
ax[0].set_xlabel('T [K]')
ax[0].set_ylabel(r'$C_p$ [cm$^3$ s$^{-1}$]')

ax[1].semilogy(1000 / T[200:], Cp[200:])
ax[1].set_xlabel(r'$1000/T$ [K$^{-1}$]')
ax[1].set_ylabel(r'$C_p$ [cm$^3$ s$^{-1}$]')

for a in ax:
    a.grid(True, ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig("Cp_T.png", dpi=600)
plt.show()

# ========================
# Capture cross section
# ========================

m_eff = 0.18

sigma = Cp / thermal_velocity(T, m_eff)   # cm^2
sigma *= 1e16                             # Å^2

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].semilogy(T, sigma)
ax[0].set_xlabel('T [K]')
ax[0].set_ylabel(r'$\sigma$ [$\AA^2$]')

ax[1].semilogy(1000 / T[200:], sigma[200:])
ax[1].set_xlabel(r'$1000/T$ [K$^{-1}$]')
ax[1].set_ylabel(r'$\sigma$ [$\AA^2$]')

for a in ax:
    a.grid(True, ls='--', alpha=0.5)

plt.tight_layout()
plt.savefig("sigma_T.png", dpi=600)
plt.show()

# ========================
# 300 K values
# ========================

idx = np.abs(T - 300).argmin()

Cp_300K = Cp[idx]
sigma_300K = sigma[idx]

print("========== 300 K ==========")
print(f"Cp     = {Cp_300K:.3e} cm^3 s^-1")
print(f"sigma  = {sigma_300K:.3e} Å^2")
print("===========================")

# ========================
# Save Cp(T)
# ========================

with open("Cp_T.dat", "w") as f:

    f.write("# T(K)      Cp(cm^3 s^-1)\n")

    for t, c in zip(T, Cp):
        f.write(f"{t:10.4f}  {c:.6e}\n")

print("Cp_T.dat saved.")