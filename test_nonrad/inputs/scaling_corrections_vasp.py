#!/usr/bin/env python

from pathlib import Path
import sys
import numpy as np

# --------------------------------------------------
# local nonrad source
# --------------------------------------------------

NONRAD_SRC = "/home/student/soft/nonrad-1.2.0"

if NONRAD_SRC not in sys.path:
    sys.path.insert(0, NONRAD_SRC)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from nonrad.scaling import (
    sommerfeld_parameter,
    charged_supercell_scaling_VASP
)

# --------------------------------------------------
# paths
# --------------------------------------------------

case_root = Path.cwd()

outdir = case_root
outdir.mkdir(parents=True, exist_ok=True)

wavecar = case_root / "relax" / "excited_dir" / "WAVECAR"

# --------------------------------------------------
# material parameters
# --------------------------------------------------

Z = -1
m_eff = 0.18
eps_static = 8.9

# --------------------------------------------------
# temperature range
# --------------------------------------------------

T_single = 300.0

Tmin = 25.0
Tmax = 800.0
nT = 1000

# --------------------------------------------------
# band settings
# --------------------------------------------------

band_index = 189
def_index = 192

# ISPIN=2:
# spin=0 -> spin-up
# spin=1 -> spin-down

spin = 1

kpoint = 1

# --------------------------------------------------
# scaling settings
# --------------------------------------------------

cutoff = 0.02
limit = 5.0
full_range = False

tag = "GaN_C_N"

# ==================================================
# Sommerfeld factor
# ==================================================

report = []

report.append("===== nonrad scaling corrections report =====")
report.append(f"case_root = {case_root}")
report.append("")

f_single = sommerfeld_parameter(
    T_single,
    Z,
    m_eff,
    eps_static
)

T_grid = np.linspace(Tmin, Tmax, nT)

f_grid = sommerfeld_parameter(
    T_grid,
    Z,
    m_eff,
    eps_static
)

report.append("===== Sommerfeld factor =====")
report.append(f"Z = {Z}")
report.append(f"m_eff = {m_eff}")
report.append(f"eps_static = {eps_static}")
report.append(f"Sommerfeld factor @ {T_single} K = {float(f_single):.8f}")
report.append("")

csv_path = outdir / f"{tag}_sommerfeld_vs_T.csv"

np.savetxt(
    csv_path,
    np.column_stack([T_grid, f_grid]),
    delimiter=",",
    header="T_K,sommerfeld_factor",
    comments=""
)

fig, ax = plt.subplots(figsize=(6,5))

ax.plot(T_grid, f_grid, lw=2)

ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Sommerfeld factor")

ax.grid(True, ls="--", alpha=0.5)

fig.tight_layout()

fig_path = outdir / f"{tag}_sommerfeld_vs_T.png"

fig.savefig(fig_path, dpi=600)

plt.close(fig)

report.append("===== Charged-supercell scaling =====")
report.append(f"WAVECAR = {wavecar}")
report.append(f"band_index = {band_index}")
report.append(f"def_index = {def_index}")
report.append(f"spin = {spin}")
report.append("")

fig = plt.figure(figsize=(12,5))

factor = charged_supercell_scaling_VASP(
    str(wavecar),
    band_index,
    def_index=def_index,
    spin=spin,
    kpoint=kpoint,
    cutoff=cutoff,
    limit=limit,
    full_range=full_range,
    fig=fig
)

plt.tight_layout()

fig_path = outdir / f"{tag}_charged_supercell_scaling.png"

fig.savefig(fig_path, dpi=600)

plt.close(fig)

scaling = 1.0 / factor

report.append(f"raw_factor = {float(factor):.12f}")
report.append(f"recommended_scaling = {float(scaling):.12f}")
report.append("")

report_path = outdir / f"{tag}_scaling_report.txt"

report_path.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print(report_path.read_text())

print(f"Wrote report: {report_path}")