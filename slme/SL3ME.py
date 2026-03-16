import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
from math import exp, pi

def calculate_SLME(material_eV_for_absorbance_data, material_absorbance_data,
                   material_direct_allowed_gap, material_indirect_gap,
                   thickness=50E-6, T=293.15, absorbance_in_inverse_centimeters=True,
                   cut_off_absorbance_below_direct_allowed_gap=True):

    if absorbance_in_inverse_centimeters:
        material_absorbance_data = material_absorbance_data * 100

    try:
        solar_spectra_wavelength, solar_spectra_irradiance = np.loadtxt(
            "am1.5G.dat", usecols=[0, 1], unpack=True, skiprows=2)
    except OSError:
        print('Could not locate am1.5G.dat datafile. Please place this file in the local directory.')
        sys.exit()

    solar_spectra_wavelength_m = solar_spectra_wavelength * 1e-9

    # Constants
    c = 299792458
    h = 6.62607004081E-34
    h_eV = 4.135667516E-15
    k = 1.3806485279E-23
    k_eV = 8.617330350E-5
    e = 1.602176620898E-19

    delta = material_direct_allowed_gap - material_indirect_gap
    fr = exp(-delta / (k_eV * T))

    solar_photon_flux = solar_spectra_irradiance * (solar_spectra_wavelength_m / (h * c))
    P_in = simpson(y=solar_spectra_irradiance, x=solar_spectra_wavelength)

    blackbody_irradiance = (2.0 * pi * h * c**2 / (solar_spectra_wavelength_m**5)) * \
                           (1.0 / (np.exp(h * c / (solar_spectra_wavelength_m * k * T)) - 1.0))
    blackbody_photon_flux = blackbody_irradiance * (solar_spectra_wavelength_m / (h * c))

    material_wavelength_nm = (c * h_eV / (material_eV_for_absorbance_data + 1e-10)) * 1e9
    
    # Remove duplicates to avoid interp1d error
    unique_wl, unique_indices = np.unique(material_wavelength_nm, return_index=True)
    material_wavelength_nm = unique_wl
    material_absorbance_data = material_absorbance_data[unique_indices]

    absorbance_func = interp1d(
        material_wavelength_nm, material_absorbance_data,
        kind='cubic',
        fill_value=(material_absorbance_data[0], material_absorbance_data[-1]),
        bounds_error=False
    )

    material_absorbance_interp = np.zeros(len(solar_spectra_wavelength))
    for i, wl in enumerate(solar_spectra_wavelength):
        if wl <= (c * h_eV / material_direct_allowed_gap) * 1e9 or not cut_off_absorbance_below_direct_allowed_gap:
            material_absorbance_interp[i] = absorbance_func(wl)

    absorbed_by_wavelength = 1.0 - np.exp(-2.0 * material_absorbance_interp * thickness)

    J_0_r = e * pi * simpson(y=blackbody_photon_flux * absorbed_by_wavelength, x=solar_spectra_wavelength_m)
    J_0 = J_0_r / fr

    J_sc = e * simpson(y=solar_photon_flux * absorbed_by_wavelength, x=solar_spectra_wavelength)

    def J(V):
        return J_sc - J_0 * (exp(e * V / (k * T)) - 1.0)

    def power(V):
        return J(V) * V

    def neg_power(V):
        return -power(V)

    result = minimize(neg_power, x0=[0.0001])
    V_Pmax = float(result.x)
    P_m = power(V_Pmax)

    efficiency = P_m / P_in
    return efficiency


if __name__ == '__main__':
    material_direct_allowed_gap = 1.73
    material_indirect_gap = 1.73
    file_name = "ABSORPTION.dat"

    try:
        material_eV_for_absorbance_data, material_absorbance_data = np.loadtxt(
            file_name, usecols=[0, 1], unpack=True)
    except OSError:
        print(f"Could not locate {file_name}. Please place this file in the local directory.")
        sys.exit()

    SLME = calculate_SLME(
        material_eV_for_absorbance_data=material_eV_for_absorbance_data,
        material_absorbance_data=material_absorbance_data,
        material_direct_allowed_gap=material_direct_allowed_gap,
        material_indirect_gap=material_indirect_gap,
        thickness=50E-6, T=293.15
    )

    print('File :', file_name)
    print('Standard SLME :', SLME)

    thickness_array = np.logspace(-8, -4)
    SLME_list = []
    for thickness in thickness_array:
        SLME_list.append(calculate_SLME(
            material_eV_for_absorbance_data=material_eV_for_absorbance_data,
            material_absorbance_data=material_absorbance_data,
            material_direct_allowed_gap=material_direct_allowed_gap,
            material_indirect_gap=material_indirect_gap,
            thickness=thickness, T=293.15
        ))

    # plt.figure(figsize=(3.2, 2.5), dpi=600)
    # plt.semilogx(thickness_array * 1e6, SLME_list)
    # plt.xlabel('Thickness ($\mu m$)')
    # plt.ylabel('SLME')
    # plt.minorticks_on()
    # ax = plt.gca()
    # ax.set_xlim(0.01, ax.get_xlim()[1])
    # ax.set_ylim(0, ax.get_ylim()[1])
    # plt.tight_layout(pad=0.5)
    # plt.show()
    
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    import numpy as np
    
    thickness_um = thickness_array * 1e6
    
    plt.figure(figsize=(6,4), dpi=150)
    plt.semilogx(thickness_um, SLME_list, color='royalblue', linewidth=2, label='SLME')
    
    plt.xlabel('Thickness (μm)', fontsize=12)
    plt.ylabel('SLME', fontsize=12)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10))
    ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.9)
    ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.legend(fontsize=10, loc='lower right')
    plt.xlim(left=0.01)
    plt.ylim(bottom=0)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=1.0)
    
    plt.show()