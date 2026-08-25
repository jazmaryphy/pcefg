# %%
"""Physical constants and unit conversion factors for point-charge EFG calculations."""

from typing import Dict
import numpy as np
import scipy.constants as const

# %%
# Gyromagnetic ratios (rad / (s * T))
GAMMAS: Dict[str, float] = {
    "mu": 2.0 * np.pi * 135.53881e6,
    "F": 2.0 * np.pi * 40.053e6,
    "H": 2.0 * np.pi * 42.577e6,
    "V": 2.0 * np.pi * 11.212944e6,
}

# %%
# Mathematical constants
PI: float = np.pi
TWOPI: float = 2.0 * np.pi

# Fundamental SI constants
MU0: float = const.mu_0  # Vacuum magnetic permeability (T m A^-1)
EPSILON0: float = const.epsilon_0  # Vacuum electric permittivity (F m^-1)
H_PLANCK: float = const.h  # Planck constant (J s)
HBAR: float = const.hbar  # Reduced Planck constant (J s)
ELEMENTARY_CHARGE: float = const.e  # Elementary charge (C)
BOHR_MAGNETON: float = const.value("Bohr magneton")  # J T^-1
BOHR_RADIUS: float = const.value("Bohr radius")  # m
BOLTZMANN_CONSTANT: float = const.k  # J K^-1
AVOGADRO_CONSTANT: float = const.N_A  # mol^-1
SPEED_OF_LIGHT: float = const.c  # m s^-1

# Distance conversion factors
ANGSTROM: float = 1e-10  # meters
ANGTOM: float = ANGSTROM
BOHR_TO_ANGSTROM: float = BOHR_RADIUS / ANGSTROM

# Energy & unit relationships
EFG_AMU_TO_SI: float = 9.7173624424e21  # V m^-2
HARTREE_ENERGY: float = const.value("Hartree energy")  # J
HARTREE_ENERGY_EV: float = const.value("Hartree energy in eV")  # eV
ELECTRON_VOLT_JOULE: float = const.eV  # J

# Derived interaction constants
MUON_GYROMAGNETIC_RATIO: float = GAMMAS["mu"]  # rad s^-1 T^-1
NUCLEAR_MAGNETON_OVER_HBAR: float = (
    TWOPI * 7.622593285e6
)  # Converts nuclear g-factor to gyromagnetic ratio (rad s^-1 T^-1)

# Second moment prefactor: (2/3) * (mu_0 / 4pi)^2 * hbar^2 * gamma_mu^2
SECOND_MOMENT_PREFACTOR: float = (
    (2.0 / 3.0)
    * (MU0 / (4.0 * np.pi)) ** 2
    * HBAR**2
    * MUON_GYROMAGNETIC_RATIO**2
)

# Fermi contact field prefactor: converts atomic unit spin density (e/bohr^3) to Tesla
# B_c = (2/3) * mu_0 * mu_B * delta_s(0) / a0^3
SPIN_DENSITY_AU_TO_TESLA: float = (
    (2.0 / 3.0) * MU0 * BOHR_MAGNETON / (BOHR_RADIUS**3)
)