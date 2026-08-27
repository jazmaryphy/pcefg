"""Unit tests for the point-charge EFG calculator."""

import pytest
import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal
from src.pcefg.point_charge import PointChargeEFG
from src.pcefg.point_charge import compute_efg, point_charge_EFG, diagonalized_EFG


@pytest.fixture
def nacl_calculator() -> PointChargeEFG:
    """Fixture providing a configured calculator for rocksalt NaCl."""
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    a = 5.640
    atoms = crystal(
        symbols=["Na", "Cl"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5)], 
        spacegroup=225, # Fm-3m
        cellpar=[a, a, a, 90, 90, 90],
    )
    charges = {"Na": +1.0, "Cl": -1.0}
    return PointChargeEFG(
        atoms=atoms, 
        charges=charges, 
        sphere_radius=50.0
    )


def test_cubic_symmetry_zero_efg(nacl_calculator: PointChargeEFG) -> None:
    """Test that EFG at a cubic symmetry site is zero."""
    res = nacl_calculator.compute_at(
        position=[0, 0, 0], 
        coords_are_cartesian=False,
        verbose=False
    )
    
    # In cubic symmetry, Vzz should be zero up to numerical cutoff noise (~1e8 V/m^2)
    assert np.isclose(res["Vzz"], 0.0, atol=1e+8), (
        f"Vzz is {res['Vzz']:.2e}, which breaks cubic symmetry!"
    )


def test_tensor_properties_laplace(nacl_calculator: PointChargeEFG) -> None:
    """Test that the EFG tensor is symmetric and traceless at an off-center site."""
    res = nacl_calculator.compute_at(
        position=[0.1, 0.25, 0.3], 
        coords_are_cartesian=False,
        verbose=False
    )
    tensor = res["EFG_tensor"]

    max_element = float(np.max(np.abs(tensor)))

    # 1. Check symmetry: V_ab == V_ba with scaled absolute tolerance for SI units (~10^21 V/m^2)
    assert np.allclose(
        tensor, 
        tensor.T, 
        rtol=1e-5, 
        atol=1e-8 * max_element
    ), "EFG tensor must be symmetric"
    
    # 2. Check Laplace equation in vacuum: Trace(V) == 0
    assert np.isclose(
        np.trace(tensor), 
        0.0, 
        rtol=1e-5, 
        atol=1e-8 * max_element
    ), "EFG tensor trace must be zero"
    
    # 3. Check Vzz is dominant principal component: |Vzz| >= |Vyy| >= |Vxx|
    vxx, vyy, vzz = res["V_aa"]
    assert abs(vzz) >= abs(vyy) - 1e-8 * max_element, "Vzz is not >= Vyy"
    assert abs(vyy) >= abs(vxx) - 1e-8 * max_element, "Vyy is not >= Vxx"


def test_missing_charges_raises_error() -> None:
    """Test that the calculator raises a ValueError if an atom is missing a charge."""
    a = 5.640
    atoms = crystal(
        symbols=["Na", "Cl"],
        basis=[(0, 0, 0), (0.5, 0.5, 0.5)], 
        spacegroup=225,  # Fm-3m
        cellpar=[a, a, a, 90, 90, 90],
    )
    # Missing 'Cl' charge
    bad_charges = {"Na": +1.0}
    calc = PointChargeEFG(atoms=atoms, charges=bad_charges)

    with pytest.raises(ValueError, match="does not cover"):
        calc.compute_at([0, 0, 0], coords_are_cartesian=False)