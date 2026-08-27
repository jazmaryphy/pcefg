"""Unit tests for the point-charge EFG calculator."""

import pytest
import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal
from src.pcefg.point_charge import PointChargeEFG
from src.pcefg.point_charge import compute_efg, point_charge_EFG, diagonalize_EFG


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


def test_point_charge_efg_returns_3x3_tensor(
    nacl_calculator: PointChargeEFG,
) -> None:
    """Test that point_charge_EFG and compute_efg return a 3x3 array directly."""
    site_frac = [0.1, 0.25, 0.3]

    tensor_direct = point_charge_EFG(
        atoms=nacl_calculator.atoms,
        site_position=site_frac,
        charges=nacl_calculator.charges,
        sphere_radius=nacl_calculator.sphere_radius,
        coords_are_cartesian=False,
        verbose=False
    )

    tensor_alias = compute_efg(
        atoms=nacl_calculator.atoms,
        probe_position=site_frac,
        atomic_charges=nacl_calculator.charges,
        sphere_radius=nacl_calculator.sphere_radius,
        coords_are_cartesian=False,
        verbose=False
    )

    # 1. Check array shape & dimension
    assert isinstance(tensor_direct, np.ndarray), "Output must be a numpy ndarray"
    assert tensor_direct.shape == (3, 3), "EFG tensor must be a 3x3 matrix"

    # 2. Verify compute_efg alias returns identical 3x3 array
    np.testing.assert_allclose(tensor_direct, tensor_alias["EFG_tensor"], rtol=1e-8)



def test_diagonalize_efg_with_and_without_q_moment(
    nacl_calculator: PointChargeEFG,
) -> None:
    """Test diagonalize_EFG tuple return with and without quadrupole moment."""
    site_frac = [0.1, 0.25, 0.3]

    tensor = point_charge_EFG(
        atoms=nacl_calculator.atoms,
        site_position=site_frac,
        charges=nacl_calculator.charges,
        sphere_radius=nacl_calculator.sphere_radius,
        coords_are_cartesian=False,
        verbose=False
    )

    q_moment = 0.055e-28  # 23Na quadrupole moment in m^2

    # --- Case 1: Quadrupole moment PROVIDED ---
    v_aa_q, p_matrix_q, chi_q_val, eta_q = diagonalize_EFG(tensor, quadrupole_moment=q_moment)

    assert isinstance(v_aa_q, np.ndarray) and v_aa_q.shape == (3,)
    assert isinstance(p_matrix_q, np.ndarray) and p_matrix_q.shape == (3, 3)
    assert isinstance(chi_q_val, float)
    assert isinstance(eta_q, float)
    assert 0.0 <= eta_q <= 1.0

    # Verify sorting rule: |Vzz| >= |Vyy| >= |Vxx| (where v_aa = [Vxx, Vyy, Vzz])
    vxx, vyy, vzz = v_aa_q
    assert abs(vzz) >= abs(vyy) - 1e-6 * abs(vzz)
    assert abs(vyy) >= abs(vxx) - 1e-6 * abs(vzz)

    # --- Case 2: Quadrupole moment OMITTED / None ---
    v_aa_no_q, p_matrix_no_q, chi_q_none, eta_no_q = diagonalize_EFG(tensor, quadrupole_moment=None)

    assert chi_q_none is None

    # V_aa, P matrix, and eta must match identically regardless of quadrupole_moment argument
    np.testing.assert_allclose(v_aa_q, v_aa_no_q, rtol=1e-6)
    np.testing.assert_allclose(p_matrix_q, p_matrix_no_q, rtol=1e-6)
    assert np.isclose(eta_q, eta_no_q, rtol=1e-6)



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



def test_missing_charges_raises_error(nacl_calculator: PointChargeEFG) -> None:
    """Test that the calculator raises a ValueError if an atom is missing a charge."""
    # Missing 'Cl' charge
    bad_charges = {"Na": +1.0}
    calc = PointChargeEFG(atoms=nacl_calculator.atoms, charges=bad_charges)

    with pytest.raises(ValueError, match="does not cover"):
        calc.compute_at([0, 0, 0], coords_are_cartesian=False)



def _test_missing_charges_raises_error() -> None:
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