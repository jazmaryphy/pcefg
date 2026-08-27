"""Unit tests for the point-charge EFG calculator."""

import pytest
import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal
from src.pcefg.point_charge import PointChargeEFG
from src.pcefg.point_charge import compute_efg, point_charge_EFG


@pytest.fixture
def nacl_calculator() -> PointChargeEFG:
    """Fixture providing a configured calculator for rocksalt NaCl."""
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    charges = {"Na": +1.0, "Cl": -1.0}
    return PointChargeEFG(
        atoms=atoms, 
        charges=charges, 
        sphere_radius=30.0
    )


def test_cubic_symmetry_zero_efg(nacl_calculator: PointChargeEFG) -> None:
    """Test that EFG at a cubic symmetry site is zero."""
    # Fractional [0, 0, 0] is the Na site
    res = nacl_calculator.compute_at(
        position=[0, 0, 0], 
        coords_are_cartesian=False,
        verbose=False
    )
    
    assert np.isclose(res["Vzz"], 0.0, atol=1e-8), "Vzz should be 0 for cubic symmetry"
    assert np.isclose(res["eta"], 0.0, atol=1e-8), "eta should be 0"


# def test_tensor_properties_laplace(nacl_calculator: PointChargeEFG) -> None:
#     """Test that the EFG tensor is symmetric and traceless at an off-center site."""
#     res = nacl_calculator.compute_at(
#         position=[0.1, 0.25, 0.3], 
#         coords_are_cartesian=False,
#         verbose=False
#     )
#     tensor = res["EFG_tensor"]

#     # remove numerical noise
#     noise_threshold=1e-8
#     max_element = np.max(np.abs(tensor))
#     if max_element > 0:
#         tensor[np.abs(tensor) < noise_threshold * max_element] = 0.0

#     # 1. Check symmetry: V_ab == V_ba
#     assert np.allclose(tensor, tensor.T, atol=1e-10), "EFG tensor must be symmetric"
    
#     # 2. Check Laplace equation in vacuum: Trace(V) == 0
#     assert np.isclose(np.trace(tensor), 0.0, atol=1e-10), "EFG tensor trace must be zero"
    
#     # 3. Check Vzz is dominant principal component
#     vxx, vyy, vzz = res["V_aa"]
#     assert abs(vzz) >= abs(vyy) >= abs(vxx), "Principal components are not sorted correctly"


# def test_missing_charges_raises_error() -> None:
#     """Test that the calculator raises a ValueError if an atom is missing a charge."""
#     atoms = bulk("NaCl", "rocksalt", a=5.64)
#     # Missing 'Cl' charge
#     bad_charges = {"Na": +1.0}
#     calc = PointChargeEFG(atoms=atoms, charges=bad_charges)

#     with pytest.raises(ValueError, match="does not cover"):
#         calc.compute_at([0, 0, 0], coords_are_cartesian=False)