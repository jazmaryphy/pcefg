"""Unit tests for the point-charge EFG calculator."""

import pytest
import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal
from src.pcefg.point_charge import PointChargeEFG
from src.pcefg.point_charge import compute_efg, point_charge_EFG


AL2O3_O_INDEX = 14


@pytest.fixture
def al2o3_calculator() -> PointChargeEFG:
    """Configured point-charge EFG calculator for alpha-Al2O3."""
    a, c = 4.754, 12.990
    atoms = crystal(
        symbols=["Al", "O"],
        basis=[(0, 0, 0.35228), (0.3064, 0, 0.25)],
        spacegroup=167,
        cellpar=[a, a, c, 90, 90, 120],
    )

    assert len(atoms) == 30

    # Assign formal charges (Al: +3.0, O: -2.0)
    charges = {"Al": 3.0, "O": -2.0}
    return PointChargeEFG(
        atoms=atoms, 
        charges=charges, 
        sphere_radius=50.0
    )


def test_al2o3_structure(al2o3_calculator: PointChargeEFG) -> None:
    """Check that the corundum structure has the expected composition."""

    atoms = al2o3_calculator.atoms

    assert len(atoms) == 30

    assert atoms.get_chemical_formula() == "Al12O18"

    assert np.count_nonzero(atoms.get_chemical_symbols() == np.array(["Al"] * len(atoms))) >= 0


def test_al2o3_composition(al2o3_calculator: PointChargeEFG) -> None:
    atoms = al2o3_calculator.atoms

    symbols = atoms.get_chemical_symbols()

    assert symbols.count("Al") == 12
    assert symbols.count("O") == 18



def test_al2o3_oxygen_efg(
    al2o3_calculator: PointChargeEFG,
) -> None:
    """Calculate the EFG at an oxygen site in alpha-Al2O3."""

    res = al2o3_calculator.compute_at(
        position=AL2O3_O_INDEX,
        coords_are_cartesian=False,
        verbose=False,
    )

    assert res["probe_index"] == AL2O3_O_INDEX
    assert res["probe_symbol"] == "O"

    assert np.isfinite(res["Vxx"])
    assert np.isfinite(res["Vyy"])
    assert np.isfinite(res["Vzz"])
    assert np.isfinite(res["eta"])


def test_al2o3_efg_tensor_is_symmetric(
    al2o3_calculator: PointChargeEFG,
) -> None:
    """The electrostatic EFG tensor must be symmetric."""

    res = al2o3_calculator.compute_at(
        position=AL2O3_O_INDEX,
        coords_are_cartesian=False,
        verbose=False,
    )

    tensor = res["EFG_tensor"]

    assert np.allclose(
        tensor,
        tensor.T,
        rtol=0.0,
        atol=1e-10,
    )


def test_al2o3_efg_is_traceless(
    al2o3_calculator: PointChargeEFG,
) -> None:
    """The EFG tensor must satisfy Laplace's equation."""

    res = al2o3_calculator.compute_at(
        position=AL2O3_O_INDEX,
        coords_are_cartesian=False,
        verbose=False,
    )

    tensor = res["EFG_tensor"]

    assert np.isclose(
        np.trace(tensor),
        0.0,
        rtol=1e-12,
        atol=1e8,
    )


def test_al2o3_principal_components_are_sorted(
    al2o3_calculator: PointChargeEFG,
) -> None:
    """Principal EFG components should be ordered by magnitude."""

    res = al2o3_calculator.compute_at(
        position=AL2O3_O_INDEX,
        coords_are_cartesian=False,
        verbose=False,
    )

    vxx, vyy, vzz = res["V_aa"]

    assert abs(vzz) >= abs(vyy) >= abs(vxx)


def test_al2o3_principal_components_are_eigenvalues(
    al2o3_calculator: PointChargeEFG,
) -> None:
    """Check that the reported principal components diagonalize the EFG."""

    res = al2o3_calculator.compute_at(
        position=AL2O3_O_INDEX,
        coords_are_cartesian=False,
        verbose=False,
    )

    tensor = res["EFG_tensor"]

    calculated = np.linalg.eigvalsh(tensor)
    reported = np.sort(res["V_aa"])

    assert np.allclose(
        calculated,
        reported,
        rtol=1e-10,
        atol=1e8,
    )


def test_al2o3_fractional_and_cartesian_coordinates_agree(
    al2o3_calculator: PointChargeEFG,
) -> None:
    """Fractional and Cartesian coordinates must give the same EFG."""

    atoms = al2o3_calculator.atoms

    frac = atoms.get_scaled_positions()[AL2O3_O_INDEX]
    cart = atoms.positions[AL2O3_O_INDEX]

    frac_result = al2o3_calculator.compute_at(
        position=frac,
        coords_are_cartesian=False,
        verbose=False,
    )

    cart_result = al2o3_calculator.compute_at(
        position=cart,
        coords_are_cartesian=True,
        verbose=False,
    )

    assert np.allclose(
        frac_result["EFG_tensor"],
        cart_result["EFG_tensor"],
        rtol=1e-10,
        atol=1e8,
    )


def test_al2o3_sternheimer_correction(
    al2o3_calculator: PointChargeEFG,
) -> None:
    """Check the multiplicative Sternheimer correction."""

    atoms = al2o3_calculator.atoms
    frac = atoms.get_scaled_positions()[AL2O3_O_INDEX]

    bare = al2o3_calculator.get_raw_tensor(
        frac,
        gamma_sternheimer=0.0,
    )

    corrected = al2o3_calculator.get_raw_tensor(
        frac,
        gamma_sternheimer=-2.2,
    )

    assert np.allclose(
        corrected,
        3.2 * bare,
        rtol=1e-10,
        atol=1e8,
    )


def test_al2o3_oxygen_regression(
    al2o3_calculator: PointChargeEFG,
) -> None:
    """Regression test for the published alpha-Al2O3 calculation."""

    res = al2o3_calculator.compute_at(
        position=AL2O3_O_INDEX,
        coords_are_cartesian=False,
        nuclear_spin=2.5,
        quadrupole_moment=-0.0265e-28,
        verbose=False,
    )

    assert np.isclose(
        res["Vxx"],
        7.247509699025467e20,
        rtol=2e-3,
    )

    assert np.isclose(
        res["Vyy"],
        3.0494015362241606e21,
        rtol=2e-3,
    )

    assert np.isclose(
        res["Vzz"],
        -3.7741525061267735e21,
        rtol=2e-3,
    )

    assert np.isclose(
        res["eta"],
        0.6159397540369369,
        rtol=2e-3,
    )

    assert np.isclose(
        res["nu_z_MHz"],
        0.3627529412726441,
        rtol=2e-3,
    )

    assert np.isclose(
        res["nu_Q_MHz"],
        0.38500728241417964,
        rtol=2e-3,
    )

    assert np.isclose(
        res["chi_Q_MHz"],
        2.4183529418176284,
        rtol=2e-3,
    )