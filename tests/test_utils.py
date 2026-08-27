"""Unit tests for the point-charge EFG calculator."""

import pytest
import numpy as np
from src.pcefg.constants import ELEMENTARY_CHARGE, EPSILON0
from src.pcefg.utils import (
    Vzz_for_unit_charge_at_distance,
    gen_radial_EFG,
)


def test_vzz_for_unit_charge():
    r = 1.0e-10

    expected = (
        2.0
        / (4.0 * np.pi * EPSILON0)
        * ELEMENTARY_CHARGE
        / r**3
    )

    result = Vzz_for_unit_charge_at_distance(r)

    assert np.isclose(result, expected, rtol=1e-12)


def test_vzz_rejects_zero_distance():
    with pytest.raises(ValueError):
        Vzz_for_unit_charge_at_distance(0.0)


def test_vzz_rejects_negative_distance():
    with pytest.raises(ValueError):
        Vzz_for_unit_charge_at_distance(-1.0)


def test_gen_radial_efg_along_z():
    charge_position = np.array([0.0, 0.0, 0.0])
    site_position = np.array([0.0, 0.0, 1.0e-10])

    tensor = gen_radial_EFG(
        charge_position=charge_position,
        site_position=site_position,
    )

    assert tensor.shape == (3, 3)

    assert np.allclose(
        tensor,
        tensor.T,
        atol=1e-10,
    )

    assert np.isclose(
        np.trace(tensor),
        0.0,
        atol=1e-10,
    )

    assert np.isclose(
        tensor[0, 0],
        -0.5 * tensor[2, 2],
    )

    assert np.isclose(
        tensor[1, 1],
        -0.5 * tensor[2, 2],
    )


def test_gen_radial_efg_rejects_coincident_positions():
    position = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        gen_radial_EFG(
            charge_position=position,
            site_position=position,
        )