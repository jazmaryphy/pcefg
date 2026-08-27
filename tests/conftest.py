import pytest
from ase import Atoms


@pytest.fixture
def simple_cubic():
    return Atoms(
        symbols=["Na", "Cl"],
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
        ],
        cell=[
            [5.64, 0.0, 0.0],
            [0.0, 5.64, 0.0],
            [0.0, 0.0, 5.64],
        ],
        pbc=True,
    )