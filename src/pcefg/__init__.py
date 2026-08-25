"""pcefg: Point-Charge (PC) Electric Field Gradient (EFG) calculations in Python."""

__version__ = "0.1.0"

from src.pcefg.point_charge import (
    compute_efg,
    diagonalize_EFG,
    point_charge_EFG,
    sphere_radius_convergence,
    PointChargeEFG,
)


__all__ = [
    "compute_efg",
    "diagonalize_EFG",
    "point_charge_EFG",
    "sphere_radius_convergence",
    "PointChargeEFG",
]