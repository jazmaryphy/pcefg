# %%
"""Point-charge lattice summation and Electric Field Gradient (EFG) tensor calculations."""

from typing import Any, Optional, Sequence, Tuple, Union
import matplotlib.axes
import numpy as np
import numpy.typing as npt
from ase import Atoms

from src.pcefg.utils import quadrupole_frequencies
from src.pcefg.lattice import check_charges_cover_atoms, get_site_labels
from src.pcefg.constants import ANGSTROM, ELEMENTARY_CHARGE, EPSILON0, H_PLANCK

# %%
def _replicate_lattice(
    atoms: Atoms,
    charges: dict[str, float],
    sphere_radius_m: float,
    exclude_indices: Sequence[int] = (),
    validate_charges: bool = True,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate positions and charges for periodic images within a cutoff sphere.

    Replicates unit cell positions across periodic boundaries to encompass a spherical
    cutoff region of radius `sphere_radius_m` around the origin.

    Args:
        atoms: Crystal structure containing atomic positions and cell parameters.
        charges: Mapping of element symbols or site labels to formal charges in units of e.
        sphere_radius_m: Spherical cutoff radius in meters.
        exclude_indices: 0-based indices of atoms in `atoms` to exclude from sum.
        validate_charges: If True, raises ValueError if `charges` does not cover all atoms.

    Returns:
        Tuple containing:
            - **pts**: Cartesian coordinates of replicated charges in meters, shape `(N, 3)`.
            - **qs**: Charge values in Coulombs, shape `(N,)`.

    Raises:
        ValueError: If `validate_charges=True` and any atom lacks an assigned charge.
    """
    if validate_charges:
        missing = check_charges_cover_atoms(atoms, charges=charges, strict=False)
        if missing:
            raise ValueError(
                f"ERROR: <charges={charges}> does not cover whole atoms/structure, "
                f"charges of: {missing} missing!"
            )
        # if missing:
        #     raise ValueError(
        #         f"Missing charge specification for site(s)/species: {missing}"
        #     )

    cell: npt.NDArray[np.float64] = np.array(atoms.get_cell())
    positions: npt.NDArray[np.float64] = atoms.get_positions()
    symbols: npt.NDArray[np.str_] = np.array(atoms.get_chemical_symbols())
    site_labels: npt.NDArray[np.str_] = np.array(get_site_labels(atoms))

    # Resolve each atom's charge ONCE (per-site charge takes priority,
    # falls back to per-element) -- this does NOT depend on which
    # periodic image an atom sits in
    q_per_atom: list[Optional[float]] = [
        charges.get(label, charges.get(sym))
        for label, sym in zip(site_labels, symbols)
    ]

    exclude_mask = np.zeros(len(atoms), dtype=bool)
    if exclude_indices:
        exclude_mask[list(exclude_indices)] = True

    nonzero_mask = np.array([q is not None and q != 0 for q in q_per_atom])
    keep_atom = nonzero_mask & ~exclude_mask

    if not np.any(keep_atom):
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)

    kept_positions = positions[keep_atom]
    kept_charges = (
        np.array([q_per_atom[i] for i in np.where(keep_atom)[0]], dtype=float)
        * ELEMENTARY_CHARGE
    )

    # how many unit cells to replicate in each direction to cover `sphere_radius`
    sphere_radius_ang = sphere_radius_m / ANGSTROM
    cell_norms = np.linalg.norm(cell, axis=1)
    n_reps = np.ceil(sphere_radius_ang / cell_norms).astype(int) + 1

    na = np.arange(-n_reps[0], n_reps[0] + 1)
    nb = np.arange(-n_reps[1], n_reps[1] + 1)
    nc = np.arange(-n_reps[2], n_reps[2] + 1)
    nx, ny, nz = np.meshgrid(na, nb, nc, indexing="ij")

    n_ints = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    shifts = np.dot(n_ints, cell)

    pts = (kept_positions[None, :, :] + shifts[:, None, :]).reshape(-1, 3) * ANGSTROM
    qs = np.tile(kept_charges, len(shifts))
    return pts, qs

# %%
def _efg_tensor_from_charges(
    pts: npt.NDArray[np.float64],
    qs: npt.NDArray[np.float64],
    site_position_m: npt.NDArray[np.float64],
    sphere_radius_m: float,
    gamma_sternheimer: float = 0.0,
    verbose: bool = False,
) -> npt.NDArray[np.float64]:
    """Compute 3x3 EFG tensor at a target site from predefined point charges.

    Args:
        pts: Point charge Cartesian coordinates in meters, shape `(N, 3)`.
        qs: Point charges in Coulombs, shape `(N,)`.
        site_position_m: Evaluation position in meters, shape `(3,)`.
        sphere_radius_m: Maximum distance cutoff in meters.
        gamma_sternheimer: Sternheimer antishielding factor.
        verbose: If True, prints diagnostic summation info.

    Returns:
        Symmetric 3x3 EFG tensor in V/m^2.
    """
    d = pts - site_position_m
    r2 = np.sum(d * d, axis=1)
    keep = (r2 > 1e-24) & (r2 < sphere_radius_m**2)

    d, r2, qs_k = d[keep], r2[keep], qs[keep]
    r5 = r2**2.5

    if verbose:
        print(
            f"Point-charge EFG: summing {len(qs_k)} charges "
            f"within radius {sphere_radius_m / ANGSTROM:.3f} Å."
        )

    w = qs_k / r5
    v_tensor = 3.0 * np.einsum("i,ia,ib->ab", w, d, d) - np.eye(3) * np.sum(w * r2)
    v_tensor *= 1.0 / (4.0 * np.pi * EPSILON0)
    v_tensor *= 1.0 - gamma_sternheimer
    return v_tensor


def point_charge_EFG(
    atoms: Atoms,
    site_position: npt.ArrayLike,
    charges: dict[str, float],
    sphere_radius: float = 50.0,
    exclude_indices: Sequence[int] = (),
    extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
    gamma_sternheimer: float = 0.0,
    verbose: bool = True,
) -> npt.NDArray[np.float64]:
    """Calculate point-charge EFG tensor [V/m^2] at a specified location.

    Args:
        atoms: Periodic crystal structure as an ASE `Atoms` instance.
        site_position: Evaluation Cartesian coordinate in Ångströms, shape `(3,)`.
        charges: Mapping of species or site labels to formal charges in units of e.
        sphere_radius: Spherical summation cutoff radius in Ångströms.
        exclude_indices: Indices of structure atoms to omit from summation.
        extra_charges: Sequence of additional `(position_angstrom, charge_e)` pairs.
        gamma_sternheimer: Sternheimer antishielding factor.
        verbose: If True, prints summation info.

    Returns:
        Symmetric traceless 3x3 EFG tensor in V/m^2.
    """
    sphere_radius_m = sphere_radius * ANGSTROM
    site_m = np.asarray(site_position, dtype=np.float64) * ANGSTROM

    pts, qs = _replicate_lattice(
        atoms, charges, sphere_radius_m, exclude_indices=exclude_indices
    )

    if extra_charges:
        extra_pts = np.array([p for p, _ in extra_charges], dtype=np.float64) * ANGSTROM
        extra_qs = np.array([q for _, q in extra_charges], dtype=np.float64) * ELEMENTARY_CHARGE
        pts = np.vstack([pts, extra_pts]) if len(pts) else extra_pts
        qs = np.concatenate([qs, extra_qs]) if len(qs) else extra_qs

    return _efg_tensor_from_charges(
        pts, qs, site_m, sphere_radius_m, gamma_sternheimer, verbose=verbose
    )


def diagonalize_EFG(
    tensor: npt.NDArray[np.float64],
    quadrupole_moment: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
    """Diagonalize EFG tensor into Principal Axis System (PAS) ordered components.

    Calculates principal values according to |Vzz| >= |Vyy| >= |Vxx|.

    Args:
        tensor: 3x3 symmetric EFG tensor in V/m^2.
        quadrupole_moment: Nuclear quadrupole moment Q in m^2.

    Returns:
        Tuple containing:
            - **V_aa**: Sorted principal components `[Vxx, Vyy, Vzz]` in V/m^2.
            - **P**: 3x3 matrix of normalized principal axes eigenvectors.
            - **chi_q**: Quadrupole coupling constant chi_Q in MHz.
            - **eta**: Asymmetry parameter eta in [0, 1].
    """
    evals, evecs = np.linalg.eigh(tensor)
    order = np.argsort(-np.abs(evals))
    v_zz, v_yy, v_xx = evals[order]
    v_aa = np.array([v_xx, v_yy, v_zz])

    p_matrix = evecs[:, order][:, ::-1]

    scale = np.abs(evals).max()
    eta = float(np.abs(v_xx - v_yy) / np.abs(v_zz)) if abs(v_zz) > 1e-6 * scale else 0.0

    chi_q = float(np.abs(v_zz * ELEMENTARY_CHARGE * quadrupole_moment / H_PLANCK))
    chi_q *= 1e-6  # Convert Hz to MHz
    return v_aa, p_matrix, chi_q, eta

# %%
def compute_efg(
    atoms: Atoms,
    probe_position: npt.ArrayLike,
    atomic_charges: dict[str, float],
    sphere_radius: float,
    gamma_sternheimer: float = 0.0,
    exclude_indices: Sequence[int] = (),
    extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
    coords_are_cartesian: bool = True,
    nuclear_spin: Optional[float] = None,
    quadrupole_moment: Optional[float] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Comprehensive EFG computation, PAS diagonalization, and frequency analysis.

    Args:
        atoms: Crystal structure as an ASE `Atoms` instance.
        probe_position: Evaluation coordinates (Cartesian or fractional).
        atomic_charges: Mapping of species/site labels to formal charges in units of e.
        sphere_radius: Summation sphere radius in Ångströms.
        gamma_sternheimer: Antishielding factor.
        exclude_indices: Indices of atoms to exclude from summation.
        extra_charges: Sequence of additional `(position, charge_e)` pairs.
        coords_are_cartesian: If True, `probe_position` is Cartesian; if False, fractional.
        nuclear_spin: Nuclear spin quantum number I.
        quadrupole_moment: Nuclear quadrupole moment Q in m^2.
        verbose: If True, prints formatted summary.

    Returns:
        Dictionary containing calculated EFG properties (`EFG_tensor`, `V_aa`, `eta`,
        `chi_Q_MHz`, `nu_z_MHz`, `nu_Q_MHz`, `principal_axes`, `probe_position`).
    """
    probe_pos_arr = np.asarray(probe_position, dtype=np.float64)
    if not coords_are_cartesian:
        cart_probe = np.dot(probe_pos_arr, atoms.get_cell())
        if extra_charges:
            extra_charges = [
                (np.dot(p, atoms.get_cell()).tolist(), q) for p, q in extra_charges
            ]
    else:
        cart_probe = probe_pos_arr

    tensor = point_charge_EFG(
        atoms,
        cart_probe,
        charges=atomic_charges,
        sphere_radius=sphere_radius,
        extra_charges=extra_charges,
        exclude_indices=exclude_indices,
        gamma_sternheimer=gamma_sternheimer,
        verbose=verbose,
    )

    v_aa = principal_axes = chi = eta = None
    v_xx = v_yy = v_zz = None
    nu_z = nu_q = None

    if quadrupole_moment is not None:
        v_aa, principal_axes, chi, eta = diagonalize_EFG(
            tensor, quadrupole_moment=quadrupole_moment
        )
        v_xx, v_yy, v_zz = v_aa

        if nuclear_spin is not None:
            props = quadrupole_frequencies(I=nuclear_spin, Q=quadrupole_moment, Vzz=v_zz, eta=eta)
            nu_z, nu_q = props.get("nu_z_MHz"), props.get("nu_Q_MHz")

    frac_pos = np.dot(cart_probe, np.linalg.inv(atoms.get_cell())) % 1.0

    results = {
        "Vxx": v_xx,
        "Vyy": v_yy,
        "Vzz": v_zz,
        "eta": eta,
        "V_aa": v_aa,
        "nu_z_MHz": nu_z,
        "nu_Q_MHz": nu_q,
        "chi_Q_MHz": chi,
        "EFG_tensor": tensor,
        "principal_axes": principal_axes,
        "probe_index": None,
        "probe_symbol": None,
        "probe_position": frac_pos,
    }

    if verbose:
        _pretty_print_efg(results=results)
    return results

# %%
def _pretty_print_efg(results: dict[str, Any]) -> None:
    """Pretty-prints EFG evaluation results table."""
    print("\n" + "=" * 70)
    probe_pos = results.get("probe_position")
    if probe_pos is not None:
        label = (
            f"atom {results['probe_index']} ({results['probe_symbol']})"
            if results.get("probe_index") is not None and results.get("probe_symbol") is not None
            else "probe site"
        )
        print(
            f"EFG analysis for {label} at frac coord. "
            f"({probe_pos[0]:.4f}, {probe_pos[1]:.4f}, {probe_pos[2]:.4f})"
        )
    print("=" * 70)

    scalar_fields = [
        ("Vzz", "V/m^2"),
        ("Vyy", "V/m^2"),
        ("Vxx", "V/m^2"),
        ("eta", "(unitless)"),
        ("chi_Q_MHz", "MHz"),
        ("nu_z_MHz", "MHz"),
        ("nu_Q_MHz", "MHz"),
    ]

    for key, unit in scalar_fields:
        val = results.get(key)
        if val is None:
            continue
        if key == "eta" or key in ("chi_Q_MHz", "nu_z_MHz", "nu_Q_MHz"):
            print(f"{key:<12} = {val: .8f} {unit}")
        else:
            print(f"{key:<12} = {val: .8e} {unit}")

    v_tensor = results.get("EFG_tensor")
    if v_tensor is not None:
        print("\nEFG tensor V_ab (V/m^2) =")
        print("-" * 70)
        for row in v_tensor:
            print(" [ " + ", ".join(f"{x: .8e}" for x in row) + " ]")
        print("-" * 70)
        print(f"Trace(V_ab) = {np.trace(v_tensor): .5e}")
        print(f"Symmetric   = {np.allclose(v_tensor, v_tensor.T)}")

    p_axes = results.get("principal_axes")
    if p_axes is not None:
        print("\nprincipal axes (unitless) = ")
        print("-" * 70)
        for row in p_axes:
            print(" [ " + ", ".join(f"{x: .8e}" for x in row) + " ]")
        print("-" * 70)
    print("=" * 70)

# %%
def sphere_radius_convergence(
    atoms: Atoms,
    site_position: npt.ArrayLike,
    charges: dict[str, float],
    exclude_indices: Sequence[int] = (),
    extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
    quadrupole_moment: float = 1.0e-28,
    sphere_radius_list: Optional[Sequence[float]] = None,
    gamma_sternheimer: float = 0.0,
    conv_thr: float = 1e-3,
    sphere_radius_step: float = 10.0,
    sphere_radius_max: float = 100.0,
    num_conv_streak: int = 3,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> Tuple[list[float], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate convergence of real-space EFG summation over varying radii.

    Args:
        atoms: Crystal structure as ASE `Atoms`.
        site_position: Target Cartesian evaluation position in Ångströms.
        charges: Atomic/site formal charge mapping in units of e.
        exclude_indices: Indices of atoms excluded from summation.
        extra_charges: Additional point charge tuples `(position, charge)`.
        quadrupole_moment: Quadrupole moment Q in m^2.
        sphere_radius_list: Initial list of radii to evaluate in Ångströms.
        gamma_sternheimer: Antishielding factor.
        conv_thr: Relative difference convergence threshold.
        sphere_radius_step: Increment step in Ångströms when extending radii.
        sphere_radius_max: Maximum search cutoff radius in Ångströms.
        num_conv_streak: Number of consecutive evaluations required to declare convergence.
        ax: Optional Matplotlib Axes to plot log relative error vs radius.

    Returns:
        Tuple containing:
            - **radii**: List of evaluated radii in Ångströms.
            - **vzz_values**: Array of Vzz values corresponding to radii.
            - **rel_errors**: Relative errors normalized against final estimate.
    """
    radii = list(sphere_radius_list) if sphere_radius_list is not None else [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]

    build_radius = max(sphere_radius_max + sphere_radius_step, max(radii))
    build_radius_m = build_radius * ANGSTROM
    pts_full, qs_full = _replicate_lattice(
        atoms, charges, build_radius_m, exclude_indices=exclude_indices
    )

    site_m = np.asarray(site_position, dtype=np.float64) * ANGSTROM

    if extra_charges:
        extra_pts = np.array([p for p, _ in extra_charges], dtype=np.float64) * ANGSTROM
        extra_qs = np.array([q for _, q in extra_charges], dtype=np.float64) * ELEMENTARY_CHARGE
        pts_full = np.vstack([pts_full, extra_pts]) if len(pts_full) else extra_pts
        qs_full = np.concatenate([qs_full, extra_qs]) if len(qs_full) else extra_qs

    def _vzz(r: float) -> float:
        r_m = r * ANGSTROM
        v_tensor = _efg_tensor_from_charges(
            pts_full, qs_full, site_m, r_m, gamma_sternheimer, verbose=False
        )
        v_aa, _, _, _ = diagonalize_EFG(v_tensor, quadrupole_moment=quadrupole_moment)
        return float(v_aa[2])

    vzz_values = [_vzz(r) for r in radii]

    def _is_converged() -> bool:
        if len(vzz_values) < num_conv_streak + 1:
            return False
        recent = vzz_values[-(num_conv_streak + 1):]
        diffs = [
            abs(recent[i + 1] - recent[i]) / max(abs(recent[i + 1]), 1e-300)
            for i in range(num_conv_streak)
        ]
        return all(d < conv_thr for d in diffs)

    while radii[-1] < sphere_radius_max and not _is_converged():
        next_r = radii[-1] + sphere_radius_step
        radii.append(next_r)
        vzz_values.append(_vzz(next_r))

    vzz_arr = np.array(vzz_values)
    best = vzz_arr[-1]
    rel_error = np.abs(vzz_arr - best) / max(abs(best), 1e-300)

    print(f"{'radius (Å)':>12} {'Vzz (V/m^2)':>16} {'rel. error vs best':>18}")
    for r, v, e in zip(radii, vzz_arr, rel_error):
        print(f"{r:>12.1f} {v:>16.4e} {e:>18.2e}")

    if not _is_converged() and radii[-1] >= sphere_radius_max:
        print(
            f"WARNING: reached sphere_radius_max={sphere_radius_max} without "
            f"{num_conv_streak} consecutive sustained-converged points."
        )

    if ax is not None:
        mask = rel_error > 0
        ax.semilogy(np.array(radii)[mask], rel_error[mask], "o-")
        ax.set_xlabel("sphere radius (Å)")
        ax.set_ylabel("relative error vs. largest-radius estimate")
        ax.set_title("EFG (Vzz): real-space sum convergence")
        ax.grid(True, which="both", alpha=0.3)

    return radii, vzz_arr, rel_error

# %%
class PointChargeEFG:
    """Calculator interface for evaluating Electric Field Gradients using point charges.

    Attributes:
        atoms (Atoms): The periodic crystal structure.
        charges (dict[str, float]): Formal atomic or site charges in units of e.
        sphere_radius (float): Default summation cutoff sphere radius in Ångströms.
        gamma_sternheimer (float): Sternheimer antishielding factor.
        exclude_indices (Tuple[int, ...]): Atomic indices excluded from lattice summations.
    """

    def __init__(
        self,
        atoms: Atoms,
        charges: dict[str, float],
        sphere_radius: float = 50.0,
        gamma_sternheimer: float = 0.0,
        exclude_indices: Sequence[int] = (),
    ) -> None:
        """Initialize the PointChargeEFG calculator."""
        self.atoms = atoms
        self.charges = charges
        self.sphere_radius = sphere_radius
        self.gamma_sternheimer = gamma_sternheimer
        self.exclude_indices = tuple(exclude_indices)

    def compute_at(
        self,
        position: npt.ArrayLike,
        coords_are_cartesian: bool = True,
        nuclear_spin: Optional[float] = None,
        quadrupole_moment: Optional[float] = None,
        extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Compute the EFG tensor and quadrupolar properties at a specific position."""
        return compute_efg(
            atoms=self.atoms,
            probe_position=position,
            atomic_charges=self.charges,
            sphere_radius=self.sphere_radius,
            gamma_sternheimer=self.gamma_sternheimer,
            exclude_indices=self.exclude_indices,
            extra_charges=extra_charges,
            coords_are_cartesian=coords_are_cartesian,
            nuclear_spin=nuclear_spin,
            quadrupole_moment=quadrupole_moment,
            verbose=verbose,
        )

    def get_raw_tensor(
        self,
        position: npt.ArrayLike,
        coords_are_cartesian: bool = True,
        extra_charges: Optional[Sequence[Tuple[npt.ArrayLike, float]]] = None,
    ) -> npt.NDArray[np.float64]:
        """Compute raw 3x3 EFG matrix [V/m^2] without full property extraction."""
        pos_arr = np.asarray(position, dtype=np.float64)
        if not coords_are_cartesian:
            pos_cart = np.dot(pos_arr, self.atoms.get_cell())
            if extra_charges:
                extra_charges = [
                    (np.dot(p, self.atoms.get_cell()).tolist(), q) for p, q in extra_charges
                ]
        else:
            pos_cart = pos_arr

        return point_charge_EFG(
            atoms=self.atoms,
            site_position=pos_cart,
            charges=self.charges,
            sphere_radius=self.sphere_radius,
            exclude_indices=self.exclude_indices,
            extra_charges=extra_charges,
            gamma_sternheimer=self.gamma_sternheimer,
            verbose=False,
        )

    def check_radius_convergence(
        self,
        position: npt.ArrayLike,
        coords_are_cartesian: bool = True,
        quadrupole_moment: float = 1.0e-28,
        sphere_radius_max: float = 100.0,
        sphere_radius_step: float = 10.0,
        conv_thr: float = 1e-3,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> Tuple[list[float], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Evaluate how Vzz converges relative to real-space summation radius."""
        pos_arr = np.asarray(position, dtype=np.float64)
        pos_cart = pos_arr if coords_are_cartesian else np.dot(pos_arr, self.atoms.get_cell())

        return sphere_radius_convergence(
            atoms=self.atoms,
            site_position=pos_cart,
            charges=self.charges,
            exclude_indices=self.exclude_indices,
            quadrupole_moment=quadrupole_moment,
            gamma_sternheimer=self.gamma_sternheimer,
            conv_thr=conv_thr,
            sphere_radius_step=sphere_radius_step,
            sphere_radius_max=sphere_radius_max,
            ax=ax,
        )