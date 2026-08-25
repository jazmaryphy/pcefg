# pcefg: Point-Charge (PC) Model for the Electric Field Gradient (EFG)

`pcefg` is a Python package for computing the Electric Field Gradient (EFG) tensor, asymmetry parameters ($\eta$), and quadrupolar coupling constants ($\chi_Q$) in crystal structures using a classical point-charge model. It serves as a lightweight, fast alternative or complementary approach to First-Principles/Density Functional Theory (DFT) calculations.

---

## Features

- **Fast EFG Tensor Calculation**: Computes lattice EFG tensors via direct lattice summation.
- **Sternheimer Antishielding**: Supports polarization corrections via $(1-\gamma_\infty)$.
- **ASE Integration**: Works directly with Atomic Simulation Environment (`ase.Atoms`) structures.
- **Crystalline Symmetry Support**: Automatically handles spacegroup site labels and site-specific charge specifications.
- **Quadrupole Coupling Utilities**: Calculates $V_{zz}$, $\eta$, and quadrupolar coupling constants ($\chi_Q$) for arbitrary spin $I > 1/2$ nuclei.

---

## Theoretical Background

### Point-Charge EFG Model

The electrostatic potential at a probe position $\mathbf{r}_0$ is:

$$V(\mathbf{r}_0)=\frac{1}{4\pi\varepsilon_0} \int \frac{\rho(\mathbf{r}')}{\vert{}\mathbf{r}'-\mathbf{r}_0\vert{}}\,d\tau'$$

Assuming a collection of point charges $\rho(\mathbf{r})=\sum_k q_k\delta(\mathbf{r}-\mathbf{r}_k)$, the potential simplifies to:

$$V(\mathbf{r}_0)= \frac{1}{4\pi\varepsilon_0} \sum_k \frac{q_k}{R_k}$$

where $\mathbf{R}_k = \mathbf{r}_0 - \mathbf{r}_k$ and $R_k = \vert{}\mathbf{R}_k\vert{}$.

The EFG tensor is defined as the Hessian of the electrostatic potential:

$$V_{ij} = \frac{\partial^2 V}{\partial x_i\partial x_j}$$

Evaluating the partial derivatives yields the explicit sum over point charges:

$$V_{ij} = \frac{1}{4\pi\varepsilon_0} \sum_k q_k \left( \frac{3R_{k,i}R_{k,j}-\delta_{ij}R_k^2}{R_k^5} \right)$$

where $\delta_{ij}$ is the Kronecker delta.

### Sternheimer Antishielding Correction

To account for the polarization of the core electronic cloud surrounding the probe nucleus, the lattice EFG is scaled using the Sternheimer antishielding factor $\gamma_\infty$:

$$V_{ij}^{\mathrm{total}} = (1-\gamma_\infty)\, V_{ij}^{\mathrm{lattice}}$$

### Quadrupolar Interaction

For nuclei with spin $I > \frac{1}{2}$, the electric quadrupole interaction contribution to the Hamiltonian is:

$$\hat{\mathcal{H}}_Q = \sum_{i}^{N_{\mathrm{nuc}}}\frac{eQ^i(1-\gamma_\infty^i)}{\hbar\,2I(2I-1)} \sum_{\alpha\beta} V_{\alpha\beta}^{i} \hat{I}_\alpha^i \hat{I}_\beta^i$$

where:
- $Q^i$ is the $i$-th nuclear electric quadrupole moment.
- $V_{\alpha\beta}^{i}$ is the external EFG tensor at the site of the $i$-th quadrupolar nucleus.
- $\hat{I}_\alpha^i$ are the nuclear spin operators.

Diagonalization of the EFG tensor yields its principal components $(V_{xx}, V_{yy}, V_{zz})$, ordered by magnitude:

$$\vert{}V_{zz}\vert{} \ge \vert{}V_{yy}\vert{} \ge \vert{}V_{xx}\vert{}$$

From these components, the asymmetry parameter $\eta$ and quadrupolar coupling constant $\chi_Q$ are calculated:

$$\eta = \frac{V_{xx} - V_{yy}}{V_{zz}}, \qquad \chi_Q = \frac{e Q V_{zz}}{h}$$

---

## Installation

### From Source
```bash
git clone [https://github.com/jazmaryphy/pcefg.git](https://github.com/jazmaryphy/pcefg.git)
cd pcefg
pip install .
```

## Usage

```python
import numpy as np
from ase.build import bulk
from pcefg.point_charge import compute_efg

# 1. Create structure (e.g., Rocksalt NaCl)
atoms = bulk("NaCl", "rocksalt", a=5.64)

# 2. Define formal oxidation states
charges = {"Na": +1.0, "Cl": -1.0}

# 3. Calculate EFG at the Na site [0, 0, 0]
probe_pos = [0.0, 0.0, 0.0]

result = compute_efg(
    atoms=atoms,
    probe_position=probe_pos,
    atomic_charges=charges,
    sphere_radius=30.0,            # Cutoff radius in Angstroms
    gamma_sternheimer=-5.5,         # Sternheimer factor
    coords_are_cartesian=False,     # Fractional coordinates
    nuclear_spin=1.5,              # 3/2 spin for 23Na
    quadrupole_moment=+0.104e-28,  # Q in m^2
    verbose=True
)
```

## Examples

For more usage see example folder
```bash
cd examples
```