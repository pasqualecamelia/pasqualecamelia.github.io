# QGT Hydrogen3D — Bilinear Atom–Light Readout + Coulomb/QGT Comparator

Public web build for the QGT Hydrogen3D simulator. This patch preserves the r07 bilinear/Walsh readout and adds a synchronized **Coulomb / QGT / Δρ** comparison mode.

## Live modes

1. **PREPARE** — imaginary-time ground-state preparation (numerical eigenprojection, not physical capture).
2. **LIVE ATOM** — coherent real-time bound-state evolution.
3. **LIGHT** — explicit near-resonant dipole coupling `H = H0 - F(t) z`.
4. **BILINEAR / WALSH 11** — four coherent TDSE branches `++`, `+-`, `-+`, `--`, with `beta11 = 1/4 Arg[(M++ M--)/(M+- M-+)]`.
5. **COULOMB ↔ QGT** — two synchronized copies start from the **same normalized Coulomb 1s grid array**, use the same grid, timestep, kinetic propagator and normalization, and differ only in the potential.

### Comparison potential

The QGT branch reproduces the structural ansatz already present in the repository `qgt_hydrogen3d.py`:

`V_QGT(r,t) = -1/r + alpha*M_eff*sech^2(r/ell_pen) + f_t*cos(2*Gamma_tau*t)*exp(-r^2/2)`

with the constants constructed in that source. The Coulomb control is `V_C=-1/r`. The view selector switches among `COULOMB`, `QGT`, and the signed density difference

`Delta rho(r,t) = |psi_QGT|^2 - |psi_C|^2`.

The scalar `D_rho = 1/2 sum_grid |Delta rho|` is the discrete-grid total-variation distance. The state-overlap fidelity `|<psi_C|psi_QGT>|^2` is shown separately because density agreement does not imply phase agreement.

**Epistemic status:** the QGT potential deformation is displayed as a declared structural model/ansatz already present in the repository. This comparator does **not** claim that it is an empirically established correction to hydrogen.

## Precision layer

`data/beta11_reference.json` contains the independent partial-wave/Cayley benchmark and Sternheimer static polarizability from `QGT_BETA11_SAME_ACTION_r01`. This benchmark remains separate from the new Coulomb/QGT visualization.

## Deployment

Upload the **contents** of this folder over `/qgt_hydrogen3d_r07/` to keep the existing public URL unchanged.

## Main files

- `index.html`
- `hydrogen3d_bilinear_engine_r07.js`
- `assets/qgt_hydrogen3d_beta11_dashboard.png`
- `data/beta11_reference.json`
- `data/hadamard_closure_reference.json`
- `downloads/QGT_BETA11_SAME_ACTION_r01.zip`
- `downloads/QGT_MINIMAL_QUBIT_HADAMARD_CLOSURE_r01.zip`
