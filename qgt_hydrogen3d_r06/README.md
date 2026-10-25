# QGT Hydrogen3D r06 — Guided Physical Hydrogen

This revision replaces the legacy single-mode Hydrogen3D narrative with three explicitly typed stages.

## 1. PREPARE — numerical bound-state preparation
A broad trial wavefunction is propagated in **imaginary time** under

\[
H_0=-\frac12\nabla^2-\frac1\rho.
\]

This is an eigenstate-projection / numerical preparation procedure. It is **not** presented as physical radiative recombination. The independent Python reference raises the 1s fidelity from about 0.5145 to 0.9767 over the displayed preparation interval.

## 2. LIVE — life of a formed atom
A coherent bound-state superposition evolves in the Coulomb sector. The visual encoding is:
- brightness = \(|\psi|^2\),
- hue = local phase \(\arg\psi\),
- filaments = probability current,
- white point = fixed proton/potential centre.

There is no classical electron orbit.

## 3. LIGHT — explicit atom–light interaction
The r06 baseline replaces the old radial breathing proxy with an explicit dipole interaction

\[
H(\tau)=H_0-F(\tau)\rho_z,
\]

using a near-resonant pulse at \(\omega=3/8\) in scaled hydrogen units. The independent reference reaches a maximum \(2p_z\) population of about 0.253 while conserving norm to ~1.7e-13.

## Epistemic firewall
- Radiative continuum capture \(e^-+p^+\to H+\gamma\) is **not** simulated in r06.
- PREPARE is numerical eigenprojection, not physical time.
- LIVE and LIGHT are real-time Schrödinger-grade dynamics.
- Absolute \(\alpha_Q\) and \(\mu_Q\) calibrate units but are not needed for the dimensionless dynamics.
- Legacy golden-clock/breathing terms are not used in the r06 baseline.

## Files
- `index.html` — guided web experience with 4 introductory slides and the 3-mode live simulator.
- `hydrogen3d_guided_engine_r06.js` — live 64^3 browser engine and renderer.
- `qgt_hydrogen3d_r06_reference.py` — independent 48^3 Python reference/verifier.
- `data/summary.json` — quantitative summary.
- `data/preparation.json`, `life.json`, `light.json` — observables by frame.
- `data/slices.npz` — reference central-slice data.
