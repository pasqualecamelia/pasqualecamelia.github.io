# QGT Hydrogen3D — Bilinear Atom–Light Readout

Public web build for the QGT Hydrogen3D simulator.

## Live modes

1. **PREPARE** — imaginary-time ground-state preparation (numerical eigenprojection, not physical capture).
2. **LIVE ATOM** — coherent real-time bound-state evolution.
3. **LIGHT** — explicit near-resonant dipole coupling `H = H0 - F(t) z`.
4. **BILINEAR / WALSH 11** — four coherent TDSE branches `++`, `+-`, `-+`, `--`, with
   `beta11 = 1/4 Arg[(M++ M--)/(M+- M-+)]`.

The third diagnostic pane switches from the ordinary `yz` density slice to the sign-normalized mixed `XZ` response in BILINEAR mode.

## Precision layer

`data/beta11_reference.json` contains the independent partial-wave/Cayley benchmark and Sternheimer static polarizability from `QGT_BETA11_SAME_ACTION_r01`.

## GitHub Pages deployment

Upload the **contents** of this folder to:

`/qgt_hydrogen3d_r07/`

Do not wrap it inside another directory.

The root homepage should link to:

`qgt_hydrogen3d_r07/`

## Main files

- `index.html`
- `hydrogen3d_bilinear_engine_r07.js`
- `assets/qgt_hydrogen3d_beta11_dashboard.png`
- `data/beta11_reference.json`
- `data/hadamard_closure_reference.json`
- `downloads/QGT_BETA11_SAME_ACTION_r01.zip`
- `downloads/QGT_MINIMAL_QUBIT_HADAMARD_CLOSURE_r01.zip`
