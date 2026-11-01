# Coulomb / QGT comparator patch

- Added synchronized `compare` mode.
- Identical initial discrete 1s state for both branches.
- Same FFT kinetic operator, grid and timestep; only potential differs.
- Added `COULOMB`, `QGT`, and signed `Delta rho` view selector.
- Added 3D signed-difference rendering and signed central slices.
- Added `D_rho`, wavefunction fidelity, `Delta<r>`, `max|Delta rho|`, and instantaneous QGT breathing amplitude.
- Preserved PREPARE, LIVE, LIGHT, BILINEAR/WALSH 11 and red-team control.
- Added explicit epistemic qualification: QGT potential deformation is a structural ansatz, not an empirically established hydrogen correction.
