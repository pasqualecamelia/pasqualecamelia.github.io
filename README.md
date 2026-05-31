# QGT Companion — Interactive companion to the monograph

**The Quantum Gauge Theory of Time** — Pasquale Camelia (2026)

This site is the interactive companion to the monograph. It is not a
publication list: it is a tool to *run* the derivation the book presents —
the six-layer qubit, the breathing flow, phase-locking gluing, and the
structural validators for hydrogen and the CMB.

## Pages

| Page | What it does |
|------|--------------|
| `index.html` | The thread, in one read: projection → breathing → gluing → hydrogen → CMB, with live canvases |
| `qgt_companion.html` | Full interactive verifier: constants, breathing, soliton, hydrogen (static + 3D split-operator), spectrum, α⁻¹ cascade, filter |
| `qgt_animation_v4.html` | Temporal fibre S¹(τ) and the four Lissajous anyon classes |
| `qgt_breathing_3d.html` | The shell breathing between the two domain walls on S²_χ |
| `M5_toroide_QGT.svg` | Toroidal texture of synchronised shells (layer 6 gluing) |

## Python

```bash
pip install numpy scipy matplotlib      # + camb for live CMB spectra
python qgt_simulator.py verify          # structural identities, epistemic levels
python qgt_simulator.py hydrogen        # dynamic 3D hydrogen (split-operator)
python qgt_simulator.py all --plot      # everything, saves figures
```

## Epistemic levels (as in the monograph)

- **A** — structural / derived
- **B** — strong structural readout
- **O** — open

Diagnostic comparisons (e.g. CMB χ² with diagonal Planck errors) are
labelled as such and are not the official likelihoods.

## Licence

Code: MIT. Scientific content accompanies the First Edition of the monograph.
