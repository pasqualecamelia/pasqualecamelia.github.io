# pasqualecamelia.github.io

Interactive companion for the **Quantum Gauge Theory of Time** (Camelia 2026, KDP First Edition).

## What this is

This repository is the companion site to the monograph *The Quantum Gauge Theory of Time*.
It provides interactive visualisations, structural validators, and numerical tools that illustrate
the core chain of the QGT:

```
∂_I J^I = 0  →  Π (rank-5)  →  projective cells  →  coherent gluing  →  ℝ⁴_eff
```

## Structure

| File / Folder | Content |
|---|---|
| `index.html` | Main companion page (breathing, gluing, hydrogen, CMB) |
| `qgt_companion.html` | Full interactive companion with verify panel and 3D H atom |
| `qgt_animation_v4.html` | Temporal fibre S¹(τ) · Lissajous solitons animation |
| `qgt_breathing_3d.html` | Majorana helix — breathing in 3D |
| `qgt_simulator.py` | Structural validator: α, a₀, T_CMB, CMB peaks, Gate-3 |
| `CamH/` | Phase-locking validator for the hydrogen channel |
| `CamCMB/` | CAMB Gate-3 validation with Planck PR3/PR4 data |
| `requirements.txt` | Python dependencies |

## Running the validators

```bash
pip install -r requirements.txt

# Main structural validator
python qgt_simulator.py
python qgt_simulator.py --plot

# Hydrogen phase-lock (CamH)
python CamH/run_camH_validation.py

# CMB Gate-3 (requires camb>=1.5)
python CamCMB/run_qgt_gate3_camb.py
```

## Key results

- α⁻¹_QGT = 20φ⁴ = 137.082 (structural); α⁻¹_phys = 137.036 (CODATA, 0.033 ppm)
- a₀_bare = Γτ² = φ²/5 = 0.5236 Å (golden temporal scale)
- a₀_phys = 0.529177228 Å (Bohr–Rydberg readout, LEVEL_A, +0.033 ppm)
- Hydrogen phase lock: sin Δ* = (13/8)δ_F (exact), δω/K ≈ 5.6×10⁻³ ≪ 1
- CMB peaks RMS 1.6%, χ²/ν = 1.179 (PR3), 0.129 (PR4)

## Monograph

*The Quantum Gauge Theory of Time* — Pasquale Camelia, 2026, KDP First Edition.
QR code in the monograph points to: https://pasqualecamelia.github.io
