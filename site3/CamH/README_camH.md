# CamH — Reproducibility Package
## QGT gluing mechanism on the hydrogen channel

**Companion to:** P. Camelia, *The Quantum Gauge Theory of Time*, KDP 2026 (v176)
**Analogous to:** CamCMB (CMB power spectra) — this package does for the
**first composite object** (H = p⁺_{k=0} + e⁻_{k=2}) what CamCMB does for the CMB.

> CamH validates the **gluing mechanism** on the hydrogen channel — the
> dynamical phase lock of the first closed boundary composite. The
> Bohr–Rydberg numbers are a metrological cross-check, **not** a proof of
> the standard Coulomb bond.

---

## What it validates

The gluing of two boundary cells is the **stable phase-locking of their
FMT-read shell amplitudes under the Noether-current cancellation condition**.

- The sinusoidal boundary current is **derived**, not assumed:
  Φ_xy = Im⟨ℱ_φψ_x, ℱ_φψ_y⟩ = sin(θ_y − θ_x), from the quantum-FMT overlap.
- The reduced dynamics is  d Δ/dt = δω − K sin Δ,  Δ = θ_y − θ_x.
- Gluing condition (dynamical form of gluing-condition 5):  |δω| ≤ K.
- Canonical, non-tunable inputs:
  - δω = Γ_τ · δ_F  (irreducible margin, physical-time normalisation)
  - K  = κ₀ · μ_Π(ker overlap),  μ_Π = dim(T₈)/dim(B₁₃) = 8/13,  κ₀ = Γ_τ

For H the result is a **sub-threshold stable lock** (δω/K ≈ 5.6×10⁻³),
with residual margin set by sin Δ* = (13/8) δ_F ≠ 0 (so Δ* ≈ (13/8)δ_F to
first order) — synchronisation **with margin**,
i.e. the arrow of time. H = p⁺(k=0)+e⁻(k=2) locks at the breathing fixed
point B⋆ = 8.

---

## Quick start

```bash
python run_camH_validation.py   # runs the three checks, writes metrics + CSV
python make_fig_camH.py         # generates the 4-panel figure
python camH_core.py             # core validation report only
```

Requires: numpy, scipy (matplotlib for the figure). No external data needed;
everything is recomputed from the monograph's structural constants.

---

## File inventory

| File | Description |
|------|-------------|
| `camH_core.py` | core physics: shell phase, FMT overlap, reduced dynamics, three checks |
| `run_camH_validation.py` | runner: metrics JSON, lock curve CSV, threshold sweep CSV, H metrology |
| `make_fig_camH.py` | 4-panel validation figure (phase, lock, threshold, portrait) |
| `metrics_camH.json` | canonical inputs, the three checks, CODATA cross-check, epistemic status |
| `camH_lock_curve.csv` | Δ(t) convergence trajectory |
| `camH_threshold.csv` | lock/drift across δω/K |
| `fig_camH.png/.pdf` | the figure |

---

## The three required checks (doc)

1. **convergence to Δ\*** — phase-lock reached from any initial Δ₀.
2. **loss of lock above threshold** — for |δω| > K the phase drifts (no H).
3. **residual margin ≠ 0 ∝ δ_F** — sin Δ\* = (13/8)δ_F exactly (Δ\* = arcsin((13/8)δ_F)); the arrow of time.

---

## Epistemic status

- internal derivation: **closed** (functional form)
- gluing threshold: **derived**
- phase sine: **derived** from quantum FMT, not imported
- minimal detuning: identified with **δ_F** (after time normalisation)
- coupling K: fixed by **kernel-ribbon measure 8/13** (non-tunable)
- Kuramoto: **NOT named** in the monograph body; the reduced equation belongs
  to the known phase-locking class — a downstream bibliographic note only.
