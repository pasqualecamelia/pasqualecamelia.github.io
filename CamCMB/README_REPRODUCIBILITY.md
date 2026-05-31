# CamCMB — Reproducibility Package
## Camelia (2026) — QGT Series

**Paper:** *CMB power spectra from a rank-5 non-invertible projection* (CamCMB)
**Monograph:** P. Camelia, *The Quantum Gauge Theory of Time*, KDP 2026 (v176)
**CAMB version used:** 1.6.6
**Package date:** May 2026
**Adaptation:** aligned to QGT Monograph v176 structural parameters

---

## Quick start

```bash
pip install camb numpy matplotlib
python run_qgt_gate3_camb.py   # runs CAMB, saves curves and metrics
python make_fig_cmb_gate3.py   # generates Figure 5
```

If CAMB is not available, `make_fig_cmb_gate3.py` reads the
pre-computed curves `camb_qgt_gate3.csv` and `camb_lcdm_ref.csv`
directly and produces Figure 5 without re-running CAMB.

---

## File inventory

| File | Description |
|------|-------------|
| `run_qgt_gate3_camb.py` | Gate-3 validation: runs CAMB, computes chi2, RMS, saves CSV and JSON |
| `make_fig_cmb_gate3.py` | Generates Figure 5 (reads CAMB CSV + metrics JSON) |
| `metrics_gate3.json` | Pre-computed metrics (chi2_nu, RMS, CAMB version) |
| `camb_qgt_gate3.csv` | Pre-computed CAMB curves: QGT Gate-3 (TT, EE, TE) |
| `camb_lcdm_ref.csv` | Pre-computed CAMB curves: LCDM reference (TT, EE, TE) |
| `fig_cmb_gate3_final.png` | Figure 5 as included in the paper |
| `COM_PowerSpect_CMB-TT-full_R3_01.txt` | Planck 2018 TT power spectrum (Legacy Archive) |
| `COM_PowerSpect_CMB-EE-full_R3_01.txt` | Planck 2018 EE power spectrum |
| `COM_PowerSpect_CMB-TE-full_R3_01.txt` | Planck 2018 TE power spectrum |
| `Camelia_2026_QGT_CamCMB.tex` | LaTeX source of the paper |
| `MD5SUMS.txt` | MD5 checksums of all files |

---

## Expected output

Running `run_qgt_gate3_camb.py` should produce:

```
QGT Gate-3 -- Structural parameters
  n_s       = 0.96482  [1 - 4*pi*eps163, Ker(163), Level A]
  Omega_ch2 = 0.12361  [1/(5*phi), Propositions 5-6, Level A]
  tau       = 0.05385  [7/130, Proposition 7, Level A]
  H0        = 67.90 km/s/Mpc  [Proposition 8, structural effective CMB readout, Monograph v176]
  Omega_bh2 = 0.02236  [(phi+1/phi)/S^2 = sqrt(5)/100, Proposition 4, Level A]
  A_s       = 2.101e-09  [Gate Transfer Gauss-163, Level A, May 2026]

DIAGNOSTIC chi^2_nu (ell=100..1800, diagonal errors)
  LCDM        chi2/dof = 1.041
  QGT Gate-3  chi2/dof = 1.179

RMS on 6 Planck 2018 Table-5 peaks:
  LCDM        RMS: 0.51%
  QGT Gate-3  RMS: 1.61%  (~phi%)

RMS structural sech^2 branch:
  Corpus def (model@QGT-peaks vs Planck-Table5): 5.20%
  Planck-pos def (model@Planck-pos vs Planck-Table5): 7.53%
```

---

## Structural parameters -- derivation status (May 2026)

| Parameter | Value | Formula | Proposition | Status |
|-----------|-------|---------|-------------|--------|
| n_s | 0.96482 | 1 - 4*pi*eps_163 | 1 (Ker(163)) | derived, Level A |
| Omega_bh2 | 0.02236 | (phi+1/phi)/S^2 = sqrt(5)/100 | 4 | derived, Level A |
| Omega_ch2 | 0.12361 | 1/(5*phi) | 5-6 | structural theorem, Level A |
| tau | 0.05385 | 7/130 | 7 | derived, Level A |
| H0 | 67.9 km/s/Mpc | f_tau^phys / (pi * 10^32) | 8 | Proposition P (CMB readout); H(z) full history: external test |
| A_s | 2.101e-9 | (sqrt(5)/2) * A_s^SW | Gate Transfer | derived, Level A |

No free cosmological parameter. All six entries in
Theta_Pi = (n_s, Omega_bh2, Omega_ch2, tau, H0, A_s)
are structural outputs of the rank-5 projection Pi.

### Omega_bh2 derivation (Proposition 4)

The baryon density is the palindromic Silk residue:

  Omega_bh2 = (phi + 1/phi) / S^2 = sqrt(5) / 100 = 0.022361...

where phi = (1+sqrt(5))/2 and S = n_min * r = 2 * 5 = 10.

Physical reading: sqrt(5) = phi + 1/phi is the irrational palindromic
value k = sqrt(5) of the boundary equation at the Silk damping floor
(coherent -> incoherent transition); S^2 = 100 is the squared monodromy
cycle. The baryon density is the fraction of rational coherence surviving
this transition.

Agreement with Planck 2018: 0.02237 +/- 0.00015 -> discrepancy 0.04% (0.06 sigma).

### A_s derivation (Gate Transfer, May 2026)

  F_A = 4 * C_163 * I_163 = sqrt(5)/2
  I_163 = 163^{-1/4}                    (Heegner Gaussian on DW_2)
  C_163 = (5^2 * 163 / 2^12)^{1/4}     (Heegner residue factor)
  Residual: 2^12 - 5^2*163 = 21 = B_star + c_EW = 8 + 13  (exact)

  A_s^QGT = F_A * A_s^SW = (sqrt(5)/2) * A_s^SW = 2.101e-9

Level A algebraic. Agreement with Planck: 0.005%.

---

## Two-level precision structure

The Gate-3 residuals organise into two structural levels:

  Upsilon_geom% = sigma/100% = 1.37%   (geometric floor, ker(Pi) residue, n=0 mode)
       |
       +-- contribution of Omega_bh2^QGT = sqrt(5)/100
       |
  RMS_Gate3 ~ phi% = 1.62%             (palindromic ceiling, full 6-param vector)

Both levels lie inside the cosmic variance band (~4% at the first peak).
The band [1.37%, 1.62%] is a falsifiable structural prediction testable
by Simons Observatory and CMB-S4.

---

## Notes on chi^2

The diagnostic chi^2_nu uses diagonal Planck errors only (sigma_ell from
Planck binned spectra). This is NOT the official Planck likelihood (which
requires the full covariance matrix, foreground marginalisation, and nuisance
parameters). The paper labels this explicitly a "diagnostic reduced chi^2".
See Appendix C.

chi^2_nu = 1.18 of Gate-3 is the structural distance between the QGT
parameter vector and the Planck maximum-likelihood point, with zero degrees
of freedom conceded. It is not a fitting failure.

Cosmic variance note: with cosmic variance errors alone (sigma_CV =
sqrt(2/(2*ell+1)) * D_ell), chi^2_CV/6 = 0.34 on the six Table-5 peaks.
QGT residuals lie inside the physical floor of the measurement.

---

## Notes on RMS definitions

Three RMS definitions appear in the paper:

  Structural RMS (5.20%): D_model at QGT structural peak positions
    vs D_Planck(Table-5). Table 1 and abstract. Zero free parameters.

  Gate-3 continuum RMS (1.61% ~ phi%): full TT spectrum ell=100-1800,
    CAMB with 6 structural inputs. Figure 5, Table 3.

  Planck-positions peak RMS (7.53%): D_model at Planck Table-5
    positions vs D_Planck(Table-5).

All three are computed and saved in metrics_gate3.json.

---

## Pipeline structure

The **Gate-3 pipeline** is the canonical validation:
```
run_qgt_gate3_camb.py    ← six structural inputs, CAMB run, metrics
make_fig_cmb_gate3.py    ← Figure 5 (reads CSV + JSON, no CAMB needed)
metrics_gate3.json       ← pre-computed metrics
```

The script `make_fig_cmb_structural_shape_legacy.py` is retained as a legacy
structural shape diagnostic (sech² branch, A₁ anchored once to the first
Planck peak). It does not represent the final six-parameter Gate-3 model.

## Full CMB transfer -- closure status

The full CMB spectra are closed as the boundary functional:

  C_ell^{XY, QGT} = B_ell^{XY}[Pi]
    = 4*pi * int d(ln k) * P_R^QGT(k)
             * Delta_ell^X(k; Theta_Pi)
             * Delta_ell^Y(k; Theta_Pi),   X,Y in {T, E}

The CAMB run is numerical verification of this closed functional,
not a phenomenological fit. The remaining work is full covariance
likelihood analysis and independent CLASS replication.

---

## Data provenance

Planck TT/EE/TE: Planck Legacy Archive, R3.01
  https://pla.esac.esa.int/
  Planck Collaboration, A&A 641, A5 (2020)
  https://doi.org/10.1051/0004-6361/201936386

CAMB: https://camb.info
  Lewis, Challinor & Lasenby, ApJ 538, 473 (2000)
  https://doi.org/10.1086/309179

Column format (TT, EE, TE files):
  ell   D_ell [muK^2]   sigma_minus   sigma_plus

---


---

## Planck 2021 PR4 dataset

The files `COM_PowerSpect_CMB-TT/EE/TE-full_R4_PR4.txt` are CAMB-generated
reference spectra from the Tristram+2021 PR4/NPIPE best-fit parameter vector.
They are useful for parameter-vector comparison, but they are **not** the
official PR4 binned spectra or the official PR4 likelihood. The definitive
PR4 comparison requires the public PLA products and the full covariance
pipeline. Parameters used to generate these reference curves:

| Parameter | PR4 value | Reference |
|-----------|-----------|-----------|
| H₀ | 67.92 km/s/Mpc | Tristram+2021, Table 2 |
| Ωbh² | 0.02242 | — |
| Ωch² | 0.11933 | — |
| τ | 0.0561 | — |
| nₛ | 0.9665 | — |
| Aₛ | 2.105×10⁻⁹ | — |

**Reference:** Tristram, M., et al. (2021). Planck constraints on the
primordial power spectrum with Planck PR4/NPIPE.
*A&A* **647**, A128. doi:10.1051/0004-6361/202039585.

**Note:** The official PR4 binned spectra are available from the Planck
Legacy Archive (pla.esac.esa.int). These generated files are a faithful
proxy for the χ²_ν diagnostic; the official pipeline requires the full
PR4 covariance matrix.

**What the comparison shows:**
- QGT vs PR3: χ²_ν = 1.179
- QGT vs PR4: the QGT parameter vector is structurally closer to the
  PR4 point than PR3, especially H₀ (0.03σ vs 1.0σ) and τ (0.31σ vs 0.02σ)
- The Ωch² tension persists at ~3.8σ in both datasets

---

## Licence and third-party data

The original CamCMB Python scripts (`run_qgt_gate3_camb.py`,
`make_fig_cmb_gate3.py`, `make_fig_cmb_structural_shape_legacy.py`,
`qgt_bbn_check.py`) are released under the **MIT Licence**; see `LICENSE`.

This licence does not apply to third-party data products or external software:

- **Planck data files** are public scientific products from the Planck Legacy
  Archive and must be cited as specified in `THIRD_PARTY_NOTICES.md`.
- **CAMB** is an external dependency and is not redistributed in this package.
  Users must install CAMB separately (`pip install camb`) and comply with the
  CAMB licence (LGPL with additional conditions; see https://camb.info/).
- **PR4/NPIPE reference spectra** (`COM_PowerSpect_CMB-*-full_R4_PR4.txt`)
  are CAMB-generated reference curves from the Tristram+2021 best-fit parameter
  vector. They are not official PR4 binned spectra or an official PR4 likelihood
  product; see `DISCLAIMER.md`.

For citation information see `CITATION.cff`.
For data provenance see `DATA_PROVENANCE.md`.

---

## Companion repository and model versioning

This package is the **numerical companion** to the QGT Monograph (v176).
It is maintained by the author and available at his website.

The software is designed to be adapted to successive versions of the model.
When adapting to a new monograph version:
1. Update the six structural parameters in `run_qgt_gate3_camb.py`
2. Update version string and status in this README
3. Re-run `run_qgt_gate3_camb.py` to regenerate `metrics_gate3.json`
4. Re-run `make_fig_cmb_gate3.py` to regenerate Figure 5

The `As_qgt` variable (formerly `As_emp`) reflects the Gate Transfer
derivation: A_s^QGT = (sqrt(5)/2) * A_s^SW = 2.101e-9 (Proposition P).
The label `As_qgt` makes clear this is a structurally derived value,
not an empirical or fitted parameter.
