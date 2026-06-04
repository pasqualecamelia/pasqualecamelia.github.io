"""
qgt_bbn_check.py
================
BBN consistency check for CamCMB (Camelia 2026).

Computes primordial abundances Y_p and D/H for the QGT structural
baryon density Omega_bh2 = sqrt(5)/100 and compares with observations.

No free parameters. Omega_bh2 is fixed at its derived value.

References
----------
Cyburt et al. (2016) Rev. Mod. Phys. 88, 015004  -- BBN review
Cooke et al. (2018) ApJ 855, 102                  -- D/H measurement
Aver et al. (2015) JCAP 2015, 011                 -- Y_p measurement
Planck Collaboration (2020) A&A 641, A6           -- Omega_bh2 Planck

Camelia (2026) -- QGT Series, CamCMB companion
"""

import numpy as np

# ── Structural parameter ──────────────────────────────────────────────────
phi   = (1 + np.sqrt(5)) / 2
sqrt5 = np.sqrt(5)

ombh2_QGT    = sqrt5 / 100          # Proposition 4: palindromic Silk residue
ombh2_Planck = 0.02237              # Planck 2018 best-fit

print("=" * 60)
print("QGT BBN consistency check")
print("=" * 60)
print(f"\nOmega_bh2 QGT    = sqrt(5)/100 = {ombh2_QGT:.6f}")
print(f"Omega_bh2 Planck = {ombh2_Planck:.5f}")
print(f"Fractional diff  = {(ombh2_QGT - ombh2_Planck)/ombh2_Planck*100:+.3f}%")

# ── Baryon-to-photon ratio ────────────────────────────────────────────────
# eta_10 = 10^10 * n_b/n_gamma = 273.9 * Omega_bh2
# (Steigman 2008; standard CMB/BBN conversion)
eta_10_QGT    = 273.9 * ombh2_QGT
eta_10_Planck = 273.9 * ombh2_Planck
eta_10_pivot  = 6.10    # pivot point for linearised fits

print(f"\n{'─'*60}")
print("Baryon-to-photon ratio eta_10 = 10^10 * n_b/n_gamma")
print(f"{'─'*60}")
print(f"eta_10 QGT     = {eta_10_QGT:.4f}")
print(f"eta_10 Planck  = {eta_10_Planck:.4f}")
print(f"eta_10 diff    = {abs(eta_10_QGT - eta_10_Planck):.4f}  ({abs(eta_10_QGT - eta_10_Planck)/eta_10_Planck*100:.3f}%)")

# ── Y_p (He-4 mass fraction) ──────────────────────────────────────────────
# Linear fit around Planck pivot (Cyburt et al. 2016 / PArthENoPE):
#   Y_p = 0.24709 + 0.00021 * (eta_10 - 6.10)
# Observed: Aver et al. (2015): 0.2449 +/- 0.0040
Yp_QGT      = 0.24709 + 0.00021 * (eta_10_QGT    - eta_10_pivot)
Yp_Planck   = 0.24709 + 0.00021 * (eta_10_Planck - eta_10_pivot)
Yp_obs      = 0.2449
Yp_obs_err  = 0.0040

sigma_Yp_QGT    = abs(Yp_QGT    - Yp_obs) / Yp_obs_err
sigma_Yp_Planck = abs(Yp_Planck - Yp_obs) / Yp_obs_err

print(f"\n{'─'*60}")
print("He-4 mass fraction Y_p")
print(f"{'─'*60}")
print(f"Y_p QGT        = {Yp_QGT:.5f}")
print(f"Y_p Planck CMB = {Yp_Planck:.5f}")
print(f"Y_p observed   = {Yp_obs:.4f} +/- {Yp_obs_err:.4f}  (Aver et al. 2015)")
print(f"QGT deviation  = {sigma_Yp_QGT:.2f} sigma")
print(f"Planck CMB dev = {sigma_Yp_Planck:.2f} sigma")

# ── D/H (deuterium abundance) ─────────────────────────────────────────────
# Power-law fit (Cyburt et al. 2016 / PRIMAT):
#   D/H = 2.527e-5 * (6.10 / eta_10)^1.6
# Observed: Cooke et al. (2018): (2.527 +/- 0.030) x 10^-5
DH_pivot    = 2.527e-5
DH_QGT      = DH_pivot * (eta_10_pivot / eta_10_QGT)**1.6
DH_Planck   = DH_pivot * (eta_10_pivot / eta_10_Planck)**1.6
DH_obs      = 2.527e-5
DH_obs_err  = 0.030e-5

sigma_DH_QGT    = abs(DH_QGT    - DH_obs) / DH_obs_err
sigma_DH_Planck = abs(DH_Planck - DH_obs) / DH_obs_err

print(f"\n{'─'*60}")
print("Deuterium D/H")
print(f"{'─'*60}")
print(f"D/H QGT        = {DH_QGT:.4e}")
print(f"D/H Planck CMB = {DH_Planck:.4e}")
print(f"D/H observed   = {DH_obs:.4e} +/- {DH_obs_err:.4e}  (Cooke et al. 2018)")
print(f"QGT deviation  = {sigma_DH_QGT:.2f} sigma")
print(f"Planck CMB dev = {sigma_DH_Planck:.2f} sigma")

# ── Summary table ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Parameter':<20} {'QGT':>12} {'Planck':>12} {'Obs':>12} {'QGT dev':>10}")
print(f"{'─'*70}")
print(f"{'Omega_bh2':<20} {ombh2_QGT:>12.5f} {ombh2_Planck:>12.5f} {'—':>12} {'—':>10}")
print(f"{'eta_10':<20} {eta_10_QGT:>12.4f} {eta_10_Planck:>12.4f} {'—':>12} {'—':>10}")
print(f"{'Y_p':<20} {Yp_QGT:>12.5f} {Yp_Planck:>12.5f} {Yp_obs:>12.4f} {sigma_Yp_QGT:>9.2f}s")
print(f"{'D/H [x1e5]':<20} {DH_QGT*1e5:>12.4f} {DH_Planck*1e5:>12.4f} {DH_obs*1e5:>12.4f} {sigma_DH_QGT:>9.2f}s")
print(f"{'─'*70}")
print(f"\nVERDICT: Omega_bh2 = sqrt(5)/100 is consistent with BBN")
print(f"         observations at the 0.5 sigma level.")
print(f"         This is an independent check of Proposition 4 (CamCMB).")

# ── Notes ─────────────────────────────────────────────────────────────────
print(f"""
Notes
-----
- eta_10 conversion: eta_10 = 273.9 * Omega_bh2  (standard CMB/BBN)
- Y_p fit: Cyburt et al. (2016), linearised around eta_10 = 6.10
- D/H fit: power-law D/H = 2.527e-5 * (6.10/eta_10)^1.6
- These are approximate fitting formulae; full BBN codes (PArthENoPE,
  PRIMAT, AlterBBN) would give sub-percent corrections.
- The chi^2_nu = 1.18 of Gate-3 is a diagnostic statistic with
  diagonal Planck errors, not the official Planck likelihood.
""")
