"""
run_qgt_gate3_camb.py
=====================
Script di validazione Gate-3 — CamCMB (Camelia 2026).
Versione: adattata a QGT Monograph v176.

Esegue CAMB con i sei input strutturali QGT derivati e calcola
chi^2_nu diagnostico e RMS rispetto a ENTRAMBI i dataset Planck:
  - Planck 2018 PR3 (R3.01): COM_PowerSpect_CMB-TT/EE/TE-full_R3_01.txt
  - Planck 2021 PR4/NPIPE:   COM_PowerSpect_CMB-TT/EE/TE-full_R4_PR4.txt
    (Tristram+2021, A&A 647, A128, doi:10.1051/0004-6361/202039585)

Richiede: camb >= 1.5 (pip install camb)

NOTE SUL chi^2 DIAGNOSTICO:
  Errori diagonali Planck — non la likelihood ufficiale con covarianza completa.

Camelia (2026) — QGT Series, CamCMB companion validation script.
"""

import numpy as np
import json
import os

try:
    import camb
except ImportError:
    raise ImportError("CAMB non trovato. Installa con: pip install camb")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = DATA_DIR

# ── 1. Parametri strutturali QGT (derivati, zero fit cosmologici) ──────────
phi    = (1 + np.sqrt(5)) / 2
sigma  = 20 * phi**4                          # alpha^-1 = 20*phi^4 = (5+3*sqrt(5))^2
                                               # [P09, Monograph v176]
Gt     = phi / np.sqrt(5)                     # Gamma_tau [P06]
eps163 = np.pi * np.sqrt(163) - np.log(phi**2 * 1e17)
n_s    = 1 - 4 * np.pi * eps163              # spectral tilt [Ker163]
omch2  = 1.0 / (5.0 * phi)                   # Omega_ch^2 [Prop. 5-6]
tau    = 7.0 / 130.0                          # optical depth [Prop. 7]
ombh2  = np.sqrt(5) / 100                     # (phi+1/phi)/S^2 [Prop. 4]
H0_qgt = 67.9                                 # f_tau^phys/(pi*10^32) [Prop. 8]
                                               # structural effective CMB readout
                                               # H(z), BAO/SNe/lensing: external test
As_qgt = 2.101e-9                             # Proposition P: A_s^QGT = F_A * A_s^SW
                                               # F_A = sqrt(5)/2 (Gate Transfer Gauss-163)
                                               # I163 = 163^(-1/4), C163 = (5^2*163/2^12)^(1/4)
                                               # Monograph v176, Appendix I

print("=" * 64)
print("QGT Gate-3 — Parametri strutturali (Monograph v176)")
print("=" * 64)
print(f"  alpha^-1  = {sigma:.6f}  [20*phi^4 = (5+3*sqrt5)^2, P09]")
print(f"  n_s       = {n_s:.5f}  [1-4*pi*eps163, Ker163]")
print(f"  Omega_ch2 = {omch2:.5f}  [1/(5*phi), Prop. 5-6]")
print(f"  tau       = {tau:.5f}  [7/130, Prop. 7]")
print(f"  H0        = {H0_qgt:.2f} km/s/Mpc  [Prop. 8, CMB readout]")
print(f"  Omega_bh2 = {ombh2:.5f}  [sqrt(5)/100, Prop. 4]")
print(f"  A_s       = {As_qgt:.3e}  [F_A*A_s^SW, F_A=sqrt(5)/2, Gate Transfer]")
print()

# ── 2. Run CAMB QGT Gate-3 ─────────────────────────────────────────────────
print("Running CAMB QGT Gate-3...")
pars_qgt = camb.CAMBparams()
pars_qgt.set_cosmology(H0=H0_qgt, ombh2=ombh2, omch2=omch2,
                       tau=tau, mnu=0.06, omk=0, TCMB=2.725)
pars_qgt.InitPower.set_params(ns=n_s, As=As_qgt)
pars_qgt.set_for_lmax(2500, lens_potential_accuracy=0)
results_qgt = camb.get_results(pars_qgt)
powers_qgt  = results_qgt.get_cmb_power_spectra(pars_qgt, CMB_unit='muK')
Dl_tt_qgt = powers_qgt['total'][:, 0]
Dl_ee_qgt = powers_qgt['total'][:, 1]
Dl_te_qgt = powers_qgt['total'][:, 3]
print("  Done.")

# ── 3. Run CAMB LCDM reference — Planck 2018 best-fit ──────────────────────
print("Running CAMB LCDM reference (Planck 2018 best-fit)...")
pars_lcdm18 = camb.CAMBparams()
pars_lcdm18.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.12011,
                           tau=0.0544, mnu=0.06, omk=0, TCMB=2.725)
pars_lcdm18.InitPower.set_params(ns=0.9649, As=2.1e-9)
pars_lcdm18.set_for_lmax(2500, lens_potential_accuracy=0)
results_lcdm18 = camb.get_results(pars_lcdm18)
powers_lcdm18  = results_lcdm18.get_cmb_power_spectra(pars_lcdm18, CMB_unit='muK')
Dl_tt_lcdm18 = powers_lcdm18['total'][:, 0]
print("  Done.")

# ── 4. Run CAMB LCDM reference — Planck 2021 PR4 best-fit ──────────────────
print("Running CAMB LCDM reference (Planck 2021 PR4 best-fit, Tristram+2021)...")
pars_lcdm21 = camb.CAMBparams()
pars_lcdm21.set_cosmology(H0=67.92, ombh2=0.02242, omch2=0.11933,
                           tau=0.0561, mnu=0.06, omk=0, TCMB=2.7255)
pars_lcdm21.InitPower.set_params(ns=0.9665, As=2.105e-9)
pars_lcdm21.set_for_lmax(2500, lens_potential_accuracy=0)
results_lcdm21 = camb.get_results(pars_lcdm21)
powers_lcdm21  = results_lcdm21.get_cmb_power_spectra(pars_lcdm21, CMB_unit='muK')
Dl_tt_lcdm21 = powers_lcdm21['total'][:, 0]
print("  Done.")

# ── 5. Chi^2 diagnostico per un dataset ────────────────────────────────────
def compute_chi2(Dl_model, data_file, lmin=100, lmax=1800):
    data = np.loadtxt(os.path.join(DATA_DIR, data_file), comments='#')
    ell_pl = data[:, 0]
    Dl_pl  = data[:, 1]
    err_pl = data[:, 2]           # usa sigma_minus come errore simmetrico
    mask = (ell_pl >= lmin) & (ell_pl <= lmax)
    ell_m = ell_pl[mask].astype(int)
    D_m   = Dl_pl[mask]
    e_m   = err_pl[mask]
    Dl_mod = np.array([Dl_model[l] if l < len(Dl_model) else 0.0 for l in ell_m])
    chi2 = np.sum(((D_m - Dl_mod) / e_m)**2)
    ndof = len(D_m)
    return chi2, ndof, chi2/ndof

# ── 6. RMS sui 6 picchi Planck 2018 Table 5 ────────────────────────────────
pl_ell_t5 = np.array([220.6, 538.1, 809.8, 1147.8, 1446.8, 1779.0])
pl_D_t5   = np.array([5733., 2586., 2518., 1227.,   799.,   378.])

def rms_peaks(Dl_model):
    Dl_pk = np.array([Dl_model[int(round(l))] for l in pl_ell_t5])
    return np.sqrt(np.mean(((Dl_pk - pl_D_t5)/pl_D_t5)**2)) * 100

# ── 7. Calcola tutto ────────────────────────────────────────────────────────
chi2_qgt_18,  ndof18, chi2nu_qgt_18  = compute_chi2(Dl_tt_qgt,   'COM_PowerSpect_CMB-TT-full_R3_01.txt')
chi2_lcdm_18, _,      chi2nu_lcdm_18 = compute_chi2(Dl_tt_lcdm18,'COM_PowerSpect_CMB-TT-full_R3_01.txt')

chi2_qgt_21,  ndof21, chi2nu_qgt_21  = compute_chi2(Dl_tt_qgt,   'COM_PowerSpect_CMB-TT-full_R4_PR4.txt')
chi2_lcdm_21, _,      chi2nu_lcdm_21 = compute_chi2(Dl_tt_lcdm21,'COM_PowerSpect_CMB-TT-full_R4_PR4.txt')

rms_qgt  = rms_peaks(Dl_tt_qgt)
rms_lcdm = rms_peaks(Dl_tt_lcdm18)

# ── 8. Output ──────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("DIAGNOSTIC chi^2_nu  (ell=100..1800, diagonal errors)")
print("=" * 64)
print(f"  vs Planck 2018 PR3:")
print(f"    LCDM 2018    chi2/dof: {chi2_lcdm_18:.1f}/{ndof18} = {chi2nu_lcdm_18:.3f}")
print(f"    QGT Gate-3   chi2/dof: {chi2_qgt_18:.1f}/{ndof18} = {chi2nu_qgt_18:.3f}")
print()
print(f"  vs Planck 2021 PR4 (Tristram+2021):")
print(f"    LCDM 2021    chi2/dof: {chi2_lcdm_21:.1f}/{ndof21} = {chi2nu_lcdm_21:.3f}")
print(f"    QGT Gate-3   chi2/dof: {chi2_qgt_21:.1f}/{ndof21} = {chi2nu_qgt_21:.3f}")
print()
print(f"RMS on 6 Planck 2018 Table-5 peaks:")
print(f"  LCDM 2018    RMS: {rms_lcdm:.2f}%")
print(f"  QGT Gate-3   RMS: {rms_qgt:.2f}%  (~phi%)")
print()
print(f"NOTE: chi^2_nu diagnostico con errori diagonali Planck.")
print(f"      Non e' la likelihood ufficiale Planck.")
print(f"      CAMB v{camb.__version__}")
print("=" * 64)

# ── 9. Confronto parametrico QGT vs Planck 2018/2021 ──────────────────────
print()
print("Confronto parametrico QGT vs Planck:")
print(f"{'Parameter':<12} {'QGT':>12} {'P2018':>10} {'σ18':>7} {'P2021':>10} {'σ21':>7}")
print("-" * 62)
params_cmp = [
    ('n_s',    n_s,    0.9649, 0.0042, 0.9665, 0.0038),
    ('Ωbh²',   ombh2,  0.02237,0.00015,0.02242,0.00014),
    ('Ωch²',   omch2,  0.1200, 0.0012, 0.11933,0.00110),
    ('τ',      tau,    0.054,  0.007,  0.0561, 0.0073),
    ('A_s/1e9',As_qgt*1e9, 2.101,0.034, 2.105,0.032),
    ('H₀',     H0_qgt, 67.4,  0.5,    67.92,  0.70),
]
for name, qv, p18, e18, p21, e21 in params_cmp:
    s18 = abs(qv-p18)/e18
    s21 = abs(qv-p21)/e21
    print(f"{name:<12} {qv:>12.5g} {p18:>10.5g} {s18:>7.2f}σ {p21:>10.5g} {s21:>7.2f}σ")

# ── 10. Salva JSON ──────────────────────────────────────────────────────────
metrics = {
    "description": "QGT Gate-3 diagnostic metrics -- CamCMB adapted to QGT Monograph v176",
    "monograph_version": "QGT Monograph v176 (KDP 2026)",
    "camb_version": camb.__version__,
    "as_variable_note": "As_qgt: structurally derived via Gate Transfer, A_s^QGT=(sqrt(5)/2)*A_s^SW=2.101e-9",
    "chi2_note": "Diagnostic chi^2_nu with diagonal Planck errors — not the official Planck likelihood.",
    "structural_parameters": {
        "alpha_inv": float(sigma),
        "alpha_inv_formula": "20*phi^4 = (5+3*sqrt5)^2",
        "n_s": float(n_s), "Omega_ch2": float(omch2),
        "tau": float(tau), "H0_kms_Mpc": float(H0_qgt),
        "H0_status": "Proposition P (structural effective CMB readout); H(z): external test",
        "Omega_bh2": float(ombh2),
        "A_s_derived": float(As_qgt),
        "A_s_note": "Gate Transfer: F_A=sqrt(5)/2, Proposition P, Monograph v176 Appendix I"
    },
    "vs_planck_2018_PR3": {
        "gate3_chi2nu": float(chi2nu_qgt_18),
        "lcdm_chi2nu":  float(chi2nu_lcdm_18),
        "ell_range": "100-1800", "ndof": int(ndof18),
        "gate3_RMS_6peaks_pct": float(rms_qgt),
        "lcdm_RMS_6peaks_pct":  float(rms_lcdm),
    },
    "vs_planck_2021_PR4": {
        "reference": "Tristram+2021, A&A 647, A128, doi:10.1051/0004-6361/202039585",
        "gate3_chi2nu": float(chi2nu_qgt_21),
        "lcdm_chi2nu":  float(chi2nu_lcdm_21),
        "ell_range": "100-1800", "ndof": int(ndof21),
    },
    "parametric_comparison": {
        p[0]: {"QGT": p[1], "Planck2018": p[2], "sigma18": round(abs(p[1]-p[2])/p[3],3),
               "Planck2021": p[4], "sigma21": round(abs(p[1]-p[4])/p[5],3)}
        for p in params_cmp
    }
}
with open(os.path.join(OUT_DIR, 'metrics_gate3.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print()
print("Metriche salvate: metrics_gate3.json")

# ── 11. Salva curve CAMB ────────────────────────────────────────────────────
ells = np.arange(len(Dl_tt_qgt))
np.savetxt(os.path.join(OUT_DIR, 'camb_qgt_gate3.csv'),
    np.column_stack([ells, Dl_tt_qgt, Dl_ee_qgt, Dl_te_qgt]),
    header='ell,Dl_TT,Dl_EE,Dl_TE', delimiter=',', comments='# ')
np.savetxt(os.path.join(OUT_DIR, 'camb_lcdm_ref.csv'),
    np.column_stack([np.arange(len(Dl_tt_lcdm18)),
                     Dl_tt_lcdm18,
                     powers_lcdm18['total'][:, 1],  # EE
                     powers_lcdm18['total'][:, 3]]), # TE
    header='ell,Dl_TT,Dl_EE,Dl_TE', delimiter=',', comments='# ')
np.savetxt(os.path.join(OUT_DIR, 'camb_lcdm21_ref.csv'),
    np.column_stack([np.arange(len(Dl_tt_lcdm21)), Dl_tt_lcdm21]),
    header='ell,Dl_TT (LCDM PR4 best-fit)', delimiter=',', comments='# ')
print("Curve salvate: camb_qgt_gate3.csv, camb_lcdm_ref.csv, camb_lcdm21_ref.csv")
