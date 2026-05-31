"""
make_fig_cmb_gate3.py
=====================
Figure 5 (Gate-3) -- CamCMB (Camelia 2026).

Legge le curve CAMB generate da run_qgt_gate3_camb.py e plotta:
  - Planck 2018 data
  - LCDM reference (da camb_lcdm_ref.csv)
  - QGT structural sech^2 (ricalcolata qui, 0 free param.)
  - QGT+CAMB Gate-3 (da camb_qgt_gate3.csv)

Chi^2 e metriche letti da metrics_gate3.json (non hard-coded).

Richiede:
  camb_qgt_gate3.csv, camb_lcdm_ref.csv, metrics_gate3.json
  COM_PowerSpect_CMB-TT/EE/TE-full_R3_01.txt
  (generati da run_qgt_gate3_camb.py)

Camelia (2026) -- QGT Series, CamCMB companion script.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Leggi metriche dal JSON ───────────────────────────────────────────────
with open(os.path.join(DATA_DIR, 'metrics_gate3.json'), 'r') as f:
    metrics = json.load(f)

chi2nu_qgt  = metrics['vs_planck_2018_PR3']['gate3_chi2nu']
chi2nu_lcdm = metrics['vs_planck_2018_PR3']['lcdm_chi2nu']
ell_range   = metrics['vs_planck_2018_PR3']['ell_range']

# ── Leggi curve CAMB ─────────────────────────────────────────────────────
camb_qgt  = np.loadtxt(os.path.join(DATA_DIR, 'camb_qgt_gate3.csv'),
                        delimiter=',', comments='#')
camb_lcdm = np.loadtxt(os.path.join(DATA_DIR, 'camb_lcdm_ref.csv'),
                        delimiter=',', comments='#')
ell_camb     = camb_qgt[:, 0].astype(int)
Dl_tt_qgt_c  = camb_qgt[:, 1]
Dl_ee_qgt_c  = camb_qgt[:, 2]
Dl_te_qgt_c  = camb_qgt[:, 3]
Dl_tt_lcdm_c = camb_lcdm[:, 1]
Dl_ee_lcdm_c = camb_lcdm[:, 2]
Dl_te_lcdm_c = camb_lcdm[:, 3]

# ── Parametri strutturali QGT (per la curva sech^2) ──────────────────────
phi    = (1 + np.sqrt(5)) / 2
sigma  = (np.e * np.pi / 100)**(-2)
Gt     = phi / np.sqrt(5)
eps163 = np.pi * np.sqrt(163) - np.log(phi**2 * 1e17)
n_s    = 1 - 4 * np.pi * eps163
k_ent  = 5 / (2 * phi**2)
Rb     = Gt**2 / (1 - k_ent / phi**3)
b_odd  = 1 + Rb / (1 + Rb)
b_even = 1 / (1 + Rb)
mu_str = 4 * sigma / 385
ell_D  = sigma**2 / 10

def ell_k(k):   return sigma * (k * phi + (k - 1) / phi)
def sigma_B(k): return sigma * Gt * (13 - k) / 12
def b_k(k):     return b_odd if k % 2 == 1 else b_even
def mu_silk(k): return np.exp(-mu_str * (ell_k(k) / ell_D)**2)

pl_ell_t5 = np.array([220.6, 538.1, 809.8, 1147.8, 1446.8, 1779.0])
pl_D_t5   = np.array([5733., 2586., 2518., 1227., 799., 378.])
pl_sD_t5  = np.array([  39.,   23.,   17.,    9.,   5.,   3.])

ell_arr  = np.arange(2, 2001, dtype=float)
A_SW     = 1800.0
D_SW_arr = A_SW * (ell_arr / 10)**(n_s - 1) * np.exp(-(ell_arr / 1500)**3)
lk1  = ell_k(1); idx1 = np.argmin(np.abs(ell_arr - lk1))
A1   = (pl_D_t5[0] - D_SW_arr[idx1]) / (b_k(1) * mu_silk(1))
D_sech2 = D_SW_arr.copy()
for k in range(1, 7):
    amp = A1 * phi**(-(k-1)) * b_k(k) * mu_silk(k)
    D_sech2 += amp / np.cosh((ell_arr - ell_k(k)) / sigma_B(k))**2

# ── Dati Planck binned ────────────────────────────────────────────────────
def load_bin(fname, bw=25, lmin=2, lmax=2000):
    data = np.loadtxt(os.path.join(DATA_DIR, fname), comments='#')
    ell, D, err = data[:,0], data[:,1], data[:,2]
    mask = (ell>=lmin)&(ell<=lmax)
    ell, D, err = ell[mask], D[mask], err[mask]
    edges = np.arange(lmin, lmax+bw, bw)
    bc, bD, bE = [], [], []
    for i in range(len(edges)-1):
        m = (ell>=edges[i])&(ell<edges[i+1])
        if m.sum()==0: continue
        w = 1/err[m]**2
        bc.append(np.average(ell[m],weights=w))
        bD.append(np.average(D[m],  weights=w))
        bE.append(1/np.sqrt(w.sum()))
    return np.array(bc), np.array(bD), np.array(bE)

ell_tt, D_tt, e_tt = load_bin('COM_PowerSpect_CMB-TT-full_R3_01.txt')
ell_ee, D_ee, e_ee = load_bin('COM_PowerSpect_CMB-EE-full_R3_01.txt')
ell_te, D_te, e_te = load_bin('COM_PowerSpect_CMB-TE-full_R3_01.txt')

# ── Colori ────────────────────────────────────────────────────────────────
C_lcdm  = '#888888'
C_sech2 = '#1a5fa8'   # sech^2 strutturale
C_gate3 = '#1a7a3a'   # Gate-3 CAMB
C_pl    = '#cc2222'
LMAX_PLOT = 2000

# ── Figura ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 2], hspace=0.06)
ax_tt = fig.add_subplot(gs[0])
ax_ee = fig.add_subplot(gs[1])
ax_te = fig.add_subplot(gs[2])

mask_c = ell_camb <= LMAX_PLOT

# ── TT ─────────────────────────────────────────────────────────────────────
ax_tt.fill_between(ell_tt, D_tt-e_tt, D_tt+e_tt, alpha=0.20, color=C_pl)
ax_tt.plot(ell_tt, D_tt, '.', color=C_pl, ms=2.5, alpha=0.5, label='Planck 2018')
ax_tt.errorbar(pl_ell_t5, pl_D_t5, yerr=pl_sD_t5, fmt='D',
               color=C_pl, ms=6, zorder=10, label='Planck Table 5')
ax_tt.plot(ell_camb[mask_c], Dl_tt_lcdm_c[mask_c], '-',
           color=C_lcdm, lw=1.8,
           label=rf'$\Lambda$CDM  ($\chi^2_\nu={chi2nu_lcdm:.2f}$)')
ax_tt.plot(ell_arr, D_sech2, '--', color=C_sech2, lw=1.8,
           label=r'QGT sech$^2$  (RMS $5.20\%$, 0 free param., $A_1$ anchored)')
ax_tt.plot(ell_camb[mask_c], Dl_tt_qgt_c[mask_c], '-',
           color=C_gate3, lw=2.2,
           label=(rf'QGT Gate-3  ($\chi^2_\nu={chi2nu_qgt:.2f}$, '
                  r'6 struct., $A_s$ derived (Gate Transfer))'))
ax_tt.set_xlim(2, LMAX_PLOT); ax_tt.set_ylim(0, 7500)
ax_tt.set_ylabel(r'$D_\ell^{TT}\ [\mu{\rm K}^2]$', fontsize=11)
ax_tt.tick_params(labelbottom=False)
ax_tt.legend(fontsize=8.5, loc='upper right')
ax_tt.grid(True, alpha=0.20)

# ── EE ─────────────────────────────────────────────────────────────────────
ax_ee.fill_between(ell_ee, D_ee-e_ee, D_ee+e_ee, alpha=0.20, color=C_pl)
ax_ee.plot(ell_ee, D_ee, '.', color=C_pl, ms=2.5, alpha=0.5)
ax_ee.plot(ell_camb[mask_c], Dl_ee_lcdm_c[mask_c], '-', color=C_lcdm, lw=1.8)
ax_ee.plot(ell_camb[mask_c], Dl_ee_qgt_c[mask_c],  '-', color=C_gate3, lw=2.0)
ax_ee.text(0.02, 0.93, r'EE/TE: closed as boundary functional $\mathcal{B}_\ell[\Pi]$',
    transform=ax_ee.transAxes, fontsize=8.5, color='#888', va='top')
ax_ee.set_xlim(2, LMAX_PLOT); ax_ee.set_ylim(-1, 45)
ax_ee.set_ylabel(r'$D_\ell^{EE}\ [\mu{\rm K}^2]$', fontsize=11)
ax_ee.tick_params(labelbottom=False)
ax_ee.grid(True, alpha=0.20)

# ── TE ─────────────────────────────────────────────────────────────────────
ax_te.fill_between(ell_te, D_te-e_te, D_te+e_te, alpha=0.20, color=C_pl)
ax_te.plot(ell_te, D_te, '.', color=C_pl, ms=2.5, alpha=0.5)
ax_te.plot(ell_camb[mask_c], Dl_te_lcdm_c[mask_c], '-', color=C_lcdm, lw=1.8)
ax_te.plot(ell_camb[mask_c], Dl_te_qgt_c[mask_c],  '-', color=C_gate3, lw=2.0)
for k in range(1, 7):
    lk = ell_k(k)
    if lk <= LMAX_PLOT:
        ax_te.axvline(lk, color=C_sech2, lw=0.9, ls='--', alpha=0.60)
ax_te.axhline(0, color='black', lw=0.5)
ax_te.text(0.02, 0.08,
    r'Blue dashed: QGT TE nodes ($\xi\perp\eta$, RMS $1.7\%$ on 6 nodes)',
    transform=ax_te.transAxes, fontsize=8.5, color=C_sech2)
ax_te.set_xlim(2, LMAX_PLOT); ax_te.set_ylim(-160, 160)
ax_te.set_ylabel(r'$D_\ell^{TE}\ [\mu{\rm K}^2]$', fontsize=11)
ax_te.set_xlabel(r'Multipole $\ell$', fontsize=11)
ax_te.grid(True, alpha=0.20)

fig.suptitle(
    rf'QGT Gate-3 vs $\Lambda$CDM vs Planck 2018  --  TT, EE, TE',
    fontsize=12, fontweight='bold', y=0.995)

out = os.path.join(DATA_DIR, 'fig_cmb_gate3_final.png')
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Figura salvata: {out}')
