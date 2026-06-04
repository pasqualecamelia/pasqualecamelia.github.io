"""
make_fig_cmb_structural_shape_legacy.py
========================================
LEGACY DIAGNOSTIC — structural sech² shape (amplitude anchored at first peak).

This script plots the structural sech² CMB branch with the amplitude A₁
anchored once to the first Planck peak. It is a shape diagnostic, not the
final Gate-3 six-parameter model.

For the canonical Gate-3 figure (six structural QGT inputs, CAMB):
  → use make_fig_cmb_gate3.py

Retained for comparison with earlier versions of the model.
Camelia (2026) — QGT Series, CamCMB companion.
"""

Ramo strutturale puro QGT vs Planck 2018
- TT: modello sech² strutturale, zero parametri cosmologici liberi, RMS=5.20%
- EE: formula strutturale (ampiezza aperta, Level B)
- TE: sqrt(D_TT*D_EE)*sin(phi)*cos(phi), nodi = picchi TT per costruzione
- Residui TT picco per picco

Dati: COM_PowerSpect_CMB-TT/TE/EE-full_R3_01.txt (Planck 2018)
      binnati con bw=25, pesatura inv-varianza

Parametri strutturali (tutti derivati, zero fit):
  phi    = (1+sqrt5)/2
  sigma  = (e*pi/100)^-2 = 137.123   [polo Pincherle, P09]
  n_s    = 1-4*pi*eps163 = 0.96482   [Ker(163), Level A]
  R_b    = Gt^2/(1-k_ent/phi^3)      [P12, Level A]
  mu_str = 4*sigma/(5*7*11)           [384 in N_adm]
  S_geom = phi^4/(5*ln2) = 1.9805    [P14, Level A]
  A1     ancorato a D_1^Pl = 5733 muK^2 (unica ancora osservativa)

Nodi TE osservati: letti da Planck 2018 VI Fig.2 (arrotondati)
  obs = [220, 535, 820, 1130, 1450, 1740]

Camelia (2026) — QGT Series
"""

import os

# ── path dati ─────────────────────────────────────────────────────────────
# I file COM_PowerSpect_CMB-*.txt devono stare nella stessa cartella
# di questo script. Per cambiare percorso, modifica DATA_DIR.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── parametri strutturali ─────────────────────────────────────────────────
phi       = (1+np.sqrt(5))/2
sqrt5     = np.sqrt(5)
Gt        = phi/sqrt5
Gt2       = phi**2/5
sigma     = (np.e*np.pi/100)**(-2)
alpha     = 1/sigma
eps163    = np.pi*np.sqrt(163) - np.log(phi**2*1e17)
n_s       = 1 - 4*np.pi*eps163
k_ent     = 5/(2*phi**2)
A_pump    = phi**3
R_b       = Gt**2/(1-k_ent/A_pump)
b_odd     = 1 + R_b/(1+R_b)
b_even    = 1/(1+R_b)
ell_D     = sigma**2/10
mu_struct = 4*sigma/(5*7*11)
S_geom    = phi**4/(5*np.log(2))

print(f"ns={n_s:.5f}, sigma={sigma:.4f}, R_b={R_b:.4f}")
print(f"S_geom={S_geom:.4f}, mu_struct={mu_struct:.6f}")

# ── funzioni modello ──────────────────────────────────────────────────────
def D_SW(ell):
    return 1800.*(np.asarray(ell,float)/10)**(n_s-1) \
           * np.exp(-(np.asarray(ell,float)/1500)**3)

def ell_k(k):
    return sigma*(k*phi + (k-1)/phi)

def sigma_B(k):
    return sigma*Gt*(13-k)/12

def silk(ell):
    return np.exp(-mu_struct*(np.asarray(ell,float)/ell_D)**2)

def sech2(ell, k):
    return 1./np.cosh((np.asarray(ell,float)-ell_k(k))/sigma_B(k))**2

def amplitude(k):
    bk = b_odd if k%2==1 else b_even
    return phi**(-(k-1))*bk*silk(ell_k(k))

# ── picchi Planck 2018 Table 5 ────────────────────────────────────────────
pk_ell = np.array([220.6, 538.1, 809.8, 1147.8, 1446.8, 1779.0])
pk_Dl  = np.array([5733., 2586., 2518., 1227.,   799.,   378.])
pk_sD  = np.array([39.,   23.,   17.,    9.,      5.,     3.])

# ── array ell e fase palindromica ─────────────────────────────────────────
ell_arr  = np.arange(2, 2500, dtype=float)
ph       = np.pi*(ell_arr-(-sigma/phi))/(sigma*sqrt5)
D_SW_arr = D_SW(ell_arr)
A1       = (pk_Dl[0]-D_SW(pk_ell[0]))/amplitude(1)
print(f"A1={A1:.1f} muK^2")

# ── TT strutturale ────────────────────────────────────────────────────────
D_pks = np.zeros_like(ell_arr)
for k in range(1, 7):
    Ak = A1*amplitude(k)
    D_pks += (Ak * sech2(ell_arr,k) / sech2(np.array([ell_k(k)]),k)[0]
              * silk(ell_arr) / silk(ell_k(k)))
D_TT = D_SW_arr + D_pks

# ── EE strutturale ────────────────────────────────────────────────────────
k_cont  = ph/np.pi - 0.5
sin_k   = np.maximum(np.sin(k_cont*Gt2), 0)
ell_EE4 = (ell_k(4)+ell_k(5))/2
w4      = 1 - (1/6)*np.exp(-0.5*((ell_arr-ell_EE4)/sigma_B(3.5))**2)
D_EE    = alpha*A1*S_geom*sin_k*w4*np.sin(ph)**2

# ── TE strutturale ────────────────────────────────────────────────────────
D_TE = np.sqrt(np.abs(D_TT*D_EE))*np.sin(ph)*np.cos(ph)

# ── statistiche TT ───────────────────────────────────────────────────────
errs_TT = [(np.interp(ell_k(k),ell_arr,D_TT)-pk_Dl[k-1])/pk_Dl[k-1]*100
           for k in range(1,7)]
rms_TT  = np.sqrt(np.mean(np.array(errs_TT)**2))

# ── nodi TE: predetti vs Planck 2018 VI Fig.2 ────────────────────────────
qgt_nodes = np.array([ell_k(k) for k in range(1,7)])
obs_nodes = np.array([220., 535., 820., 1130., 1450., 1740.])
resid_TE  = (qgt_nodes - obs_nodes)/obs_nodes*100
rms_TE    = np.sqrt(np.mean(resid_TE**2))

print(f"TT RMS sui 6 picchi: {rms_TT:.2f}%")
print(f"TE RMS sui 6 nodi:   {rms_TE:.2f}%")

# ── binning dati Planck ───────────────────────────────────────────────────
def bin_spec(fname, bw=25):
    d   = np.loadtxt(fname)
    ell = d[:,0]; Dl = d[:,1]; err = (d[:,2]+d[:,3])/2
    rows = []
    for i in range(len(ell)):
        if ell[i] <= 30:
            rows.append([ell[i], Dl[i], err[i]])
    ls = 31
    while ls <= ell[-1]:
        m = (ell>=ls) & (ell<=ls+bw-1)
        if m.sum() > 0:
            w = 1/err[m]**2
            rows.append([np.sum(ell[m]*w)/np.sum(w),
                         np.sum(Dl[m]*w) /np.sum(w),
                         1/np.sqrt(np.sum(w))])
        ls += bw
    return np.array(rows)

TT_b = bin_spec(os.path.join(DATA_DIR, 'COM_PowerSpect_CMB-TT-full_R3_01.txt'))
TE_b = bin_spec(os.path.join(DATA_DIR, 'COM_PowerSpect_CMB-TE-full_R3_01.txt'))
EE_b = bin_spec(os.path.join(DATA_DIR, 'COM_PowerSpect_CMB-EE-full_R3_01.txt'))

mk_b_tt = (TT_b[:,0]>=30)&(TT_b[:,0]<=2000)
mk_b_te = (TE_b[:,0]>=30)&(TE_b[:,0]<=2000)
mk_b_ee = (EE_b[:,0]>=30)&(EE_b[:,0]<=2000)
mk_arr  = (ell_arr>=30)&(ell_arr<=2000)

# ── stile ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':'serif','font.size':11,
    'axes.linewidth':0.8,'axes.edgecolor':'#333',
    'xtick.direction':'in','ytick.direction':'in',
    'xtick.top':True,'ytick.right':True,
    'legend.framealpha':0.95,'legend.edgecolor':'#ccc',
})

col_d = '#d62728'
col_s = '#1f77b4'
col_n = '#2ca02c'

fig = plt.figure(figsize=(11,13))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(4, 1, height_ratios=[3, 2, 2, 1.4], hspace=0.06)

# ── TT ───────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor('white')
ax1.errorbar(TT_b[mk_b_tt,0], TT_b[mk_b_tt,1], yerr=TT_b[mk_b_tt,2],
             fmt='o', color='#bbbbbb', ms=2.0, elinewidth=0.5,
             capsize=0, alpha=0.55, zorder=1, label='Planck 2018 binned (Δℓ=25)')
ax1.errorbar(pk_ell, pk_Dl, yerr=pk_sD, fmt='D', color=col_d, ms=6,
             markeredgecolor='k', markeredgewidth=0.4, capsize=3,
             elinewidth=1.2, zorder=4, label='Planck 2018 Table 5')
# Sachs-Wolfe plateau — fondo 1/f della proiezione
ax1.fill_between(ell_arr[mk_arr], D_SW_arr[mk_arr], alpha=0.18,
                 color='#ff9933', zorder=0,
                 label=rf'S-W plateau ($n_s=1-4\pi\varepsilon_{{163}}$, $A_{{\rm pump}}=\varphi^3$)')
ax1.plot(ell_arr[mk_arr], D_TT[mk_arr], color=col_s, lw=2.0, ls='--', zorder=3,
         label=f'QGT structural sech² — RMS={rms_TT:.2f}%  (0 free cosmological param.)')
# polo sigma = alpha^-1
ax1.axvline(sigma, color='#9966cc', lw=1.2, ls='-', alpha=0.55,
            label=rf'$\sigma=\alpha^{{-1}}\approx{sigma:.1f}$ (projection pole)')
for k in range(1,7):
    ax1.axvline(ell_k(k), color='gray', ls=':', lw=0.6, alpha=0.25)
ax1.set_xlim(30,2000); ax1.set_ylim(-100,7000)
ax1.set_ylabel(r'$D_\ell^{TT}\ [\mu\mathrm{K}^2]$', fontsize=11)
ax1.legend(fontsize=8.5, loc='upper right')
ax1.set_xticklabels([])
ax1.grid(alpha=0.10)
ax1.text(0.012,0.97,'TT',transform=ax1.transAxes,fontsize=13,fontweight='bold',va='top')

# ── EE ───────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor('white')
ax2.errorbar(EE_b[mk_b_ee,0], EE_b[mk_b_ee,1], yerr=EE_b[mk_b_ee,2],
             fmt='o', color='#bbbbbb', ms=2.0, elinewidth=0.5,
             capsize=0, alpha=0.55, zorder=1, label='Planck 2018 binned')
ax2.plot(ell_arr[mk_arr], D_EE[mk_arr], color=col_s, lw=2.0, ls='--', zorder=3,
         label='QGT structural — shape and nodes (Level B); amplitude closed as $\mathcal{B}_\ell[\Pi]$')
ax2.set_xlim(30,2000); ax2.set_ylim(-1,55)
ax2.set_ylabel(r'$D_\ell^{EE}\ [\mu\mathrm{K}^2]$', fontsize=11)
ax2.legend(fontsize=8.5, loc='upper right')
ax2.set_xticklabels([])
ax2.grid(alpha=0.10)
ax2.text(0.012,0.97,'EE',transform=ax2.transAxes,fontsize=13,fontweight='bold',va='top')
ax2.text(700,38,'Shape and nodes: Level B\nAmplitude: closed as $\\mathcal{B}_\\ell[\\Pi]$\n(full covariance likelihood pending)',
         fontsize=8,color='#555',style='italic',
         bbox=dict(boxstyle='round,pad=0.3',fc='#fffbe6',ec='#ccaa00',lw=0.8))

# ── TE ───────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor('white')
ax3.errorbar(TE_b[mk_b_te,0], TE_b[mk_b_te,1], yerr=TE_b[mk_b_te,2],
             fmt='o', color='#bbbbbb', ms=2.0, elinewidth=0.5,
             capsize=0, alpha=0.55, zorder=1, label='Planck 2018 binned')
ax3.plot(ell_arr[mk_arr], D_TE[mk_arr], color=col_s, lw=2.0, ls='--', zorder=3,
         label='QGT structural — amplitude closed as $\mathcal{B}_\\ell[\\Pi]$')
ax3.axhline(0, color='k', lw=0.5, alpha=0.35)
for k,(qn,on) in enumerate(zip(qgt_nodes, obs_nodes),1):
    ax3.axvline(qn, color=col_n, lw=1.3, ls=':', alpha=0.8, zorder=4)
    ax3.plot(on, 0, 'x', color=col_d, ms=7, markeredgewidth=1.8, zorder=5)
from matplotlib.lines import Line2D
nleg = [
    Line2D([0],[0],color=col_n,lw=1.3,ls=':',
           label=f'QGT predicted nodes (= TT peaks)'),
    Line2D([0],[0],marker='x',color=col_d,ms=7,lw=0,markeredgewidth=1.8,
           label=f'Observed nodes Planck 2018 VI Fig.2 — RMS={rms_TE:.2f}%'),
]
h,l = ax3.get_legend_handles_labels()
ax3.legend(handles=h+nleg, fontsize=8, loc='upper right')
ax3.set_xlim(30,2000); ax3.set_ylim(-180,160)
ax3.set_ylabel(r'$D_\ell^{TE}\ [\mu\mathrm{K}^2]$', fontsize=11)
ax3.set_xticklabels([])
ax3.grid(alpha=0.10)
ax3.text(0.012,0.97,'TE',transform=ax3.transAxes,fontsize=13,fontweight='bold',va='top')
ax3.text(700,-120,'Structural amplitude ~×3 below data\n(closed as $\\mathcal{B}_\\ell[\\Pi]$;\nfull likelihood analysis pending)',
         fontsize=8,color='#555',style='italic',
         bbox=dict(boxstyle='round,pad=0.3',fc='#fffbe6',ec='#ccaa00',lw=0.8))

# ── residui TT ───────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3]); ax4.set_facecolor('white')
ax4.axhline(0, color='k', lw=0.7)
ax4.axhspan(-rms_TT, rms_TT, alpha=0.08, color=col_s)
for k in range(1,7):
    e = errs_TT[k-1]
    c = col_s if abs(e)<=rms_TT else col_d
    ax4.bar(ell_k(k), e, width=55, color=c, alpha=0.75,
            edgecolor='k', linewidth=0.5)
    ax4.text(ell_k(k), e+(0.9 if e>=0 else -1.7),
             f'{e:+.1f}%', ha='center', fontsize=8, color='#222')
ax4.set_xlim(30,2000); ax4.set_ylim(-13,13)
ax4.set_ylabel('TT Residual [%]', fontsize=10)
ax4.set_xlabel(r'Multipole $\ell$', fontsize=11)
ax4.text(0.012,0.96,'TT peak-by-peak residuals vs Planck 2018 Table 5',
         transform=ax4.transAxes,fontsize=8.5,va='top',style='italic')
ax4.grid(alpha=0.10, axis='y')

fig.suptitle(
    'QGT palindromic structure vs Planck 2018 CMB — Primary observational result\n'
    r'TT: sech$^2$ structural model, RMS=5.20%, zero free cosmological parameters  |  '
    r'TE: nodes = TT peaks by construction ($\xi\perp\eta$, Level A)  |  '
    r'EE/TE amplitudes: closed as $\mathcal{B}_\ell[\Pi]$',
    fontsize=9.5, fontweight='bold', y=0.999)

plt.savefig('fig_cmb_main.png', dpi=170, bbox_inches='tight',
            facecolor='white')
print("Salvato: fig_cmb_main.png")

