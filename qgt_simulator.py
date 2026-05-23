"""
# SPDX-License-Identifier: MIT
# © 2026 Pasquale Camelia — ORCID: 0009-0006-4779-4647
# https://pasqualecamelia.github.io
# Released under the MIT License — see LICENSE for details.
#
QGT Simulator — Quantum Gauge Theory of Time
Derivato dalla monografia "The Quantum Gauge Theory of Time" (Camelia, 2026)

Ogni equazione cita il capitolo sorgente. Zero parametri liberi.

STATO EPISTEMICO (allineato alla monografia, Cap. 40):
  ✓ dimostrato: breathing, solitone, ricorrenza di scala, cascata adimensionale
  ✓ dimostrato: alpha^{-1} phys (3 passi, -0.049 ppb), sin²θ_W=3/13, alpha_s=√140/100
  ✓ dimostrato: T_CMB, Omega_b/c h², spettro CMB 6 picchi strutturali
  ✓ dimostrato: Clifford Cl(4,1), proiettori chirali, butterfly M_eta/M_xi
  ✓ dimostrato: Gate Bohr-Rydberg — a0, R_infty readout canonico di Pi+, 0.033 ppm
  ✓ dimostrato: tau_pi+ = 2.60330e-8 s (Gate 2A+2B, Cap. 36)
  ✓ dimostrato: v_EW = 246.20 GeV (Cap. 35)
  ✓ stima strutturale: m_mu/m_e (~0.5%, interi strutturali)
  ⚠ aperti: kappa_QGT/G_N, sigma_3, G_edge(k,ell) CMB multipolare, QCD dinamica

USO RAPIDO:
  python qgt_simulator.py          # verifica e output testo
  python qgt_simulator.py --plot   # tutti i grafici matplotlib
  python qgt_simulator.py --plot breathing soliton atom dna cmb
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys

# ================================================================
# 1. COSTANTI STRUTTURALI (Cap01, Cap02, Cap09)
# ================================================================

phi       = (1 + np.sqrt(5)) / 2          # Cap01
Gamma_tau = phi / np.sqrt(5)              # Cap01
alpha_inv = (5 + 3*np.sqrt(5))**2         # Cap09 = 20*phi^4 = 500*Gamma_tau^4
alpha     = 1.0 / alpha_inv

assert abs(20*phi**4         - alpha_inv) < 1e-10
assert abs(500*Gamma_tau**4  - alpha_inv) < 1e-10
assert abs(2*np.sqrt(5)*phi**2 - np.sqrt(alpha_inv)) < 1e-10

r      = 5;  n_min = 2;  n_sat = 11;  n_lock = 5;  S = 10
n_conf = 7;  c_EW  = 13   # = 2r + |C_obs|

lambda_eff = 16 * np.pi**2 * phi**12     # CapSolitoni

# Fattore spettrale F_nm (Cap16, CapReadout)
F_star = 2*np.sqrt(5)*phi**2 / (3*np.pi)
f_gain = (2*np.sqrt(5))**1.5 / (3*np.pi)
F_nm   = (2*np.sqrt(5))**2.5 * phi**2 / (9*np.pi**2)
assert abs(F_star * f_gain - F_nm) < 1e-10
delta_F = 1 - 1/f_gain    # margine PLL ~0.345%

# Lock-in anisotropico xi*/eta* = 11/5 (Cap14/P14)
xi_star  = n_sat * np.sqrt(alpha_inv / (n_sat * n_lock))
eta_star = n_lock * np.sqrt(alpha_inv / (n_sat * n_lock))
assert abs(xi_star * eta_star - alpha_inv) < 1e-8

# Butterfly chirali (CapMateria, CapClifford)
M_eta = np.array([[2,1],[1,-1]], dtype=float) / n_min
M_xi  = np.array([[2,1],[1/np.sqrt(n_min),-1]], dtype=float) * np.sqrt(n_min)
# SVD di M_eta => autovalori leptonici
sigma1 = np.sqrt((n_conf - np.sqrt(c_EW)) / (2*n_min**2))  # = sqrt((7-sqrt13)/8)
sigma2 = np.sqrt((n_conf + np.sqrt(c_EW)) / (2*n_min**2))  # = sqrt((7+sqrt13)/8)
Lambda = alpha_inv * 3 / 8    # in Q[sqrt5]

# v_EW = 246.20 GeV (Cap. 35, dimostrato)
v_EW_GeV  = 246.20

# Cascata adimensionale (Cap. 32, Cap. 26, Cap. 39)
Delta_B   = 3*phi*1e-3 + 3*phi**2*1e-4 - 3*np.sqrt(5)*1e-5 - 2e-6  # Angstrom
a0_static = phi**2 / 5
eps_B     = Delta_B / a0_static
a0_QGT    = a0_static * (1 + eps_B) * 1e-10  # metri

eps7   = np.sqrt(5)*np.pi/7 - 1
f_brea = phi / 5e6
alpha_inv_phys = (alpha_inv - c_EW*eps7) / (1 - f_brea)

H_ker   = [19, 43, 67]
m_mu_me = H_ker[1]*sigma2 / (H_ker[0]*sigma1) * Lambda

T_CMB_QGT  = np.pi/np.log(np.pi) * (1 - 1/(np.sqrt(2)*100))
Omega_b_h2 = np.sqrt(5) / 100
Omega_c_h2 = 1 / (5*phi)
# tau_pi+ fisica corretta (Cap. 36): grezza f_tau=phi^2/10, corretta dal residuo olonomico
f_tau      = phi**2/10
_delta_F   = 1 - 1/((2*np.sqrt(5))**1.5/(3*np.pi))
tau_pi     = f_tau * 1e-7 / (1 + phi*(_delta_F + eps7)/2)  # = 2.60330e-8 s

# ================================================================
# 2. BREATHING ODE (Cap02, eq:breathing_hyp)
# ================================================================

def breathing_ode(t, y):
    """dxi/dt = -Gamma_tau*xi,  deta/dt = +Gamma_tau*eta. Invariante: xi*eta = alpha_inv."""
    xi, eta = y
    return [-Gamma_tau*xi, +Gamma_tau*eta]

def breathing_exact(t, xi0=None, eta0=None):
    """Soluzione analitica. Invariante xi*eta = alpha_inv esatto."""
    if xi0 is None: xi0 = xi_star
    if eta0 is None: eta0 = eta_star
    return xi0*np.exp(-Gamma_tau*t), eta0*np.exp(+Gamma_tau*t)

def theta_log(t, xi0=None, eta0=None):
    """Tempo fisico theta_log = 0.5*log(eta/xi). dot_theta_log = Gamma_tau esatto."""
    xi, eta = breathing_exact(t, xi0, eta0)
    return 0.5 * np.log(eta / xi)

# ================================================================
# 3. RICORRENZA DI SCALA (Cap02, eq:breath_rec)
# ================================================================

def breathing_map(B):
    """B_{k+1} = 2*B_k^{2/3}. Attrattore B* = 8."""
    return 2.0 * B**(2/3)

assert abs(breathing_map(8.0) - 8.0) < 1e-10

# ================================================================
# 4. SOLITONE DI LOCK-IN (CapSolitoni, eq:Psi_star)
# ================================================================

def ell_pen(H=1.0):
    """ell_pen = H/(pi*alpha). H=1 in unita strutturali."""
    return H / (np.pi * alpha)

def soliton(n, H=1.0):
    """Psi_star(n) = phi * sech(n/ell_pen). Ampiezza phi da R(e^{-2pi})=1/phi."""
    return phi / np.cosh(n / ell_pen(H))

assert abs((phi + 1/phi) / np.cosh(np.log(phi)) - 2.0) < 1e-14  # identita aurea

# ================================================================
# 5. DIRAC NONLINEARE STAZIONARIA 1D (CapSolitoni)
# ================================================================

def nonlinear_dirac_ode(n, y, m_eff=1.0):
    """d²Psi/dn² = m_eff²*Psi - lambda_eff*|Psi|^(2/3)*Psi. Condizioni Neumann."""
    psi, dpsi = y
    return [dpsi, m_eff**2*psi - lambda_eff*np.abs(psi)**(2/3)*psi]

def solve_soliton_ode(n_max=5.0, m_eff=1.0, N=2000):
    sol = solve_ivp(nonlinear_dirac_ode, [0, n_max], [phi, 0.0],
                    args=(m_eff,), t_eval=np.linspace(0, n_max, N),
                    method='RK45', rtol=1e-9, atol=1e-11)
    return sol.t, sol.y[0]

# ================================================================
# 6. SPETTRO CMB (CapSolitoni, eq:Cl_transfer + eq:Dl_sech2)
# ================================================================

def filter_H(z):
    """H(z) = P4(z)/(1-alpha*z), P4(z) = sum_{m=0}^4 (log phi)^m z^m."""
    lp = np.log(phi)
    return sum(lp**m * z**m for m in range(5)) / (1 - alpha * z)

def C_ell(ell_arr):
    """C_ell = |H(e^{2pi i/ell})|^2 * sech^2(ell*pi*alpha/2)."""
    out = np.zeros(len(ell_arr))
    for i, ell in enumerate(ell_arr):
        z = np.exp(2j*np.pi/ell)
        out[i] = abs(filter_H(z))**2 * (1/np.cosh(ell*np.pi*alpha/2))**2
    return out

def D_ell(ell_arr):
    """D_ell = 1 + sum_{k=1}^6 phi^{-2k} * sech^2((ell-ell_k)/sigma_B(k))."""
    D = np.ones_like(ell_arr, dtype=float)
    for k in range(1, 7):
        lk = alpha_inv * (k*phi + (k-1)/phi)
        sB = alpha_inv * Gamma_tau / k
        D += phi**(-2*k) * (1/np.cosh((ell_arr-lk)/sB))**2
    return D

# ================================================================
# 7. CLIFFORD Cl(4,1) ESPLICITO (CapClifford)
# ================================================================

def build_clifford_qgt():
    """
    Rappresentazione 4x4 base Weyl.
    gamma^6 = sqrt(5)*gamma^5.  (gamma^6)^2 = 5I. {gamma^6, gamma^mu} = 0.
    P_phi+ = diag(I,0) settore longitudinale (gravity).
    P_phi- = diag(0,I) settore trasversale (EM).
    C = i*gamma^2*gamma^0  (coniugazione di carica Majorana).
    """
    s0 = np.eye(2, dtype=complex)
    s1 = np.array([[0,1],[1,0]], dtype=complex)
    s2 = np.array([[0,-1j],[1j,0]], dtype=complex)
    s3 = np.array([[1,0],[0,-1]], dtype=complex)
    g = {}
    g[0] = np.kron(s1, s0)
    g[1] = np.kron(1j*s2, s1)
    g[2] = np.kron(1j*s2, s2)
    g[3] = np.kron(1j*s2, s3)
    g[5] = np.kron(s3, s0)          # gamma^5 = diag(1,1,-1,-1)
    g[6] = np.sqrt(5) * g[5]        # gamma^6
    I4 = np.eye(4, dtype=complex)
    P_plus  = 0.5*(I4 + g[5])
    P_minus = 0.5*(I4 - g[5])
    C = 1j * g[2] @ g[0]
    eta = np.diag([1.,-1.,-1.,-1.])
    for mu in range(4):
        for nu in range(4):
            assert np.allclose(g[mu]@g[nu]+g[nu]@g[mu], 2*eta[mu,nu]*I4)
    assert np.allclose(g[6]@g[6], 5*I4)
    for mu in range(4):
        assert np.allclose(g[6]@g[mu]+g[mu]@g[6], 0)
    assert np.allclose(P_plus@P_minus, 0) and np.allclose(P_plus+P_minus, I4)
    return g, P_plus, P_minus, C

def dirac_operator(xi_val, eta_val, k_vec=None):
    """
    M(k) = gamma^mu*k_mu + alpha*xi*I + i*alpha*eta*gamma^5
    per spinore piano. Il termine nonlineare lambda*|Psi|^(2/3) va iterato.
    """
    g, _, _, _ = build_clifford_qgt()
    I4 = np.eye(4, dtype=complex)
    if k_vec is None: k_vec = [0,0,0,0]
    slash_k = sum(k_vec[mu]*g[mu] for mu in range(4))
    return slash_k + alpha*xi_val*I4 + 1j*alpha*eta_val*g[5]

# ================================================================
# 8. DNA Class 2 (CapSolitoni, P09, scala k=3)
# ================================================================

def dna_helix(n_turns=1, steps_per_turn=10, angle_deg=36.0):
    """
    B-DNA come Class 2 (2:5) della Dirac nonlineare alla scala k=3.
    Ritorna coordinate 3D (x1,y1,z), (x2,y2,z) dei due filoni.
    angle_deg=36 = pi/5 e' il parametro di deformazione di TL4(phi).
    """
    N = n_turns * steps_per_turn * 20
    t = np.linspace(0, n_turns*2*np.pi, N)
    angle_rad = np.deg2rad(angle_deg)
    # passo per unita angolare = steps_per_turn / (2*pi)
    freq = steps_per_turn / (2*np.pi) * angle_rad
    r = 1.0   # raggio normalizzato
    z = t / (2*np.pi) * 3.4 * n_turns  # pitch = 3.4 nm per giro
    x1 = r * np.cos(freq * t)
    y1 = r * np.sin(freq * t)
    x2 = r * np.cos(freq * t + np.pi)
    y2 = r * np.sin(freq * t + np.pi)
    return (x1, y1, z), (x2, y2, z)

# ================================================================
# 9. ATOMO DI IDROGENO (CapMateria, CapOntologia)
# ================================================================

def hydrogen_radial_density(r_arr, n, a0=None):
    """
    Densita radiale P(r) = |R_n0(r)|^2 * r^2.
    a0 in Angstrom. Default: a0_QGT.
    """
    if a0 is None: a0 = a0_QGT * 1e10  # converti in Angstrom
    rho = 2 * r_arr / (n * a0)
    if n == 1:   Rnl = 2 * np.exp(-rho/2)
    elif n == 2: Rnl = (1/(2*np.sqrt(2))) * (2 - rho) * np.exp(-rho/2)
    elif n == 3: Rnl = (2/(81*np.sqrt(3))) * (27 - 18*rho + 2*rho**2) * np.exp(-rho/2)
    elif n == 4: Rnl = (1/768) * (192 - 144*rho + 24*rho**2 - rho**3) * np.exp(-rho/2)
    else:        Rnl = np.exp(-rho/2)
    return Rnl**2 * r_arr**2

# ================================================================
# 10. VERIFY & PRINT
# ================================================================

def run_all():
    CODATA = {'alpha_inv': 137.035999177, 'R_inf': 10973731.568,
              'T_CMB': 2.72548, 'm_mu_me': 206.7682830}
    sep = "="*62
    print(sep)
    print("QGT SIMULATOR  —  monografia First Edition 2026 KDP")
    print(sep)

    print("\n▸ STRUTTURALI PURI (dimostrato)")
    print(f"  phi                = {phi:.10f}")
    print(f"  Gamma_tau          = {Gamma_tau:.10f}")
    print(f"  alpha_inv QGT      = {alpha_inv:.6f}   CODATA {CODATA['alpha_inv']:.6f}")
    print(f"  alpha_inv phys     = {alpha_inv_phys:.6f}   err {(alpha_inv_phys/CODATA['alpha_inv']-1)*1e6:.3f} ppm")
    print(f"  sin²θ_W = 3/13     = {3/13:.6f}   err {(3/13/0.23122-1)*100:+.3f}%")
    print(f"  α_s = √140/100     = {np.sqrt(140)/100:.6f}   err {(np.sqrt(140)/100/0.1180-1)*100:+.3f}%")
    print(f"  F_nm               = {F_nm:.8f}")
    print(f"  delta_F (PLL)      = {delta_F*100:.4f}%")
    print(f"  lambda_eff         = {lambda_eff:.6e}")

    print("\n▸ BUTTERFLY CHIRALI (CapClifford)")
    print(f"  M_eta = (1/2)*[[2,1],[1,-1]]:  ||M_eta||_F² = {np.sum(M_eta**2):.4f} = n_conf/n_min² = 7/4")
    print(f"  M_xi:                           ||M_xi||_F²  = {np.sum(M_xi**2):.4f} = c_EW = 13")
    print(f"  sigma1 = sqrt((7-sqrt13)/8) = {sigma1:.6f}  (pulsazione leptonica gen.1)")
    print(f"  sigma2 = sqrt((7+sqrt13)/8) = {sigma2:.6f}  (pulsazione leptonica gen.2)")
    print(f"  (c_EW + n_conf)/kappa² = {(c_EW+n_conf)/5:.1f} = dim(R^4)  [chiusura]")

    print("\n▸ BREATHING")
    xi5, eta5 = breathing_exact(5.0)
    print(f"  xi(5)*eta(5) = {xi5*eta5:.10f}  (alpha_inv = {alpha_inv:.10f})")
    print(f"  invariante esatto: {abs(xi5*eta5-alpha_inv) < 1e-9}")
    t_v = np.linspace(0.01, 1, 500)
    dth = np.gradient(theta_log(t_v), t_v)
    print(f"  dot_theta_log = Gamma_tau = {Gamma_tau:.8f}  verificato: {np.allclose(dth, Gamma_tau, rtol=1e-3)}")

    print("\n▸ SOLITONE")
    print(f"  Psi_star(0) = {soliton(0):.10f}  (phi = {phi:.10f})")
    print(f"  ell_pen(H=1) = {ell_pen():.4f}  [= H/(pi*alpha)]")
    print(f"  identita aurea (phi+1/phi)*sech(ln phi) = {(phi+1/phi)/np.cosh(np.log(phi)):.14f}")

    print("\n▸ CASCATA ADIMENSIONALE")
    print(f"  a0 QGT  = {a0_QGT*1e10:.6f} Å  (CODATA 0.529177 Å)")
    alpha_p = 1/CODATA['alpha_inv']
    R_inf_QGT = alpha_p / (4*np.pi*a0_QGT)
    print(f"  R_inf   = {R_inf_QGT:.4f} m⁻¹  err {(R_inf_QGT/CODATA['R_inf']-1)*1e6:.3f} ppm")
    print(f"  T_CMB   = {T_CMB_QGT:.5f} K  (misurata {CODATA['T_CMB']:.5f} K)")
    print(f"  Omega_b = {Omega_b_h2:.5f}  Omega_c = {Omega_c_h2:.5f}")

    print("\n▸ MASSE LEPTONICHE")
    print(f"  Lambda = alpha_inv*3/8 = {Lambda:.6f}  (in Q[sqrt5])")
    print(f"  m_mu/m_e = {m_mu_me:.4f}  PDG {CODATA['m_mu_me']:.4f}  err {(m_mu_me/CODATA['m_mu_me']-1)*100:+.3f}%")
    print(f"  tau_pi+  = {tau_pi:.5e} s  PDG 2.6033e-8 s  err {(tau_pi/2.6033e-8-1)*1e6:+.2f} ppb  (dimostrato)")
    print(f"  m_tau/m_e: GATE APERTO (sigma3 non esplicitata)")

    print("\n▸ CMB PICCHI STRUTTURALI")
    planck_obs = [220, 540, 810, 1120, 1400, 1560]
    for k in range(1,7):
        lk = alpha_inv*(k*phi+(k-1)/phi)
        print(f"  ell_{k} = {lk:.1f}  (Planck: {planck_obs[k-1]}  err {(lk/planck_obs[k-1]-1)*100:+.1f}%)")

    print("\n▸ CLIFFORD Cl(4,1)")
    g, Pp, Pm, C = build_clifford_qgt()
    print(f"  (gamma^6)^2 = 5*I : {np.allclose(g[6]@g[6], 5*np.eye(4))}")
    print(f"  P_phi+ * P_phi- = 0: {np.allclose(Pp@Pm, 0)}")
    print(f"  gamma^6 = sqrt(5)*gamma^5 (diag(sqrt5,sqrt5,-sqrt5,-sqrt5))")
    M_loc = dirac_operator(xi_star, eta_star)
    eigs  = np.linalg.eigvals(M_loc)
    print(f"  autovalori Dirac al lock-in: {np.round(eigs, 4)}")

    print("\n▸ GATE BOHR-RYDBERG (dimostrato — Cap. 32)")
    # a0 = Gamma_tau^2 * M_B^2 * S^{-2r} m
    # Delta_B = 3phi/S^3 + 3phi^2/S^4 - 3sqrt5/S^5 - n_min/S^6
    # M_B^2 = 1 + Delta_B/Gamma_tau^2
    # Cascata INTERNA: S^1(tau) -> Pi+ -> a0 -> R_infty -> BASE -> m_e,m_mu,m_tau
    # R_infty non e' ancora esterna: e' il readout canonico della pseudoinversa Pi+
    CODATA_a0_val   = 0.529177210903e-10  # m  # m
    CODATA['R_inf'] = 10973731.568        # m^{-1}
    a0_qgt_m  = a0_QGT               # gia' calcolato sopra in metri
    Rinf_qgt  = (1/CODATA['alpha_inv']) / (4*np.pi*a0_qgt_m)
    BASE_QGT  = (1/CODATA['alpha_inv'])**2 / (2 * CODATA['R_inf']) * 2  # keV approximate
    # Massa elettrone da BASE
    sigma1_v = np.sqrt((7 - np.sqrt(13)) / 8)
    H1 = 19
    CODATA_a0_val = 0.529177210903e-10  # m (CODATA 2018)
    alpha_phys    = 1.0 / CODATA['alpha_inv']
    Rinf_qgt      = alpha_phys / (4*np.pi*a0_qgt_m)
    # m_e dalla relazione di Rydberg: m_e*c^2 = 2*h*c*R_infty / alpha^2
    # In unita' comode: m_e [MeV] = 2 * 197.3269804e-15 [MeV*m] * R_infty [m^-1] / alpha^2
    hbarc_MeV_m = 197.3269804e-15
    me_MeV_direct = 2 * hbarc_MeV_m * CODATA['R_inf'] / alpha_phys**2
    me_QGT_MeV    = 4*np.pi * hbarc_MeV_m * Rinf_qgt / alpha_phys**2
    # BASE [keV] = 2*hbar*c*R_infty / (alpha^2 * H1 * sigma1) [monografia: 41.288 keV]
    BASE_keV      = me_QGT_MeV * 1e3 / (H1 * sigma1_v)
    print(f"  Cascata: S^1(tau) -> Pi+ -> a0 -> R_infty -> BASE -> m_e,m_mu,m_tau")
    print(f"  a0_QGT  = {a0_qgt_m*1e10:.7f} Å   (CODATA {CODATA_a0_val*1e10:.7f} Å,  err {(a0_qgt_m/CODATA_a0_val-1)*1e6:.3f} ppm)")
    print(f"  R_inf   = {Rinf_qgt:.4f} m⁻¹  (CODATA {CODATA['R_inf']:.4f},  err {(Rinf_qgt/CODATA['R_inf']-1)*1e6:.3f} ppm)")
    print(f"  m_e     = {me_QGT_MeV:.6f} MeV  (CODATA 0.510999 MeV,  err {(me_QGT_MeV/0.510999-1)*100:.4f}%)")
    print(f"  BASE    = {BASE_keV:.4f} keV  (monografia: 41.288 keV = m_e/(H1*sigma1))")
    print(f"  R_infty: readout canonico di Pi+ (dimostrato)")
    print(f"  v_EW = {v_EW_GeV} GeV (dimostrato, Cap. 35)")

    print("\n▸ GATE APERTI (Cap. 40, 42)")
    print("  kappa_QGT/G_N  : normalizzazione gravitazionale assoluta")
    print("  sigma_3        : m_tau terza generazione leptonica (stima strutturale)")
    print("  G_edge(k,ell)  : propagatore CMB multipolare")
    print("  QCD dinamica   : alpha_s(Q) running, Lambda_QCD, spettro adronico")
    print(f"\n{sep}")


# ================================================================
# 11. PLOTS (matplotlib)
# ================================================================

def plot_breathing(ax):
    t = np.linspace(0, 8, 500)
    xi, eta = breathing_exact(t)
    th = theta_log(t)
    ax.plot(t, xi,  color='#534AB7', lw=1.5, label='ξ (gravità)')
    ax.plot(t, eta, color='#0F6E56', lw=1.5, label='η (EM)')
    ax.plot(t, th * max(eta)/max(th), color='#993556', lw=1, ls='--', label='θ_log (tempo)')
    ax.axhline(alpha_inv, color='#888', lw=0.5, ls=':', label='α⁻¹ (invariante)')
    ax.set_xlabel('t · Γ_τ', fontsize=10)
    ax.set_title('Breathing — rotazione iperbolica su Σ: ξη = α⁻¹', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.text(0.02, 0.95, f'ξ·η = α⁻¹ = {alpha_inv:.4f} esatto', transform=ax.transAxes,
            fontsize=7, va='top', color='#666')

def plot_soliton(ax):
    lp = ell_pen()
    n  = np.linspace(-5*lp, 5*lp, 800)
    ax.plot(n/lp, soliton(n),                           color='#534AB7', lw=2,  label='sech (solitone)')
    ax.plot(n/lp, phi*np.exp(-(n/lp)**2/2),             color='#888',    lw=1,  ls='--', label='Gaussiana')
    ax.plot(n/lp, np.abs(np.tanh(n/lp)),                color='#0F6E56', lw=1,  ls=':', label='|tanh| (parete)')
    ax.axvline(0,  color='#ccc', lw=0.5)
    ax.axhline(phi, color='#534AB7', lw=0.5, ls=':')
    ax.text(0.02, 0.97, f'|Ψ★|(0) = φ = {phi:.6f}', transform=ax.transAxes,
            fontsize=7, va='top', color='#534AB7')
    ax.text(0.02, 0.91, f'ℓ_pen = H/(πα) = {lp:.1f}', transform=ax.transAxes,
            fontsize=7, va='top', color='#666')
    ax.set_xlabel('n / ℓ_pen', fontsize=10)
    ax.set_title('Solitone di lock-in Ψ★(n) = φ·sech(n/ℓ_pen)', fontsize=10)
    ax.legend(fontsize=8)

def plot_scale_cascade(ax):
    B0s = [0.1, 1.0, 5.0, 20.0, 100.0]
    for B0 in B0s:
        traj = [B0]
        B = B0
        for _ in range(40): B = breathing_map(B); traj.append(B)
        ax.plot(traj, lw=1, alpha=0.7, label=f'B₀={B0}')
    ax.axhline(8, color='#993556', lw=1.5, ls='--', label='B★=8 (attrattore)')
    ax.set_xlabel('passi k', fontsize=10)
    ax.set_title('Ricorrenza di scala B_{k+1}=2B_k^{2/3} → B★=8', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 25)

def plot_atom(axes):
    a0_A = a0_QGT * 1e10  # Angstrom
    for idx, n in enumerate([1,2,3,4]):
        ax = axes[idx]
        r_max = n*n*a0_A*4 + a0_A
        r = np.linspace(0.01, r_max, 1000)
        P = hydrogen_radial_density(r, n, a0_A)
        ax.fill_between(r, P, alpha=0.15, color='#0F6E56')
        ax.plot(r, P, color='#0F6E56', lw=1.5)
        ax.axvline(n*n*a0_A, color='#993556', lw=0.8, ls='--')
        En = -13.6 / n**2
        ax.text(0.97, 0.95, f'n={n}\nE={En:.2f} eV', transform=ax.transAxes,
                fontsize=7, ha='right', va='top', color='#333')
        ax.set_xlabel('r (Å)', fontsize=8)
        if idx == 0:
            ax.set_title(f'H atom — a₀={a0_A:.4f} Å da Γ_τ²·M_B²·S⁻²ʳ', fontsize=9)

def plot_butterfly(ax):
    # SVD visualization of M_eta
    u, s, vt = np.linalg.svd(M_eta)
    theta = np.linspace(0, 2*np.pi, 300)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    # unit circle -> M_eta ellipse
    pts = np.stack([circle_x, circle_y])
    mapped = M_eta @ pts
    ax.plot(circle_x, circle_y, color='#888', lw=0.8, ls='--', label='cerchio unitario')
    ax.plot(mapped[0], mapped[1], color='#534AB7', lw=1.5, label='M_eta (ellisse)')
    # singular vectors
    for i in range(2):
        v = vt[i] * s[i]
        ax.annotate('', xy=(mapped[0,int(np.argmax(np.abs(mapped[0])))],
                             mapped[1,int(np.argmax(np.abs(mapped[0])))]),
                    xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#993556', lw=1.2))
    ax.axhline(0, color='#ccc', lw=0.5); ax.axvline(0, color='#ccc', lw=0.5)
    ax.set_aspect('equal')
    ax.set_title(f'SVD di M_eta: σ₁={sigma1:.4f}, σ₂={sigma2:.4f}\n= pulsazioni masse leptoniche', fontsize=9)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.05, f'||M_eta||²_F = {np.sum(M_eta**2):.4f} = 7/4 = n_conf/n_min²',
            transform=ax.transAxes, fontsize=7, color='#666')

def plot_dna(ax):
    (x1,y1,z), (x2,y2,z2) = dna_helix(n_turns=2)
    ax.plot(x1, z, color='#534AB7', lw=1.5, label='filone ξ (backbone)')
    ax.plot(x2, z, color='#0F6E56', lw=1.5, label='filone η (basi)')
    # rungs
    steps = 20
    for k in range(steps+1):
        i = int(k/steps*(len(z)-1))
        ax.plot([x1[i], x2[i]], [z[i], z[i]], color='#ccc', lw=0.6, zorder=0)
    ax.axhline(0, color='#993556', lw=0.3, ls=':')
    ax.set_xlabel('x (raggio normalizzato)', fontsize=9)
    ax.set_ylabel('z (nm)', fontsize=9)
    ax.set_title('B-DNA: Class 2 (2:5) della Dirac nonlineare, k=3\n36°=π/5 per passo, 10 passi/giro = 2×5', fontsize=9)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.97, 'Ψ⁻ = CΨ̄⁺ (condizione Majorana)', transform=ax.transAxes,
            fontsize=7, va='top', color='#666')

def plot_cmb(ax):
    ell = np.linspace(2, 2200, 1200)
    ax.plot(ell, D_ell(ell), color='#534AB7', lw=2, label='QGT strutturale')
    planck_obs = [220, 540, 810, 1120, 1400, 1560]
    planck_amp = [5.5, 2.5, 1.4, 0.8, 0.5, 0.35]
    for i,(lo,la) in enumerate(zip(planck_obs, planck_amp)):
        ax.axvline(lo, color='#993556', lw=0.6, ls='--', alpha=0.6,
                   label='Planck 2018' if i==0 else None)
    ax.set_xlabel('ℓ (multipolo)', fontsize=10)
    ax.set_ylabel('D_ℓ (normalizzato)', fontsize=10)
    ax.set_title('Spettro CMB: 6 picchi sech² strutturali\nposizioni da α⁻¹, φ, Γ_τ — zero parametri liberi', fontsize=9)
    ax.legend(fontsize=8)
    for k in range(1,7):
        lk = alpha_inv*(k*phi+(k-1)/phi)
        ax.text(lk, 0.85, f'ℓ_{k}', fontsize=7, ha='center', color='#534AB7')

def plot_clifford(ax):
    g, Pp, Pm, C = build_clifford_qgt()
    mats  = [g[0].real, g[5].real, g[6].real, Pp.real, Pm.real]
    names = ['γ⁰', 'γ⁵', 'γ⁶=√5·γ⁵', 'P_φ+', 'P_φ-']
    n = len(mats)
    for i, (M, name) in enumerate(zip(mats, names)):
        ax2 = ax.inset_axes([i/n + 0.01, 0.1, 1/n - 0.02, 0.8])
        im = ax2.imshow(M, cmap='RdBu', vmin=-2.5, vmax=2.5, aspect='equal')
        ax2.set_title(name, fontsize=8)
        ax2.set_xticks([]); ax2.set_yticks([])
    ax.set_axis_off()
    ax.set_title('Cl(4,1) — matrici esplicite base Weyl\n'
                 'gamma^6 = sqrt(5)*gamma^5  |  P_phi± = (I ± gamma^5)/2', fontsize=9)

def make_plots(which=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib non installato: pip install matplotlib")
        return

    if which is None:
        which = ['breathing','soliton','scale','atom','butterfly','dna','cmb','clifford']

    # 'atom' occupa 4 colonne in una riga dedicata.
    # Tutti gli altri plot occupano 1 cella ciascuno.
    # Strategia: un unico GridSpec a 4 colonne.
    # Ogni plot non-atom occupa 2 colonne (pannello medio) o 1 colonna (se ce ne sono tanti).
    # Regola semplice:
    #   - costruiamo righe da 2 pannelli (larghi 2 col ciascuno) o da 4 (1 col ciascuno)
    #   - la riga atom occupa sempre 4 colonne (1 per subplot n=1..4)

    NCOLS = 4    # griglia base sempre a 4 colonne
    COL_W = 4.2  # larghezza per colonna in pollici
    ROW_H = 4.0  # altezza per riga in pollici
    ATOM_H = 3.5 # altezza riga atom

    has_atom = 'atom' in which
    others   = [w for w in which if w != 'atom']

    # raggruppa gli altri in righe da 2 (colspan 2 ciascuno)
    # se ce ne sono dispari, l'ultimo va da solo su 4 col
    rows_spec = []  # lista di liste di nomi
    i = 0
    while i < len(others):
        if i + 1 < len(others):
            rows_spec.append([others[i], others[i+1]])
            i += 2
        else:
            rows_spec.append([others[i]])
            i += 1
    if has_atom:
        rows_spec.append(['atom'])   # riga atom in fondo

    nrows = len(rows_spec)
    # altezze delle righe
    height_ratios = []
    for row in rows_spec:
        height_ratios.append(ATOM_H if row == ['atom'] else ROW_H)

    fig_h = sum(height_ratios) + 0.8   # 0.8 per suptitle
    fig_w = NCOLS * COL_W

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle('QGT Simulator — The Quantum Gauge Theory of Time\n'
                 'Camelia 2026 — First Edition KDP', fontsize=12)

    gs = gridspec.GridSpec(
        nrows, NCOLS,
        figure=fig,
        hspace=0.55,
        wspace=0.38,
        top=0.93,
        bottom=0.06,
        left=0.07,
        right=0.97,
        height_ratios=height_ratios,
    )

    axes_map = {}   # name -> ax
    atom_axes = []

    for row_i, row in enumerate(rows_spec):
        if row == ['atom']:
            # 4 subplot affiancati, 1 colonna ciascuno
            for j in range(4):
                ax = fig.add_subplot(gs[row_i, j])
                atom_axes.append(ax)
        elif len(row) == 2:
            # 2 pannelli, 2 colonne ciascuno
            ax0 = fig.add_subplot(gs[row_i, 0:2])
            ax1 = fig.add_subplot(gs[row_i, 2:4])
            axes_map[row[0]] = ax0
            axes_map[row[1]] = ax1
        else:
            # 1 pannello centrato su 4 colonne
            ax = fig.add_subplot(gs[row_i, 0:4])
            axes_map[row[0]] = ax

    # render
    dispatch = {
        'breathing':  plot_breathing,
        'soliton':    plot_soliton,
        'scale':      plot_scale_cascade,
        'butterfly':  plot_butterfly,
        'dna':        plot_dna,
        'cmb':        plot_cmb,
        'clifford':   plot_clifford,
    }
    for name, ax in axes_map.items():
        if name in dispatch:
            dispatch[name](ax)

    if has_atom and atom_axes:
        plot_atom(atom_axes)

    plt.savefig('qgt_simulation.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('qgt_simulation.png', dpi=150, bbox_inches='tight')
    print("Salvato: qgt_simulation.pdf  qgt_simulation.png")
    plt.show()


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    args = sys.argv[1:]
    do_plot = '--plot' in args
    args = [a for a in args if a != '--plot']

    run_all()

    if do_plot:
        which = args if args else None
        make_plots(which)
