"""
camH_core.py
============
CamH — validation of the QGT gluing mechanism on the hydrogen channel.

CamH validates the QGT gluing mechanism on the hydrogen channel, understood
as the first closed boundary composite (p+_{k=0} + e-_{k=2}), by showing
that the FMT-induced shell phases satisfy the Noether-current cancellation
condition through a stable sub-threshold phase lock.

This is NOT a derivation of standard atomic hydrogen: the Bohr-Rydberg
numbers (a0, R_inf) are a metrological CROSS-CHECK, not a proof of the
Coulomb bond. What is validated is the dynamical gluing of the first closed
boundary composite. Companion to CamCMB (same mechanism, cosmological scale).

PHYSICS (Monograph + session derivation, NO free parameters)
------------------------------------------------------------
The gluing of two boundary cells is the stable phase-locking of their
FMT-read shell amplitudes under the Noether-current cancellation condition.

1. Shell phase. On the lock-in surface xi*eta = alpha^{-1} the breathing
   flow (c05_mother eq:breathing) dot_xi=-Gamma_tau xi, dot_eta=+Gamma_tau eta
   has the circular representation (B6.14, eq:bulk_gd):
       Xi_bulk = phi * e^{-i gd(Gamma_tau t)} ,
   so the shell carries a genuine S^1 phase
       theta(t) = gd(Gamma_tau t)            [gd = Gudermannian]
   with sech^2 + tanh^2 = 1 (eq:lock_identity).

2. Sinusoidal boundary current (NOT assumed). The shared-face flux is the
   imaginary overlap of the two shell amplitudes read in the quantum-FMT
   basis (v2orig_c10 eq:FMT-low/high), where amplitudes are pure phases:
       Phi_xy = Im < F_phi psi_x , F_phi psi_y >_phi = sin(theta_y - theta_x).
   The sine is a boundary current, not a coupling force put in by hand.

3. Coupled phase dynamics + reduced equation:
       dot_theta_x = omega_x + (K_xy/2) sin(theta_y - theta_x)
       dot_theta_y = omega_y + (K_xy/2) sin(theta_x - theta_y)
       => dot_Delta = delta_omega_xy - K_xy sin(Delta),  Delta = theta_y - theta_x.

4. Gluing condition (dynamical form of gluing-condition 5):
       |delta_omega_xy| <= K_xy
   -> stable fixed point sin(Delta*) = delta_omega/K, the normal flux cancels
      dynamically, and K_tau = xi*eta = alpha^{-1} is transported (cond. 4).
   Otherwise the phase drifts: cells do not glue.

CANONICAL (non-tunable) inputs for H
------------------------------------
  Gamma_tau = phi/sqrt5                              tick rate            [c05]
  delta_F   = 1 - 3pi/(2 sqrt5)^{3/2}                irreducible margin   [c10a]
  detuning  delta_omega = Gamma_tau * delta_F        (physical-time norm) [doc]
  coupling  K_xy = kappa_0 * mu_Pi(ker overlap)                          [doc]
            mu_Pi = dim(T_8)/dim(B_13) = 8/13   (shared kernel ribbon)   [c12]
            kappa_0 = Gamma_tau           (only FMT frequency scale)
  H modes:  p+ (k=0, xi>>eta),  e- (k=2, eta>>xi); lock at B*=8, f'(8)=2/3 [appB]

Released under MIT (original code). © 2026 P. Camelia logic; this validation
module written as a companion to the monograph.
"""
import numpy as np
from scipy.integrate import solve_ivp

# ── structural constants (all from the monograph) ─────────────────────────
phi       = (1 + np.sqrt(5)) / 2
Gamma_tau = phi / np.sqrt(5)
alpha_inv = (5 + 3*np.sqrt(5))**2          # 20 phi^4 (structural pole)
alpha_inv_phys = 137.035999177             # physical (CODATA)
delta_F   = 1 - 3*np.pi/(2*np.sqrt(5))**1.5

# B13 = S5 + T8 ribbon split  [c12_matter]
DIM_S5, DIM_T8, DIM_B13 = 5, 8, 13
assert DIM_S5 + DIM_T8 == DIM_B13

# breathing fixed point  [c05_mother]
B_STAR = 8
def breathing_map(B):           # B_{k+1} = 2 B^{2/3}, attractor 8, f'(8)=2/3
    return 2.0 * B**(2/3)


# ── 1. shell phase = gd(Gamma_tau t) ──────────────────────────────────────
def gudermann(x):
    return 2*np.arctan(np.tanh(x/2))

def shell_phase(t):
    """theta(t) = gd(Gamma_tau t), the S^1 phase of the breathing shell."""
    return gudermann(Gamma_tau * t)

def lock_identity_residual(t):
    """sech^2 + tanh^2 - 1 on the breathing trajectory (must be ~0)."""
    s = 1/np.cosh(Gamma_tau*t); c = np.tanh(Gamma_tau*t)
    return np.max(np.abs(s**2 + c**2 - 1))


# ── 2. FMT quantum overlap -> sinusoidal boundary current ──────────────────
def fmt_amplitude(theta):
    """Shell amplitude read in the quantum-FMT basis: a pure phase e^{i theta}
    (modulus fixed by the lock-in xi*eta = alpha^{-1})."""
    return np.exp(1j*theta)

def boundary_flux(theta_x, theta_y):
    """Phi_xy = Im < F_phi psi_x, F_phi psi_y > = sin(theta_y - theta_x).
    Derived (not assumed): the imaginary overlap of two FMT-read phases."""
    ax, ay = fmt_amplitude(theta_x), fmt_amplitude(theta_y)
    return np.imag(np.conj(ax) * ay)       # = sin(theta_y - theta_x)


# ── 3. canonical detuning and coupling (non-tunable) ──────────────────────
def detuning(physical_time=True):
    """delta_omega between the two dual modes.
    physical_time=True : delta_omega = Gamma_tau * delta_F   (doc, physical t)
    physical_time=False: delta_omega = delta_F               (rescaled s=Gamma t)
    """
    return Gamma_tau * delta_F if physical_time else delta_F

def kernel_overlap_measure():
    """mu_Pi(ker Pi_x ∩ ker Pi_y) for the two dual modes of the hydrogen
    channel (n=5,k=2).

    STRUCTURAL CLAIM (required for K=Gamma_tau*(8/13) to be Level A):
    in the hydrogen channel the shared kernel ribbon is precisely the T_8
    sector of the branch B_13 = S_5 (+) T_8 [c12_matter]. With that
    identification the normalised overlap measure is
        mu_Pi = dim(T_8)/dim(B_13) = 8/13,
    and the coupling is fixed (non-tunable). For channels other than H this
    measure must be recomputed per bond/pair; only the H value is closed here.
    """
    return DIM_T8 / DIM_B13

def coupling(physical_time=True):
    """K_xy = kappa_0 * mu_Pi(ker overlap), kappa_0 = Gamma_tau.
    Non-tunable: fixed by the FMT frequency scale and the ribbon split."""
    kappa_0 = Gamma_tau if physical_time else 1.0
    return kappa_0 * kernel_overlap_measure()


# ── 4. reduced phase-difference dynamics ──────────────────────────────────
def reduced_rhs(t, D, dw, K):
    return dw - K*np.sin(D)

def integrate_delta(D0=2.5, t_max=4000.0, physical_time=True):
    """Integrate dot_Delta = delta_omega - K sin(Delta) to the fixed point."""
    dw = detuning(physical_time); K = coupling(physical_time)
    sol = solve_ivp(reduced_rhs, [0, t_max], [D0], args=(dw, K),
                    dense_output=True, rtol=1e-12, atol=1e-14, max_step=t_max/2e4)
    return sol, dw, K

def fixed_point(physical_time=True):
    """Delta* = arcsin(delta_omega / K), the stable locked phase difference."""
    dw = detuning(physical_time); K = coupling(physical_time)
    if abs(dw) > K:
        return None                       # no lock (drift)
    return np.arcsin(dw / K)

def is_glued(physical_time=True):
    """Dynamical gluing condition |delta_omega| <= K."""
    return abs(detuning(physical_time)) <= coupling(physical_time)


# ── 5. validation bundle ───────────────────────────────────────────────────
def validate_hydrogen(verbose=True):
    """Run the three required checks (doc):
       (a) convergence to Delta*,
       (b) loss of lock above threshold,
       (c) residual margin != 0 proportional to delta_F.
    Returns a results dict."""
    res = {}

    # breathing fixed point
    res['B_star_check'] = abs(breathing_map(B_STAR) - B_STAR) < 1e-12
    res['B_star_slope'] = (4/3)*(B_STAR**(-1/3))      # f'(8) = 2/3

    # shell phase is a genuine S^1 phase
    res['lock_identity_residual'] = lock_identity_residual(np.linspace(-8, 8, 4001))

    # canonical numbers
    dw = detuning(); K = coupling()
    res['Gamma_tau'] = Gamma_tau
    res['delta_F'] = delta_F
    res['delta_omega'] = dw
    res['mu_overlap'] = kernel_overlap_measure()
    res['K'] = K
    res['ratio'] = dw/K
    res['glued'] = is_glued()

    # (a) convergence to Delta*
    sol, _, _ = integrate_delta(D0=2.5)
    Dstar = fixed_point()
    D_end = sol.y[0, -1]
    wrap = lambda a: (a + np.pi) % (2*np.pi) - np.pi
    res['Delta_star'] = Dstar
    res['Delta_end'] = D_end
    res['converged'] = abs(wrap(D_end - Dstar)) < 1e-4

    # (b) loss of lock above threshold (force dw' = 1.5 K)
    K_ = coupling()
    sol2 = solve_ivp(reduced_rhs, [0, 4000], [0.0], args=(1.5*K_, K_),
                     rtol=1e-10, atol=1e-12)
    res['drift_phase_advance'] = sol2.y[0, -1] - sol2.y[0, 0]
    res['drifts_above_threshold'] = res['drift_phase_advance'] > 10

    # (c) residual margin proportional to delta_F
    #     Delta* ~ delta_omega/K = (Gamma_tau delta_F)/K  for small ratio
    res['margin_nonzero'] = abs(Dstar) > 1e-9
    # EXACT relation: sin(Delta*) = (delta_omega/K) = (13/8) delta_F.
    # The ANGLE ratio Delta*/delta_F = arcsin((13/8)dF)/dF = 13/8 + O(dF^2),
    # i.e. 13/8 only to first order — NOT exact for the angle.
    res['sin_Delta_star'] = np.sin(Dstar)                 # = (13/8) delta_F, exact
    res['sin_over_deltaF'] = np.sin(Dstar)/delta_F        # = 13/8 exactly
    res['margin_over_deltaF'] = Dstar/delta_F             # = 13/8 + O(dF^2), first order only

    # lock-in invariant transported across the pair
    res['lockin_invariant'] = alpha_inv                 # xi*eta shared

    if verbose:
        _print_report(res)
    return res


def _print_report(r):
    sep = "="*64
    print(sep)
    print("CamH — QGT gluing on the hydrogen channel (first closed composite)")
    print(sep)
    print("\n▸ STRUCTURAL ANCHORS (monograph)")
    print(f"  Gamma_tau = phi/sqrt5            = {r['Gamma_tau']:.8f}")
    print(f"  delta_F   = 1-3pi/(2sqrt5)^1.5   = {r['delta_F']:.6e}")
    print(f"  breathing fixed point B*=8       : {r['B_star_check']}  (f'(8)={r['B_star_slope']:.4f})")
    print(f"  shell phase gd: sech^2+tanh^2=1  : residual {r['lock_identity_residual']:.2e}")

    print("\n▸ CANONICAL GLUING INPUTS (non-tunable)")
    print(f"  detuning  delta_omega = Gamma_tau*delta_F   = {r['delta_omega']:.6e}")
    print(f"  overlap   mu_Pi = dim(T8)/dim(B13) = 8/13   = {r['mu_overlap']:.6f}")
    print(f"  coupling  K = Gamma_tau*mu_Pi               = {r['K']:.6f}")
    print(f"  ratio     delta_omega/K                     = {r['ratio']:.6e}")
    print(f"  gluing condition |delta_omega| <= K         : {r['glued']}")

    print("\n▸ VALIDATION (the three required checks)")
    print(f"  (a) convergence to Delta*        : {r['converged']}")
    print(f"      Delta*  = arcsin(dw/K)       = {r['Delta_star']:.8f} rad")
    print(f"      Delta(t->inf)                = {r['Delta_end']:.8f} rad (mod 2pi)")
    print(f"  (b) drift above threshold        : {r['drifts_above_threshold']}")
    print(f"      phase advance (dw=1.5K, t=4000)= {r['drift_phase_advance']:.1f} rad")
    print(f"  (c) residual margin != 0         : {r['margin_nonzero']}")
    print(f"      sin(Delta*) / delta_F        = {r['sin_over_deltaF']:.6f}  = 13/8 EXACT")
    print(f"      Delta* / delta_F             = {r['margin_over_deltaF']:.6f}  = 13/8 + O(dF^2) (first order)")

    print("\n▸ PHYSICAL READOUT")
    print(f"  H = p+ (k=0) + e- (k=2) lock at B*=8; sub-threshold stable lock.")
    print(f"  Locked phase Delta* != 0 -> synchronisation WITH margin = arrow of time.")
    print(f"  Lock-in invariant transported: xi*eta = alpha^-1 = {r['lockin_invariant']:.4f}")
    print(f"\n  NOTE: the sine is a boundary current from the FMT overlap, not an")
    print(f"        assumed coupling. K and delta_omega are fixed by the kernel")
    print(f"        ribbon and delta_F respectively; nothing is tuned to the answer.")
    print(sep)


if __name__ == '__main__':
    validate_hydrogen()
