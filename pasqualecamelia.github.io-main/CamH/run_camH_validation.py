"""
run_camH_validation.py
======================
CamH validation runner — analogous to run_qgt_gate3_camb.py (CamCMB).

Runs the QGT phase-locking gluing on hydrogen, writes:
  - metrics_camH.json   : all canonical inputs, the three checks, CODATA cross-checks
  - camH_lock_curve.csv : Delta(t) convergence trajectory
  - camH_threshold.csv  : lock vs drift sweep across detuning

The hydrogen CROSS-CHECK uses the monograph's own derived observables
(a0, R_inf, alpha^-1) compared to CODATA, so the module also reports
the metrological side of "H as the first closed QGT object" (appB).
"""
import numpy as np
import json, os
import camH_core as H

OUT = os.path.dirname(os.path.abspath(__file__))

# ── CODATA reference for the hydrogen cross-check ─────────────────────────
CODATA = {
    'alpha_inv': 137.035999177,
    'R_inf_m':   10973731.568,        # m^-1
    'a0_A':      0.529177210903,      # Angstrom
    'm_e_MeV':   0.51099895,
}

def hydrogen_metrology():
    """Monograph-derived H observables vs CODATA (appB Bohr-Rydberg gate)."""
    phi = H.phi; Gt = H.Gamma_tau
    # a0 = Gamma_tau^2 + Delta_B  (structural cascade, appB / c-cascade)
    Delta_B = 3*phi*1e-3 + 3*phi**2*1e-4 - 3*np.sqrt(5)*1e-5 - 2e-6   # Angstrom
    a0_static = phi**2 / 5
    a0_A = a0_static*(1 + Delta_B/a0_static)
    alpha_phys = 1/CODATA['alpha_inv']
    R_inf = alpha_phys / (4*np.pi*a0_A*1e-10)
    return {
        'a0_A': a0_A,
        'a0_err_ppm': (a0_A/CODATA['a0_A']-1)*1e6,
        'R_inf_m': R_inf,
        'R_inf_err_ppm': (R_inf/CODATA['R_inf_m']-1)*1e6,
    }

def main():
    r = H.validate_hydrogen(verbose=True)
    met = hydrogen_metrology()

    print("\n▸ HYDROGEN METROLOGY CROSS-CHECK (appB, vs CODATA)")
    print(f"  a0     = {met['a0_A']:.7f} A   err {met['a0_err_ppm']:+.3f} ppm")
    print(f"  R_inf  = {met['R_inf_m']:.4f} m^-1   err {met['R_inf_err_ppm']:+.3f} ppm")

    # ── trajectory CSV ────────────────────────────────────────────────────
    sol, dw, K = H.integrate_delta(D0=2.5, t_max=4000.0)
    tt = np.linspace(0, 4000, 4000)
    Dt = sol.sol(tt)[0]
    np.savetxt(os.path.join(OUT, 'camH_lock_curve.csv'),
               np.column_stack([tt, Dt]),
               header='t, Delta(t)  [phase-difference convergence to Delta*]',
               delimiter=',', comments='# ')

    # ── threshold sweep CSV ───────────────────────────────────────────────
    ratios = np.linspace(0, 2.0, 81)            # delta_omega / K
    locked = []
    for rho in ratios:
        dw_s = rho*K
        s = H.solve_ivp(H.reduced_rhs, [0, 2000], [0.0], args=(dw_s, K),
                        rtol=1e-9, atol=1e-11)
        adv = s.y[0, -1] - s.y[0, 0]
        locked.append(0.0 if adv > 5 else 1.0)  # 1=lock, 0=drift
    np.savetxt(os.path.join(OUT, 'camH_threshold.csv'),
               np.column_stack([ratios, locked]),
               header='delta_omega/K, locked(1)/drift(0)',
               delimiter=',', comments='# ')

    # ── metrics JSON ──────────────────────────────────────────────────────
    metrics = {
        "description": "CamH — QGT phase-locking gluing validated on hydrogen",
        "monograph_version": "QGT Monograph v176 (KDP 2026)",
        "formulation": ("Gluing of two boundary cells = stable phase-locking of "
                        "FMT-read shell amplitudes under Noether-current cancellation."),
        "canonical_inputs": {
            "Gamma_tau": float(H.Gamma_tau),
            "delta_F":   float(H.delta_F),
            "delta_omega_formula": "Gamma_tau * delta_F  (physical-time normalisation)",
            "delta_omega": float(r['delta_omega']),
            "mu_Pi_overlap": "dim(T8)/dim(B13) = 8/13",
            "mu_Pi_value": float(r['mu_overlap']),
            "kappa_0": "Gamma_tau (FMT frequency scale)",
            "K": float(r['K']),
            "ratio_dw_over_K": float(r['ratio']),
            "note": "sine = boundary current from FMT overlap; nothing tuned."
        },
        "checks": {
            "breathing_fixed_point_B8": bool(r['B_star_check']),
            "shell_phase_S1_residual": float(r['lock_identity_residual']),
            "gluing_condition_met": bool(r['glued']),
            "a_converges_to_Delta_star": bool(r['converged']),
            "Delta_star_rad": float(r['Delta_star']),
            "b_drift_above_threshold": bool(r['drifts_above_threshold']),
            "drift_phase_advance_rad": float(r['drift_phase_advance']),
            "c_margin_nonzero": bool(r['margin_nonzero']),
            "sin_Delta_star": float(r['sin_Delta_star']),
            "sin_Delta_star_over_delta_F": float(r['sin_over_deltaF']),
            "sin_Delta_star_EXACT": "sin(Delta*) = (13/8) delta_F  [exact, = dim(B13)/dim(T8) * delta_F]",
            "Delta_star_over_delta_F": float(r['margin_over_deltaF']),
            "Delta_star_over_delta_F_note": "= 13/8 + O(delta_F^2); the 13/8 ratio is exact for sin(Delta*), first-order for the angle",
        },
        "hydrogen_metrology_vs_CODATA": {
            "a0_Angstrom": met['a0_A'], "a0_err_ppm": met['a0_err_ppm'],
            "R_inf_m": met['R_inf_m'], "R_inf_err_ppm": met['R_inf_err_ppm'],
            "lockin_invariant_alpha_inv": float(H.alpha_inv),
        },
        "epistemic_status": {
            "internal_derivation": "closed (functional form)",
            "gluing_threshold": "derived",
            "phase_sine": "derived from quantum FMT, not imported",
            "minimal_detuning": "identified with delta_F (after time normalisation)",
            "coupling_K": "fixed by kernel-ribbon measure 8/13 (non-tunable)",
            "kuramoto": "NOT named in monograph; reduced eq. belongs to the known "
                        "phase-locking class — bibliographic note only, downstream."
        }
    }
    with open(os.path.join(OUT, 'metrics_camH.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved: metrics_camH.json, camH_lock_curve.csv, camH_threshold.csv")


if __name__ == '__main__':
    main()
