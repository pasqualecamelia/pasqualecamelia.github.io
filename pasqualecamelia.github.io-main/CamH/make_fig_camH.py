"""
make_fig_camH.py
================
Generates the CamH validation figure (4 panels):
  A  shell phase theta = gd(Gamma_tau t)  and  sech/tanh quadrature
  B  convergence of Delta(t) -> Delta*  (the lock)
  C  lock / drift diagram across detuning ratio (the threshold)
  D  phase portrait dot_Delta vs Delta  (fixed point + flow)

Reads nothing external; recomputes from camH_core.
"""
import numpy as np
import camH_core as H

def make(save='fig_camH.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    C_xi, C_eta, C_lock, C_drift = '#534AB7', '#0F6E56', '#534AB7', '#993556'
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), facecolor='white')
    fig.suptitle('CamH — QGT phase-locking gluing on hydrogen  (H = p$^+_{k=0}$ + e$^-_{k=2}$)',
                 fontsize=13)

    # ── A: shell phase gd ─────────────────────────────────────────────────
    a = ax[0, 0]
    t = np.linspace(-8, 8, 1000)
    a.plot(t, H.shell_phase(t), color=C_xi, lw=2, label=r'$\theta=\mathrm{gd}(\Gamma_\tau t)$')
    a.plot(t, np.tanh(H.Gamma_tau*t), color=C_eta, lw=1, ls='--', label=r'$\tanh(\Gamma_\tau t)=\sin\theta$')
    a.plot(t, 1/np.cosh(H.Gamma_tau*t), color='#999', lw=1, ls=':', label=r'$\mathrm{sech}(\Gamma_\tau t)=\cos\theta$')
    a.axhline(np.pi/2, color='#ccc', lw=0.5); a.axhline(-np.pi/2, color='#ccc', lw=0.5)
    a.set_title('A. Shell phase: breathing → genuine $S^1$ angle', fontsize=10)
    a.set_xlabel(r'$t$'); a.legend(fontsize=8, loc='lower right')
    a.text(0.03, 0.95, r'$\mathrm{sech}^2+\tanh^2=1$', transform=a.transAxes,
           fontsize=8, va='top', color='#666')

    # ── B: convergence to Delta* ──────────────────────────────────────────
    b = ax[0, 1]
    Dstar = H.fixed_point()
    for D0, col in [(2.5, C_xi), (-2.0, C_eta), (1.0, '#B77A4A')]:
        sol, dw, K = H.integrate_delta(D0=D0, t_max=3000.0)
        tt = np.linspace(0, 3000, 3000)
        b.plot(tt, ((sol.sol(tt)[0]+np.pi) % (2*np.pi))-np.pi, color=col, lw=1.3,
               label=fr'$\Delta_0={D0}$')
    b.axhline(Dstar, color=C_drift, lw=1.2, ls='--',
              label=fr'$\Delta^*={Dstar:.4f}$')
    b.set_title('B. Phase-lock: $\\Delta(t)\\to\\Delta^*$ (gluing)', fontsize=10)
    b.set_xlabel(r'$t$'); b.set_ylabel(r'$\Delta=\theta_y-\theta_x$')
    b.legend(fontsize=8, loc='upper right')
    b.text(0.03, 0.06,
           fr'$\delta\omega/K={H.detuning()/H.coupling():.4f}$  (sub-threshold)',
           transform=b.transAxes, fontsize=8, color='#666')

    # ── C: lock / drift threshold ─────────────────────────────────────────
    c = ax[1, 0]
    K = H.coupling()
    ratios = np.linspace(0, 2.0, 161)
    lock = []
    for rho in ratios:
        s = H.solve_ivp(H.reduced_rhs, [0, 1500], [0.0], args=(rho*K, K),
                        rtol=1e-9, atol=1e-11)
        lock.append(1.0 if (s.y[0,-1]-s.y[0,0]) <= 5 else 0.0)
    lock = np.array(lock)
    c.fill_between(ratios, 0, lock, where=lock>0.5, color=C_lock, alpha=0.25, step='mid')
    c.fill_between(ratios, 0, 1-lock, where=lock<0.5, color=C_drift, alpha=0.20, step='mid')
    c.axvline(1.0, color='k', lw=1.2, ls='--', label=r'threshold $|\delta\omega|=K$')
    c.axvline(H.detuning()/K, color=C_eta, lw=1.5, label=r'H ($\delta\omega/K$)')
    c.set_title('C. Gluing threshold: lock (blue) vs drift (red)', fontsize=10)
    c.set_xlabel(r'$\delta\omega/K$'); c.set_yticks([])
    c.legend(fontsize=8, loc='center right')
    c.set_xlim(0, 2)

    # ── D: phase portrait ─────────────────────────────────────────────────
    d = ax[1, 1]
    dw = H.detuning(); K = H.coupling()
    D = np.linspace(-np.pi, np.pi, 400)
    d.plot(D, dw - K*np.sin(D), color=C_xi, lw=2, label=r'$\dot\Delta=\delta\omega-K\sin\Delta$')
    d.axhline(0, color='#ccc', lw=0.7)
    d.plot(Dstar, 0, 'o', color=C_lock, ms=8, label=fr'stable $\Delta^*$')
    d.plot(np.pi-Dstar, 0, 'o', mfc='white', mec=C_drift, ms=8, label='unstable')
    d.set_title('D. Phase portrait: stable & unstable fixed points', fontsize=10)
    d.set_xlabel(r'$\Delta$'); d.set_ylabel(r'$\dot\Delta$')
    d.legend(fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.savefig(save.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved: {save}, {save.replace('.png','.pdf')}")


if __name__ == '__main__':
    make()
