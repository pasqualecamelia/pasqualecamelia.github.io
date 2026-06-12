"""
qgt_hydrogen3d.py — Simulazione 3D dinamica dell'atomo di idrogeno QGT
Derivato da: The Quantum Gauge Theory of Time (Camelia, 2026)
P08, P11; CapSolitoni; CapMateria; CapQM

══════════════════════════════════════════════════════════════════
EQUAZIONI
══════════════════════════════════════════════════════════════════
Equazione di Schrödinger 3D con potenziale QGT (unità atomiche):

    iħ ∂Ψ/∂t = H_QGT Ψ

    H_QGT = -½∇² + V_QGT(r,t)

    V_QGT(r,t) = -1/r                          (Coulomb)
               + α·M_eff · sech²(r/ℓ_pen)      (back-action QGT, LEVEL_A)
               + f_t·cos(2Γτ·t)·exp(-r²/2)     (breathing modulazione, LEVEL_A)

dove:
  M_eff = α·√(ξ★²+η★²)  — massa effettiva al lock-in
  ℓ_pen = H/(π·α)        — lunghezza di penetrazione
  f_t   = Γτ²/2          — frequenza di breathing
  Γτ    = φ/√5            — connessione temporale dorata

Il termine sech² è il back-action del kernel ker(Π) sulla funzione d'onda
di bordo — il residuo proiettivo che modifica il potenziale Coulomb puro.

Il termine cos(2Γτ·t) è la modulazione di breathing: ξ e η oscillano
sul piano invariante Σ:ξη=α⁻¹ con frequenza Γτ, e la loro modulazione
produce una correzione periodica alla funzione d'onda.

══════════════════════════════════════════════════════════════════
METODO NUMERICO
══════════════════════════════════════════════════════════════════
Split-operator (Fourier):
  Ψ(t+dt) = exp(-iV·dt/2) · IFFT[exp(-iK·dt) · FFT[exp(-iV·dt/2)·Ψ]]

dove K = |k|²/2 è l'operatore cinetico in spazio k.
Questo metodo è:
  - unitario (norma conservata esattamente)
  - secondo ordine in dt
  - scalabile: O(N³ log N) per step

══════════════════════════════════════════════════════════════════
LIVELLI EPISTEMICI
══════════════════════════════════════════════════════════════════
LEVEL_A:
  - Potenziale Coulomb -1/r
  - Scala a0_QGT = Γτ² a0 (dal corpus, CapMateria)
  - Frequenza di breathing f_t = Γτ²/2
  - Termine sech² dal back-action di ker(Π)
  - Stato ground: E1 = -0.5 a.u. = -13.606 eV

LEVEL_B:
  - Ampiezza del termine breathing (usa f_t come stima)

OPEN:
  - Accoppiamento completo con la Dirac nonlineare 4D
  - Back-action su (ξ,η) dalla soluzione
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq

from qgt_companion.qgt_core import (
    phi, Gamma_tau, alpha, alpha_inv,
    xi_star, eta_star, ell_pen,
)

# ════════════════════════════════════════════════════════════════
# Parametri fisici (unità atomiche: ħ=m=e=1, a0=1)
# ════════════════════════════════════════════════════════════════
A0_QGT  = Gamma_tau**2          # Bohr radius QGT ≈ 0.5236 a0  [CapMateria]
F_T     = Gamma_tau**2 / 2      # frequenza breathing           [P08]
M_EFF   = alpha * np.sqrt(xi_star**2 + eta_star**2)  # massa lock-in [P11]
ELL_PEN = ell_pen()             # lunghezza di penetrazione


def build_grid(N=48, L=20.0):
    """
    Griglia 3D cartesiana centrata in 0.
    N: numero di punti per lato
    L: semi-estensione in unità a0
    """
    dx   = 2*L / N
    x    = np.linspace(-L + dx/2, L - dx/2, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R    = np.sqrt(X**2 + Y**2 + Z**2) + 1e-6   # evita divisione per zero
    return X, Y, Z, R, dx


def hydrogen_1s(X, Y, Z, a0=None):
    """
    Stato 1s dell'idrogeno: Ψ₁ₛ = (1/√π·a0³) · exp(-r/a0)
    Usa a0_QGT come scala (LEVEL_A).
    """
    if a0 is None: a0 = A0_QGT
    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-10
    psi = np.exp(-R / a0)
    psi /= np.sqrt((np.abs(psi)**2).sum())    # normalizzazione discreta (L²)
    return psi.astype(complex)


def hydrogen_2pz(X, Y, Z, a0=None):
    """
    Stato 2pz: Ψ₂ₚz = (1/4√2π·a0³) · (r/a0)·exp(-r/2a0)·cos(θ)
    """
    if a0 is None: a0 = A0_QGT
    R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-10
    psi = (R / (2*a0)) * np.exp(-R / (2*a0)) * Z / R
    psi /= np.sqrt((np.abs(psi)**2).sum())
    return psi.astype(complex)


def superposition_1s_2pz(X, Y, Z, a0=None, c1=0.7, c2=0.7):
    """
    Sovrapposizione 1s + 2pz per mostrare la dinamica.
    """
    psi1 = hydrogen_1s(X, Y, Z, a0)
    psi2 = hydrogen_2pz(X, Y, Z, a0)
    psi  = c1*psi1 + c2*psi2
    psi /= np.sqrt((np.abs(psi)**2).sum())
    return psi


def potential_qgt(R, t=0.0):
    """
    Potenziale QGT completo (LEVEL_A):

        V(r,t) = -1/r
               + α·M_eff · sech²(r/ℓ_pen)        [back-action ker(Π)]
               + f_t·cos(2Γτ·t)·exp(-r²/2)         [breathing modulazione]

    Il termine sech² è la correzione strutturale QGT al potenziale Coulomb.
    Il termine cos oscilla con il breathing ξ↔η al lock-in.
    """
    V_coulomb  = -1.0 / R
    V_backreact = alpha * M_EFF * (1.0 / np.cosh(R / ELL_PEN))**2
    V_breathing = F_T * np.cos(2 * Gamma_tau * t) * np.exp(-R**2 / 2.0)
    return V_coulomb + V_backreact + V_breathing


def kinetic_operator_k2(N, dx):
    """
    Operatore cinetico in spazio k: K = |k|²/2
    """
    kx = fftfreq(N, d=dx) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
    return (KX**2 + KY**2 + KZ**2) / 2.0


def evolve_split_operator(psi, V, K2, dt, n_steps,
                          R=None, t0=0.0, callback=None, callback_every=10):
    """
    Evoluzione in tempo reale con split-operator (Fourier).

    Ψ(t+dt) = exp(-iV·dt/2) · IFFT[exp(-iK²·dt) · FFT[exp(-iV·dt/2)·Ψ]]

    Parametri
    ---------
    psi         : funzione d'onda iniziale (complessa, N³)
    V           : potenziale al t=0 (aggiornato ad ogni step se R fornito)
    K2          : |k|²/2 in spazio k
    dt          : passo temporale
    n_steps     : numero di passi
    R           : griglia radiale (per aggiornare V con breathing)
    t0          : tempo iniziale
    callback    : funzione chiamata ogni callback_every passi con (t, psi, norm)
    callback_every : frequenza callback

    Ritorna
    -------
    psi_final, t_final, norms (lista delle norme ad ogni callback)
    """
    exp_K = np.exp(-1j * K2 * dt)
    norms = []
    t = t0

    for step in range(n_steps):
        # Aggiorna potenziale con breathing (tempo-dipendente)
        if R is not None:
            V = potential_qgt(R, t)

        # Half-step nel potenziale
        psi = psi * np.exp(-0.5j * V * dt)

        # Step completo in k
        psi = ifftn(fftn(psi) * exp_K)

        # Half-step nel potenziale
        psi = psi * np.exp(-0.5j * V * dt)

        t += dt

        if step % callback_every == 0:
            norm = float(np.sqrt((np.abs(psi)**2).sum()))
            norms.append((t, norm))
            if callback is not None:
                callback(t, psi, norm)

    return psi, t, norms


def energy_expectation(psi, V, K2, dx):
    """
    Valore atteso dell'energia: <E> = <T> + <V>
    in unità atomiche.
    """
    dV = dx**3

    # <V>
    E_V = float(np.real(np.sum(np.conj(psi) * V * psi))) * dV

    # <T> via FFT
    psi_k = fftn(psi) * dx**3
    E_T   = float(np.real(np.sum(np.conj(psi_k) * K2 * psi_k))) / (2*np.pi)**3 * (2*np.pi/dx)**3 / (dx**3)
    # Semplificato: usa identità di Parseval
    E_T   = float(np.real(np.sum(K2 * np.abs(fftn(psi))**2))) * dV / (dx**3 * np.prod(psi.shape))

    return E_T + E_V, E_T, E_V


def run_hydrogen3d_simulation(N=48, L=15.0, dt=0.02, n_steps=200,
                               initial='superposition', verbose=True):
    """
    Simulazione 3D completa dell'idrogeno QGT.

    Parametri
    ---------
    N        : punti per lato della griglia (default 48 → 48³ ≈ 110k punti)
    L        : semi-estensione della griglia in a0 (default 15)
    dt       : passo temporale in a.u. (default 0.02)
    n_steps  : numero di passi (default 200 = tempo totale 4 a.u.)
    initial  : stato iniziale ('1s', '2pz', 'superposition')
    verbose  : stampa output

    Ritorna
    -------
    dict con: psi_t (lista snapshot), times, density_t, norms,
              E_initial, grid_params
    """
    sep = "═" * 60

    if verbose:
        print(sep)
        print("  QGT IDROGENO 3D — Simulazione dinamica")
        print(sep)
        print(f"  Griglia: {N}³ = {N**3:,} punti, L={L} a0")
        print(f"  dt = {dt} a.u.,  T_totale = {dt*n_steps:.2f} a.u.")
        print(f"  Stato iniziale: {initial}")
        print(f"  a0_QGT = Γτ² = {A0_QGT:.6f} a0")
        print(f"  f_t    = {F_T:.6f}  (freq. breathing)")
        print()

    # ── Griglia ──────────────────────────────────────────────
    X, Y, Z, R, dx = build_grid(N, L)

    # ── Stato iniziale ────────────────────────────────────────
    if initial == '1s':
        psi0 = hydrogen_1s(X, Y, Z)
    elif initial == '2pz':
        psi0 = hydrogen_2pz(X, Y, Z)
    else:
        psi0 = superposition_1s_2pz(X, Y, Z)

    # ── Operatori ─────────────────────────────────────────────
    V0 = potential_qgt(R, t=0.0)
    K2 = kinetic_operator_k2(N, dx)

    # ── Energia iniziale ──────────────────────────────────────
    E0, ET0, EV0 = energy_expectation(psi0, V0, K2, dx)

    if verbose:
        print(f"  Energia iniziale: E = {E0:.6f} a.u. = {E0*27.211:.4f} eV")
        print(f"    <T> = {ET0:.6f} a.u.")
        print(f"    <V> = {EV0:.6f} a.u.")
        print(f"  E1 atteso: -0.500 a.u. = -13.606 eV")
        print()

    # ── Evoluzione ────────────────────────────────────────────
    snapshots_psi  = [psi0.copy()]
    snapshots_dens = [np.abs(psi0)**2]
    snapshot_times = [0.0]

    save_every = max(1, n_steps // 20)   # 20 snapshot totali

    def on_step(t, psi, norm):
        pass   # callback vuoto — usiamo save_every sotto

    psi = psi0.copy()
    norms_list = []
    t = 0.0

    if verbose:
        print(f"  Evoluzione in corso ({n_steps} passi)...")

    exp_K = np.exp(-1j * K2 * dt)

    for step in range(n_steps):
        V = potential_qgt(R, t)
        psi = psi * np.exp(-0.5j * V * dt)
        psi = ifftn(fftn(psi) * exp_K)
        psi = psi * np.exp(-0.5j * V * dt)
        t += dt

        norm = float(np.sqrt((np.abs(psi)**2).sum()))  # norma L² discreta
        norms_list.append((t, norm))

        if (step + 1) % save_every == 0:
            snapshots_psi.append(psi.copy())
            snapshots_dens.append(np.abs(psi)**2)
            snapshot_times.append(t)

    # Energia finale
    Ef, ETf, EVf = energy_expectation(psi, potential_qgt(R, t), K2, dx)

    if verbose:
        print(f"  ✓ Evoluzione completata")
        print()
        print(f"  Energia finale:   E = {Ef:.6f} a.u. = {Ef*27.211:.4f} eV")
        print(f"  Norma finale:     {norms_list[-1][1]:.10f}  (deve essere 1.0000)")
        print(f"  Snapshot salvati: {len(snapshots_psi)}")
        print(sep)

    return {
        'psi_t':       snapshots_psi,
        'density_t':   snapshots_dens,
        'times':       np.array(snapshot_times),
        'norms':       norms_list,
        'E_initial':   E0,
        'E_final':     Ef,
        'grid':        (X, Y, Z, R, dx),
        'N':           N,
        'L':           L,
        'dt':          dt,
        'a0_qgt':      A0_QGT,
        'f_t':         F_T,
        'initial':     initial,
    }


def plot_hydrogen3d(result, t_idx=0, iso_level=0.02,
                    title=None, show=True):
    """
    Visualizza la densità |Ψ|² come isosuperficie 3D con plotly.

    Parametri
    ---------
    result    : output di run_hydrogen3d_simulation
    t_idx     : indice del snapshot temporale (0=iniziale)
    iso_level : valore dell'isosuperficie (relativo al massimo)
    title     : titolo del grafico
    show      : mostra il grafico
    """
    try:
        import plotly.graph_objects as go
        from skimage.measure import marching_cubes
    except ImportError:
        print("Installa plotly e scikit-image per la visualizzazione 3D:")
        print("  pip install plotly scikit-image")
        return None

    X, Y, Z, R, dx = result['grid']
    N = result['N']
    dens = result['density_t'][t_idx]
    t    = result['times'][t_idx]

    # Isosuperficie con marching cubes
    level = iso_level * dens.max()
    try:
        verts, faces, _, _ = marching_cubes(dens, level=level, spacing=(dx,dx,dx))
    except Exception:
        print(f"Marching cubes fallito con level={level:.4e}")
        return None

    # Centra i vertici
    L = result['L']
    verts -= L

    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            opacity=0.5,
            colorscale='Blues',
            intensity=verts[:,2],
            showscale=False,
        )
    ])

    if title is None:
        title = f"QGT Idrogeno 3D — |Ψ|² — t={t:.3f} a.u."

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x (a₀)',
            yaxis_title='y (a₀)',
            zaxis_title='z (a₀)',
            aspectmode='cube',
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='#0a0a14',
        font_color='white',
    )

    if show:
        fig.show()
    return fig


def plot_density_slices(result, t_idx=0, show=True):
    """
    Visualizza la densità |Ψ|² su tre piani di taglio (xy, xz, yz)
    con matplotlib — funziona senza plotly.
    """
    import matplotlib.pyplot as plt

    X, Y, Z, R, dx = result['grid']
    N = result['N']
    dens = result['density_t'][t_idx]
    t    = result['times'][t_idx]
    mid  = N // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"QGT Idrogeno 3D — |Ψ(r,t)|² — t = {t:.3f} a.u.  "
                 f"[a₀_QGT = {result['a0_qgt']:.4f}]",
                 color='white', fontsize=13)
    fig.patch.set_facecolor('#0a0a14')

    L = result['L']
    ext = [-L, L, -L, L]

    titles = ['Piano z=0 (xy)', 'Piano y=0 (xz)', 'Piano x=0 (yz)']
    slices = [dens[:,:,mid], dens[:,mid,:], dens[mid,:,:]]

    for ax, sl, ttl in zip(axes, slices, titles):
        im = ax.imshow(sl.T, extent=ext, origin='lower',
                       cmap='inferno', interpolation='bilinear')
        ax.set_title(ttl, color='white', fontsize=11)
        ax.set_xlabel('a₀', color='white')
        ax.set_ylabel('a₀', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def run_animation_data(N=40, L=12.0, dt=0.05, n_steps=100, initial='superposition'):
    """
    Genera i dati per un'animazione della densità 3D.
    Ritorna solo le sezioni centrali (piano z=0) per leggerezza.
    Usato dalla GUI qgt_companion.html.
    """
    result = run_hydrogen3d_simulation(N, L, dt, n_steps, initial, verbose=False)
    mid    = N // 2
    L_grid = result['L']

    frames = []
    for i, (t, dens) in enumerate(zip(result['times'], result['density_t'])):
        frames.append({
            't':     float(t),
            'slice': dens[:,:,mid].tolist(),   # piano z=0
            'norm':  float(result['norms'][i*max(1,n_steps//20)][1])
                     if i < len(result['norms'])//(max(1,n_steps//20)) else 1.0,
        })

    return {
        'frames':   frames,
        'L':        L_grid,
        'a0_qgt':   result['a0_qgt'],
        'f_t':      result['f_t'],
        'E0':       result['E_initial'],
        'initial':  initial,
    }


if __name__ == '__main__':
    print("QGT Idrogeno 3D — test rapido (N=32, 50 passi)")
    result = run_hydrogen3d_simulation(N=32, L=12.0, dt=0.05, n_steps=50,
                                       initial='superposition', verbose=True)
    print(f"\nDensità massima finale: {result['density_t'][-1].max():.6e}")
    print(f"Tempo finale: {result['times'][-1]:.3f} a.u.")
