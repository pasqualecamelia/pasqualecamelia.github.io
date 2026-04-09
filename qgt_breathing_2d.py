"""
QGT — Bordo che respira (animazione 2D)
ξ(t) = (1/√α) · exp(+A · sin(Γτ · t))
η(t) = (1/√α) · exp(−A · sin(Γτ · t))
ξη = α⁻¹  conservato ad ogni istante

Γτ = φ/√5,  φ = (1+√5)/2,  α⁻¹ ≈ 137.036
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# ── costanti fisiche ───────────────────────────────────────────────
alpha    = 1 / 137.036
phi      = (1 + np.sqrt(5)) / 2
Gamma    = phi / np.sqrt(5)
inv_a    = 1 / alpha
sq_a     = np.sqrt(inv_a)          # 1/√α ≈ 11.71  (asse del respiro)
A        = 1.8                      # ampiezza log-oscillazione

def xi_eta(t):
    s   = A * np.sin(Gamma * t)
    xi  = sq_a * np.exp( s)
    eta = sq_a * np.exp(-s)
    return xi, eta

# ── figura ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 7), facecolor='#0f0f0e')
gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
               left=0.07, right=0.97, top=0.93, bottom=0.08)

ax_time  = fig.add_subplot(gs[0, :])   # serie temporale (larghezza piena)
ax_phase = fig.add_subplot(gs[1, 0])   # piano di fase
ax_prod  = fig.add_subplot(gs[1, 1])   # prodotto ξη nel tempo

for ax in (ax_time, ax_phase, ax_prod):
    ax.set_facecolor('#0f0f0e')
    for spine in ax.spines.values():
        spine.set_color('#444440')
    ax.tick_params(colors='#888880', labelsize=8)
    ax.xaxis.label.set_color('#888880')
    ax.yaxis.label.set_color('#888880')

col_xi   = '#5DCAA5'
col_eta  = '#7F77DD'
col_axis = '#888880'
col_prod = '#EF9F27'

# ── istantanee iniziali ───────────────────────────────────────────
HIST   = 400
t_arr  = np.zeros(HIST)
xi_arr = np.full(HIST, sq_a)
eta_arr= np.full(HIST, sq_a)
t_hist = np.zeros(HIST)

# Serie temporale
xi_line,  = ax_time.plot([], [], color=col_xi,  lw=1.8, label=r'$\xi$ longitudinale (gravitazionale)')
eta_line, = ax_time.plot([], [], color=col_eta, lw=1.8, label=r'$\eta$ trasverso (EM)')
ax_time.axhline(sq_a,  color=col_axis, lw=0.8, ls='--', alpha=0.6)
ax_time.text(0.01, sq_a + 0.3, r'$1/\sqrt{\alpha} \approx 11.71$',
             color=col_axis, fontsize=8, transform=ax_time.get_yaxis_transform())
ax_time.set_ylim(sq_a * np.exp(-A) * 0.9, sq_a * np.exp(A) * 1.1)
ax_time.set_xlim(0, HIST)
ax_time.set_xlabel(r'$\tau$ (tempo)', fontsize=9)
ax_time.set_ylabel(r'ampiezza', fontsize=9)
ax_time.set_title('Bordo che respira — ξ e η in controfase',
                  color='#ccc8c0', fontsize=11, pad=8)
ax_time.legend(loc='upper right', fontsize=8, framealpha=0.0,
               labelcolor='white')

# Valore corrente
xi_dot,  = ax_time.plot([], [], 'o', color=col_xi,  ms=6, zorder=5)
eta_dot, = ax_time.plot([], [], 'o', color=col_eta, ms=6, zorder=5)
text_info = ax_time.text(0.01, 0.97, '', transform=ax_time.transAxes,
                          color='#ccc8c0', fontsize=8, va='top')

# Piano di fase
xi_range = np.linspace(sq_a * np.exp(-A) * 0.92, sq_a * np.exp(A) * 1.08, 400)
eta_hyp  = inv_a / xi_range
ax_phase.plot(xi_range, eta_hyp, color=col_prod, lw=0.9, ls='--', alpha=0.5,
              label=r'$\xi\eta = \alpha^{-1}$')

# 10 nodi (codoni)
for s in range(10):
    tt = s * (2 * np.pi / Gamma) / 10
    xi_n, eta_n = xi_eta(tt)
    ax_phase.plot(xi_n, eta_n, 'o', color=col_prod, ms=5, zorder=4)

ax_phase.axvline(sq_a, color=col_axis, lw=0.7, ls='--', alpha=0.5)
ax_phase.axhline(sq_a, color=col_axis, lw=0.7, ls='--', alpha=0.5)
ax_phase.set_xlim(sq_a * np.exp(-A) * 0.88, sq_a * np.exp(A) * 1.12)
ax_phase.set_ylim(sq_a * np.exp(-A) * 0.88, sq_a * np.exp(A) * 1.12)
ax_phase.set_xlabel(r'$\xi$ (longitudinale)', fontsize=9)
ax_phase.set_ylabel(r'$\eta$ (trasverso)', fontsize=9)
ax_phase.set_title('Piano di fase', color='#ccc8c0', fontsize=10, pad=6)
ax_phase.legend(fontsize=7, framealpha=0, labelcolor='#EF9F27')

phase_traj, = ax_phase.plot([], [], color='#444440', lw=0.8, alpha=0.7)
phase_dot,  = ax_phase.plot([], [], 'o', color=col_eta, ms=7, zorder=6)

# Prodotto ξη
prod_line, = ax_prod.plot([], [], color=col_prod, lw=1.6)
ax_prod.axhline(inv_a, color=col_axis, lw=0.8, ls='--', alpha=0.6)
ax_prod.text(0.01, 0.96, r'$\alpha^{-1} = 137.036$',
             transform=ax_prod.transAxes, color=col_axis, fontsize=8, va='top')
ax_prod.set_xlim(0, HIST)
ax_prod.set_ylim(inv_a * 0.985, inv_a * 1.015)
ax_prod.set_xlabel(r'$\tau$', fontsize=9)
ax_prod.set_ylabel(r'$\xi \cdot \eta$', fontsize=9)
ax_prod.set_title(r'Invariante $\xi\eta = \alpha^{-1}$ (costante)',
                  color='#ccc8c0', fontsize=10, pad=6)

fig.patch.set_facecolor('#0f0f0e')

# ── animazione ────────────────────────────────────────────────────
frame_t = [0.0]

def animate(frame):
    frame_t[0] += 0.06

    t_hist[:-1] = t_hist[1:]
    t_hist[-1]  = frame_t[0]

    xi_arr[:-1]  = xi_arr[1:]
    eta_arr[:-1] = eta_arr[1:]
    xi, eta      = xi_eta(frame_t[0])
    xi_arr[-1]   = xi
    eta_arr[-1]  = eta

    idx = np.arange(HIST)

    xi_line.set_data(idx, xi_arr)
    eta_line.set_data(idx, eta_arr)
    xi_dot.set_data([HIST-1], [xi])
    eta_dot.set_data([HIST-1], [eta])

    phase_traj.set_data(xi_arr[-150:], eta_arr[-150:])
    phase_dot.set_data([xi], [eta])

    prod_line.set_data(idx, xi_arr * eta_arr)

    cycle = int(frame_t[0] * Gamma / (2 * np.pi))
    step  = int(((frame_t[0] * Gamma) % (2 * np.pi)) / (2 * np.pi / 10))
    text_info.set_text(
        f'ξ = {xi:.3f}   η = {eta:.3f}   '
        f'√(ξη) = {np.sqrt(xi*eta):.3f} = 1/√α   '
        f'giro {cycle}   passo {step}/10'
    )

    return (xi_line, eta_line, xi_dot, eta_dot,
            phase_traj, phase_dot, prod_line, text_info)

ani = animation.FuncAnimation(fig, animate, interval=20,
                               blit=True, cache_frame_data=False)

plt.show()
