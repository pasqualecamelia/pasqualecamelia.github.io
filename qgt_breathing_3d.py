"""
QGT — Doppia elica come bordo che respira (animazione 3D)
I due filamenti oscillano in controfase: raggio ∝ ξ(t) e η(t)
10 nodi per giro (codoni), angolo 36° = π/5

Richiede: matplotlib, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gridspec

# ── costanti fisiche ───────────────────────────────────────────────
alpha  = 1 / 137.036
phi    = (1 + np.sqrt(5)) / 2
Gamma  = phi / np.sqrt(5)
inv_a  = 1 / alpha
sq_a   = np.sqrt(inv_a)        # 1/√α ≈ 11.71
A      = 2.2                    # ampiezza log-oscillazione (più visibile)
STEPS  = 10                     # passi per giro (codoni)
N_TURNS= 1.5                    # numero di giri visualizzati
N_PTS  = 400                    # punti per filamento

def xi_eta(t):
    s = A * np.sin(Gamma * t)
    return sq_a * np.exp(s), sq_a * np.exp(-s)

def radius_norm(v):
    """Normalizza ampiezza in raggio visivo 0.25..0.90"""
    vmin = sq_a * np.exp(-A)
    vmax = sq_a * np.exp( A)
    return 0.20 + (v - vmin) / (vmax - vmin) * 0.72

# ── figura ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 7), facecolor='#0f0f0e')
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.05,
                         left=0.02, right=0.98, top=0.94, bottom=0.04)

ax3d  = fig.add_subplot(gs[0, 0], projection='3d')
ax_ph = fig.add_subplot(gs[0, 1])

# Stile 3D
ax3d.set_facecolor('#0f0f0e')
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor('#222220')
ax3d.yaxis.pane.set_edgecolor('#222220')
ax3d.zaxis.pane.set_edgecolor('#222220')
ax3d.tick_params(colors='#444440', labelsize=0)
ax3d.grid(False)
ax3d.set_xlim(-1.1, 1.1)
ax3d.set_ylim(-1.1, 1.1)
ax3d.set_zlim(-1.2, 1.2)
ax3d.set_xlabel('', fontsize=0)
ax3d.set_ylabel('', fontsize=0)
ax3d.set_zlabel('τ', color='#666660', fontsize=9)
ax3d.view_init(elev=22, azim=-55)

# Stile piano di fase
ax_ph.set_facecolor('#0f0f0e')
for sp in ax_ph.spines.values():
    sp.set_color('#444440')
ax_ph.tick_params(colors='#888880', labelsize=8)

col_xi   = '#5DCAA5'
col_eta  = '#7F77DD'
col_node = '#EF9F27'
col_axis = '#666660'
col_hyp  = '#EF9F27'

# ── piano di fase (statico + punto mobile) ─────────────────────────
xi_range = np.linspace(sq_a * np.exp(-A)*0.90, sq_a * np.exp(A)*1.10, 400)
ax_ph.plot(xi_range, inv_a / xi_range, color=col_hyp, lw=1.0, ls='--',
           alpha=0.45, label=r'$\xi\eta = \alpha^{-1}$')
for s in range(STEPS):
    tt = s * (2 * np.pi / Gamma) / STEPS
    xn, yn = xi_eta(tt)
    ax_ph.plot(xn, yn, 'o', color=col_node, ms=5, zorder=4)
ax_ph.axvline(sq_a, color=col_axis, lw=0.7, ls='--', alpha=0.4)
ax_ph.axhline(sq_a, color=col_axis, lw=0.7, ls='--', alpha=0.4)
ax_ph.set_xlim(sq_a * np.exp(-A)*0.85, sq_a * np.exp(A)*1.15)
ax_ph.set_ylim(sq_a * np.exp(-A)*0.85, sq_a * np.exp(A)*1.15)
ax_ph.set_xlabel(r'$\xi$ (longitudinale, gravitazionale)', color='#888880', fontsize=9)
ax_ph.set_ylabel(r'$\eta$ (trasverso, EM)', color='#888880', fontsize=9)
ax_ph.set_title('Piano di fase — orbita sull\'iperbole',
                color='#ccc8c0', fontsize=10, pad=6)
ax_ph.text(sq_a * 0.75, sq_a * np.exp(A) * 1.02,
           r'$\sqrt{\xi\eta} = 1/\sqrt{\alpha} \approx 11.71$',
           color=col_axis, fontsize=8)
ax_ph.legend(fontsize=8, framealpha=0, labelcolor=col_hyp, loc='upper right')

traj_ph,  = ax_ph.plot([], [], color='#333330', lw=1.0, alpha=0.8)
dot_ph,   = ax_ph.plot([], [], 'o', color=col_eta, ms=8, zorder=6)

# ── elementi 3D (linee aggiornate ad ogni frame) ──────────────────
# asse centrale
zv = np.linspace(-1.2, 1.2, 2)
ax3d.plot([0, 0], [0, 0], zv, color=col_axis, lw=0.8, ls='--', alpha=0.5)
ax3d.text(0, 0, 1.3, r'$1/\sqrt{\alpha}$', color=col_axis, fontsize=8, ha='center')

# nodi codoni statici (calcolati a t=0, posizione angolare fissa)
node_angles = np.arange(int(N_TURNS * STEPS)) * (2*np.pi/STEPS)
node_angles = node_angles[node_angles < N_TURNS * 2 * np.pi]
node_zs = -1 + node_angles / (N_TURNS * 2 * np.pi) * 2
# nodi iniziali a equilibrio
r0 = radius_norm(sq_a)
node_sc = ax3d.scatter(
    r0 * np.cos(node_angles),
    r0 * np.sin(node_angles),
    node_zs,
    c=col_node, s=22, zorder=5, depthshade=False
)

fil_plus,  = ax3d.plot([], [], [], color=col_xi,  lw=2.5, alpha=0.92)
fil_minus, = ax3d.plot([], [], [], color=col_eta, lw=2.5, alpha=0.92)
# scatter supplementari per mostrare il gonfiamento con dimensione punto
scat_plus  = ax3d.scatter([], [], [], c=col_xi,  s=[], alpha=0.30, depthshade=False)
scat_minus = ax3d.scatter([], [], [], c=col_eta, s=[], alpha=0.30, depthshade=False)
head_plus, = ax3d.plot([], [], [], 'o', color='white', ms=6, mec=col_xi, mew=2.0, zorder=6)

title_txt  = ax3d.text2D(0.02, 0.97, '', transform=ax3d.transAxes,
                          color='#ccc8c0', fontsize=9, va='top')
info_txt   = ax_ph.text(0.02, 0.04, '', transform=ax_ph.transAxes,
                         color='#ccc8c0', fontsize=8, va='bottom')

fig.patch.set_facecolor('#0f0f0e')
fig.suptitle('QGT — Doppia elica come bordo che respira',
             color='#ccc8c0', fontsize=12, y=0.98)

# ── storia piano di fase ──────────────────────────────────────────
HIST = 280
xi_hist  = np.full(HIST, sq_a)
eta_hist = np.full(HIST, sq_a)

# ── animazione ────────────────────────────────────────────────────
frame_t = [0.0]
azim    = [-55.0]

def animate(frame):
    frame_t[0] += 0.055
    t = frame_t[0]
    azim[0] = (azim[0] + 0.18) % 360
    ax3d.view_init(elev=22, azim=azim[0])

    xi_c, eta_c = xi_eta(t)

    # storia piano di fase
    xi_hist[:-1]  = xi_hist[1:];  xi_hist[-1]  = xi_c
    eta_hist[:-1] = eta_hist[1:]; eta_hist[-1] = eta_c
    traj_ph.set_data(xi_hist[-120:], eta_hist[-120:])
    dot_ph.set_data([xi_c], [eta_c])

    # elica a finestra scorrevole: z fisso -1..+1, angolo avanza con t
    ang_start = Gamma * t
    ang_arr   = np.linspace(ang_start - N_TURNS*2*np.pi, ang_start, N_PTS)
    z_arr     = np.linspace(-1.0, 1.0, N_PTS)

    # raggio locale: ogni punto è nel passato di (ang_start-ang)/Gamma
    t_local = t - (ang_start - ang_arr) / Gamma

    xi_loc  = sq_a * np.exp( A * np.sin(Gamma * t_local))
    eta_loc = sq_a * np.exp(-A * np.sin(Gamma * t_local))
    r_plus  = radius_norm(xi_loc)
    r_minus = radius_norm(eta_loc)

    x_plus  = r_plus  * np.cos(ang_arr)
    y_plus  = r_plus  * np.sin(ang_arr)
    x_minus = r_minus * np.cos(ang_arr + np.pi)
    y_minus = r_minus * np.sin(ang_arr + np.pi)

    fil_plus.set_data_3d(x_plus, y_plus, z_arr)
    fil_minus.set_data_3d(x_minus, y_minus, z_arr)

    # scatter con size ∝ raggio per mostrare gonfiamento
    # campiono ogni 8 punti per non sovraccaricare
    step = 12
    s_plus  = (r_plus[::step]  * 28)**1.8
    s_minus = (r_minus[::step] * 28)**1.8
    scat_plus._offsets3d  = (x_plus[::step],  y_plus[::step],  z_arr[::step])
    scat_minus._offsets3d = (x_minus[::step], y_minus[::step], z_arr[::step])
    scat_plus.set_sizes(s_plus)
    scat_minus.set_sizes(s_minus)

    # testa = cima dell'elica, sempre a z=+1
    head_plus.set_data_3d([x_plus[-1]], [y_plus[-1]], [1.0])

    # nodi su entrambi i filamenti — stessa finestra scorrevole
    ang_nodes_abs = ang_start - N_TURNS*2*np.pi + node_angles
    t_node = t - (ang_start - ang_nodes_abs) / Gamma
    xi_node = sq_a * np.exp( A * np.sin(Gamma * t_node))
    eta_node= sq_a * np.exp(-A * np.sin(Gamma * t_node))
    r_plus_n  = radius_norm(xi_node)
    r_minus_n = radius_norm(eta_node)
    # interleave: nodi sul verde (filamento +) e sul blu (filamento −, sfasato π)
    all_x = np.concatenate([r_plus_n  * np.cos(ang_nodes_abs),
                             r_minus_n * np.cos(ang_nodes_abs + np.pi)])
    all_y = np.concatenate([r_plus_n  * np.sin(ang_nodes_abs),
                             r_minus_n * np.sin(ang_nodes_abs + np.pi)])
    frac_nodes = (ang_nodes_abs - (ang_start - N_TURNS*2*np.pi)) / (N_TURNS*2*np.pi)
    z_nodes_cur = -1.0 + 2.0 * np.clip(frac_nodes, 0, 1)
    all_z = np.concatenate([z_nodes_cur, z_nodes_cur])
    node_sc._offsets3d = (all_x, all_y, all_z)

    cycle = int(t * Gamma / (2 * np.pi))
    step  = int(((t * Gamma) % (2 * np.pi)) / (2 * np.pi / STEPS))
    direction = 'ξ↑ η↓ — verde gonfio, blu stretto' if xi_c > sq_a else 'ξ↓ η↑ — blu gonfio, verde stretto'
    title_txt.set_text(f'θ = {((t*Gamma) % (2*np.pi))*180/np.pi:.1f}°   '
                        f'passo {step}/10   giro {cycle}\n{direction}')
    info_txt.set_text(f'ξ = {xi_c:.3f}   η = {eta_c:.3f}   '
                       f'ξη = {xi_c*eta_c:.2f}   √(ξη) = {np.sqrt(xi_c*eta_c):.3f}')

    return fil_plus, fil_minus, scat_plus, scat_minus, head_plus, traj_ph, dot_ph, title_txt, info_txt, node_sc

ani = animation.FuncAnimation(fig, animate, interval=18,
                               blit=False, cache_frame_data=False)

plt.show()
