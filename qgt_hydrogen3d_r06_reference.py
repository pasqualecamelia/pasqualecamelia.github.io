#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from numpy.fft import fftn, ifftn, fftfreq

OUT=Path('/mnt/data/QGT_Hydrogen3D_r06_GUIDED_PHYSICAL')
DATA=OUT/'data'; DATA.mkdir(parents=True,exist_ok=True)

# Dimensionless Schrödinger-grade QGT hydrogen coordinates:
# rho = r/a_Q, tau = E_Q t/hbar, H0/E_Q = -1/2 nabla_rho^2 - 1/rho.
N=48; dx=0.5; dt=0.18; dtau=0.05
xs=(np.arange(N)-N/2+0.5)*dx
X,Y,Z=np.meshgrid(xs,xs,xs,indexing='ij')
R=np.sqrt(X*X+Y*Y+Z*Z)
V=-1.0/R
k=fftfreq(N,d=dx)*2*np.pi
KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij')
K2=(KX*KX+KY*KY+KZ*KZ)/2

# Numerical basis states, normalized on the same finite grid.
def norm(psi): return psi/np.sqrt(np.sum(np.abs(psi)**2))
psi1=norm(np.exp(-R).astype(complex))
psi2=norm((Z*np.exp(-R/2)).astype(complex))
psi3=norm(((3*Z*Z-R*R)*np.exp(-R/3)).astype(complex))

# --- A. Numerical eigenstate preparation (imaginary time, not physical time) ---
prep=norm(np.exp(-(R**2)/(2*3.0**2)).astype(complex))
kinI=np.exp(-K2*dtau); potI=np.exp(-0.5*V*dtau)
prep_rows=[]; prep_snaps=[]
for step in range(121):
    if step%4==0:
        fid=float(abs(np.vdot(psi1,prep))**2)
        meanr=float(np.real(np.vdot(prep,R*prep)))
        prep_rows.append({'step':step,'tau_imag':step*dtau,'fidelity_1s':fid,'mean_r':meanr})
        prep_snaps.append(np.abs(prep[:,:,N//2])**2)
    prep=potI*prep
    prep=ifftn(fftn(prep)*kinI)
    prep=potI*prep
    prep=norm(prep)

# --- B. Bound-atom life: exact hydrogenic phase evolution of a prepared superposition ---
E1,E2,E3=-0.5,-0.125,-1/18
c1,c2,c3=1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)
life_rows=[]; life_snaps=[]
for j,t in enumerate(np.linspace(0,32,65)):
    psi=norm(c1*psi1*np.exp(-1j*E1*t)+c2*psi2*np.exp(-1j*E2*t)+c3*psi3*np.exp(-1j*E3*t))
    p=np.abs(psi)**2
    z=float(np.real(np.vdot(psi,Z*psi)))
    meanr=float(np.real(np.vdot(psi,R*psi)))
    life_rows.append({'frame':j,'t':float(t),'dipole_z':z,'mean_r':meanr})
    life_snaps.append(p[:,:,N//2])

# --- C. True dipole light pulse in the Schrödinger Hamiltonian ---
# H(t)=H0 - E(t) z, with near-resonant omega = E2-E1 = 3/8.
light=psi1.copy()
kinR=np.exp(-1j*K2*dt)
E0=0.12; omega=3/8; Tpulse=30.0
light_rows=[]; light_snaps=[]
for step in range(181):
    t=step*dt
    env=math.sin(math.pi*t/Tpulse)**2 if 0<=t<=Tpulse else 0.0
    field=E0*env*math.cos(omega*t)
    if step%4==0:
        den=float(np.sum(np.abs(light)**2))
        z=float(np.real(np.vdot(light,Z*light))/den)
        p1=float(abs(np.vdot(psi1,light))**2/den)
        p2=float(abs(np.vdot(psi2,light))**2/den)
        light_rows.append({'step':step,'t':t,'field':field,'dipole_z':z,'P_1s':p1,'P_2pz':p2,'norm':den})
        light_snaps.append(np.abs(light[:,:,N//2])**2)
    Vt=V-field*Z
    ph=np.exp(-0.5j*Vt*dt)
    light=ph*light
    light=ifftn(fftn(light)*kinR)
    light=ph*light

summary={
 'model':'QGT Schrödinger-grade dimensionless hydrogen',
 'scaled_equation':'i dpsi/dtau = [-1/2 nabla_rho^2 - 1/rho - F(tau) rho_z] psi',
 'grid':{'N':N,'dx':dx,'extent':float(N*dx/2)},
 'preparation':{
   'status':'numerical eigenstate preparation, not physical radiative recombination',
   'initial_fidelity_1s':prep_rows[0]['fidelity_1s'],
   'final_fidelity_1s':prep_rows[-1]['fidelity_1s'],
   'initial_mean_r':prep_rows[0]['mean_r'],'final_mean_r':prep_rows[-1]['mean_r']
 },
 'life':{
   'status':'analytic unitary hydrogenic superposition in the closed Coulomb sector',
   'dipole_min':min(x['dipole_z'] for x in life_rows),
   'dipole_max':max(x['dipole_z'] for x in life_rows)
 },
 'light':{
   'status':'real-time split-operator with explicit dipole coupling -E(t) z',
   'E0_scaled':E0,'omega_scaled':omega,'pulse_duration_scaled':Tpulse,
   'max_abs_dipole':max(abs(x['dipole_z']) for x in light_rows),
   'max_P_2pz':max(x['P_2pz'] for x in light_rows),
   'min_P_1s':min(x['P_1s'] for x in light_rows),
   'max_norm_drift':max(abs(x['norm']-1) for x in light_rows)
 },
 'epistemic_firewall':{
   'radiative_capture':'not simulated; requires photon-emission/open-QED extension',
   'old_legacy_breathing_term':'not used in r06 baseline',
   'absolute_alpha_Q_and_mu_Q':'scale calibration remains separate from the dimensionless dynamics'
 }
}
(DATA/'summary.json').write_text(json.dumps(summary,indent=2))
(DATA/'preparation.json').write_text(json.dumps(prep_rows,indent=2))
(DATA/'life.json').write_text(json.dumps(life_rows,indent=2))
(DATA/'light.json').write_text(json.dumps(light_rows,indent=2))
np.savez_compressed(DATA/'slices.npz', prep=np.asarray(prep_snaps), life=np.asarray(life_snaps), light=np.asarray(light_snaps), xs=xs)
print(json.dumps(summary,indent=2))
