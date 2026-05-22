# QGT Companion Simulator v2.0

**The Quantum Gauge Theory of Time** — Pasquale Camelia (2026)  
ORCID: [0009-0006-4779-4647](https://orcid.org/0009-0006-4779-4647)  
Repository: [pasqualecamelia.github.io](https://pasqualecamelia.github.io)

---

## Installazione

```bash
pip install numpy>=1.24 scipy>=1.10 matplotlib>=3.7
```

Testato con: Python 3.10–3.12, NumPy 1.26/2.4, SciPy 1.11/1.14.

---

## Uso rapido

```bash
# Verifica strutturale con livelli epistemici espliciti
python -m qgt_companion verify

# Simulazione dinamica atomo di idrogeno (ITP Crank-Nicolson)
python -m qgt_companion hydrogen

# Tutto insieme
python -m qgt_companion all

# Con grafici matplotlib (salva qgt_simulation.pdf e .png)
python -m qgt_companion all --plot

# Grafici selezionati
python -m qgt_companion --plot breathing soliton cmb hydrogen
```

---

## Struttura

```
qgt_companion/
  qgt_core.py       Costanti strutturali φ, Γ_τ, α⁻¹, a₀_QGT,
                    breathing, solitone, CMB, CODATA 2018
  qgt_clifford.py   Algebra Cl(4,1), proiettori chirali P_φ±,
                    coniugazione Majorana, verifica spinore fisico
  qgt_hydrogen.py   Simulazione ITP Crank-Nicolson, cascata Bohr-Rydberg
  qgt_plots.py      Visualizzazioni matplotlib
  qgt_verify.py     Verifica strutturale con livelli epistemici
  __main__.py       Entry point
requirements.txt
README.md
```

---

## Livelli epistemici

Ogni risultato è etichettato esplicitamente:

| Simbolo | Livello | Significato |
|---------|---------|-------------|
| ✓ | **LEVEL_A** | Derivazione chiusa, zero parametri liberi |
| ~ | **LEVEL_B** | Connessione fisica + assunzioni addizionali |
| ⚠ | **OPEN** | Porta aperta / gate residuo |

---

## Simulazione idrogeno — cosa fa e cosa NON fa

### Cosa fa (strumento numerico di verifica)

Verifica che `a₀_QGT` — derivato dalla catena interna LEVEL_A

```
S¹(τ) → Π⁺ → a₀_QGT → R_∞ → BASE → mₑ, mμ, mτ
```

sia consistente con la meccanica quantistica standard dell'idrogeno,
usandolo come unica scala di lunghezza nell'Hamiltoniano radiale.

Metodo: **Imaginary-Time Propagation (ITP)** con schema Crank-Nicolson.

```
H = -1/2 · d²/dρ² - 1/ρ,    ρ = r / a₀_QGT

(I + dτ/2·H) u_{k+1} = (I - dτ/2·H) u_k    [Crank-Nicolson, stabile]
u_{k+1} ← u_{k+1} / ‖u_{k+1}‖              [rinormalizzazione]
```

### Risultati attesi (con N=600, ρ_max=25)

| Osservabile | ITP | Esatto | Errore |
|-------------|-----|--------|--------|
| E₁ | −13.6016 eV | −13.6057 eV | 4 meV |
| ρ_peak | 0.998 a₀ | 1.000 a₀ | 0.2% |
| ⟨ρ⟩ | 1.501 a₀ | 1.500 a₀ | 0.1% |
| 2⟨T⟩+⟨V⟩ | −0.0036 a.u. | 0 | — |

L'errore di 4 meV è interamente da discretizzazione O(dρ²) con
dρ = 25/601 ≈ 0.042. È molto minore di kT_ambiente = 26 meV.

### Cosa NON fa

Questo modulo **non** risolve le equazioni della QGT. Non è una
derivazione della struttura atomica dalla teoria. È un test di
consistenza numerico: verifica che `a₀_QGT` sia la scala corretta
per l'idrogeno standard. La derivazione di `a₀_QGT` è in `qgt_core.py`
e nella monografia (Cap. Gate Bohr-Rydberg, LEVEL_A).

---

## Note sulla griglia ITP

L'errore di discretizzazione scala come O(dρ²) = O((ρ_max/N)²):

| N | ρ_max | dρ | Errore E₁ atteso |
|---|-------|-----|-----------------|
| 300 | 25 | 0.083 | ~16 meV |
| 600 | 25 | 0.042 | ~4 meV  |
| 1200 | 25 | 0.021 | ~1 meV  |

Per N=600 il costo computazionale è < 5 s su qualsiasi laptop moderno.

---

## Correzioni rispetto a v1 (script singolo)

| Problema | v1 | v2 |
|----------|----|----|
| Struttura | script singolo | 5 moduli separati |
| Idrogeno | densità statica | ITP Crank-Nicolson dinamica |
| Viriale | inconsistente (dot vs trapez.) | tutto con trapezoid |
| DNA z-bug | `z *= n_turns` due volte | corretto |
| numpy 2.x | `np.trapz` (rimosso) | `np.trapezoid` |
| Livelli epistemici | nel docstring | stampati e distinti |
| Clifford | test algebrico | + spinore Majorana fisico |
| Δ_B | formula senza commenti | formula con derivazione esplicita |
| m_mu/m_e | errore non motivato | nota su H_ker come interi strutturali |
| convergence | threshold implicita | 10 meV < kT, giustificata |

---

## Riferimenti

- Koonin & Meredith, *Computational Physics* (Addison-Wesley, 1990)
- Baye & Heenen, J. Phys. A **19** (1986) 2041
- Lehtovaara et al., J. Comput. Phys. **221** (2007) 148
- CODATA 2018: https://physics.nist.gov/cuu/Constants/
- Camelia, P. *The Quantum Gauge Theory of Time* (2026), First Edition KDP

---

## Licenza / License

© 2026 Pasquale Camelia — ORCID: [0009-0006-4779-4647](https://orcid.org/0009-0006-4779-4647)

Questo codice e la monografia associata sono distribuiti sotto licenza
**[Creative Commons Attribuzione 4.0 Internazionale (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.

This code and the associated monograph are licensed under the
**[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**.

È consentita la riproduzione, distribuzione, adattamento e uso commerciale,
a condizione di citare:

> Camelia, P. *The Quantum Gauge Theory of Time*, First Edition, 2026.  
> ORCID: 0009-0006-4779-4647 — https://pasqualecamelia.github.io

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
