# QGT — ERRATA & ADDENDA (nota master)
## Pasquale Camelia · 13 giugno 2026 · v1
Registro unico di correzioni e aggiunte emerse dopo la stampa delle due
edizioni della monografia (EN 648pp, IT 432pp) e durante l'audit del
companion. Ogni voce ha: contenuto esatto, dove vive l'errore/novità,
destinazione di correzione, priorità. Le edizioni cartacee NON si toccano:
i veicoli vivi sono i paper Zenodo (versionati) → ResearchGate, e il sito.

---

## E — ERRATA (cose da correggere)

### E1 · P10, abstract e §16: il residuo di α senza baseline CODATA
**Stato attuale (P10):** "α⁻¹_CODATA = (α⁻¹_QGT − 13ε₇)/(1 − 2φ×10⁻⁷),
error 8.6×10⁻⁸, Level A".
**Problema:** il numero è giusto ma riferito a **CODATA 2018**
(137.035999084) senza dirlo. Dal 2024 esiste CODATA 2022 (137.035999177):
contro quello il residuo è **−6.7×10⁻⁹ = 0.32σ**. Citare 8.6×10⁻⁸ "nudo"
oggi sottostima la teoria di un fattore 13 e si fa correggere dal primo
lettore col companion aperto.
**Correzione (testo pronto per P10 v2):**
> "Against CODATA 2018 (137.035999084) the residual is +8.6×10⁻⁸; the
> CODATA 2022 adjustment (137.035999177) moved α⁻¹ by +9.3×10⁻⁸, landing
> at −6.7×10⁻⁹ = 0.32σ from the QGT value."
**Nota strategica:** la frase "the data moved toward the formula" è il
claim più forte del corpus, MA si usa solo se la catena è documentabile
come fissata **prima** del rilascio CODATA 2022. Da verificare con date di
deposito/commit. Se non documentabile: limitarsi a "0.32σ from CODATA 2022".
**Destinazione:** Zenodo P10 v2 → ResearchGate. Priorità: ALTA.

### E2 · P10, Theorem 7.1: mp/me = 6π⁵
**Stato attuale:** P10 lo presenta come Teorema (0.002%), con
giustificazione fattoriale (6=2×3 per SU(2)×SU(3); π⁵ = cinque windings).
**Problema (triplo):** (a) è la coincidenza di **Lenz (1951)** — l'esempio
da manuale di numerologia che i referee citano; (b) **non è nel registro
epistemico della monografia** (Appendix U, entrambe le edizioni); (c) non è
tracciato nel Working Note v63 (il registro autoritativo aperti/chiusi).
P10 contiene quindi un "Teorema" che il resto del corpus deliberatamente
non riconosce.
**Opzioni per P10 v2 (decisione dell'autore):**
- (A) *Downgrade*: da Theorem a "structural observation", con citazione
  esplicita di W. Lenz, Phys. Rev. 82, 554 (1951), e status dichiarato
  "numerical coincidence, derivation open";
- (B) *Rimozione* dalla v2;
- (C) *Derivazione vera* dal framework (se esiste, è un altro paper).
**Nel lancio non entra in nessun caso.** Priorità: ALTA.

### E3 · P10, Implication VII: T_CMB dichiarata "partial" ma ormai chiusa
**Stato attuale (P10):** "the effective mode degeneracy N_eff … not yet
available; the single remaining step".
**Problema:** il Working Note v63 (sessione v26, aprile 2026) chiude
l'Open Block 3 a **Level A strutturale senza R∞**: N_eff = π,
S_osc = ln π, Γτ⁽⁰⁾ = 4π (Cayley + PNT-E),
T_CMB = (π/ln π)(1 − 1/(√2·100)) = 2.724991 K. P10 è stantio rispetto al
proprio corpus: depositarlo così significa auto-smentirsi al ribasso.
**Correzione:** aggiornare §14 e la tabella "What Is Derived" citando la
chiusura (WN v63 §26 o il paper dedicato quando esce). Priorità: ALTA.

### E4 · P10, etichetta "Level A" per la catena α
**Problema:** il registro della monografia classifica α⁻¹_phys =
137.035999170 come **Proposizione** (Cap. 13, condizionata), non Teorema /
Level A pieno; i Teoremi sono a₀ e R∞ a 0.033 ppm (Cap. 32). P10 dice
"Level A".
**Correzione:** allineare l'etichetta di P10 alla scala della monografia
(o motivare esplicitamente la promozione). Priorità: MEDIA.

### E5 · Sito/simulatore: mμ/mₑ con la Λ sbagliata — **CORRETTO OGGI**
**Problema trovato:** `qgt_simulator.py` e il companion calcolavano
Λ = α⁻¹_strutturale·3/8 → mμ/mₑ = 205.6405 (−0.545%), mentre il canone
(Appendix U, entrambe le edizioni stampate; HKY v4) è
Λ = α⁻¹_phys·3/8 → **mμ/mₑ = 205.5714 (−0.58%)**.
**Correzione applicata** (13/06/2026): simulatore e companion ora usano
α_phys, con commento esplicito al canone; verificato live (205.5714).
Tutte le altre costanti invariate (T_CMB 2.72499, a₀ 0.5291772,
α_phys 137.035999170). Destinazione: già nel sito; citare in pagina
errata. Priorità: chiusa.

### E6 · Companion (storico, già corretto il 13/06): scala CMB e R∞
Per il registro: il grafico CMB del companion usava σ=137.082 strutturale
e σ_B=σΓτ/k invece del canone σ=α_phys⁻¹ e σ_B(k)=σΓτ(13−k)/12; il modulo
Spectrum calcolava R∞ via rapporto-α (−0.05 ppb) invece del canonico
α_p/(4π·a₀_QGT) (+32.7 ppb su Lyman-α, +3.977 fm). Entrambi corretti e
verificati headless. Priorità: chiusa.

---

## A — ADDENDA (risultati post-stampa, da propagare)

### A1 · F_π = 130.065 MeV (−0.10% da PDG) — HKY v4, Prop. 30
Test di consistenza di precisione (nessun parametro forte; |Vud| e mπ±
input esterni dichiarati, Gate 4 pendente). **Aggiunto oggi ai gates del
companion** (livello ~ stima strutturale). Destinazione futura: registro
2ª ed. (sezione V.b) quando/se Gate 2B-4 avanzano.

### A2 · Electroweak Lift dimostrato unico — HKY v4, Lemma 22
M_EW = α⁻²_QGT·πS·f_PLL, (p,q,s)=(2,1,1) forzati. Promuove v_EW=246.20
GeV (−0.008%) e G_F=1.1666×10⁻⁵ (−0.015%) a "esatti condizionati a R∞".
Il registro stampato li ha come Pc: la 2ª ed. può citare il lemma di
unicità. Destinazione: 2ª ed.; il sito li ha già.

### A3 · T_CMB senza R∞ (N_eff = π) — WN v63 §26
Vedi E3. Oltre alla correzione di P10, è un upgrade di status per il
registro: T_CMB da P/IS a Level A strutturale "R∞ not required".
Destinazione: paper dedicato o 2ª ed.

### A4 · Identità esatte aggiunte al verify del companion (13/06)
Beat theorem σ₊−σ₋ = 1/2 (HKY v4 Th. 17; verificato: diff 0.0 a precisione
macchina) e q_H¹⁴ = (√2√5)⁻¹⁴ = 10⁻⁷ (diff 4×10⁻²³). Più, a registro:
ΣH_ker+B★ = 19+43+67+8 = 137 = σ; H_μ=(H_e+H_τ)/2=43 esatto;
163 = 137+2·13. Tutte machine-checkable in pagina.

### A5 · τ_π⁺ fisico = 2.60330×10⁻⁸ s — HKY v4 (80)
Con la correzione di residuo (1+ℜz_res): recupera PDG alla precisione
dichiarata. Già nei gates del sito. Gate 2B (τ_h = φ²/10 dal network
completo) resta l'aperto dichiarato.

### A6 · Narrativa CODATA 2018→2022 (vedi E1)
Da usare nel kit/playbook SOLO previa verifica di datazione. Già inserita
nella NOTA NUMERI di 02_primo_commento.md con il caveat esplicito.

---

## Destinazioni — riepilogo operativo
1. **Subito (fatto oggi):** sito allineato (E5, E6, A1, A4) — zip
   aggiornati; kit con nota numeri (E1/A6, E2).
2. **Quando deciderai di metterci mano:** P10 v2 su Zenodo (E1, E2, E3,
   E4 — i testi pronti sono in questa nota) → mirror su ResearchGate.
3. **Pagina "Errata & Addenda" sul sito:** questa nota, tradotta in EN e
   datata, è il contenuto della pagina — il ponte ufficiale tra le
   edizioni stampate e il corpus vivo.
4. **2ª edizione (EN e IT), quando ci sarà massa critica:** A1, A2, A3 +
   prefazione datata con questo registro. Trigger suggerito: chiusura di
   Gate 2B o esito del ciclo arXiv/journal.
