# REGISTRAZIONE — Frozen Recoil-Branch Prediction (Register A: discretezza)

Blindaggio di priorita' della predizione cieca QGT sul reticolo alpha^-1.
Contenuto cieco (Register A): una determinazione futura di alpha^-1 da recoil
cade sul reticolo w in {-0.7236, 0, +0.2764}, NON off-lattice. La parte Rb/Cs
e' dichiarata CONDIZIONALE (Register B) nel file stesso.

## Gli artefatti e i loro hash (i BYTE ESATTI da ancorare)

- SORGENTE (canonico):  frozen_recoil_prediction.tex
  SHA-256: 9d68a090008606e1f31b659bb870c9236c528c7d733a1e84e8abb7aa22f14572

- PDF (deterministico, SOURCE_DATE_EPOCH=1783123200, bit-riproducibile):
  frozen_recoil_prediction.pdf
  SHA-256: 56b2d567c4804256d494ee7ec80276f7f337316e33a85159b432c52b74c9bd53

Nota: il PDF si rigenera identico da chiunque con
  SOURCE_DATE_EPOCH=1783123200 pdflatex frozen_recoil_prediction.tex (x2)
quindi anche il suo hash e' verificabile in modo indipendente.

## PASSO 1 — OpenTimestamps (la priorita' vera; gratis, senza account)

Via client (consigliato):
  pip install opentimestamps-client
  ots stamp frozen_recoil_prediction.tex        # -> frozen_recoil_prediction.tex.ots
  ots stamp frozen_recoil_prediction.pdf        # -> frozen_recoil_prediction.pdf.ots

Via web (nessuna installazione):
  https://opentimestamps.org  -> trascina i due file, scarica i .ots

Dopo qualche ora (quando il blocco Bitcoin conferma) aggiorna la prova:
  ots upgrade frozen_recoil_prediction.tex.ots
  ots upgrade frozen_recoil_prediction.pdf.ots

Verifica (mostra la data del blocco Bitcoin):
  ots verify frozen_recoil_prediction.tex.ots
  ots verify frozen_recoil_prediction.pdf.ots

CONSERVA per sempre i file .ots: SONO la prova. Backup in piu' posti.
Non serve nessun account, nessuna piattaforma, nessun permesso.

## PASSO 2 — Archivio immutabile con DOI (record citabile, secondario)

Deposita lo STESSO identico file (hash 9d68a090...), byte per byte.
Ordine di ripiego (usa il primo che accetta):
  1. HAL (CCSD/CNRS)  -> immutabile, ma MODERATO: puo' rifiutare/riclassificare
  2. OSF (osf.io)     -> pensato per la preregistrazione, ideale, nessuna moderazione fisica
  3. figshare         -> DOI, immutabile
  4. Software Heritage -> archivia il repo GitHub in modo permanente e citabile
Allega i .ots al deposito.

## PASSO 3 — Disseminazione (NON prova di priorita')

ResearchGate / pasqualecamelia.github.io: copia vetrina, in cima a tutto.
Zenodo: solo se il support sblocca l'account; non e' sul percorso critico.

## REGOLA D'ORO

Ancora e deposita SEMPRE lo stesso file esatto che hai timestampato.
Un solo byte diverso -> hash diverso -> il timestamp non copre quel documento.
Il file congelato puo' cambiare SOLO per registrare un verdetto
PASS / FAIL / INCONCLUSIVE del dato, MAI il suo contenuto numerico.

## COSA NON timestampare oggi

La mappa canale->ramo (tabella Rb->w_i, Cs->w_j, ...) NON esiste ancora come
lemma: e' il gate A1 da chiudere. Congelarla ora = postdizione travestita.
Quando A1 chiude, sara' un SECONDO timestamp, con la sua data, che promuove
la predizione da "discretezza cieca" a "ramo assegnato a priori".
