# README_DEPLOY — aggiornamento sito per il lancio (13 giugno 2026)

## Contenuto dello zip (drop-in sul repo pasqualecamelia.github.io)

MODIFICATI:
- index.html
- qgt_companion.html   (Hydrogen 3D: da player di 25 frame pre-calcolati a
  SOLVER LIVE split-operator 64x64x64 (render bilineare a 256²; vista volumetrica 3D phase-coloured con rotazione) in JS puro, eseguito nel browser.
  Stessa UI, stessi controlli, stessa fisica dichiarata nell'intestazione.
  Validazione headless (Node): |Delta norm|/norm < 4e-14 su 200 passi,
  ~57 ms/passo, respirazione visibile in <r>. Pagina da 240 KB a 89 KB.
  Softening Coulomb eps = dx/2 dichiarato nel sorgente; loop 0-6 a.u.)

NUOVI:
- favicon.png                    (48x48, sfera qubit)
- apple-touch-icon.png           (180x180)
- assets/cover_en_front.jpg      (copertina EN, fronte, 620px)
- assets/cover_it_front.jpg      (copertina IT, fronte, 620px)
- assets/og_image.png            (1200x630, EN+IT su nero — og:image/twitter)
- assets/thumb_hydrogen3d.jpg    (thumbnail sezione Watch)
- assets/thumb_breathing3d.jpg
- assets/thumb_lissajous.jpg

NON TOCCATI: papers.html, animazioni, CamCMB/, script,
tutti i contenuti scientifici e i validatori. La scala A/B/O è invariata.

## PRIMA del commit — 2 sostituzioni obbligatorie in index.html

Il placeholder {{AMAZON_IT_URL}} compare DUE volte (bottone hero + bottone
sezione Book). Sostituire con il link Amazon.it dell'edizione italiana:

    sed -i 's|{{AMAZON_IT_URL}}|https://www.amazon.it/dp/XXXXXXXXXX|g' index.html

NON deployare con il placeholder presente.

## Cosa è cambiato in index.html (riepilogo)

1. HERO: tagline "Time is what the boundary reads when the kernel
   remembers." + 3 bottoni sopra la piega: Buy English / Edizione italiana /
   Watch the hydrogen atom (→ qgt_companion.html#hydrogen3d).
   Nota: l'etichetta del bottone è "Watch the hydrogen atom — 3D, real time"
   e non "Watch hydrogen being born": coerente con la correzione epistemica
   del post (il simulatore mostra la dinamica, non la nascita).
2. NAV sticky ridotta a 6 voci: Book · Watch · Thread · Validators ·
   Papers · Tools. Le vecchie ancore (#start #qubit #breathing #gluing
   #hydrogen #cmb #alpha #gallery) restano tutte attive: nessun link
   esterno esistente si rompe. Aggiunto scroll-margin-top per la barra.
3. Nuova sezione WATCH (00C): 3 card con thumbnail — Hydrogen 3D
   (#hydrogen3d del companion), Breathing 3D, Lissajous.
4. Sezione BOOK (00B): due edizioni affiancate con copertine reali, ISBN,
   pagine, link d'acquisto; nota sulle edizioni gemelle.
5. SEO/social: og:image → assets/og_image.png (1200x630, formato corretto
   per LinkedIn), og:description aggiornata al lancio, twitter:image,
   favicon + apple-touch-icon.
6. Mobile: watch-grid → 1 colonna sotto 760px; editions → 1 colonna sotto
   860px; copertina centrata sotto 480px; bottoni hero già flex-wrap.
7. Ancora #validators (div vuoto, solo navigazione) prima della sezione
   Hydrogen: porta a Hydrogen → CMB → Alpha in sequenza.

## Verifiche post-deploy (5 minuti)

1. https://pasqualecamelia.github.io/ da TELEFONO: tagline + 3 bottoni
   visibili senza scroll; nav scorrevole; card Watch impilate.
2. Click sui 6 link della nav e sui 3 bottoni hero.
3. og:image: incollare l'URL del sito nel composer LinkedIn (senza
   pubblicare) e verificare l'anteprima con le due copertine; se LinkedIn
   mostra la cache vecchia, usare https://www.linkedin.com/post-inspector/
   per forzare il refresh. FARE QUESTO PRIMA del post di lancio.
4. Favicon visibile nel tab del browser (Ctrl+F5 se serve).

## Limite dichiarato di questa consegna
Nel sandbox non è disponibile un browser: il render pixel-perfect non è
stato eseguito. I controlli effettuati: parsing HTML senza errori, tag
bilanciati, tutte le ancore verificate, tutti i path degli asset risolti,
CSS riusato dalle classi esistenti del sito (rischio basso). La verifica
visiva finale è il punto 1 qui sopra, su desktop e telefono, prima del post.
