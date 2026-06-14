# Workflow revisione Team 8 - Overleaf

## Obiettivo

Aggiornare e compilare il report Research alla luce della nuova struttura del
codice Hawkes, mantenendo un solo modello Hawkes nel confronto principale con
Black--Scholes, Heston e Bates e raccogliendo le altre calibrazioni Hawkes in un
paragrafo metodologico separato.

## Path autorevoli

- Progetto Overleaf da modificare:
  `C:\Users\salvm\Desktop\SF\Research projects\Barrick gold\team 8\Overleaf`
- Sorgente principale:
  `C:\Users\salvm\Desktop\SF\Research projects\Barrick gold\team 8\Overleaf\Articolo.tex`
- PDF finale:
  `C:\Users\salvm\Desktop\SF\Research projects\Barrick gold\team 8\Overleaf\Articolo.pdf`
- Immagini usate dal report:
  `C:\Users\salvm\Desktop\SF\Research projects\Barrick gold\team 8\Overleaf\img`
- Diagnostiche articolo:
  `C:\Users\salvm\Desktop\SF\Research projects\Barrick gold\team 8\Overleaf\img\diagnostics`
- Materiale Hawkes preesistente da consultare, non da trattare come sorgente
  finale del report:
  `C:\Users\salvm\Desktop\SF\Research projects\Barrick gold\team 8\Overleaf\Implementiamo hawkes`
- Cartella temporanea e QA:
  `C:\Users\salvm\Desktop\SF\Research projects\Barrick gold\team 8\Overleaf\TMP`
- Repository codice autorevole:
  `C:\Users\salvm\Desktop\SF\GITHUB\Research\Barrick-Gold\Gold-Options-Stochastic-Modeling`
- Implementazioni Hawkes e calibrazioni:
  `...\Gold-Options-Stochastic-Modeling\Hawkes.py`
- Pricing affine esatto:
  `...\Gold-Options-Stochastic-Modeling\BatesHawkesExact.py`
- Notebook di calibrazione e grafici Hawkes:
  `...\Gold-Options-Stochastic-Modeling\Hawkes Calibration.ipynb`
- Risultati numerici e figure del codice:
  `...\Gold-Options-Stochastic-Modeling\Data`
- Guida Research locale prioritaria:
  `C:\Users\salvm\Desktop\SF\Utilities\Guide\Pubblicazione\GUIDA PUBBLICAZIONE_Research.tex`

## Decisioni editoriali e tecniche

1. Conservare nel confronto principale un solo Hawkes: il modello affine
   Heston--Hawkes con kernel esponenziale, perché è l'unica specificazione
   direttamente confrontabile con Heston e Bates sulla stessa superficie di
   opzioni e mantiene una characteristic function affine utilizzabile dal
   pricer COS/Fourier.
2. Presentare separatamente le calibrazioni Hawkes disponibili:
   - likelihood point-process con kernel esponenziale;
   - likelihood point-process con rough power-law kernel;
   - benchmark affine esatto a volatilità costante;
   - calibrazione affine completa Heston--Hawkes.
3. Non confrontare il rough Hawkes direttamente con Heston/Bates nei grafici di
   option pricing: il rough kernel è un modello diagnostico del point process e
   non usa l'attuale pricer affine finito-dimensionale.
4. Spiegare che l'aggiunta dei salti separa la variabilità in due canali:
   diffusione continua/stocastica e componente discontinua. Di conseguenza i
   parametri Heston non sono invarianti tra Heston e Bates/Hawkes; possono
   ridursi perché parte di skew, code e movimenti estremi viene assorbita dai
   parametri di salto. Evitare di descrivere questo risultato come una prova
   causale o universale.
5. Verificare i valori numerici contro i JSON/CSV in `Data` prima di modificare
   tabelle o affermazioni.

## Sequenza di lavoro

1. Estrarre struttura, tabelle, figure, riferimenti e passaggi Hawkes da
   `Articolo.tex`.
2. Leggere i risultati correnti in `Data` e scegliere il modello Hawkes del
   confronto principale sulla base di compatibilità e metriche disponibili.
3. Aggiornare abstract, roadmap e descrizione della model stack.
4. Riscrivere il capitolo Hawkes con un paragrafo separato sulle quattro route
   di calibrazione e una motivazione esplicita della scelta del full affine
   Heston--Hawkes per il confronto.
5. Correggere capitoli di calibrazione, risultati, interpretazione dei parametri
   e conclusioni, chiarendo la riallocazione della volatilità tra diffusione e
   jump component.
6. Eliminare riferimenti alla vecchia cartella `calibrations/`; indicare classi
   `.py` e notebook root come struttura riproducibile.
7. Compilare `Articolo.tex` almeno due volte con `pdflatex`.
8. Controllare il log per errori, undefined references, overfull box e figure
   mancanti.
9. Renderizzare tutte le pagine del PDF in `TMP/qa_hawkes_revision` e svolgere
   QA visivo pagina per pagina.
10. Consegnare `Articolo.tex`, `Articolo.pdf` e l'intero progetto Overleaf
    corretto, lasciando i file temporanei in `TMP`.

## Rischi noti

- La calibrazione affine Heston--Hawkes può collocarsi vicino al limite Bates;
  il testo deve presentarlo come risultato di identificazione del dataset.
- I parametri Heston tra modelli non sono direttamente interpretabili come
  stime strutturali immutabili, perché cambiano insieme alla specificazione dei
  salti.
- Le calibrazioni point-process richiedono event times; la calibrazione della
  superficie di opzioni usa invece errori di prezzo/volatilità pesati per vega.
  Le due famiglie non devono essere confuse.
- Il PDF finale deve chiudersi con le References e il riferimento GitHub deve
  rimanere l'ultima voce.

## Stato

- [x] Skill e guida Research lette.
- [x] Sorgente Overleaf e struttura immagini individuati.
- [x] Risultati numerici verificati contro i file in `Data`.
- [x] Testo LaTeX aggiornato e riallineato alla struttura corrente del codice.
- [x] Compilazione completata con due passaggi `pdflatex`.
- [x] QA visivo completato su tutte le 36 pagine.

## Esito finale

- Modello Hawkes nel confronto principale: full affine Heston--Hawkes con
  kernel esponenziale, scelto per confrontabilita metodologica con Heston e
  Bates, non perche vinca la calibrazione.
- Le route exponential likelihood, rough likelihood ed exact constant-volatility
  sono documentate separatamente e non entrano nel ranking tra modelli.
- Il report chiarisce la riallocazione della variabilita tra diffusione Heston e
  jump component e riporta le riduzioni dei parametri osservate nel passaggio
  Heston--Bates.
- `Articolo.pdf` compilato: 36 pagine, riferimenti risolti, nessun errore LaTeX,
  immagine mancante o overfull box.
- Render QA: `TMP\qa_hawkes_revision\page-01.png` fino a `page-36.png`.
- Backup precedente alla revisione:
  `TMP\Articolo_before_hawkes_revision_20260614.tex`.
