# Raw local data

I record LSE e gli altri input licenziati restano in questa directory locale e
sono esclusi da Git. Non copiare dati row-level in `Overleaf`, `Drive`, nei file
Markdown o nei pacchetti di pubblicazione.

La pipeline corrente salva ogni acquisizione in una directory immutabile
`lse_local/<run-id>/`, separando `barrick_equity_candles.json` da
`gld_market_inputs.json`. I manifest pubblici versionati sono in
`../../manifests/<run-id>/` e contengono hash, cutoff, parametri endpoint,
versioni del client/codice e soli aggregati; non contengono righe LSE.
