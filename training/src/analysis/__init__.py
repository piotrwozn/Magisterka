"""
Analizy do pracy na ukończonych runach — refity z najlepszymi parametrami, BEZ tuningu.

    seeds           (C1) zmienność między seedami: scripts/09_seed_refits.py
    site_out        (C3) leave-site-out + korekta rozkładu klas (znany / EM): scripts/10_leave_site_out.py
    learning_curve  (C9) krzywa uczenia + dopasowanie potęgowe: scripts/11_learning_curve.py
    priors          korekta rozkładu klas (Saerens i in. 2002)

Wyniki: runs/analyses/<analiza>__<cechy>__<tag>/ (plan.json, units/*.json, results.json,
summary.md). Każda analiza jest wznawialna — gotowe jednostki są pomijane (chyba że --force).
Refity: zwykłe katalogi runs/<run>/ z DONE.json `custom: true`, `refit_of: <run źródłowy>`
(`src.models.train.refit_from_run`).

Ten plik celowo niczego nie importuje: skrypty ustawiają SOR_AI_N_JOBS (src.analysis.cli)
PRZED importem config.
"""
