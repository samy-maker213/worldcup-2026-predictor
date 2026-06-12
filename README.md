# worldcup-2026-predictor

**FIFA World Cup 2026 prediction model — Elo ratings + Poisson distribution + Monte-Carlo simulation, updated after every match day.**

🇫🇷 *Version française ci-dessous.*

## Method (English)

- **Team strength**: Elo ratings (eloratings.net), updated after every result (K=60, goal-difference multiplier).
- **Context**: home advantage (+80 Elo), altitude bonus ≥1500 m (+50), rest days (±10/day, capped ±30).
- **Score model**: Poisson distribution, λ1 = 2.6 × We, λ2 = 2.6 × (1−We).
- **Tournament**: Monte-Carlo simulation (5000 runs — group stage → 8 best third-placed teams → knockout bracket).
- **Honest tracking**: predictions archived with timestamps BEFORE matches, Brier score, comparison against the naive "always pick the Elo favourite" strategy.

### Known limitations

Simplified Elo→goals mapping; group tie-breakers without head-to-head; greedy third-place bracket allocation (not the exact FIFA table); static Elo within a simulation. **Learning project — do not use for betting.**

### Install & run (Windows)

    py -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python -m pytest        # 27 tests should pass

    .venv\Scripts\python predire.py matchs    # predict the next match day
    .venv\Scripts\python predire.py tournoi   # tournament win probabilities
    .venv\Scripts\python predire.py maj       # recompute Elo after entering results
    .venv\Scripts\python predire.py bilan     # model performance vs naive strategy

*Code identifiers and reports are in French (French learning project).*

---

# Prono-CDM — Modèle de prédiction Coupe du Monde 2026 (Français)

Modèle évolutif : probabilités 1N2 + scores probables (Elo → Poisson) et
probabilités de parcours (Monte-Carlo, 5000 simulations). Mise à jour après
chaque journée de matchs (voir GUIDE.md).

## Méthode

- Force des équipes : rating Elo (eloratings.net), mis à jour après chaque résultat
  (K=60, multiplicateur d'écart de buts).
- Contexte : avantage domicile (+80 Elo), bonus altitude ≥1500 m (+50), repos (±10/jour, plafonné ±30).
- Score : loi de Poisson, λ1 = 2.6 × We, λ2 = 2.6 × (1−We).
- Tournoi : simulation Monte-Carlo (groupes → 8 meilleurs 3èmes → tableau final).
- Suivi honnête : prédictions archivées AVANT les matchs, score de Brier,
  comparaison à la stratégie naïve « toujours le favori Elo ».

## Limites assumées

- Mapping Elo→buts simplifié ; départage de groupes sans head-to-head ;
  affectation des 3èmes au tableau par algorithme glouton (≠ table FIFA exacte) ;
  Elo statique à l'intérieur d'une simulation.
- Projet d'APPRENTISSAGE. Ne pas utiliser pour parier de l'argent.

## Installation (Windows)

    py -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python -m pytest        # 27 tests doivent passer

## Commandes

    .venv\Scripts\python predire.py matchs    # prédit la prochaine journée
    .venv\Scripts\python predire.py tournoi   # probabilités de parcours
    .venv\Scripts\python predire.py maj       # recalcule les Elo après saisie des résultats
    .venv\Scripts\python predire.py bilan     # performance vs stratégie naïve
