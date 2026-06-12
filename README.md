# Prono-CDM — Modèle de prédiction Coupe du Monde 2026

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

## Commandes

    .venv\Scripts\python predire.py matchs    # prédit la prochaine journée
    .venv\Scripts\python predire.py tournoi   # probabilités de parcours
    .venv\Scripts\python predire.py maj       # recalcule les Elo après saisie des résultats
    .venv\Scripts\python predire.py bilan     # performance vs stratégie naïve
