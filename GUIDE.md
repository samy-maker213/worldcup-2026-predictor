# GUIDE — Mise à jour après chaque journée de matchs

À faire le soir de chaque journée (ou le lendemain matin), dans Claude Code,
depuis le dossier Prono-CDM :

1. **Saisir les résultats** — demander à Claude :
   « Mets à jour data/matchs.json avec les résultats d'hier (recherche web,
   source officielle FIFA ou Wikipedia). Pour les matchs à élimination directe
   joués, remplis aussi equipe1/equipe2. »
2. **Recalculer les Elo** :

       .venv\Scripts\python predire.py maj

3. **Prédire la journée suivante** (archive automatiquement les prédictions) :

       .venv\Scripts\python predire.py matchs

4. **Re-simuler le tournoi** :

       .venv\Scripts\python predire.py tournoi

5. **Suivre la performance** :

       .venv\Scripts\python predire.py bilan

6. **Sauvegarder** :

       git add -A
       git commit -m "maj: resultats du <date> + predictions"
       git push

## Règles d'honnêteté

- Ne JAMAIS modifier `predictions/archive.json` à la main (c'est le juge de paix).
- Lancer `matchs` AVANT le coup d'envoi, jamais après.
- À la fin des groupes : demander à Claude de remplir `equipe1`/`equipe2` des 16es
  selon les classements réels et la table officielle FIFA des troisièmes.

## Rappels sur les données

- `elo_initial` = relevé eloratings.net du 12 juin 2026 (matchs des 11-12 juin
  des groupes A déjà inclus → `inclus_elo: true` sur les matchs 1 et 2).
- Tout nouveau résultat saisi doit rester `inclus_elo: false` : c'est le rejeu
  (`maj`) qui l'intègre aux Elo.
