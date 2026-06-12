"""Prédiction d'un match : Elo + contexte (domicile, altitude, repos) -> Poisson."""

from datetime import date

from src import elo, poisson

TOTAL_BUTS = 2.6          # moyenne de buts/match en Coupe du Monde
AVANTAGE_DOMICILE = 80    # points Elo si l'équipe joue dans son pays
BONUS_ALTITUDE = 50       # points Elo en plus à domicile si ville >= 1500 m
SEUIL_ALTITUDE_M = 1500
POINTS_PAR_JOUR_REPOS = 10
PLAFOND_REPOS = 30


def avantage_contextuel(nom1, nom2, ville, repos1=None, repos2=None):
    """Avantage Elo net de l'équipe 1 (négatif si le contexte favorise l'équipe 2)."""
    av = 0
    bonus_ville = AVANTAGE_DOMICILE
    if ville.get("altitude_m", 0) >= SEUIL_ALTITUDE_M:
        bonus_ville += BONUS_ALTITUDE
    if ville["pays"] == nom1:
        av += bonus_ville
    if ville["pays"] == nom2:
        av -= bonus_ville
    if repos1 is not None and repos2 is not None:
        diff_repos = (repos1 - repos2) * POINTS_PAR_JOUR_REPOS
        av += max(-PLAFOND_REPOS, min(PLAFOND_REPOS, diff_repos))
    return av


def jours_repos(nom, date_match, matchs):
    """Jours depuis le match précédent de l'équipe dans matchs (None si aucun)."""
    cible = date.fromisoformat(date_match)
    precedents = [
        date.fromisoformat(m["date"])
        for m in matchs
        if nom in (m.get("equipe1"), m.get("equipe2"))
        and date.fromisoformat(m["date"]) < cible
    ]
    if not precedents:
        return None
    return (cible - max(precedents)).days


def predire_match(nom1, elo1, nom2, elo2, ville, repos1=None, repos2=None):
    """Prédiction complète : probas 1N2, lambdas Poisson, 3 scores les plus probables."""
    av = avantage_contextuel(nom1, nom2, ville, repos1, repos2)
    we = elo.probabilite_victoire(elo1, elo2, av)
    lam1 = TOTAL_BUTS * we
    lam2 = TOTAL_BUTS * (1 - we)
    matrice = poisson.matrice_scores(lam1, lam2)
    return {
        "equipe1": nom1,
        "equipe2": nom2,
        "we": we,
        "avantage": av,
        "lambdas": (lam1, lam2),
        "probas": poisson.probas_1n2(matrice),
        "scores": poisson.scores_probables(matrice, 3),
    }
