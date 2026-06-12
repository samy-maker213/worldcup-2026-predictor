"""Suivi honnête de la performance : score de Brier + comparaison à la stratégie naïve."""

ISSUES = ("1", "N", "2")


def resultat_1n2(buts1, buts2):
    if buts1 > buts2:
        return "1"
    if buts1 < buts2:
        return "2"
    return "N"


def brier(probas, resultat):
    """Score de Brier multi-classes : 0 = parfait, 2 = pire possible."""
    cible = [1.0 if issue == resultat else 0.0 for issue in ISSUES]
    return sum((p - c) ** 2 for p, c in zip(probas, cible))


def evaluer(archive, matchs):
    """Compare le modèle (issue la plus probable) à la stratégie naïve (favori Elo)."""
    nb = bons_modele = bons_naif = 0
    somme_brier = 0.0
    for m in matchs:
        cle = str(m["id"])
        if m.get("score") is None or cle not in archive:
            continue
        reel = resultat_1n2(*m["score"])
        probas = archive[cle]["probas"]
        choix_modele = ISSUES[probas.index(max(probas))]
        nb += 1
        bons_modele += int(choix_modele == reel)
        bons_naif += int(archive[cle]["favori_naif"] == reel)
        somme_brier += brier(probas, reel)
    return {
        "nb_matchs": nb,
        "bons_modele": bons_modele,
        "bons_naif": bons_naif,
        "brier_moyen": somme_brier / nb if nb else None,
    }
