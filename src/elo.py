"""Rating Elo football — formules d'eloratings.net (K=60 pour une Coupe du Monde)."""

K_COUPE_DU_MONDE = 60


def probabilite_victoire(elo_a, elo_b, avantage_a=0):
    """Probabilité que A batte B (We). avantage_a en points Elo (domicile, etc.)."""
    d = elo_a + avantage_a - elo_b
    return 1 / (1 + 10 ** (-d / 400))


def facteur_buts(ecart):
    """Multiplicateur selon l'écart de buts : 1, 1.5, puis (11+N)/8."""
    if ecart <= 1:
        return 1.0
    if ecart == 2:
        return 1.5
    return (11 + ecart) / 8


def mise_a_jour(elo_a, elo_b, buts_a, buts_b, avantage_a=0, k=K_COUPE_DU_MONDE):
    """Nouveaux Elo (a, b) après un résultat réel. Somme des Elo conservée."""
    we = probabilite_victoire(elo_a, elo_b, avantage_a)
    if buts_a > buts_b:
        w = 1.0
    elif buts_a < buts_b:
        w = 0.0
    else:
        w = 0.5
    delta = k * facteur_buts(abs(buts_a - buts_b)) * (w - we)
    return elo_a + delta, elo_b - delta
