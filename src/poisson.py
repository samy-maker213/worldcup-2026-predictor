"""Loi de Poisson : le modèle statistique standard du nombre de buts au football."""

import math

MAX_BUTS = 10  # au-delà, probabilité négligeable


def pmf(k, lam):
    """P(X = k) pour X ~ Poisson(lam)."""
    return math.exp(-lam) * lam ** k / math.factorial(k)


def matrice_scores(lam_a, lam_b, max_buts=MAX_BUTS):
    """matrice[i][j] = P(score i-j), buts indépendants A~Poisson(lam_a), B~Poisson(lam_b)."""
    return [
        [pmf(i, lam_a) * pmf(j, lam_b) for j in range(max_buts + 1)]
        for i in range(max_buts + 1)
    ]


def probas_1n2(matrice):
    """(P victoire A, P nul, P victoire B) à partir de la matrice de scores."""
    n = len(matrice)
    p1 = sum(matrice[i][j] for i in range(n) for j in range(n) if i > j)
    pn = sum(matrice[i][i] for i in range(n))
    p2 = sum(matrice[i][j] for i in range(n) for j in range(n) if i < j)
    return p1, pn, p2


def scores_probables(matrice, n=3):
    """Les n scores les plus probables : [((buts_a, buts_b), proba), ...] décroissant."""
    tous = [
        ((i, j), matrice[i][j])
        for i in range(len(matrice))
        for j in range(len(matrice))
    ]
    return sorted(tous, key=lambda x: -x[1])[:n]


def tirer_poisson(rng, lam):
    """Tirage aléatoire d'un nombre de buts ~ Poisson(lam) (algorithme de Knuth)."""
    seuil = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= seuil:
            return k
        k += 1
