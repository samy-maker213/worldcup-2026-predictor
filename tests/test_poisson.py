import random

from src import poisson


def test_pmf_valeurs_connues():
    # Poisson(λ=1) : P(0) = e^-1 ≈ 0.3679
    assert abs(poisson.pmf(0, 1.0) - 0.3679) < 0.001
    assert abs(poisson.pmf(2, 1.5) - 0.2510) < 0.001


def test_matrice_somme_a_environ_1():
    m = poisson.matrice_scores(1.5, 1.1)
    total = sum(sum(ligne) for ligne in m)
    assert 0.999 < total <= 1.0


def test_probas_1n2_coherentes():
    p1, pn, p2 = poisson.probas_1n2(poisson.matrice_scores(1.5, 1.1))
    assert abs(p1 + pn + p2 - 1.0) < 0.001
    assert p1 > p2  # λ1 > λ2 => équipe 1 favorite


def test_scores_probables_tries():
    scores = poisson.scores_probables(poisson.matrice_scores(1.3, 1.3), n=3)
    assert len(scores) == 3
    assert scores[0][1] >= scores[1][1] >= scores[2][1]
    assert scores[0][0] == (1, 1)  # match équilibré : 1-1 score le plus probable


def test_tirage_poisson_moyenne_proche_de_lambda():
    rng = random.Random(42)
    tirages = [poisson.tirer_poisson(rng, 2.6) for _ in range(5000)]
    assert all(t >= 0 for t in tirages)
    assert abs(sum(tirages) / len(tirages) - 2.6) < 0.1
