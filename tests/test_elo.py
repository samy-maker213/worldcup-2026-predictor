from src import elo


def test_elos_egaux_donnent_50_pourcent():
    assert elo.probabilite_victoire(1800, 1800) == 0.5


def test_400_points_d_ecart_donnent_91_pourcent():
    assert abs(elo.probabilite_victoire(2000, 1600) - 0.909) < 0.001


def test_avantage_s_ajoute_a_l_elo():
    assert elo.probabilite_victoire(1800, 1880, avantage_a=80) == 0.5


def test_facteur_buts():
    assert elo.facteur_buts(0) == 1.0
    assert elo.facteur_buts(1) == 1.0
    assert elo.facteur_buts(2) == 1.5
    assert elo.facteur_buts(3) == 1.75  # (11+3)/8


def test_mise_a_jour_victoire_attendue_gagne_peu():
    a, b = elo.mise_a_jour(2000, 1600, 1, 0)
    assert 2000 < a < 2010          # favori net : gain faible
    assert abs((a - 2000) + (b - 1600)) < 1e-9  # somme conservée


def test_mise_a_jour_surprise_gagne_beaucoup():
    a, b = elo.mise_a_jour(1600, 2000, 3, 0)
    assert a > 1600 + 80            # outsider qui gagne 3-0 : gros gain
