from src import modele

VILLE_MEXICO = {"pays": "Mexique", "altitude_m": 2240, "toit": False}
VILLE_BOSTON = {"pays": "États-Unis", "altitude_m": 89, "toit": False}


def test_avantage_domicile_et_altitude():
    # Le Mexique joue à Mexico (2240 m) : +80 domicile +50 altitude
    assert modele.avantage_contextuel("Mexique", "France", VILLE_MEXICO) == 130
    # La France contre le Mexique à Mexico : effet symétrique négatif
    assert modele.avantage_contextuel("France", "Mexique", VILLE_MEXICO) == -130
    # Terrain neutre, pas d'altitude : zéro
    assert modele.avantage_contextuel("France", "Brésil", VILLE_BOSTON) == 0


def test_avantage_repos_plafonne():
    av = modele.avantage_contextuel("France", "Brésil", VILLE_BOSTON,
                                    repos1=8, repos2=2)
    assert av == 30  # 6 jours x 10, plafonné à 30


def test_predire_match_structure_et_coherence():
    p = modele.predire_match("France", 2074, "Brésil", 2034, VILLE_BOSTON)
    assert abs(sum(p["probas"]) - 1.0) < 0.001
    assert p["probas"][0] > p["probas"][2]          # France légèrement favorite
    assert abs(p["lambdas"][0] + p["lambdas"][1] - modele.TOTAL_BUTS) < 1e-9
    assert len(p["scores"]) == 3


def test_jours_repos():
    matchs = [
        {"id": 1, "date": "2026-06-11", "equipe1": "Mexique", "equipe2": "X", "score": [1, 0]},
        {"id": 9, "date": "2026-06-17", "equipe1": "Y", "equipe2": "Mexique", "score": None},
    ]
    assert modele.jours_repos("Mexique", "2026-06-17", matchs) == 6
    assert modele.jours_repos("Mexique", "2026-06-11", matchs) is None  # 1er match
