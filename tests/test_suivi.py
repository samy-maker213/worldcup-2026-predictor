from src import suivi


def test_resultat_1n2():
    assert suivi.resultat_1n2(2, 1) == "1"
    assert suivi.resultat_1n2(0, 0) == "N"
    assert suivi.resultat_1n2(0, 3) == "2"


def test_brier_prediction_parfaite_et_pire():
    assert suivi.brier([1.0, 0.0, 0.0], "1") == 0.0
    assert suivi.brier([0.0, 0.0, 1.0], "1") == 2.0
    assert abs(suivi.brier([1 / 3, 1 / 3, 1 / 3], "N") - 2 / 3) < 0.001


def test_evaluer_compare_modele_et_naif():
    archive = {
        "1": {"probas": [0.6, 0.25, 0.15], "favori_naif": "1"},
        "2": {"probas": [0.2, 0.3, 0.5], "favori_naif": "2"},
        "3": {"probas": [0.5, 0.3, 0.2], "favori_naif": "1"},
    }
    matchs = [
        {"id": 1, "score": [2, 0]},   # "1" : modèle OK, naïf OK
        {"id": 2, "score": [1, 1]},   # "N" : modèle KO, naïf KO
        {"id": 3, "score": [0, 1]},   # "2" : modèle KO, naïf KO
        {"id": 4, "score": None},     # pas joué : ignoré
    ]
    bilan = suivi.evaluer(archive, matchs)
    assert bilan["nb_matchs"] == 3
    assert bilan["bons_modele"] == 1
    assert bilan["bons_naif"] == 1
    assert 0 < bilan["brier_moyen"] < 2
