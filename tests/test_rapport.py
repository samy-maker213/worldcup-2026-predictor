from src import rapport

PREDICTION = {
    "equipe1": "France", "equipe2": "Brésil", "we": 0.55, "avantage": 0,
    "lambdas": (1.43, 1.17), "probas": (0.45, 0.27, 0.28),
    "scores": [((1, 1), 0.12), ((1, 0), 0.11), ((2, 1), 0.09)],
}


def test_rapport_journee_contient_l_essentiel():
    md = rapport.rapport_journee([PREDICTION], "2026-06-17")
    assert "France" in md and "Brésil" in md
    assert "45" in md            # probabilité en %
    assert "1-1" in md           # score le plus probable


def test_rapport_tournoi_trie_par_proba_champion():
    probas = {
        "Espagne": {"16es": 0.99, "8es": 0.8, "quarts": 0.6,
                    "demies": 0.4, "finale": 0.25, "champion": 0.15},
        "Canada": {"16es": 0.7, "8es": 0.3, "quarts": 0.1,
                   "demies": 0.04, "finale": 0.01, "champion": 0.005},
    }
    md = rapport.rapport_tournoi(probas, 5000)
    assert md.index("Espagne") < md.index("Canada")
    assert "15.0" in md
