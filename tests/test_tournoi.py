from src import tournoi

EQUIPES = {
    "A1": {"groupe": "A", "elo": 2000},
    "A2": {"groupe": "A", "elo": 1900},
    "A3": {"groupe": "A", "elo": 1800},
    "A4": {"groupe": "A", "elo": 1700},
}

MATCHS_GROUPE_A = [
    {"equipe1": "A1", "equipe2": "A2", "score": [2, 0]},
    {"equipe1": "A3", "equipe2": "A4", "score": [1, 1]},
    {"equipe1": "A1", "equipe2": "A3", "score": [1, 0]},
    {"equipe1": "A2", "equipe2": "A4", "score": [3, 0]},
    {"equipe1": "A1", "equipe2": "A4", "score": [0, 0]},
    {"equipe1": "A2", "equipe2": "A3", "score": [2, 1]},
]


def test_classement_groupe():
    cl = tournoi.classement_groupe(["A1", "A2", "A3", "A4"], MATCHS_GROUPE_A, EQUIPES)
    # A4 : 2 nuls = 2 pts ; A3 : 1 nul = 1 pt -> A4 devant A3
    assert [e["nom"] for e in cl] == ["A1", "A2", "A4", "A3"]
    assert cl[0]["pts"] == 7 and cl[1]["pts"] == 6
    assert cl[0]["diff"] == 3


def test_classement_depart_par_elo():
    # Zéro match joué : tout le monde à 0 pts -> départage par Elo décroissant
    cl = tournoi.classement_groupe(["A4", "A1", "A3", "A2"], [], EQUIPES)
    assert [e["nom"] for e in cl] == ["A1", "A2", "A3", "A4"]


def test_meilleurs_troisiemes_prend_les_8_premiers():
    classements = {}
    for i, g in enumerate("ABCDEFGHIJKL"):
        classements[g] = [
            {"nom": f"{g}1", "pts": 9, "diff": 5, "bp": 6},
            {"nom": f"{g}2", "pts": 6, "diff": 1, "bp": 4},
            {"nom": f"{g}3", "pts": i, "diff": 0, "bp": 2},  # pts = 0..11
            {"nom": f"{g}4", "pts": 0, "diff": -6, "bp": 1},
        ]
    tiers = tournoi.meilleurs_troisiemes(classements)
    assert len(tiers) == 8
    assert ("L", "L3") in tiers and ("E", "E3") in tiers  # pts 11 et 4
    assert ("A", "A3") not in tiers                        # pts 0 : éliminé


def test_simulation_probabilites_coherentes():
    lettres = "ABCDEFGHIJKL"
    equipes = {
        f"E{i}": {"groupe": lettres[i // 4], "elo": 2000 - i * 15}
        for i in range(48)
    }
    villes = {"Ville": {"pays": "Nulle-part", "altitude_m": 0}}
    matchs = tournoi.calendrier_synthetique(equipes)
    resultat = tournoi.simuler_tournoi(equipes, matchs, villes,
                                       n_simulations=200, graine=42)
    assert abs(sum(r["champion"] for r in resultat.values()) - 1.0) < 1e-9
    assert resultat["E0"]["champion"] > resultat["E47"]["champion"]  # plus fort Elo
    # reproductibilité : même graine => même résultat
    resultat2 = tournoi.simuler_tournoi(equipes, matchs, villes,
                                        n_simulations=200, graine=42)
    assert resultat == resultat2
