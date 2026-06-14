import json
from datetime import date
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"


def charger(nom):
    return json.loads((DATA / nom).read_text(encoding="utf-8"))


def test_48_equipes_12_groupes_de_4():
    equipes = charger("equipes.json")
    assert len(equipes) == 48
    groupes = {}
    for infos in equipes.values():
        groupes.setdefault(infos["groupe"], 0)
        groupes[infos["groupe"]] += 1
    assert sorted(groupes) == list("ABCDEFGHIJKL")
    assert all(n == 4 for n in groupes.values())
    for infos in equipes.values():
        # elo_initial = relevé immuable (eloratings.net 12 juin), sert de base au rejeu.
        assert 1300 < infos["elo_initial"] < 2300
        # elo évolue après chaque journée (maj) ; il doit rester un nombre plausible.
        assert isinstance(infos["elo"], (int, float))
        assert 1000 < infos["elo"] < 2500


def test_104_matchs_references_valides():
    equipes = charger("equipes.json")
    matchs = charger("matchs.json")
    villes = charger("villes.json")
    assert len(matchs) == 104
    assert len(villes) == 16
    phases = {}
    for m in matchs:
        phases.setdefault(m["phase"], 0)
        phases[m["phase"]] += 1
        assert m["ville"] in villes, f"ville inconnue: {m['ville']} (match {m['id']})"
        date.fromisoformat(m["date"])  # lève une erreur si date invalide
        for cle in ("equipe1", "equipe2"):
            if m.get(cle) is not None:
                assert m[cle] in equipes, f"equipe inconnue: {m[cle]} (match {m['id']})"
        if m["phase"] == "groupes":
            assert m["equipe1"] and m["equipe2"] and m.get("groupe")
        else:
            assert m.get("source1") and m.get("source2")
        if m["score"] is not None:
            assert len(m["score"]) == 2 and all(isinstance(b, int) for b in m["score"])
    assert phases == {"groupes": 72, "16es": 16, "8es": 8,
                      "quarts": 4, "demies": 2, "match3": 1, "finale": 1}


def test_ids_uniques_et_croissants():
    ids = [m["id"] for m in charger("matchs.json")]
    assert ids == sorted(ids) and len(set(ids)) == 104
