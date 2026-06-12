"""Classements de groupes, meilleurs troisièmes, tableau final, simulation Monte-Carlo."""

import random

from src import modele, poisson


def _cle_classement(stats, equipes):
    """Critères : points, différence, buts marqués, puis Elo (approximation FIFA)."""
    return (-stats["pts"], -stats["diff"], -stats["bp"],
            -equipes[stats["nom"]]["elo"], stats["nom"])


def classement_groupe(noms, matchs_joues, equipes):
    """Classement d'un groupe à partir des matchs joués (score non nul)."""
    stats = {n: {"nom": n, "pts": 0, "diff": 0, "bp": 0} for n in noms}
    for m in matchs_joues:
        if m.get("score") is None:
            continue
        b1, b2 = m["score"]
        e1, e2 = m["equipe1"], m["equipe2"]
        stats[e1]["bp"] += b1
        stats[e2]["bp"] += b2
        stats[e1]["diff"] += b1 - b2
        stats[e2]["diff"] += b2 - b1
        if b1 > b2:
            stats[e1]["pts"] += 3
        elif b2 > b1:
            stats[e2]["pts"] += 3
        else:
            stats[e1]["pts"] += 1
            stats[e2]["pts"] += 1
    return sorted(stats.values(), key=lambda s: _cle_classement(s, equipes))


def meilleurs_troisiemes(classements):
    """Les 8 meilleurs 3èmes : liste de (groupe, nom), du mieux classé au moins bon."""
    tiers = [(g, cl[2]) for g, cl in classements.items()]
    tiers.sort(key=lambda t: (-t[1]["pts"], -t[1]["diff"], -t[1]["bp"], t[1]["nom"]))
    return [(g, s["nom"]) for g, s in tiers[:8]]


def _affecter_troisiemes(matchs_ko, troisiemes):
    """Affectation gloutonne des 3èmes qualifiés aux slots '3XYZW' du tableau.

    Approximation de la table officielle FIFA : pour chaque slot (ordre des ids),
    on prend le meilleur 3ème restant dont le groupe est autorisé, sinon le
    meilleur restant tout court.
    """
    restants = list(troisiemes)
    affectation = {}
    for m in sorted(matchs_ko, key=lambda m: m["id"]):
        for cle in ("source1", "source2"):
            src = m.get(cle)
            if src and src.startswith("3"):
                permis = src[1:]
                choix = next((t for t in restants if t[0] in permis),
                             restants[0] if restants else None)
                if choix:
                    restants.remove(choix)
                    affectation[(m["id"], cle)] = choix[1]
    return affectation


def _tirer_score(rng, lam1, lam2):
    return poisson.tirer_poisson(rng, lam1), poisson.tirer_poisson(rng, lam2)


def _jouer_elimination(rng, pred):
    """Score 90 min ; si nul : prolongation (λ/3) ; si encore nul : tirs au but selon We."""
    b1, b2 = _tirer_score(rng, *pred["lambdas"])
    if b1 == b2:
        p1, p2 = _tirer_score(rng, pred["lambdas"][0] / 3, pred["lambdas"][1] / 3)
        b1, b2 = b1 + p1, b2 + p2
    if b1 == b2:
        return (1, 0) if rng.random() < pred["we"] else (0, 1)
    return (b1, b2)


def calendrier_synthetique(equipes):
    """Calendrier minimal (tests uniquement) : 72 matchs de groupes + 32 KO génériques."""
    matchs, mid = [], 1
    groupes = {}
    for nom, infos in equipes.items():
        groupes.setdefault(infos["groupe"], []).append(nom)
    for g, noms in sorted(groupes.items()):
        for i in range(4):
            for j in range(i + 1, 4):
                matchs.append({"id": mid, "phase": "groupes", "groupe": g,
                               "date": "2026-06-15", "ville": "Ville",
                               "equipe1": noms[i], "equipe2": noms[j],
                               "score": None, "inclus_elo": False})
                mid += 1
    lettres = "ABCDEFGHIJKL"
    sources = []
    for g in lettres:                          # 12 vainqueurs + 12 deuxièmes
        sources.append("1" + g)
        sources.append("2" + g)
    for _ in range(8):                         # 8 meilleurs 3èmes (groupe libre)
        sources.append("3" + lettres)
    paires = [(sources[i], sources[i + 1]) for i in range(0, 32, 2)]
    ids_tour = []
    for s1, s2 in paires:                      # 16es de finale
        matchs.append({"id": mid, "phase": "16es", "date": "2026-06-29",
                       "ville": "Ville", "equipe1": None, "equipe2": None,
                       "score": None, "inclus_elo": False,
                       "source1": s1, "source2": s2})
        ids_tour.append(mid)
        mid += 1
    for phase, nb in (("8es", 8), ("quarts", 4), ("demies", 2)):
        nouveaux = []
        for i in range(nb):
            matchs.append({"id": mid, "phase": phase, "date": "2026-07-04",
                           "ville": "Ville", "equipe1": None, "equipe2": None,
                           "score": None, "inclus_elo": False,
                           "source1": f"V{ids_tour[2 * i]}",
                           "source2": f"V{ids_tour[2 * i + 1]}"})
            nouveaux.append(mid)
            mid += 1
        ids_tour = nouveaux
    matchs.append({"id": mid, "phase": "match3", "date": "2026-07-18",
                   "ville": "Ville", "equipe1": None, "equipe2": None,
                   "score": None, "inclus_elo": False,
                   "source1": f"P{ids_tour[0]}", "source2": f"P{ids_tour[1]}"})
    mid += 1
    matchs.append({"id": mid, "phase": "finale", "date": "2026-07-19",
                   "ville": "Ville", "equipe1": None, "equipe2": None,
                   "score": None, "inclus_elo": False,
                   "source1": f"V{ids_tour[0]}", "source2": f"V{ids_tour[1]}"})
    return matchs


def simuler_tournoi(equipes, matchs, villes, n_simulations=5000, graine=42):
    """Probabilités de parcours par équipe : {nom: {phase: proba, "champion": proba}}.

    Les matchs déjà joués (score réel) sont fixés ; le reste est tiré au sort
    selon le modèle. L'Elo reste statique pendant une simulation.
    """
    rng = random.Random(graine)
    compteur = {n: {p: 0 for p in ["16es", "8es", "quarts", "demies",
                                   "finale", "champion"]} for n in equipes}
    matchs_groupes = [m for m in matchs if m["phase"] == "groupes"]
    matchs_ko = sorted((m for m in matchs if m["phase"] != "groupes"),
                       key=lambda m: m["id"])
    groupes = {}
    for nom, infos in equipes.items():
        groupes.setdefault(infos["groupe"], []).append(nom)
    cache_pred = {}

    def prediction(nom1, nom2, nom_ville):
        cle = (nom1, nom2, nom_ville)
        if cle not in cache_pred:
            ville = villes.get(nom_ville, {"pays": "?", "altitude_m": 0})
            cache_pred[cle] = modele.predire_match(
                nom1, equipes[nom1]["elo"], nom2, equipes[nom2]["elo"], ville)
        return cache_pred[cle]

    for _ in range(n_simulations):
        # 1) Phase de groupes : résultats réels conservés, le reste tiré au sort.
        resultats_groupes = {}
        for m in matchs_groupes:
            if m["score"] is not None:
                score = tuple(m["score"])
            else:
                pred = prediction(m["equipe1"], m["equipe2"], m["ville"])
                score = _tirer_score(rng, *pred["lambdas"])
            resultats_groupes.setdefault(m["groupe"], []).append(
                {"equipe1": m["equipe1"], "equipe2": m["equipe2"],
                 "score": list(score)})
        # 2) Classements et qualifiés.
        classements, qualifies = {}, {}
        for g, noms in groupes.items():
            cl = classement_groupe(noms, resultats_groupes.get(g, []), equipes)
            classements[g] = cl
            qualifies["1" + g] = cl[0]["nom"]
            qualifies["2" + g] = cl[1]["nom"]
        tiers = meilleurs_troisiemes(classements)
        affectation_tiers = _affecter_troisiemes(matchs_ko, tiers)
        # 3) Tableau final.
        vainqueurs, perdants = {}, {}

        def resoudre(m, cle_source):
            reel = m.get("equipe" + cle_source[-1])
            if reel:
                return reel                      # match KO déjà joué en vrai
            src = m[cle_source]
            if src.startswith("V"):
                return vainqueurs[int(src[1:])]
            if src.startswith("P"):
                return perdants[int(src[1:])]
            if src.startswith("3"):
                return affectation_tiers[(m["id"], cle_source)]
            return qualifies[src]

        for m in matchs_ko:
            nom1 = resoudre(m, "source1")
            nom2 = resoudre(m, "source2")
            if m["phase"] in compteur[nom1]:
                compteur[nom1][m["phase"]] += 1
                compteur[nom2][m["phase"]] += 1
            if m["score"] is not None:
                b1, b2 = m["score"]
                if b1 == b2:  # nul réel en KO : départage simulé (t.a.b. inconnus)
                    b1, b2 = _jouer_elimination(rng, prediction(nom1, nom2, m["ville"]))
            else:
                b1, b2 = _jouer_elimination(rng, prediction(nom1, nom2, m["ville"]))
            gagnant, perdant = (nom1, nom2) if b1 > b2 else (nom2, nom1)
            vainqueurs[m["id"]] = gagnant
            perdants[m["id"]] = perdant
            if m["phase"] == "finale":
                compteur[gagnant]["champion"] += 1

    return {
        nom: {phase: nb / n_simulations for phase, nb in phases.items()}
        for nom, phases in compteur.items()
    }
