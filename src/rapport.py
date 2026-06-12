"""Génération des rapports Markdown (journée, tournoi, bilan)."""


def _pc(x):
    return f"{100 * x:.1f}"


def rapport_journee(predictions, date_journee):
    lignes = [f"# Prédictions — journée du {date_journee}", ""]
    for p in predictions:
        p1, pn, p2 = p["probas"]
        lignes.append(f"## {p['equipe1']} – {p['equipe2']}")
        lignes.append("")
        lignes.append(f"| {p['equipe1']} | Nul | {p['equipe2']} |")
        lignes.append("|---|---|---|")
        lignes.append(f"| **{_pc(p1)} %** | {_pc(pn)} % | **{_pc(p2)} %** |")
        lignes.append("")
        scores = ", ".join(f"{a}-{b} ({_pc(pr)} %)" for (a, b), pr in p["scores"])
        lignes.append(f"Scores les plus probables : {scores}")
        if p["avantage"]:
            lignes.append(f"Avantage contextuel équipe 1 : {p['avantage']:+d} points Elo")
        lignes.append("")
    return "\n".join(lignes)


def rapport_tournoi(probas, n_simulations):
    lignes = [
        f"# Probabilités de parcours — {n_simulations} simulations Monte-Carlo", "",
        "| Équipe | 16es | 8es | Quarts | Demies | Finale | CHAMPION |",
        "|---|---|---|---|---|---|---|",
    ]
    tri = sorted(probas.items(), key=lambda kv: -kv[1]["champion"])
    for nom, p in tri:
        lignes.append(
            f"| {nom} | {_pc(p['16es'])} | {_pc(p['8es'])} | {_pc(p['quarts'])} "
            f"| {_pc(p['demies'])} | {_pc(p['finale'])} | **{_pc(p['champion'])}** |")
    return "\n".join(lignes)


def rapport_bilan(bilan):
    if not bilan["nb_matchs"]:
        return "# Bilan\n\nAucun match évaluable pour l'instant."
    nb = bilan["nb_matchs"]
    return "\n".join([
        "# Bilan de performance", "",
        f"Matchs évalués : **{nb}**", "",
        f"- Bons pronostics du modèle : **{bilan['bons_modele']}/{nb}**"
        f" ({_pc(bilan['bons_modele'] / nb)} %)",
        f"- Bons pronostics du naïf (favori Elo) : **{bilan['bons_naif']}/{nb}**"
        f" ({_pc(bilan['bons_naif'] / nb)} %)",
        f"- Score de Brier moyen : **{bilan['brier_moyen']:.3f}**"
        " (0 = parfait, 0.667 = hasard uniforme, 2 = pire)",
    ])
