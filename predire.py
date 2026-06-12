"""CLI Prono-CDM : matchs | tournoi | maj | bilan."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src import elo, modele, rapport, suivi, tournoi

RACINE = Path(__file__).parent
DATA = RACINE / "data"
PRED = RACINE / "predictions"


def charger(nom):
    return json.loads((DATA / nom).read_text(encoding="utf-8"))


def sauver_pred(nom, contenu):
    PRED.mkdir(exist_ok=True)
    (PRED / nom).write_text(contenu, encoding="utf-8")
    print(f"-> predictions/{nom}")


def charger_archive():
    chemin = PRED / "archive.json"
    return json.loads(chemin.read_text(encoding="utf-8")) if chemin.exists() else {}


def cmd_matchs(args):
    equipes = charger("equipes.json")
    matchs = charger("matchs.json")
    villes = charger("villes.json")
    a_venir = [m for m in matchs
               if m["score"] is None and m.get("equipe1") and m.get("equipe2")]
    if not a_venir:
        print("Aucun match prêt à prédire (équipes inconnues ou tout est joué).")
        return
    date_cible = args.date or min(m["date"] for m in a_venir)
    journee = [m for m in a_venir if m["date"] == date_cible]
    archive = charger_archive()
    predictions = []
    for m in journee:
        r1 = modele.jours_repos(m["equipe1"], m["date"], matchs)
        r2 = modele.jours_repos(m["equipe2"], m["date"], matchs)
        p = modele.predire_match(
            m["equipe1"], equipes[m["equipe1"]]["elo"],
            m["equipe2"], equipes[m["equipe2"]]["elo"],
            villes[m["ville"]], r1, r2)
        predictions.append(p)
        cle = str(m["id"])
        if cle not in archive:  # on garde la PREMIÈRE prédiction (anti-triche)
            archive[cle] = {
                "horodatage": datetime.now().isoformat(timespec="seconds"),
                "equipe1": m["equipe1"], "equipe2": m["equipe2"],
                "probas": list(p["probas"]),
                "score_probable": list(p["scores"][0][0]),
                "favori_naif": "1" if p["we"] >= 0.5 else "2",
            }
    PRED.mkdir(exist_ok=True)
    (PRED / "archive.json").write_text(
        json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")
    sauver_pred(f"journee-{date_cible}.md",
                rapport.rapport_journee(predictions, date_cible))


def cmd_tournoi(args):
    equipes = charger("equipes.json")
    matchs = charger("matchs.json")
    villes = charger("villes.json")
    probas = tournoi.simuler_tournoi(equipes, matchs, villes,
                                     n_simulations=args.simulations,
                                     graine=args.graine)
    date_jour = datetime.now().date().isoformat()
    sauver_pred(f"tournoi-{date_jour}.md",
                rapport.rapport_tournoi(probas, args.simulations))


def cmd_maj(args):
    """Recalcule tous les Elo par rejeu depuis elo_initial (idempotent)."""
    equipes = charger("equipes.json")
    matchs = charger("matchs.json")
    villes = charger("villes.json")
    for infos in equipes.values():
        infos["elo"] = infos["elo_initial"]
    joues = [m for m in matchs
             if m["score"] is not None and not m.get("inclus_elo")
             and m.get("equipe1") and m.get("equipe2")]
    for m in sorted(joues, key=lambda m: (m["date"], m["id"])):
        e1, e2 = m["equipe1"], m["equipe2"]
        av = modele.avantage_contextuel(e1, e2, villes[m["ville"]])
        equipes[e1]["elo"], equipes[e2]["elo"] = elo.mise_a_jour(
            equipes[e1]["elo"], equipes[e2]["elo"], *m["score"], avantage_a=av)
    (DATA / "equipes.json").write_text(
        json.dumps(equipes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Elo recalculés à partir de {len(joues)} matchs joués.")
    haut = sorted(equipes.items(), key=lambda kv: -kv[1]["elo"])[:5]
    for nom, infos in haut:
        print(f"  {nom}: {infos['elo']:.0f}")


def cmd_bilan(args):
    matchs = charger("matchs.json")
    bilan = suivi.evaluer(charger_archive(), matchs)
    contenu = rapport.rapport_bilan(bilan)
    sauver_pred("bilan.md", contenu)
    print(contenu)


def main():
    parser = argparse.ArgumentParser(description="Prono-CDM 2026")
    sous = parser.add_subparsers(dest="commande", required=True)
    p_matchs = sous.add_parser("matchs", help="prédit la prochaine journée")
    p_matchs.add_argument("date", nargs="?",
                          help="AAAA-MM-JJ (défaut: prochaine journée)")
    p_matchs.set_defaults(fonction=cmd_matchs)
    p_tournoi = sous.add_parser("tournoi", help="simulation Monte-Carlo du tournoi")
    p_tournoi.add_argument("--simulations", type=int, default=5000)
    p_tournoi.add_argument("--graine", type=int, default=42)
    p_tournoi.set_defaults(fonction=cmd_tournoi)
    sous.add_parser("maj", help="recalcule les Elo après saisie des résultats") \
        .set_defaults(fonction=cmd_maj)
    sous.add_parser("bilan", help="performance du modèle vs naïf") \
        .set_defaults(fonction=cmd_bilan)
    args = parser.parse_args()
    args.fonction(args)


if __name__ == "__main__":
    main()
