# Modèle de Prédiction Coupe du Monde 2026 — Plan d'Implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire un modèle de prédiction évolutif de la CdM 2026 (104 matchs, 48 équipes) produisant des probabilités 1N2 + scores probables par match (Elo → Poisson) et des probabilités de parcours par équipe (simulation Monte-Carlo), mis à jour après chaque journée, avec suivi de performance vs stratégie naïve.

**Architecture:** Données en JSON (`data/`), modules de calcul purs en Python (`src/`), CLI unique (`predire.py`) avec 4 commandes (`matchs`, `tournoi`, `maj`, `bilan`), rapports Markdown horodatés (`predictions/`). L'Elo est recalculé par rejeu déterministe des matchs joués depuis l'Elo initial (idempotent, pas de dérive). Les prédictions sont archivées AVANT les matchs pour un suivi honnête (pas de data leakage).

**Tech Stack:** Python 3 (stdlib uniquement : `math`, `random`, `json`, `argparse`, `datetime`), pytest pour les tests, Git pour l'historique. Aucune dépendance lourde (pas de numpy/pandas — YAGNI).

**Limites assumées (documentées dans README):**
- λ = TOTAL_BUTS × We est un mapping simplifié Elo→buts (transparent, pas optimal).
- Départage des groupes : pts → diff → buts marqués → Elo (approximation des règles FIFA, pas de head-to-head).
- Affectation des 8 meilleurs 3èmes aux slots du tableau : algorithme glouton (approximation de la table officielle FIFA à 495 combinaisons).
- L'Elo n'évolue pas À L'INTÉRIEUR d'une simulation Monte-Carlo (statique par run).

---

## Structure des fichiers

```
Prono-CDM/
├── .venv/                      # environnement virtuel (gitignored)
├── conftest.py                 # vide — rend `src` importable par pytest
├── predire.py                  # CLI : matchs | tournoi | maj | bilan
├── README.md                   # objectif, méthode, limites
├── GUIDE.md                    # SOP : que faire après chaque journée
├── data/
│   ├── equipes.json            # 48 équipes : groupe, elo_initial, elo
│   ├── matchs.json             # 104 matchs : phase, date, ville, équipes/sources, score
│   └── villes.json             # 16 villes : pays, altitude, toit
├── src/
│   ├── __init__.py
│   ├── elo.py                  # probabilité de victoire + mise à jour Elo
│   ├── poisson.py              # loi de Poisson : matrice de scores, 1N2, tirages
│   ├── modele.py               # prédiction d'un match (Elo + domicile + altitude + repos)
│   ├── tournoi.py              # classements, 3èmes, tableau final, Monte-Carlo
│   ├── suivi.py                # score de Brier, évaluation vs stratégie naïve
│   └── rapport.py              # génération des rapports Markdown
├── tests/
│   ├── test_elo.py
│   ├── test_poisson.py
│   ├── test_modele.py
│   ├── test_tournoi.py
│   ├── test_suivi.py
│   └── test_donnees.py         # validation d'intégrité des JSON réels
└── predictions/
    ├── archive.json            # prédictions horodatées AVANT match (anti data-leakage)
    ├── journee-YYYY-MM-DD.md   # rapport par journée
    ├── tournoi-YYYY-MM-DD.md   # probabilités de parcours Monte-Carlo
    └── bilan.md                # performance modèle vs naïf
```

## Schémas de données (contrat entre toutes les tâches)

`data/equipes.json` — clé = nom français du pays :
```json
{
  "Mexique":  {"groupe": "A", "elo_initial": 1810, "elo": 1810},
  "France":   {"groupe": "I", "elo_initial": 2074, "elo": 2074}
}
```

`data/villes.json` — clé = nom de ville utilisé dans matchs.json :
```json
{
  "Mexico":    {"pays": "Mexique",     "altitude_m": 2240, "toit": false},
  "Dallas":    {"pays": "États-Unis",  "altitude_m": 184,  "toit": true}
}
```

`data/matchs.json` — liste de 104 matchs. Phases : `"groupes"`, `"16es"` (= round of 32), `"8es"`, `"quarts"`, `"demies"`, `"match3"`, `"finale"`.
```json
[
  {"id": 1,  "phase": "groupes", "groupe": "A", "date": "2026-06-11", "ville": "Mexico",
   "equipe1": "Mexique", "equipe2": "…", "score": [2, 1], "inclus_elo": true},
  {"id": 40, "phase": "groupes", "groupe": "E", "date": "2026-06-17", "ville": "Boston",
   "equipe1": "…", "equipe2": "…", "score": null, "inclus_elo": false},
  {"id": 73, "phase": "16es", "date": "2026-06-28", "ville": "Los Angeles",
   "equipe1": null, "equipe2": null, "score": null, "inclus_elo": false,
   "source1": "1A", "source2": "3CEFH"},
  {"id": 89, "phase": "8es", "date": "2026-07-03", "ville": "Houston",
   "equipe1": null, "equipe2": null, "score": null, "inclus_elo": false,
   "source1": "V73", "source2": "V74"}
]
```
Conventions `source` : `"1A"`/`"2A"` = 1er/2e du groupe A ; `"3CEFH"` = un 3e parmi les groupes C, E, F, H ; `"V73"` = vainqueur du match 73 ; `"P101"` = perdant du match 101 (petite finale). Quand un match à élimination directe est réellement joué, `equipe1`/`equipe2` sont remplis avec les vrais noms.
`inclus_elo: true` = le résultat de ce match est déjà reflété dans `elo_initial` (matchs joués avant la collecte des ratings) → le rejeu Elo le saute.

`predictions/archive.json` — clé = id du match en chaîne :
```json
{
  "40": {"horodatage": "2026-06-16T21:00:00", "equipe1": "…", "equipe2": "…",
          "probas": [0.55, 0.25, 0.20], "score_probable": [2, 1], "favori_naif": "1"}
}
```

---

### Task 1 : Initialisation du projet

**Files:**
- Create: `.gitignore`, `conftest.py`, `src/__init__.py`, `README.md`
- Create: dossiers `data/`, `src/`, `tests/`, `predictions/`

- [ ] **Step 1 : Initialiser Git et l'environnement virtuel**

```powershell
git init
py -m venv .venv
.venv\Scripts\python -m pip install pytest
```
Attendu : `Successfully installed pytest-…`. Toutes les commandes de test suivantes utilisent `.venv\Scripts\python -m pytest`.

- [ ] **Step 2 : Créer la structure et les fichiers de base**

`.gitignore` :
```
.venv/
__pycache__/
*.pyc
```

`conftest.py` : fichier **vide** (sa seule présence ajoute la racine du projet au chemin d'import de pytest).

`src/__init__.py` : fichier vide.

`README.md` :
```markdown
# Prono-CDM — Modèle de prédiction Coupe du Monde 2026

Modèle évolutif : probabilités 1N2 + scores probables (Elo → Poisson) et
probabilités de parcours (Monte-Carlo, 5000 simulations). Mise à jour après
chaque journée de matchs (voir GUIDE.md).

## Méthode
- Force des équipes : rating Elo (eloratings.net), mis à jour après chaque résultat
  (K=60, multiplicateur d'écart de buts).
- Contexte : avantage domicile (+80 Elo), bonus altitude ≥1500 m (+50), repos (±10/jour, plafonné ±30).
- Score : loi de Poisson, λ1 = 2.6 × We, λ2 = 2.6 × (1−We).
- Tournoi : simulation Monte-Carlo (groupes → 8 meilleurs 3èmes → tableau final).
- Suivi honnête : prédictions archivées AVANT les matchs, score de Brier,
  comparaison à la stratégie naïve « toujours le favori Elo ».

## Limites assumées
- Mapping Elo→buts simplifié ; départage de groupes sans head-to-head ;
  affectation des 3èmes au tableau par algorithme glouton (≠ table FIFA exacte) ;
  Elo statique à l'intérieur d'une simulation.
- Projet d'APPRENTISSAGE. Ne pas utiliser pour parier de l'argent.

## Commandes
    .venv\Scripts\python predire.py matchs    # prédit la prochaine journée
    .venv\Scripts\python predire.py tournoi   # probabilités de parcours
    .venv\Scripts\python predire.py maj       # recalcule les Elo après saisie des résultats
    .venv\Scripts\python predire.py bilan     # performance vs stratégie naïve
```

- [ ] **Step 3 : Commit**

```powershell
git add -A; git commit -m "chore: initialisation projet Prono-CDM (venv, pytest, structure)"
```

---

### Task 2 : Module Elo (`src/elo.py`)

**Files:**
- Create: `src/elo.py`
- Test: `tests/test_elo.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

`tests/test_elo.py` :
```python
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
```

- [ ] **Step 2 : Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_elo.py -v`
Attendu : `ModuleNotFoundError` ou `AttributeError` (module inexistant).

- [ ] **Step 3 : Implémenter `src/elo.py`**

```python
"""Rating Elo football — formules d'eloratings.net (K=60 pour une Coupe du Monde)."""

K_COUPE_DU_MONDE = 60


def probabilite_victoire(elo_a, elo_b, avantage_a=0):
    """Probabilité que A batte B (We). avantage_a en points Elo (domicile, etc.)."""
    d = elo_a + avantage_a - elo_b
    return 1 / (1 + 10 ** (-d / 400))


def facteur_buts(ecart):
    """Multiplicateur selon l'écart de buts : 1, 1.5, puis (11+N)/8."""
    if ecart <= 1:
        return 1.0
    if ecart == 2:
        return 1.5
    return (11 + ecart) / 8


def mise_a_jour(elo_a, elo_b, buts_a, buts_b, avantage_a=0, k=K_COUPE_DU_MONDE):
    """Nouveaux Elo (a, b) après un résultat réel. Somme des Elo conservée."""
    we = probabilite_victoire(elo_a, elo_b, avantage_a)
    if buts_a > buts_b:
        w = 1.0
    elif buts_a < buts_b:
        w = 0.0
    else:
        w = 0.5
    delta = k * facteur_buts(abs(buts_a - buts_b)) * (w - we)
    return elo_a + delta, elo_b - delta
```

- [ ] **Step 4 : Vérifier que les tests passent**

Run: `.venv\Scripts\python -m pytest tests/test_elo.py -v` — Attendu : 6 PASS.

- [ ] **Step 5 : Commit**

```powershell
git add src/elo.py tests/test_elo.py; git commit -m "feat: module Elo (We, facteur de buts, mise a jour K=60)"
```

---

### Task 3 : Module Poisson (`src/poisson.py`)

**Files:**
- Create: `src/poisson.py`
- Test: `tests/test_poisson.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

`tests/test_poisson.py` :
```python
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
```

- [ ] **Step 2 : Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_poisson.py -v` — Attendu : échec (module inexistant).

- [ ] **Step 3 : Implémenter `src/poisson.py`**

```python
"""Loi de Poisson : le modèle statistique standard du nombre de buts au football."""

import math

MAX_BUTS = 10  # au-delà, probabilité négligeable


def pmf(k, lam):
    """P(X = k) pour X ~ Poisson(lam)."""
    return math.exp(-lam) * lam ** k / math.factorial(k)


def matrice_scores(lam_a, lam_b, max_buts=MAX_BUTS):
    """matrice[i][j] = P(score i-j), buts indépendants A~Poisson(lam_a), B~Poisson(lam_b)."""
    return [
        [pmf(i, lam_a) * pmf(j, lam_b) for j in range(max_buts + 1)]
        for i in range(max_buts + 1)
    ]


def probas_1n2(matrice):
    """(P victoire A, P nul, P victoire B) à partir de la matrice de scores."""
    n = len(matrice)
    p1 = sum(matrice[i][j] for i in range(n) for j in range(n) if i > j)
    pn = sum(matrice[i][i] for i in range(n))
    p2 = sum(matrice[i][j] for i in range(n) for j in range(n) if i < j)
    return p1, pn, p2


def scores_probables(matrice, n=3):
    """Les n scores les plus probables : [((buts_a, buts_b), proba), ...] décroissant."""
    tous = [
        ((i, j), matrice[i][j])
        for i in range(len(matrice))
        for j in range(len(matrice))
    ]
    return sorted(tous, key=lambda x: -x[1])[:n]


def tirer_poisson(rng, lam):
    """Tirage aléatoire d'un nombre de buts ~ Poisson(lam) (algorithme de Knuth)."""
    seuil = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= seuil:
            return k
        k += 1
```

- [ ] **Step 4 : Vérifier que les tests passent**

Run: `.venv\Scripts\python -m pytest tests/test_poisson.py -v` — Attendu : 5 PASS.

- [ ] **Step 5 : Commit**

```powershell
git add src/poisson.py tests/test_poisson.py; git commit -m "feat: module Poisson (matrice de scores, 1N2, tirages)"
```

---

### Task 4 : Module de prédiction d'un match (`src/modele.py`)

**Files:**
- Create: `src/modele.py`
- Test: `tests/test_modele.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

`tests/test_modele.py` :
```python
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
```

- [ ] **Step 2 : Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_modele.py -v` — Attendu : échec (module inexistant).

- [ ] **Step 3 : Implémenter `src/modele.py`**

```python
"""Prédiction d'un match : Elo + contexte (domicile, altitude, repos) -> Poisson."""

from datetime import date

from src import elo, poisson

TOTAL_BUTS = 2.6          # moyenne de buts/match en Coupe du Monde
AVANTAGE_DOMICILE = 80    # points Elo si l'équipe joue dans son pays
BONUS_ALTITUDE = 50       # points Elo en plus à domicile si ville >= 1500 m
SEUIL_ALTITUDE_M = 1500
POINTS_PAR_JOUR_REPOS = 10
PLAFOND_REPOS = 30


def avantage_contextuel(nom1, nom2, ville, repos1=None, repos2=None):
    """Avantage Elo net de l'équipe 1 (négatif si le contexte favorise l'équipe 2)."""
    av = 0
    bonus_ville = AVANTAGE_DOMICILE
    if ville.get("altitude_m", 0) >= SEUIL_ALTITUDE_M:
        bonus_ville += BONUS_ALTITUDE
    if ville["pays"] == nom1:
        av += bonus_ville
    if ville["pays"] == nom2:
        av -= bonus_ville
    if repos1 is not None and repos2 is not None:
        diff_repos = (repos1 - repos2) * POINTS_PAR_JOUR_REPOS
        av += max(-PLAFOND_REPOS, min(PLAFOND_REPOS, diff_repos))
    return av


def jours_repos(nom, date_match, matchs):
    """Jours depuis le match précédent de l'équipe dans matchs (None si aucun)."""
    cible = date.fromisoformat(date_match)
    precedents = [
        date.fromisoformat(m["date"])
        for m in matchs
        if nom in (m.get("equipe1"), m.get("equipe2"))
        and date.fromisoformat(m["date"]) < cible
    ]
    if not precedents:
        return None
    return (cible - max(precedents)).days


def predire_match(nom1, elo1, nom2, elo2, ville, repos1=None, repos2=None):
    """Prédiction complète : probas 1N2, lambdas Poisson, 3 scores les plus probables."""
    av = avantage_contextuel(nom1, nom2, ville, repos1, repos2)
    we = elo.probabilite_victoire(elo1, elo2, av)
    lam1 = TOTAL_BUTS * we
    lam2 = TOTAL_BUTS * (1 - we)
    matrice = poisson.matrice_scores(lam1, lam2)
    return {
        "equipe1": nom1,
        "equipe2": nom2,
        "we": we,
        "avantage": av,
        "lambdas": (lam1, lam2),
        "probas": poisson.probas_1n2(matrice),
        "scores": poisson.scores_probables(matrice, 3),
    }
```

- [ ] **Step 4 : Vérifier que les tests passent**

Run: `.venv\Scripts\python -m pytest tests/test_modele.py -v` — Attendu : 4 PASS.

- [ ] **Step 5 : Commit**

```powershell
git add src/modele.py tests/test_modele.py; git commit -m "feat: prediction d'un match (Elo + domicile/altitude/repos -> Poisson)"
```

---

### Task 5 : Collecte des données réelles (recherche web)

**Files:**
- Create: `data/equipes.json`, `data/matchs.json`, `data/villes.json`
- Test: `tests/test_donnees.py`

> Cette tâche n'est pas du TDD classique : c'est de l'acquisition de données,
> validée par un test d'intégrité. L'exécutant DOIT avoir accès au web
> (WebFetch/WebSearch) et DOIT dater chaque source.

- [ ] **Step 1 : Écrire le test d'intégrité**

`tests/test_donnees.py` :
```python
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
        assert 1300 < infos["elo_initial"] < 2300
        assert infos["elo"] == infos["elo_initial"]


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
```

- [ ] **Step 2 : Collecter les données sur le web**

Sources (dans cet ordre, croiser si incohérence) :
1. `https://en.wikipedia.org/wiki/2026_FIFA_World_Cup` — les 12 groupes, les 48 équipes, le calendrier complet (dates, stades), les résultats déjà joués, la structure du tableau final (quels 1ers/2es/3es vont dans quel match — remplir `source1`/`source2`).
2. `https://en.wikipedia.org/wiki/World_Football_Elo_Ratings` — ratings Elo actuels. **Noter la date « as of » du tableau.** Pour toute équipe absente du top affiché, repli : `https://www.eloratings.net/` ; en dernier recours estimer depuis le classement FIFA (documenter dans le commit).
3. Pages groupes (`…World_Cup_Group_A`, etc.) si la page principale ne suffit pas pour les résultats.

Règles impératives :
- Noms de pays **en français** et identiques partout (« États-Unis », « Mexique », « Canada »…). Le nom d'équipe doit être égal au champ `pays` des villes hôtes pour que l'avantage domicile fonctionne.
- `inclus_elo: true` UNIQUEMENT pour les matchs dont le résultat est déjà reflété dans le rating récupéré (date du match ≤ date « as of » du tableau Elo). Sinon `false`.
- Villes (clé courte, stade → ville) : Atlanta, Boston, Dallas, Guadalajara, Houston, Kansas City, Los Angeles, Mexico, Miami, Monterrey, New York, Philadelphie, San Francisco, Seattle, Toronto, Vancouver. Altitudes approximatives à vérifier : Mexico 2240 m, Guadalajara 1566 m, Monterrey 540 m, les autres < 400 m. `toit: true` pour Atlanta, Dallas, Houston, Los Angeles, Vancouver (vérifier).

- [ ] **Step 3 : Vérifier que le test d'intégrité passe**

Run: `.venv\Scripts\python -m pytest tests/test_donnees.py -v` — Attendu : 3 PASS.

- [ ] **Step 4 : Commit (avec sources et dates dans le message)**

```powershell
git add data/ tests/test_donnees.py
git commit -m "data: 48 equipes, 104 matchs, 16 villes (Wikipedia + eloratings.net, releve du 2026-06-12)"
```

---

### Task 6 : Module tournoi (`src/tournoi.py`)

**Files:**
- Create: `src/tournoi.py`
- Test: `tests/test_tournoi.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

`tests/test_tournoi.py` :
```python
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
    assert [e["nom"] for e in cl] == ["A1", "A2", "A3", "A4"]
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
            {"nom": f"{g}1", "pts": 9, "diff": 5, "bp": 6, "elo": 1900},
            {"nom": f"{g}2", "pts": 6, "diff": 1, "bp": 4, "elo": 1850},
            {"nom": f"{g}3", "pts": i, "diff": 0, "bp": 2, "elo": 1800},  # pts = 0..11
            {"nom": f"{g}4", "pts": 0, "diff": -6, "bp": 1, "elo": 1700},
        ]
    tiers = tournoi.meilleurs_troisiemes(classements)
    assert len(tiers) == 8
    assert ("L", "L3") in tiers and ("E", "E3") in tiers  # pts 11 et 4
    assert ("A", "A3") not in tiers                        # pts 0 : éliminé


def test_simulation_probabilites_coherentes(equipes_simulation=None):
    equipes = {
        f"E{i}": {"groupe": g, "elo": 2000 - i * 15}
        for i, g in enumerate(
            [g for g in "ABCDEFGHIJKL" for _ in range(4)]
        )
    }
    matchs = tournoi.calendrier_synthetique(equipes)
    resultat = tournoi.simuler_tournoi(equipes, matchs,
                                       {"Ville": {"pays": "Nulle-part", "altitude_m": 0}},
                                       n_simulations=200, graine=42)
    assert abs(sum(r["champion"] for r in resultat.values()) - 1.0) < 1e-9
    assert resultat["E0"]["champion"] > resultat["E47"]["champion"]  # plus fort Elo
    # reproductibilité : même graine => même résultat
    resultat2 = tournoi.simuler_tournoi(equipes, matchs,
                                        {"Ville": {"pays": "Nulle-part", "altitude_m": 0}},
                                        n_simulations=200, graine=42)
    assert resultat == resultat2
```

- [ ] **Step 2 : Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_tournoi.py -v` — Attendu : échec (module inexistant).

- [ ] **Step 3 : Implémenter `src/tournoi.py`**

```python
"""Classements de groupes, meilleurs troisièmes, tableau final, simulation Monte-Carlo."""

import random

from src import modele, poisson

PHASES_KO = ["16es", "8es", "quarts", "demies", "match3", "finale"]


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
    for i, g in enumerate(lettres):           # 12 vainqueurs + 12 deuxièmes + 8 tiers
        sources.append("1" + g)
        sources.append("2" + g)
    for i in range(8):
        sources.append("3" + lettres)          # n'importe quel groupe (synthétique)
    paires = [(sources[i], sources[i + 1]) for i in range(0, 32, 2)]
    ids_tour = []
    for s1, s2 in paires:                      # 16es
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
    matchs_ko = [m for m in matchs if m["phase"] != "groupes"]
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
                {"equipe1": m["equipe1"], "equipe2": m["equipe2"], "score": list(score)})
        # 2) Classements et qualifiés.
        classements, qualifies = {}, {}
        groupes = {}
        for nom, infos in equipes.items():
            groupes.setdefault(infos["groupe"], []).append(nom)
        for g, noms in groupes.items():
            cl = classement_groupe(noms, resultats_groupes.get(g, []), equipes)
            classements[g] = cl
            qualifies["1" + g] = cl[0]["nom"]
            qualifies["2" + g] = cl[1]["nom"]
        tiers = meilleurs_troisiemes(classements)
        affectation_tiers = _affecter_troisiemes(matchs_ko, tiers)
        # 3) Tableau final.
        vainqueurs, perdants = {}, {}

        def resoudre(m, cle):
            if m.get(cle.replace("source", "equipe")):
                return m[cle.replace("source", "equipe")]   # match KO déjà joué en vrai
            src = m[cle]
            if src.startswith("V"):
                return vainqueurs[int(src[1:])]
            if src.startswith("P"):
                return perdants[int(src[1:])]
            if src.startswith("3"):
                return affectation_tiers[(m["id"], cle)]
            return qualifies[src]

        for m in sorted(matchs_ko, key=lambda x: x["id"]):
            nom1 = resoudre(m, "source1")
            nom2 = resoudre(m, "source2")
            if m["phase"] in compteur[nom1]:
                compteur[nom1][m["phase"]] += 1
                compteur[nom2][m["phase"]] += 1
            if m["score"] is not None:
                b1, b2 = m["score"]
                if b1 == b2:  # score réel nul en KO : départage aux tirs au but inconnu
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
```

- [ ] **Step 4 : Vérifier que les tests passent**

Run: `.venv\Scripts\python -m pytest tests/test_tournoi.py -v` — Attendu : 4 PASS.
(Si lenteur > 30 s : réduire `n_simulations` du test à 100.)

- [ ] **Step 5 : Commit**

```powershell
git add src/tournoi.py tests/test_tournoi.py; git commit -m "feat: simulation Monte-Carlo du tournoi (groupes, 3emes, tableau final)"
```

---

### Task 7 : Module de suivi de performance (`src/suivi.py`)

**Files:**
- Create: `src/suivi.py`
- Test: `tests/test_suivi.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

`tests/test_suivi.py` :
```python
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
```

- [ ] **Step 2 : Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_suivi.py -v` — Attendu : échec.

- [ ] **Step 3 : Implémenter `src/suivi.py`**

```python
"""Suivi honnête de la performance : score de Brier + comparaison à la stratégie naïve."""

ISSUES = ("1", "N", "2")


def resultat_1n2(buts1, buts2):
    if buts1 > buts2:
        return "1"
    if buts1 < buts2:
        return "2"
    return "N"


def brier(probas, resultat):
    """Score de Brier multi-classes : 0 = parfait, 2 = pire possible."""
    cible = [1.0 if issue == resultat else 0.0 for issue in ISSUES]
    return sum((p - c) ** 2 for p, c in zip(probas, cible))


def evaluer(archive, matchs):
    """Compare le modèle (issue la plus probable) à la stratégie naïve (favori Elo)."""
    nb = bons_modele = bons_naif = 0
    somme_brier = 0.0
    for m in matchs:
        cle = str(m["id"])
        if m.get("score") is None or cle not in archive:
            continue
        reel = resultat_1n2(*m["score"])
        probas = archive[cle]["probas"]
        choix_modele = ISSUES[probas.index(max(probas))]
        nb += 1
        bons_modele += int(choix_modele == reel)
        bons_naif += int(archive[cle]["favori_naif"] == reel)
        somme_brier += brier(probas, reel)
    return {
        "nb_matchs": nb,
        "bons_modele": bons_modele,
        "bons_naif": bons_naif,
        "brier_moyen": somme_brier / nb if nb else None,
    }
```

- [ ] **Step 4 : Vérifier que les tests passent**

Run: `.venv\Scripts\python -m pytest tests/test_suivi.py -v` — Attendu : 3 PASS.

- [ ] **Step 5 : Commit**

```powershell
git add src/suivi.py tests/test_suivi.py; git commit -m "feat: suivi de performance (Brier, modele vs favori naif)"
```

---

### Task 8 : Rapports Markdown (`src/rapport.py`) et CLI (`predire.py`)

**Files:**
- Create: `src/rapport.py`, `predire.py`
- Test: `tests/test_rapport.py`

- [ ] **Step 1 : Écrire les tests qui échouent**

`tests/test_rapport.py` :
```python
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
```

- [ ] **Step 2 : Vérifier l'échec**

Run: `.venv\Scripts\python -m pytest tests/test_rapport.py -v` — Attendu : échec.

- [ ] **Step 3 : Implémenter `src/rapport.py`**

```python
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
```

- [ ] **Step 4 : Vérifier que les tests passent, puis écrire le CLI**

Run: `.venv\Scripts\python -m pytest tests/test_rapport.py -v` — Attendu : 2 PASS.

`predire.py` :
```python
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
    equipes, matchs, villes = charger("equipes.json"), charger("matchs.json"), charger("villes.json")
    a_venir = [m for m in matchs if m["score"] is None and m.get("equipe1") and m.get("equipe2")]
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
            p1, _, p2 = p["probas"]
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
    sauver_pred(f"journee-{date_cible}.md", rapport.rapport_journee(predictions, date_cible))


def cmd_tournoi(args):
    equipes, matchs, villes = charger("equipes.json"), charger("matchs.json"), charger("villes.json")
    probas = tournoi.simuler_tournoi(equipes, matchs, villes,
                                     n_simulations=args.simulations, graine=args.graine)
    date_jour = datetime.now().date().isoformat()
    sauver_pred(f"tournoi-{date_jour}.md", rapport.rapport_tournoi(probas, args.simulations))


def cmd_maj(args):
    """Recalcule tous les Elo par rejeu depuis elo_initial (idempotent)."""
    equipes, matchs = charger("equipes.json"), charger("matchs.json")
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
    p_matchs.add_argument("date", nargs="?", help="AAAA-MM-JJ (défaut: prochaine journée)")
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
```

- [ ] **Step 5 : Vérification complète et fumée du CLI**

```powershell
.venv\Scripts\python -m pytest -v                      # TOUT doit passer
.venv\Scripts\python predire.py maj                    # recalcul Elo sans erreur
.venv\Scripts\python predire.py matchs                 # génère journee-….md
.venv\Scripts\python predire.py tournoi --simulations 1000
.venv\Scripts\python predire.py bilan
```
Attendu : aucun crash, fichiers créés dans `predictions/`.

- [ ] **Step 6 : Commit**

```powershell
git add src/rapport.py tests/test_rapport.py predire.py
git commit -m "feat: rapports Markdown et CLI (matchs, tournoi, maj, bilan)"
```

---

### Task 9 : Guide d'utilisation et première prédiction réelle

**Files:**
- Create: `GUIDE.md`
- Create (générés) : `predictions/journee-….md`, `predictions/tournoi-….md`, `predictions/archive.json`

- [ ] **Step 1 : Écrire `GUIDE.md`**

```markdown
# GUIDE — Mise à jour après chaque journée de matchs

À faire le soir de chaque journée (ou le lendemain matin), dans Claude Code,
depuis le dossier Prono-CDM :

1. **Saisir les résultats** — demander à Claude :
   « Mets à jour data/matchs.json avec les résultats d'hier (recherche web,
   source officielle FIFA ou Wikipedia). Pour les matchs à élimination directe
   joués, remplis aussi equipe1/equipe2. »
2. **Recalculer les Elo** :
       .venv\Scripts\python predire.py maj
3. **Prédire la journée suivante** (archive automatiquement les prédictions) :
       .venv\Scripts\python predire.py matchs
4. **Re-simuler le tournoi** :
       .venv\Scripts\python predire.py tournoi
5. **Suivre la performance** :
       .venv\Scripts\python predire.py bilan
6. **Sauvegarder** :
       git add -A
       git commit -m "maj: resultats du <date> + predictions"

Règles d'honnêteté :
- Ne JAMAIS modifier predictions/archive.json à la main (c'est le juge de paix).
- Lancer `matchs` AVANT le coup d'envoi, jamais après.
- À la fin des groupes : demander à Claude de remplir equipe1/equipe2 des 16es
  selon les classements réels et la table officielle FIFA des troisièmes.
```

- [ ] **Step 2 : Générer les premières prédictions réelles**

```powershell
.venv\Scripts\python predire.py maj
.venv\Scripts\python predire.py matchs
.venv\Scripts\python predire.py tournoi
```
Attendu : rapports créés. **Contrôle de cohérence humain** : les ~5 premiers du
classement « champion » doivent être des nations crédibles (Espagne, Argentine,
France, Angleterre, Brésil…). Si un outsider improbable sort en tête → bug ou
données fausses, NE PAS committer, déboguer (vérifier les Elo importés).

- [ ] **Step 3 : Commit final**

```powershell
git add -A; git commit -m "docs: guide de mise a jour + premieres predictions reelles"
```

---

## Auto-revue effectuée

- **Couverture du cahier des charges** : probabilités 1N2 ✓ (Task 4/8), scores probables ✓ (Task 3/4), favoris/parcours Monte-Carlo ✓ (Task 6), avantage domicile/altitude/repos ✓ (Task 4), mise à jour par journée ✓ (Task 8 `maj` + Task 9 GUIDE), suivi vs naïf + Brier ✓ (Task 7), anti data-leakage ✓ (archive première prédiction, Task 8).
- **Placeholder scan** : aucun TBD/TODO ; tout le code est écrit.
- **Cohérence des noms** : `probabilite_victoire/facteur_buts/mise_a_jour` (elo), `pmf/matrice_scores/probas_1n2/scores_probables/tirer_poisson` (poisson), `avantage_contextuel/jours_repos/predire_match/TOTAL_BUTS` (modele), `classement_groupe/meilleurs_troisiemes/simuler_tournoi/calendrier_synthetique` (tournoi), `resultat_1n2/brier/evaluer` (suivi), `rapport_journee/rapport_tournoi/rapport_bilan` (rapport) — vérifiés identiques entre tâches.
