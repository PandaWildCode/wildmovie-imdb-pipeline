"""
Moteur de recommandation WildMovie.

Situation de cold start : aucun historique de préférences client n'existe.
La recommandation repose donc uniquement sur les caractéristiques des films —
ici, les personnes qui les ont faits (acteurs, réalisateurs, compositeurs),
pondérées par rôle, plus un bonus de notoriété.

Ce module ne dépend pas de Streamlit : il est testable seul.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

# Poids par rôle dans le calcul de similarité.
# Le réalisateur pèse le plus : c'est lui qui porte l'univers d'un film.
POIDS_ROLE = {"D_IDS": 3.0, "C_IDS": 1.6, "A_IDS": 1.0}

# Prior bayésien pour la note : un film à 9/10 sur 600 votes ne vaut pas
# un film à 9/10 sur 600 000 votes.
VOTES_PRIOR = 5_000


def charger_films(chemin: str) -> pd.DataFrame:
    """Charge le catalogue et normalise les colonnes de listes."""
    df = pd.read_csv(chemin, compression="gzip")

    for col in ("ACTEURS", "REALISATEURS", "COMPOSITEURS", "A_IDS", "D_IDS", "C_IDS"):
        df[col] = df[col].fillna("")

    for col in ("NOTE_MOYENNE_IMDB", "NOTE_MOYENNE_TMDB", "MEILLEURE_NOTE"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("NOMBRE_DE_VOTES_IMDB", "NOMBRE_DE_VOTES_TMDB"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    df["TITRE"] = df["TITRE"].fillna("Sans titre")
    df["TITRE_VO"] = df["TITRE_VO"].fillna(df["TITRE"])
    df["NOTE_BAYES"] = note_bayesienne(df)
    df["RECHERCHE"] = (
        df["TITRE"].str.lower() + " " + df["TITRE_VO"].str.lower()
    ).str.strip()
    return df


def note_bayesienne(df: pd.DataFrame) -> pd.Series:
    """Note lissée : tire la note vers la moyenne du catalogue quand les votes sont rares."""
    notes = df["MEILLEURE_NOTE"].fillna(df["NOTE_MOYENNE_IMDB"])
    votes = df["NOMBRE_DE_VOTES_IMDB"].astype(float)
    moyenne_globale = float(np.nanmean(notes)) if len(notes) else 5.0
    lissee = (votes * notes.fillna(moyenne_globale) + VOTES_PRIOR * moyenne_globale) / (
        votes + VOTES_PRIOR
    )
    return lissee.round(2)


def construire_matrice(df: pd.DataFrame):
    """
    Matrice creuse films × personnes, pondérée par rôle et normalisée L2.

    Normaliser chaque ligne rend la similarité comparable entre un film à
    trois intervenants connus et un film au générique fourni.
    """
    vocabulaire: dict[str, int] = {}
    lignes, colonnes, valeurs = [], [], []

    for i, row in enumerate(df.itertuples(index=False)):
        vus: dict[int, float] = {}
        for colonne, poids in POIDS_ROLE.items():
            brut = getattr(row, colonne)
            if not brut:
                continue
            for identifiant in brut.split("|"):
                if not identifiant:
                    continue
                j = vocabulaire.setdefault(identifiant, len(vocabulaire))
                # Une personne créditée deux fois ne compte qu'une, au poids le plus fort.
                vus[j] = max(vus.get(j, 0.0), poids)
        for j, v in vus.items():
            lignes.append(i)
            colonnes.append(j)
            valeurs.append(v)

    matrice = sparse.csr_matrix(
        (valeurs, (lignes, colonnes)),
        shape=(len(df), max(len(vocabulaire), 1)),
        dtype=np.float32,
    )

    normes = np.sqrt(matrice.multiply(matrice).sum(axis=1)).A.ravel()
    normes[normes == 0] = 1.0
    inverse = sparse.diags(1.0 / normes)
    return inverse @ matrice, vocabulaire


def recommander(
    df: pd.DataFrame,
    matrice,
    index_film: int,
    nb: int = 12,
    poids_notoriete: float = 0.25,
    votes_min: int = 0,
) -> pd.DataFrame:
    """
    Renvoie les films les plus proches du film choisi.

    Score = (1 - poids_notoriete) × similarité de générique
          +      poids_notoriete  × note bayésienne normalisée
    """
    similarites = (matrice @ matrice[index_film].T).toarray().ravel()
    similarites[index_film] = -1.0  # ne jamais se recommander soi-même

    notes = df["NOTE_BAYES"].to_numpy(dtype=np.float32)
    etendue = float(notes.max() - notes.min()) or 1.0
    notes_norm = (notes - notes.min()) / etendue

    score = (1.0 - poids_notoriete) * similarites + poids_notoriete * notes_norm
    score[similarites <= 0] = -1.0  # aucun intervenant en commun → hors sujet

    if votes_min > 0:
        score[df["NOMBRE_DE_VOTES_IMDB"].to_numpy() < votes_min] = -1.0

    nb_valides = int((score > -1.0).sum())
    if nb_valides == 0:
        return df.head(0).assign(SCORE=[], SIMILARITE=[])

    k = min(nb, nb_valides)
    candidats = np.argpartition(-score, k - 1)[:k]
    candidats = candidats[np.argsort(-score[candidats])]

    resultat = df.iloc[candidats].copy()
    resultat["SCORE"] = np.round(score[candidats] * 100, 1)
    resultat["SIMILARITE"] = np.round(similarites[candidats] * 100, 1)
    return resultat


def intervenants_communs(df: pd.DataFrame, i: int, j: int) -> list[str]:
    """Noms partagés par deux films — sert à expliquer une recommandation."""
    def personnes(k: int) -> dict[str, str]:
        paires: dict[str, str] = {}
        for col_id, col_nom in (
            ("D_IDS", "REALISATEURS"),
            ("C_IDS", "COMPOSITEURS"),
            ("A_IDS", "ACTEURS"),
        ):
            ids = [x for x in str(df.iloc[k][col_id]).split("|") if x]
            noms = [x for x in str(df.iloc[k][col_nom]).split("|") if x]
            paires.update(dict(zip(ids, noms)))
        return paires

    a, b = personnes(i), personnes(j)
    return [a[k] for k in a.keys() & b.keys()]


def chercher(df: pd.DataFrame, requete: str, limite: int = 40) -> pd.DataFrame:
    """Recherche par titre, français ou version originale."""
    requete = (requete or "").strip().lower()
    if not requete:
        return df.head(0)
    masque = df["RECHERCHE"].str.contains(requete, regex=False, na=False)
    trouves = df[masque]
    exact = trouves["TITRE"].str.lower() == requete
    return pd.concat([trouves[exact], trouves[~exact]]).head(limite)
