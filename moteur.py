"""
CinéMad — moteur de recommandation.

Portage du moteur du Projet 2 (Wild Code School, RNCP 37429) vers une version
déployable : mêmes caractéristiques, mêmes pondérations, mêmes modes de tri.

Trois choses ont changé par rapport au script d'origine :
  1. le catalogue est chargé depuis des morceaux compressés (limite de taille
     de fichier de GitHub) au lieu d'un CSV de 119 Mo ;
  2. les identifiants de personnes sont déjà résolus en noms dans les données,
     ce qui évite de charger un référentiel de 450 Mo au démarrage ;
  3. `CountVectorizer(tokenizer=...)`, déprécié puis retiré de scikit-learn,
     est remplacé par `analyzer=...`, qui fait la même chose sans avertissement.
"""

from __future__ import annotations

import glob
import os

import pandas as pd
import streamlit as st
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


RACINE = os.path.dirname(os.path.abspath(__file__))
DOSSIER_DONNEES = os.path.join(RACINE, "data")

# Les affiches TMDB partagent toutes le même préfixe : on ne stocke que le suffixe.
PREFIXE_AFFICHE = "https://image.tmdb.org/t/p/w342/"

COLONNES_NUMERIQUES = [
    "IMDB_RATING",
    "TMDB_RATING",
    "IMDB_VOTE_COUNT",
    "TMDB_VOTE_COUNT",
    "YEAR",
    "DURATION_MINUTES",
]

# Pondération des blocs de caractéristiques, reprise du moteur d'origine.
POIDS = {
    "genres": 1.0,
    "resume": 2.0,
    "numerique": 2.0,
    "realisateur": 1.0,
    "acteurs": 1.0,
    "compositeur": 0.15,
    "titre": 1.0,
}


def _par_barre(texte: str) -> list[str]:
    """Découpe une chaîne « a|b|c » en jetons, en ignorant les vides."""
    return [jeton for jeton in str(texte).split("|") if jeton]


def _tranche(valeur, bornes, prefixe: str) -> str:
    """Range une valeur numérique dans une tranche nommée."""
    if pd.isna(valeur):
        return f"{prefixe}_inconnu"
    for borne in bornes:
        if valeur < borne:
            return f"{prefixe}_moins_{borne}"
    return f"{prefixe}_plus_{bornes[-1]}"


def jetons_numeriques(ligne) -> str:
    """
    Traduit les caractéristiques chiffrées en jetons comparables.

    Pourquoi ne pas les utiliser telles quelles : mises à l'échelle entre 0 et 1,
    l'année devient une valeur quasi identique pour tous les films (2001/2025 ≈
    0,99). Sous une similarité cosinus, cette quasi-constante domine le calcul et
    les plus proches voisins deviennent mécaniquement les films au vecteur le plus
    pauvre — des œuvres confidentielles sans rapport avec la demande. Découpées en
    tranches, les mêmes informations redeviennent discriminantes : deux films
    partagent leur décennie, leur format et leur niveau de notoriété, ou ne les
    partagent pas.
    """
    annee = ligne["YEAR"]
    decennie = f"decennie_{int(annee) // 10 * 10}" if pd.notna(annee) and annee > 1800 else "decennie_inconnue"

    votes = ligne["TOTAL_VOTES"]
    if votes >= 500_000:
        notoriete = "notoriete_mondiale"
    elif votes >= 100_000:
        notoriete = "notoriete_grand_public"
    elif votes >= 10_000:
        notoriete = "notoriete_connue"
    elif votes >= 1_000:
        notoriete = "notoriete_confidentielle"
    else:
        notoriete = "notoriete_rare"

    return "|".join(
        [
            decennie,
            notoriete,
            _tranche(ligne["DURATION_MINUTES"], [60, 90, 120, 150], "duree"),
            _tranche(ligne["BEST_RATING"], [5, 6, 7, 8], "note"),
        ]
    )


# --------------------------------------------------------------------------- #
#  Chargement
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Chargement du catalogue…")
def charger_catalogue() -> pd.DataFrame:
    # Le catalogue est découpé en morceaux pour tenir sous la limite de taille
    # de fichier de GitHub. On les cherche dans data/ puis à la racine.
    morceaux = sorted(
        glob.glob(os.path.join(DOSSIER_DONNEES, "films_*.csv.gz"))
        or glob.glob(os.path.join(RACINE, "films_*.csv.gz"))
    )
    if not morceaux:
        raise FileNotFoundError(
            f"Aucun fichier films_*.csv.gz dans {DOSSIER_DONNEES} ni dans {RACINE}."
        )

    df = pd.concat(
        (pd.read_csv(m, dtype=str, compression="gzip") for m in morceaux),
        ignore_index=True,
    )
    df.columns = df.columns.str.strip()

    for colonne in ("SUMMARY", "GENRES", "TITLE_ORIGINAL", "TITLE_FR", "TITLE_EN",
                    "POSTER_PATH", "ACTOR_NAME", "DIRECTOR_NAME", "COMPOSER_NAME"):
        if colonne in df.columns:
            df[colonne] = df[colonne].fillna("").astype(str)

    for colonne in COLONNES_NUMERIQUES + ["BEST_RATING"]:
        df[colonne] = pd.to_numeric(df[colonne], errors="coerce")

    # TITLE_FR et TITLE_EN concatènent toutes les variantes régionales d'un même
    # film (« Le chevalier noir|The Dark Knight : Le Chevalier noir »). Pour
    # l'affichage on prend la première ; toutes servent à la recherche.
    df["TITRE_AFFICHE"] = df["TITLE_FR"].str.split("|").str[0].str.strip()
    df["TITRE_AFFICHE"] = df["TITRE_AFFICHE"].where(
        df["TITRE_AFFICHE"] != "", df["TITLE_ORIGINAL"]
    )

    df["TOTAL_VOTES"] = (
        df["IMDB_VOTE_COUNT"].fillna(0) + df["TMDB_VOTE_COUNT"].fillna(0)
    ).astype("int64")
    df["YEAR_NUM"] = df["YEAR"].fillna(0).astype("int64")
    df["BEST_RATING"] = df["BEST_RATING"].fillna(0)

    # Index de recherche : toutes les variantes de titre, en minuscules,
    # encadrées de barres pour permettre une comparaison exacte par jeton.
    colonnes_titres = [c for c in ("TITLE_ORIGINAL", "TITLE_FR", "TITLE_EN") if c in df.columns]
    concat = df[colonnes_titres[0]].str.lower()
    for colonne in colonnes_titres[1:]:
        concat = concat + "|" + df[colonne].str.lower()
    df["_TITRES"] = "|" + concat.str.replace(r"\|+", "|", regex=True).str.strip("|") + "|"
    return df


def url_affiche(suffixe: str) -> str:
    """Reconstruit l'URL complète d'une affiche TMDB, ou une chaîne vide."""
    suffixe = str(suffixe).strip().lstrip("/")
    return PREFIXE_AFFICHE + suffixe if suffixe else ""


# --------------------------------------------------------------------------- #
#  Modèle
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Construction du modèle de recommandation…")
def construire_modele():
    """
    Assemble les sept blocs de caractéristiques et entraîne le KNN cosinus.

    Le résumé pèse double parce que c'est lui qui porte le sujet du film ;
    le compositeur pèse 0,15 parce qu'un même compositeur signe des films
    très différents et rapprocherait à tort des œuvres sans rapport.
    """
    df = charger_catalogue()

    bloc_resume = TfidfVectorizer(stop_words="english").fit_transform(df["SUMMARY"])
    bloc_genres = CountVectorizer(analyzer=_par_barre).fit_transform(df["GENRES"])
    bloc_acteurs = CountVectorizer(analyzer=_par_barre).fit_transform(df["ACTOR_NAME"])
    bloc_realisateur = CountVectorizer(analyzer=_par_barre).fit_transform(df["DIRECTOR_NAME"])
    bloc_compositeur = CountVectorizer(analyzer=_par_barre).fit_transform(df["COMPOSER_NAME"])

    # Bigrammes sur le titre : c'est ce qui rattrape les sagas
    # (« Harry Potter and the… », « The Godfather Part… »).
    bloc_titre = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(df["TITLE_ORIGINAL"])

    bloc_numerique = CountVectorizer(analyzer=_par_barre).fit_transform(
        df.apply(jetons_numeriques, axis=1)
    )

    combine = hstack(
        [
            bloc_genres * POIDS["genres"],
            bloc_resume * POIDS["resume"],
            bloc_numerique * POIDS["numerique"],
            bloc_realisateur * POIDS["realisateur"],
            bloc_acteurs * POIDS["acteurs"],
            bloc_compositeur * POIDS["compositeur"],
            bloc_titre * POIDS["titre"],
        ]
    ).tocsr()

    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(combine)
    return df, combine, knn


# --------------------------------------------------------------------------- #
#  Tri
# --------------------------------------------------------------------------- #

CLES_DE_TRI = {
    "recent": ["YEAR_NUM", "TOTAL_VOTES", "BEST_RATING"],
    "rating": ["BEST_RATING", "TOTAL_VOTES", "YEAR_NUM"],
    "votes": ["TOTAL_VOTES", "BEST_RATING", "YEAR_NUM"],
}


def trier(df: pd.DataFrame, tri: str = "recent") -> pd.DataFrame:
    cles = CLES_DE_TRI.get(tri, CLES_DE_TRI["recent"])
    return df.sort_values(cles, ascending=False, kind="mergesort")


def trier_recommandations(df: pd.DataFrame, tri: str = "similar") -> pd.DataFrame:
    """Comme `trier`, mais la similarité sert toujours de garde-fou."""
    if tri == "similar":
        cles = ["SIM", "TOTAL_VOTES", "YEAR_NUM"]
    else:
        cles = [CLES_DE_TRI.get(tri, CLES_DE_TRI["recent"])[0], "SIM", "TOTAL_VOTES"]
    return df.sort_values(cles, ascending=False, kind="mergesort")


# --------------------------------------------------------------------------- #
#  Recherche par film
# --------------------------------------------------------------------------- #

def index_du_titre(df: pd.DataFrame, titre: str) -> int | None:
    """
    Index du film dont l'une des variantes de titre correspond exactement.

    Plusieurs films partagent un même titre — « Les Évadés » désigne aussi bien
    le film de 1994 qu'un documentaire confidentiel. On retient le plus voté :
    c'est presque toujours celui que l'utilisateur avait en tête.
    """
    cible = str(titre).strip().lower()
    if not cible:
        return None

    correspond = df["_TITRES"].str.contains(f"|{cible}|", regex=False, na=False)
    if not correspond.any():
        return None
    return int(df.loc[correspond, "TOTAL_VOTES"].idxmax())


def recommander_par_titre(
    titre: str, n: int = 5, tri: str = "similar", votes_min: int = 1_000
) -> pd.DataFrame:
    """
    Films les plus proches du titre demandé.

    `votes_min` écarte du résultat les œuvres que presque personne n'a notées.
    Sans ce garde-fou, des films au générique et au résumé très pauvres
    remontent régulièrement : leur vecteur est si creux que la moindre
    caractéristique partagée suffit à les rapprocher de n'importe quoi.
    """
    df, combine, knn = construire_modele()

    index = index_du_titre(df, titre)
    if index is None:
        raise ValueError(f"Film « {titre} » introuvable.")

    # Vivier large : il faut de la marge pour filtrer puis trier sans
    # perdre les suites d'une saga, qui ne sont pas toujours les plus proches.
    vivier = min(max(400, n * 20), len(df) - 1)
    distances, indices = knn.kneighbors(combine[index], n_neighbors=vivier + 1)

    recos = df.loc[indices[0][1:]].copy()
    recos["SIM"] = 1.0 - distances[0][1:]

    retenus = recos[recos["TOTAL_VOTES"] >= votes_min]
    if len(retenus) < n:          # seuil trop strict : on rend ce qu'on a
        retenus = recos
    return trier_recommandations(retenus, tri).head(n)


def chercher_saga(requete: str, n: int = 20) -> pd.DataFrame:
    """
    Recherche par titre partiel — le filet de sécurité quand le titre exact
    n'existe pas (« Harry Potter », « Star Wars »). N'utilise pas le modèle.
    """
    df = charger_catalogue()
    cible = str(requete).strip()
    if not cible:
        raise ValueError("Entrez un texte.")

    trouves = df[df["_TITRES"].str.contains(cible.lower(), regex=False, na=False)].copy()
    if trouves.empty:
        raise ValueError(f"Aucun film ne contient « {requete} » dans le titre.")

    # Les documentaires polluent ce genre de recherche.
    trouves = trouves[~trouves["GENRES"].str.contains("documentary", case=False, na=False)]
    if trouves.empty:
        raise ValueError(f"Aucun film de fiction ne contient « {requete} » dans le titre.")

    return trier(trouves, "votes").head(n)


# --------------------------------------------------------------------------- #
#  Recherche par personne
# --------------------------------------------------------------------------- #

def films_par_personne(requete: str, colonne: str, n: int = 5, tri: str = "recent") -> pd.DataFrame:
    df = charger_catalogue()
    cible = str(requete).strip()
    if not cible:
        raise ValueError("Entrez un nom.")

    correspond = df[colonne].str.contains(cible, case=False, na=False, regex=False)
    if not correspond.any():
        raise ValueError(f"« {requete} » est introuvable dans la base.")

    return trier(df[correspond].copy(), tri).head(n)


def films_par_acteur(nom, n=5, tri="recent"):
    return films_par_personne(nom, "ACTOR_NAME", n, tri)


def films_par_realisateur(nom, n=5, tri="recent"):
    return films_par_personne(nom, "DIRECTOR_NAME", n, tri)


def films_par_compositeur(nom, n=5, tri="recent"):
    return films_par_personne(nom, "COMPOSER_NAME", n, tri)


# --------------------------------------------------------------------------- #
#  Suggestions de saisie
# --------------------------------------------------------------------------- #

def suggerer(requete: str, colonne: str = "_TITRES", limite: int = 8) -> list[str]:
    """Propositions de titres pendant la frappe, les plus populaires d'abord."""
    df = charger_catalogue()
    cible = str(requete).strip().lower()
    if len(cible) < 2:
        return []
    trouves = df[df[colonne].str.contains(cible, regex=False, na=False)]
    trouves = trouves.nlargest(limite, "TOTAL_VOTES")
    return trouves["TITLE_ORIGINAL"].tolist()
