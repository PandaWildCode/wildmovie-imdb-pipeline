"""
WildMovie — moteur de recommandation de films pour un cinéma indépendant.

Projet 2 du titre professionnel Data Analyst (RNCP 37429), Wild Code School.
Données : IMDb (notes, votes, génériques) et TMDB (notes complémentaires).

Lancer en local :   streamlit run app.py
"""

import hashlib
import os

import numpy as np
import pandas as pd
import streamlit as st

import moteur

# Le fichier est à la racine du dépôt (contrainte de déploiement Streamlit Cloud),
# mais on accepte aussi data/ pour un usage local rangé.
RACINE = os.path.dirname(os.path.abspath(__file__))
CHEMIN_DONNEES = next(
    chemin
    for chemin in (
        os.path.join(RACINE, "films.csv.gz"),
        os.path.join(RACINE, "data", "films.csv.gz"),
    )
    if os.path.exists(chemin)
)

st.set_page_config(
    page_title="WildMovie",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
#  Chargement
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Chargement du catalogue…")
def charger():
    return moteur.charger_films(CHEMIN_DONNEES)


@st.cache_resource(show_spinner="Construction du modèle de similarité…")
def modele(nb_lignes: int):
    # nb_lignes ne sert qu'à invalider le cache si le catalogue change.
    return moteur.construire_matrice(charger())


films = charger()
matrice, vocabulaire = modele(len(films))


# --------------------------------------------------------------------------- #
#  Style
# --------------------------------------------------------------------------- #

st.markdown(
    """
<style>
  :root {
    --nuit:    #0E1116;
    --carte:   #171C24;
    --carte-2: #1F2632;
    --trait:   #2A3341;
    --texte:   #ECEFF3;
    --gris:    #93A0B1;
    --or:      #F2B705;
    --or-fonce:#8A6A05;
  }

  .stApp { background: var(--nuit); }
  .block-container { padding-top: 2rem; max-width: 1280px; }
  html, body, [class*="css"] { color: var(--texte); }

  /* --- En-tête --- */
  .wm-header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 4px; }
  .wm-logo {
    font-size: 34px; font-weight: 800; letter-spacing: -0.02em;
    color: var(--texte); margin: 0;
  }
  .wm-logo span { color: var(--or); }
  .wm-baseline { color: var(--gris); font-size: 14px; }
  .wm-rule { height: 3px; width: 64px; background: var(--or); margin: 14px 0 22px; }

  /* --- Fiche du film sélectionné --- */
  .wm-fiche {
    display: flex; gap: 26px; background: var(--carte);
    border: 1px solid var(--trait); border-radius: 10px; padding: 22px; margin-bottom: 10px;
  }
  .wm-affiche {
    width: 132px; min-width: 132px; height: 196px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 46px; font-weight: 800; color: rgba(255,255,255,.82);
  }
  .wm-fiche h2 { margin: 0 0 2px; font-size: 27px; line-height: 1.2; }
  .wm-vo { color: var(--gris); font-size: 14px; font-style: italic; margin-bottom: 14px; }
  .wm-ligne { margin: 7px 0; font-size: 14.5px; }
  .wm-ligne b { color: var(--gris); font-weight: 500; margin-right: 7px; }

  .wm-notes { display: flex; gap: 26px; margin: 16px 0 4px; }
  .wm-note-bloc { text-align: left; }
  .wm-note-val { font-size: 25px; font-weight: 800; line-height: 1; }
  .wm-note-src { font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--gris); margin-top: 5px; }
  .wm-or { color: var(--or); }

  /* --- Cartes de recommandation --- */
  .wm-carte {
    background: var(--carte); border: 1px solid var(--trait); border-radius: 10px;
    overflow: hidden; height: 100%; display: flex; flex-direction: column;
  }
  .wm-vignette {
    height: 108px; display: flex; align-items: center; justify-content: center;
    font-size: 34px; font-weight: 800; color: rgba(255,255,255,.8);
  }
  .wm-corps { padding: 13px 15px 16px; display: flex; flex-direction: column; gap: 7px; flex: 1; }
  .wm-titre { font-size: 15px; font-weight: 700; line-height: 1.28; }
  .wm-meta  { font-size: 12.5px; color: var(--gris); line-height: 1.4; }
  .wm-commun { font-size: 12px; color: var(--or); line-height: 1.4; }

  .wm-barre { height: 5px; background: var(--carte-2); border-radius: 3px; overflow: hidden; margin-top: auto; }
  .wm-barre > div { height: 100%; background: linear-gradient(90deg, var(--or-fonce), var(--or)); }
  .wm-score { font-size: 11px; letter-spacing: .07em; text-transform: uppercase; color: var(--gris); }

  /* --- Divers --- */
  .wm-section { font-size: 19px; font-weight: 700; margin: 26px 0 14px; }
  .wm-note-bas { color: var(--gris); font-size: 13px; line-height: 1.6; }
  .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--trait); }
  .stTabs [data-baseweb="tab"] { color: var(--gris); }
  .stTabs [aria-selected="true"] { color: var(--or); }
</style>
""",
    unsafe_allow_html=True,
)


def couleurs(identifiant: str) -> tuple[str, str]:
    """Deux teintes stables déduites de l'identifiant du film."""
    graine = int(hashlib.md5(str(identifiant).encode()).hexdigest()[:8], 16)
    teinte = graine % 360
    return (
        f"hsl({teinte}, 42%, 26%)",
        f"hsl({(teinte + 38) % 360}, 46%, 15%)",
    )


def initiale(titre: str) -> str:
    for caractere in str(titre):
        if caractere.isalnum():
            return caractere.upper()
    return "?"


def liste(valeur: str, separateur: str = ", ") -> str:
    return separateur.join([x for x in str(valeur).split("|") if x]) or "—"


def entier(n) -> str:
    return f"{int(n):,}".replace(",", " ")


# --------------------------------------------------------------------------- #
#  En-tête
# --------------------------------------------------------------------------- #

st.markdown(
    """
<div class="wm-header">
  <p class="wm-logo">Wild<span>Movie</span></p>
  <span class="wm-baseline">Le moteur de recommandation du cinéma de la Creuse</span>
</div>
<div class="wm-rule"></div>
""",
    unsafe_allow_html=True,
)

onglet_reco, onglet_catalogue, onglet_projet = st.tabs(
    ["  Recommandations  ", "  Explorer le catalogue  ", "  Le projet  "]
)


# --------------------------------------------------------------------------- #
#  Onglet 1 — Recommandations
# --------------------------------------------------------------------------- #

with onglet_reco:
    colonne_recherche, colonne_reglages = st.columns([3, 2])

    with colonne_recherche:
        requete = st.text_input(
            "Un film que vous avez aimé",
            value="Inception",
            placeholder="Tapez un titre, en français ou en version originale…",
        )

    resultats = moteur.chercher(films, requete)

    if resultats.empty:
        st.info(
            "Aucun titre ne correspond. Essayez un extrait plus court, "
            "ou le titre en version originale."
        )
        st.stop()

    with colonne_recherche:
        etiquettes = {
            int(idx): f"{ligne.TITRE}  ·  {ligne.REALISATEURS.split('|')[0] or 'réalisateur inconnu'}"
            for idx, ligne in resultats.iterrows()
        }
        index_film = st.selectbox(
            f"{len(resultats)} film(s) trouvé(s)",
            options=list(etiquettes.keys()),
            format_func=lambda k: etiquettes[k],
        )

    with colonne_reglages:
        nb_reco = st.slider("Nombre de recommandations", 3, 18, 9, step=3)
        poids = st.slider(
            "Part de la notoriété dans le score",
            0.0, 0.6, 0.25, step=0.05,
            help=(
                "À 0, seule la proximité de générique compte. "
                "En montant, les films très bien notés remontent."
            ),
        )
        votes_min = st.select_slider(
            "Popularité minimale (votes IMDb)",
            options=[500, 1000, 5000, 25000, 100000],
            value=1000,
        )

    film = films.loc[index_film]
    haut, bas = couleurs(film.ID_FILM)

    note_imdb = "—" if pd.isna(film.NOTE_MOYENNE_IMDB) else f"{film.NOTE_MOYENNE_IMDB:.1f}"
    note_tmdb = "—" if pd.isna(film.NOTE_MOYENNE_TMDB) else f"{film.NOTE_MOYENNE_TMDB:.1f}"
    vo = (
        f'<div class="wm-vo">Titre original : {film.TITRE_VO}</div>'
        if film.TITRE_VO and film.TITRE_VO != film.TITRE
        else ""
    )

    st.markdown(
        f"""
<div class="wm-fiche">
  <div class="wm-affiche" style="background: linear-gradient(160deg, {haut}, {bas});">
    {initiale(film.TITRE)}
  </div>
  <div style="flex:1; min-width:0;">
    <h2>{film.TITRE}</h2>
    {vo}
    <div class="wm-notes">
      <div class="wm-note-bloc">
        <div class="wm-note-val wm-or">{note_imdb}</div>
        <div class="wm-note-src">IMDb · {entier(film.NOMBRE_DE_VOTES_IMDB)} votes</div>
      </div>
      <div class="wm-note-bloc">
        <div class="wm-note-val">{note_tmdb}</div>
        <div class="wm-note-src">TMDB · {entier(film.NOMBRE_DE_VOTES_TMDB)} votes</div>
      </div>
      <div class="wm-note-bloc">
        <div class="wm-note-val">{film.NOTE_BAYES:.2f}</div>
        <div class="wm-note-src">Note lissée</div>
      </div>
    </div>
    <div class="wm-ligne"><b>Réalisation</b>{liste(film.REALISATEURS)}</div>
    <div class="wm-ligne"><b>Avec</b>{liste(film.ACTEURS)}</div>
    <div class="wm-ligne"><b>Musique</b>{liste(film.COMPOSITEURS)}</div>
    <div class="wm-ligne"><b>Source de note retenue</b>{film.SOURCE_TO_KEEP}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    reco = moteur.recommander(
        films, matrice, index_film, nb=nb_reco, poids_notoriete=poids, votes_min=votes_min
    )

    st.markdown('<div class="wm-section">Vous aimerez sans doute</div>', unsafe_allow_html=True)

    if reco.empty:
        st.warning(
            "Aucun film ne partage d'intervenant avec celui-ci au niveau de popularité "
            "demandé. Baissez le seuil de votes."
        )
    else:
        positions = {identifiant: i for i, identifiant in enumerate(films.ID_FILM)}
        lignes = [reco.iloc[i : i + 3] for i in range(0, len(reco), 3)]

        for groupe in lignes:
            colonnes = st.columns(3)
            for colonne, (_, r) in zip(colonnes, groupe.iterrows()):
                j = positions[r.ID_FILM]
                communs = moteur.intervenants_communs(films, index_film, j)[:3]
                h, b = couleurs(r.ID_FILM)
                note = "—" if pd.isna(r.MEILLEURE_NOTE) else f"{r.MEILLEURE_NOTE:.1f}"
                realisation = r.REALISATEURS.split("|")[0] or "Réalisateur inconnu"

                with colonne:
                    st.markdown(
                        f"""
<div class="wm-carte">
  <div class="wm-vignette" style="background: linear-gradient(160deg, {h}, {b});">
    {initiale(r.TITRE)}
  </div>
  <div class="wm-corps">
    <div class="wm-titre">{r.TITRE}</div>
    <div class="wm-meta">{realisation}<br>
      <span class="wm-or">★ {note}</span> · {entier(r.NOMBRE_DE_VOTES_IMDB)} votes</div>
    <div class="wm-commun">En commun : {', '.join(communs) if communs else '—'}</div>
    <div class="wm-score">Correspondance {r.SCORE:.0f} %</div>
    <div class="wm-barre"><div style="width:{max(0, min(100, r.SCORE)):.0f}%"></div></div>
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )
            st.write("")

    st.markdown(
        """
<div class="wm-note-bas">
  <b>Comment le score est calculé.</b> Aucun client n'a renseigné ses préférences :
  c'est une situation de <i>cold start</i>. La recommandation ne peut donc pas s'appuyer
  sur des historiques de visionnage, seulement sur ce que les films ont en commun.
  Chaque film est représenté par les personnes qui l'ont fait — réalisateurs (poids 3),
  compositeurs (1,6), acteurs principaux (1). Deux films sont proches quand ces vecteurs
  pointent dans la même direction (similarité cosinus). La note lissée ajoute un bonus
  de qualité, dosable avec le curseur.
</div>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Onglet 2 — Catalogue
# --------------------------------------------------------------------------- #

with onglet_catalogue:
    filtre_1, filtre_2, filtre_3 = st.columns([2, 1, 1])

    with filtre_1:
        personne = st.text_input(
            "Filtrer par personne (acteur, réalisateur, compositeur)",
            placeholder="Ex. : Jean Dujardin, Agnès Varda…",
        )
    with filtre_2:
        note_min = st.slider("Note minimale", 0.0, 10.0, 7.0, step=0.5)
    with filtre_3:
        popularite_min = st.select_slider(
            "Votes minimum", options=[500, 1000, 5000, 25000, 100000, 500000], value=5000
        )

    vue = films[
        (films.MEILLEURE_NOTE.fillna(0) >= note_min)
        & (films.NOMBRE_DE_VOTES_IMDB >= popularite_min)
    ]

    if personne.strip():
        cible = personne.strip().lower()
        equipe = (
            vue.ACTEURS.str.lower()
            + "|"
            + vue.REALISATEURS.str.lower()
            + "|"
            + vue.COMPOSITEURS.str.lower()
        )
        vue = vue[equipe.str.contains(cible, regex=False, na=False)]

    st.caption(f"{entier(len(vue))} films correspondent — les 200 mieux notés sont affichés.")

    affichage = (
        vue.sort_values("NOTE_BAYES", ascending=False)
        .head(200)[
            [
                "TITRE",
                "REALISATEURS",
                "ACTEURS",
                "MEILLEURE_NOTE",
                "NOMBRE_DE_VOTES_IMDB",
                "SOURCE_TO_KEEP",
            ]
        ]
        .rename(
            columns={
                "TITRE": "Titre",
                "REALISATEURS": "Réalisation",
                "ACTEURS": "Distribution",
                "MEILLEURE_NOTE": "Note",
                "NOMBRE_DE_VOTES_IMDB": "Votes IMDb",
                "SOURCE_TO_KEEP": "Source",
            }
        )
    )
    for colonne in ("Réalisation", "Distribution"):
        affichage[colonne] = affichage[colonne].str.replace("|", ", ", regex=False)

    st.dataframe(affichage, use_container_width=True, hide_index=True, height=520)


# --------------------------------------------------------------------------- #
#  Onglet 3 — Le projet
# --------------------------------------------------------------------------- #

with onglet_projet:
    st.markdown(
        """
Un cinéma indépendant de la Creuse veut proposer un service en ligne à ses
spectateurs. Il n'a **aucun historique de préférences** — la recommandation doit
donc reposer entièrement sur les caractéristiques des films. La matière première :
les bases publiques IMDb et TMDB.
"""
    )

    st.markdown('<div class="wm-section">Du référentiel brut au catalogue</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Films candidats en entrée", "682 989")
    k2.metric("Films notés retenus", "202 936")
    k3.metric("Associations film–personne", "202 278")
    k4.metric("Catalogue de l'application", entier(len(films)))

    st.caption(
        "Le catalogue de cette application est volontairement restreint aux films "
        "réunissant au moins 500 votes IMDb : en dessous, la note n'est pas fiable "
        "et la recommandation devient du bruit."
    )

    st.markdown('<div class="wm-section">Ce que contient le catalogue</div>', unsafe_allow_html=True)

    gauche, droite = st.columns(2)

    with gauche:
        st.markdown("**Distribution des notes**")
        tranches = pd.cut(
            films.MEILLEURE_NOTE.dropna(),
            bins=[0, 4, 5, 6, 7, 8, 9, 10],
            labels=["<4", "4–5", "5–6", "6–7", "7–8", "8–9", "9–10"],
        )
        st.bar_chart(tranches.value_counts().sort_index(), color="#F2B705")

    with droite:
        st.markdown("**Source de note retenue**")
        st.bar_chart(films.SOURCE_TO_KEEP.value_counts(), color="#F2B705")
        st.caption(
            "Pour chaque film, la source la mieux dotée en votes a été retenue, et "
            "l'arbitrage est tracé dans la colonne `SOURCE_TO_KEEP` — plutôt qu'un "
            "choix implicite au moment de la jointure."
        )

    st.markdown('<div class="wm-section">Les personnes les plus présentes</div>', unsafe_allow_html=True)

    onglet_r, onglet_a, onglet_c = st.tabs(["Réalisateurs", "Acteurs", "Compositeurs"])
    for onglet, colonne in (
        (onglet_r, "REALISATEURS"),
        (onglet_a, "ACTEURS"),
        (onglet_c, "COMPOSITEURS"),
    ):
        with onglet:
            eclate = films[colonne].str.split("|").explode().dropna()
            comptes = eclate[eclate != ""].value_counts().head(15)
            st.bar_chart(comptes, color="#F2B705")

    st.markdown('<div class="wm-section">Limites assumées</div>', unsafe_allow_html=True)
    st.markdown(
        """
- **Pas de genre ni d'année de sortie.** La table `title.basics` d'IMDb n'a pas été
  conservée dans les exports du projet : la recommandation repose donc sur les
  génériques seuls. Ajouter les genres améliorerait nettement la diversité des
  suggestions.
- **Pas d'affiches.** Le chemin vers les visuels TMDB n'a pas été récupéré ;
  les vignettes sont des aplats générés à partir de l'identifiant du film.
- **Huit acteurs par film au maximum**, pour garder le fichier de données sous
  les limites de versionnement de GitHub.
"""
    )

    st.markdown(
        """
<div class="wm-note-bas" style="margin-top:22px">
  Projet 2 du titre professionnel Data Analyst (RNCP 37429) — Wild Code School.<br>
  Données : IMDb et TMDB. Travail d'équipe ; la chaîne de préparation des données
  présentée ici est ma contribution.
</div>
""",
        unsafe_allow_html=True,
    )
