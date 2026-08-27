"""
CinéMad — l'assistant de programmation du cinéma de la Creuse.

Projet 2 du titre professionnel Data Analyst (RNCP 37429), Wild Code School.
Données : IMDb (types, titres, génériques, notes) et TMDB (résumés, affiches).

Lancer en local :   streamlit run app.py
"""

import pandas as pd
import streamlit as st

import moteur

st.set_page_config(page_title="CinéMad", page_icon="🎬", layout="wide")


# --------------------------------------------------------------------------- #
#  Style
# --------------------------------------------------------------------------- #

st.markdown(
    """
<style>
  :root {
    --nuit:   #0D1014;
    --carte:  #161B22;
    --trait:  #262D38;
    --texte:  #E9EDF2;
    --gris:   #8D9AAB;
    --or:     #F5C518;   /* le jaune des notes IMDb */
    --rouge:  #C8102E;
  }

  .stApp { background: var(--nuit); }
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
  html, body, [class*="css"] { color: var(--texte); }

  /* --- En-tête ---
     Streamlit applique ses propres règles aux paragraphes du markdown ;
     il faut donc forcer la taille du titre pour qu'elle tienne. */
  .cm-titre {
    font-size: clamp(30px, 4.5vw, 44px) !important;
    font-weight: 800 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.025em;
    margin: 0 0 10px !important;
    color: var(--texte);
  }
  .cm-titre span { color: var(--or); }
  .cm-baseline {
    color: var(--gris); font-size: 15.5px !important; margin: 0 !important;
    max-width: 68ch; line-height: 1.6 !important;
  }
  .cm-filet { height: 3px; width: 58px; background: var(--rouge); margin: 20px 0 28px; }

  /* --- Fiche film --- */
  .cm-fiche {
    background: var(--carte); border: 1px solid var(--trait); border-radius: 10px;
    padding: 18px 20px; height: 100%;
  }
  .cm-fiche h3 { margin: 0 0 3px; font-size: 21px; line-height: 1.25; font-weight: 700; }
  .cm-vo { color: var(--gris); font-size: 13px; font-style: italic; margin-bottom: 12px; }

  .cm-chiffres { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .cm-puce {
    font-size: 12px; padding: 3px 9px; border-radius: 999px;
    border: 1px solid var(--trait); color: var(--gris); white-space: nowrap;
  }
  .cm-puce.note { border-color: var(--or); color: var(--or); font-weight: 700; }
  .cm-puce.sim  { border-color: var(--rouge); color: #FF8095; font-weight: 700; }

  .cm-genres { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 13px; }
  .cm-genre {
    font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
    padding: 3px 8px; border-radius: 4px; background: #1E2530; color: var(--gris);
  }

  .cm-equipe { font-size: 13.5px; line-height: 1.75; }
  .cm-equipe b { color: var(--gris); font-weight: 500; }
  .cm-resume { font-size: 13.5px; color: var(--gris); line-height: 1.6; margin-top: 12px; }

  .cm-sans-affiche {
    background: #1A212B; border: 1px dashed var(--trait); border-radius: 8px;
    height: 100%; min-height: 210px; display: flex; align-items: center;
    justify-content: center; color: var(--gris); font-size: 12px; text-align: center;
  }

  .cm-pied {
    color: var(--gris); font-size: 12.5px; line-height: 1.65;
    border-top: 1px solid var(--trait); padding-top: 18px; margin-top: 34px;
  }

  .stRadio [role="radiogroup"] { gap: 18px; }
  div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
#  En-tête
# --------------------------------------------------------------------------- #

st.markdown(
    """
<p class="cm-titre">🎬 Ciné<span>Mad</span></p>
<p class="cm-baseline">
  Bienvenue dans les salles de cinéma de la Creuse. CinéMad cherche les films
  qui correspondent à vos envies du moment — par titre, par acteur, par
  réalisateur ou par compositeur.
</p>
<div class="cm-filet"></div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
#  Formulaire
# --------------------------------------------------------------------------- #

COLONNE_PAR_MODE = {
    "Acteur": "ACTOR_NAME",
    "Réalisateur": "DIRECTOR_NAME",
    "Compositeur": "COMPOSER_NAME",
}

TRIS = {
    "Pertinence": "similar",
    "Plus récents": "recent",
    "Mieux notés": "rating",
    "Plus populaires": "votes",
}

POPULARITE = {
    "Aucune": 0,
    "Au moins 1 000 votes": 1_000,
    "Au moins 10 000 votes": 10_000,
    "Au moins 100 000 votes": 100_000,
}

with st.form("recherche"):
    mode = st.radio(
        "Mode de recherche",
        ["Film", "Acteur", "Réalisateur", "Compositeur"],
        horizontal=True,
    )

    col_requete, col_tri, col_pop, col_nb = st.columns([3, 1.3, 1.4, 0.9])

    with col_requete:
        requete = st.text_input(
            "Votre recherche",
            placeholder="Inception · Leonardo DiCaprio · Christopher Nolan · Hans Zimmer",
        )
    with col_tri:
        libelle_tri = st.selectbox("Trier les résultats", list(TRIS.keys()), index=0)
    with col_pop:
        libelle_pop = st.selectbox(
            "Notoriété minimale",
            list(POPULARITE.keys()),
            index=1,
            help=(
                "Écarte les films que presque personne n'a notés. Leur fiche est "
                "si pauvre qu'ils se retrouvent proches de n'importe quel film."
            ),
        )
    with col_nb:
        nb = st.selectbox("Résultats", [5, 10, 15, 20], index=0)

    lance = st.form_submit_button("Rechercher", type="primary")


# --------------------------------------------------------------------------- #
#  Affichage d'un film
# --------------------------------------------------------------------------- #

def raccourcir(texte, longueur=300):
    texte = "" if pd.isna(texte) else str(texte).replace("\n", " ").strip()
    if len(texte) <= longueur:
        return texte
    return texte[:longueur].rsplit(" ", 1)[0] + "…"


def accorde(noms: str, singulier: str, pluriel: str) -> str:
    liste = [n for n in str(noms).split("|") if n]
    if not liste:
        return ""
    etiquette = pluriel if len(liste) > 1 else singulier
    return f"<b>{etiquette} :</b> {raccourcir(', '.join(liste), 110)}<br>"


def afficher_film(film):
    col_affiche, col_infos = st.columns([1, 3.4], gap="medium")

    with col_affiche:
        url = moteur.url_affiche(film.get("POSTER_PATH", ""))
        if url:
            st.image(url, use_container_width=True)
        else:
            st.markdown(
                '<div class="cm-sans-affiche">🎞️<br>Pas d\'affiche</div>',
                unsafe_allow_html=True,
            )

    with col_infos:
        titre = film.get("TITRE_AFFICHE") or film.get("TITLE_ORIGINAL") or "Titre inconnu"
        original = film.get("TITLE_ORIGINAL", "")
        vo = (
            f'<div class="cm-vo">Titre original : {original}</div>'
            if original and original != titre
            else ""
        )

        annee = int(film["YEAR_NUM"]) if film.get("YEAR_NUM") else None
        entete = f"{titre} ({annee})" if annee else titre

        puces = []
        note = film.get("BEST_RATING") or 0
        if note:
            puces.append(f'<span class="cm-puce note">★ {float(note):.1f}</span>')
        votes = int(film.get("TOTAL_VOTES") or 0)
        if votes:
            puces.append(f'<span class="cm-puce">{votes:,} votes</span>'.replace(",", " "))
        duree = film.get("DURATION_MINUTES")
        if pd.notna(duree) and duree:
            puces.append(f'<span class="cm-puce">{int(duree)} min</span>')
        if "SIM" in film and pd.notna(film["SIM"]):
            puces.append(
                f'<span class="cm-puce sim">Correspondance {float(film["SIM"]) * 100:.0f} %</span>'
            )
        source = film.get("SOURCE_TO_KEEP")
        if isinstance(source, str) and source:
            puces.append(f'<span class="cm-puce">note {source}</span>')

        genres = "".join(
            f'<span class="cm-genre">{g}</span>'
            for g in str(film.get("GENRES", "")).split("|")
            if g
        )

        equipe = (
            accorde(film.get("DIRECTOR_NAME", ""), "Réalisateur", "Réalisateurs")
            + accorde(film.get("ACTOR_NAME", ""), "Acteur", "Acteurs")
            + accorde(film.get("COMPOSER_NAME", ""), "Compositeur", "Compositeurs")
        )

        resume = raccourcir(film.get("SUMMARY", ""), 300)
        bloc_resume = (
            f'<div class="cm-resume">{resume}</div>'
            if resume
            else '<div class="cm-resume">Pas de résumé disponible.</div>'
        )

        st.markdown(
            f"""
<div class="cm-fiche">
  <h3>{entete}</h3>
  {vo}
  <div class="cm-chiffres">{"".join(puces)}</div>
  <div class="cm-genres">{genres}</div>
  <div class="cm-equipe">{equipe}</div>
  {bloc_resume}
</div>
""",
            unsafe_allow_html=True,
        )

    st.write("")


# --------------------------------------------------------------------------- #
#  Recherche
# --------------------------------------------------------------------------- #

if lance:
    texte = requete.strip()
    if not texte:
        st.warning("Entrez un titre ou un nom pour lancer la recherche.")
        st.stop()

    tri = TRIS[libelle_tri]
    votes_min = POPULARITE[libelle_pop]

    try:
        with st.spinner("Recherche en cours…"):
            if mode == "Film":
                try:
                    resultats = moteur.recommander_par_titre(
                        texte, n=nb, tri=tri, votes_min=votes_min
                    )
                    entete = f"{len(resultats)} films proches de « {texte} »"
                    aide = (
                        "Classés par proximité de contenu : résumé, genres, équipe, "
                        "année et durée."
                    )
                except ValueError:
                    resultats = moteur.chercher_saga(texte, n=nb)
                    entete = f"{len(resultats)} films dont le titre contient « {texte} »"
                    aide = (
                        "Aucun film ne porte exactement ce titre — voici une recherche "
                        "par titre partiel. Choisissez-en un et relancez pour obtenir "
                        "de vraies recommandations."
                    )
            else:
                fonction = {
                    "Acteur": moteur.films_par_acteur,
                    "Réalisateur": moteur.films_par_realisateur,
                    "Compositeur": moteur.films_par_compositeur,
                }[mode]
                resultats = fonction(texte, n=nb, tri=tri if tri != "similar" else "votes")
                entete = f"{len(resultats)} films avec « {texte} »"
                aide = f"Filmographie filtrée sur le rôle : {mode.lower()}."

        st.success(entete)
        st.caption(aide)
        st.write("")

        for _, film in resultats.iterrows():
            afficher_film(film)

    except ValueError as erreur:
        st.error(str(erreur))
        st.caption(
            "Vérifiez l'orthographe, ou essayez le titre en version originale — "
            "le catalogue vient d'IMDb."
        )

else:
    st.caption(
        "Choisissez un mode, tapez une recherche, puis « Rechercher ». "
        "Essayez « Inception », « Jean Dujardin » ou « Hans Zimmer »."
    )


# --------------------------------------------------------------------------- #
#  Pied de page
# --------------------------------------------------------------------------- #

st.markdown(
    """
<div class="cm-pied">
  <b>Comment fonctionnent les recommandations.</b> Aucun spectateur n'a renseigné
  ses préférences : c'est une situation de <i>cold start</i>. Chaque film est donc
  représenté par son contenu — résumé (pondéré ×2), genres, équipe, titre en
  bigrammes pour rattraper les sagas, et caractéristiques numériques (notes, votes,
  année, durée, ×2). Les plus proches voisins au sens de la similarité cosinus
  deviennent les recommandations. Le compositeur ne pèse que 0,15 : un même
  compositeur signe des films trop différents pour être un bon signal de proximité.
  <br><br>
  Catalogue : 202 643 films IMDb enrichis des résumés et affiches TMDB, après
  exclusion des séries, des contenus pour adultes, des émissions de plateau et des
  œuvres de moins de 20 ou plus de 300 minutes.
  <br><br>
  Projet 2 du titre professionnel Data Analyst (RNCP 37429) — Wild Code School.
</div>
""",
    unsafe_allow_html=True,
)
