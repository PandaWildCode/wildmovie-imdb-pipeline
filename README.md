# 🎬 WildMovie — Pipeline IMDb & moteur de recommandation

> Transformer les bases publiques IMDb et TMDB en un socle de données exploitable, puis alimenter un moteur de recommandation de films pour un cinéma indépendant de la Creuse.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## Contexte

Un cinéma en perte de vitesse veut créer un service en ligne pour ses spectateurs : quelques statistiques sur les films et les acteurs, et surtout un moteur de recommandation. Aucun historique de préférences client n'existe — c'est une situation de **cold start**, la recommandation doit donc reposer uniquement sur les caractéristiques des films.

Le client fournit les datasets publics IMDb (plus de 7 M de films et 10 M de personnes référencés) et un complément TMDB.

## Mon rôle dans le projet

Projet mené en équipe, avec un mode de travail volontairement parallèle : chacun a produit **sa propre version** de sa partie, les versions ont ensuite été comparées et consolidées en un livrable unique. La version publiée ici est la mienne.

Mon périmètre : la **chaîne de préparation des données** — nettoyage du référentiel des personnes (`name.basics`), construction des tables de dimension et de fait, et normalisation des clés de jointure qui alimentent la partie machine learning.

## Le problème à résoudre

`name.basics.tsv.gz` pèse 294 Mo compressés et mélange tous les métiers du cinéma dans une seule colonne texte. Impossible de joindre quoi que ce soit tant que :

- les professions ne sont pas séparées,
- les années de naissance et de décès sont stockées en texte avec des `\N` en guise de valeurs manquantes,
- la clé `nconst` n'est pas normalisée (casse, espaces parasites) — une jointure sur clé sale perd silencieusement des lignes.

## Démarche

**1. Nettoyage du référentiel des personnes**

```python
df = pd.read_csv("name.basics.tsv.gz", sep="\t", na_values="\\N", low_memory=False)

# séparation par profession
actors    = df[df["primaryProfession"].str.contains("actor",    na=False)].copy()
actress   = df[df["primaryProfession"].str.contains("actress",  na=False)].copy()
directors = df[df["primaryProfession"].str.contains("director", na=False)].copy()
composers = df[df["primaryProfession"].str.contains("composer", na=False)].copy()

# typage des années, nettoyage syntaxique, contrôle des doublons, suppression des noms vides
for d in (actors, actress, directors, composers):
    d["birthYear"] = pd.to_numeric(d["birthYear"], errors="coerce")
    d["deathYear"] = pd.to_numeric(d["deathYear"], errors="coerce")
    d["primaryName"] = d["primaryName"].str.strip()
    d.dropna(subset=["primaryName"], inplace=True)
```

Étapes de contrôle à chaque passe : `unicité de nconst`, comptage des valeurs manquantes, vérification des types avant export.

**2. Normalisation des clés de jointure**

```python
df["nconst"] = df["nconst"].astype(str).str.strip().str.lower()
```

**3. Construction du schéma en étoile**

| Table | Rôle | Volume |
|---|---|---|
| `Film_short_list_ID_for_Filters` | Périmètre de films candidats | **682 989 films** |
| `DIM_FILM_RATINGS` | Notes IMDb + TMDB, nombre de votes, source retenue | **202 936 films** |
| `FACT_MOVIE_PERSON` | Association film ↔ acteurs / réalisateur / compositeur | **202 278 films** |
| `DIM_PERSON_LIST` | Une ligne par personne, avec indicateurs de rôle et filmographie | dérivée |

**4. Passage d'un format « liste dans une cellule » à une table relationnelle**

Les colonnes `ACTOR`, `DIRECTOR`, `COMPOSER` contenaient des identifiants concaténés par virgules. Le traitement les transforme en lignes atomiques via `explode`, puis reconstruit une dimension personne agrégée :

```python
dim_person_list = fact_prete.groupby(["nconst", "primaryName"]).agg(
    AS_ACTOR=("AS_ACTOR", "max"),
    AS_DIRECTOR=("AS_DIRECTOR", "max"),
    AS_COMPOSER=("AS_COMPOSER", "max"),
    PART_OF_THE_FILM_LIST=("ID_FILM", lambda x: list(x)),
).reset_index()
```

Un même individu peut donc être acteur *et* réalisateur sans être dupliqué — condition nécessaire au calcul du score de correspondance entre films.

**5. Documentation du pipeline**
L'ensemble de la chaîne (sources → zones de préparation → tables dimensionnelles → modèle de recommandation) est documenté dans un schéma de flux versionné (`WSC_PROJECT2_IMDB_DATA_PIPELINE_V1.5`).

## Résultats

- Référentiel réduit de **682 989 films candidats à 202 936 films notés**, soit un socle 3× plus léger sans perte d'information utile.
- **202 278 associations film ↔ personne** exploitables, contre une colonne texte non joignable au départ.
- Arbitrage de notation explicite entre IMDb et TMDB (colonne `SOURCE_TO_KEEP`) plutôt qu'un choix implicite : la source retenue est traçable pour chaque film.
- Pipeline reproductible : un notebook « version en ligne » rejoue la chaîne complète du fichier brut à l'export.

## Stack

`Python` · `Pandas` · `NumPy` · `Jupyter / Anaconda` · `scikit-learn` · `Streamlit` · `Git`

## Structure du dépôt

```
├── notebooks/
│   ├── 01_name_basics_exploration.ipynb   # exploration et nettoyage pas à pas
│   ├── 02_name_basics_pipeline.ipynb      # version consolidée, rejouable
│   └── 03_dim_person_list.ipynb           # explode + construction de la dimension
├── src/
│   └── dim_person_list.py                 # script de production de DIM_PERSON_LIST
├── docs/
│   └── pipeline_v1.5.png                  # schéma de flux du pipeline
└── README.md
```

> ⚠️ Les fichiers sources IMDb ne sont pas versionnés (plusieurs centaines de Mo). Ils se téléchargent sur [datasets.imdbws.com](https://datasets.imdbws.com/).

## Auteur

**Eddy Faucher** — Data Analyst (RNCP 37429)
[LinkedIn](https://www.linkedin.com/in/eddy-f) · [Portfolio](https://pandawildcode.github.io)
