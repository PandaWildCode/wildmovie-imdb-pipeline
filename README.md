# 🎬 WildMovie — Pipeline IMDb & moteur de recommandation

> Transformer les bases publiques IMDb et TMDB en un socle de données exploitable, puis alimenter un moteur de recommandation de films pour un cinéma indépendant de la Creuse.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 🎬 Application de démonstration

**➡️ [Ouvrir WildMovie](LIEN_STREAMLIT)**

Trois onglets :

| Onglet | Ce qu'il montre |
|---|---|
| **Recommandations** | On saisit un film aimé, l'application renvoie les plus proches avec un score de correspondance et la raison de chaque suggestion (« en commun : Christopher Nolan, Hans Zimmer ») |
| **Explorer le catalogue** | Filtrage des 66 742 films par note, popularité et par personne — acteur, réalisateur ou compositeur |
| **Le projet** | Les chiffres du pipeline, la distribution des notes, l'arbitrage IMDb / TMDB et les limites assumées |

Pour la lancer en local :

```bash
pip install -r requirements.txt
streamlit run app.py
```

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

## Limites assumées

- **Pas de genre ni d'année de sortie.** La table `title.basics` d'IMDb n'a pas été conservée dans les exports du projet : la recommandation repose donc sur les génériques seuls. Les ajouter améliorerait nettement la diversité des suggestions — c'est la première évolution à faire.
- **Pas d'affiches.** Le chemin vers les visuels TMDB n'a pas été récupéré ; les vignettes de l'application sont des aplats générés à partir de l'identifiant du film.
- **Un film sans générique connu (28 sur 66 742) ne reçoit aucune recommandation.** L'application le dit explicitement plutôt que d'inventer une liste.

## Le moteur de recommandation

Le score qui classe les suggestions se lit en une phrase :

```
score = (1 − p) × similarité de générique  +  p × note lissée normalisée
```

**La similarité de générique.** Chaque film devient un vecteur creux sur l'ensemble des personnes du catalogue — 220 934 dimensions, pondérées par rôle : réalisateur ×3, compositeur ×1,6, acteur principal ×1. Le réalisateur pèse le plus parce que c'est lui qui porte l'univers d'un film. Les vecteurs sont normalisés en L2, ce qui rend comparables un film à trois intervenants connus et un film au générique fourni ; la proximité est alors une similarité cosinus, calculée par un simple produit matriciel creux — **moins de 10 ms sur 66 742 films**.

**La note lissée.** Une moyenne bayésienne, pour qu'un 9/10 sur 600 votes ne passe pas devant un 9/10 sur 600 000 :

```
note_lissée = (votes × note + 5000 × moyenne_catalogue) / (votes + 5000)
```

**Le curseur `p`** laisse l'utilisateur arbitrer entre proximité pure et notoriété. À 0, seul le générique compte.

Un film sans aucun intervenant en commun est écarté plutôt que classé bas : mieux vaut ne rien proposer qu'une suggestion sans rapport.

## Stack

`Python` · `Pandas` · `NumPy` · `SciPy (matrices creuses)` · `Streamlit` · `Jupyter / Anaconda` · `Git`

## Structure du dépôt

```
├── app.py                 # l'application Streamlit (interface, 3 onglets)
├── moteur.py              # la logique de recommandation, testable sans Streamlit
├── films.csv.gz           # catalogue allégé : 66 742 films, 10 Mo
├── requirements.txt
├── .streamlit/
│   └── config.toml        # thème de l'application
├── notebooks/
│   ├── 01_name_basics_exploration.ipynb   # exploration et nettoyage pas à pas
│   ├── 02_name_basics_pipeline.ipynb      # version consolidée, rejouable
│   └── 03_dim_person_list.ipynb           # explode + construction de la dimension
├── src/
│   └── dim_person_list.py                 # script de production de DIM_PERSON_LIST
└── README.md
```

### Le catalogue embarqué

`films.csv.gz` est un extrait volontairement réduit des tables du pipeline, pour que l'application tienne dans un dépôt Git et démarre en une seconde :

- seuls les films réunissant **au moins 500 votes IMDb** — en dessous, la note n'est pas fiable et la recommandation devient du bruit ;
- **8 acteurs, 3 réalisateurs et 2 compositeurs** au maximum par film ;
- titre français quand il existe, titre original sinon.

> ⚠️ Les fichiers sources IMDb ne sont pas versionnés (plusieurs centaines de Mo). Ils se téléchargent sur [datasets.imdbws.com](https://datasets.imdbws.com/).

## Auteur

**Eddy Faucher** — Data Analyst (RNCP 37429)
[LinkedIn](https://www.linkedin.com/in/eddy-f) · [Portfolio](https://pandawildcode.github.io)
