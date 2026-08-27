# 🎬 CinéMad — pipeline IMDb et moteur de recommandation

> Transformer les bases publiques IMDb et TMDB en un socle exploitable, puis alimenter un moteur de recommandation pour un cinéma indépendant de la Creuse.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## ➡️ [Ouvrir CinéMad](https://pandawildmovie.streamlit.app/)

Quatre modes de recherche — par **film**, **acteur**, **réalisateur** ou **compositeur** — et quatre tris : pertinence, plus récents, mieux notés, plus populaires.

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Contexte

Un cinéma en perte de vitesse veut créer un service en ligne pour ses spectateurs : des statistiques sur les films et les acteurs, et surtout un moteur de recommandation.

Aucun client n'a renseigné ses préférences : c'est une situation de **cold start**. La recommandation ne peut donc reposer que sur les caractéristiques des films eux-mêmes. Le client fournit les bases publiques IMDb (plus de 7 M de titres, 10 M de personnes) et un complément TMDB.

## Mon rôle dans le projet

Projet mené en équipe, avec un mode de travail volontairement parallèle : chacun a produit **sa propre version** de sa partie, les versions ont ensuite été comparées et consolidées en un livrable unique.

Mon périmètre : la **chaîne de préparation des données** — nettoyage du référentiel des personnes (`name.basics`), construction des tables de dimension et de fait, et normalisation des clés de jointure qui alimentent le modèle.

## Le pipeline

### 1. Sélection des œuvres

Les tables `title.basics` (IMDb) et `tmdb_full` (TMDB) sont fusionnées, puis passées dans une cascade de **28 filtres auditée à chaque étape** — chaque filtre journalise le nombre de lignes qu'il retire, ce qui rend la sélection défendable ligne par ligne :

| Ce qu'on exclut | Pourquoi |
|---|---|
| Contenus pour adultes (IMDb et TMDB) | Un cinéma familial |
| Séries, épisodes, mini-séries, spéciaux TV, jeux vidéo | On ne programme que des films |
| Talk-shows, jeux télévisés, télé-réalité | Pas des œuvres de cinéma |
| Durées hors de la fourchette 20–300 minutes | Ni un clip, ni un marathon |
| Œuvres annulées, planifiées, en production | On ne programme pas ce qui n'existe pas |
| Années de sortie postérieures à 2025 | Projets non sortis |

**Résultat : 202 643 films.**

### 2. Nettoyage du référentiel des personnes

`name.basics.tsv.gz` pèse 294 Mo compressés et mélange tous les métiers du cinéma dans une seule colonne texte.

```python
df = pd.read_csv("name.basics.tsv.gz", sep="\t", na_values="\\N", low_memory=False)

actors    = df[df["primaryProfession"].str.contains("actor",    na=False)].copy()
actress   = df[df["primaryProfession"].str.contains("actress",  na=False)].copy()
directors = df[df["primaryProfession"].str.contains("director", na=False)].copy()
composers = df[df["primaryProfession"].str.contains("composer", na=False)].copy()

for d in (actors, actress, directors, composers):
    d["birthYear"] = pd.to_numeric(d["birthYear"], errors="coerce")
    d["deathYear"] = pd.to_numeric(d["deathYear"], errors="coerce")
    d["primaryName"] = d["primaryName"].str.strip()
    d.dropna(subset=["primaryName"], inplace=True)
```

Puis la normalisation des clés, sans laquelle les jointures perdent des lignes en silence :

```python
df["nconst"] = df["nconst"].astype(str).str.strip().str.lower()
```

### 3. Schéma en étoile

| Table | Rôle | Volume |
|---|---|---|
| `DIM_FILM_LIST_FINAL` | Une ligne par film : titres, année, durée, genres, résumé, affiche, notes | **202 643** |
| `FACT_FILM_PERSON` | Association film ↔ acteurs / réalisateur / compositeur | **202 278** |
| `DIM_PERSON_LIST_FINAL` | Une ligne par personne, avec indicateurs de rôle et filmographie | **720 593** |

Les colonnes `ACTOR`, `DIRECTOR` et `COMPOSER` contenaient des identifiants concaténés par virgules. Le traitement les transforme en lignes atomiques via `explode`, puis reconstruit une dimension personne agrégée — un même individu peut ainsi être acteur *et* réalisateur sans être dupliqué.

### 4. Arbitrage des notes

Chaque film porte une note IMDb et une note TMDB. Plutôt qu'un choix implicite au moment de la jointure, la source la mieux dotée en votes est retenue et **tracée** dans la colonne `SOURCE_TO_KEEP` : 99 % IMDb, 1 % TMDB.

---

## Le moteur de recommandation

Un **KNN à similarité cosinus** sur sept blocs de caractéristiques assemblés en une matrice creuse :

| Bloc | Encodage | Poids |
|---|---|---:|
| Résumé | TF-IDF, *stop words* anglais | **2,0** |
| Caractéristiques chiffrées | jetons (décennie, notoriété, durée, note) | **2,0** |
| Genres | sac de mots | 1,0 |
| Réalisateur | sac de mots | 1,0 |
| Acteurs | sac de mots | 1,0 |
| Titre | TF-IDF en bigrammes | 1,0 |
| Compositeur | sac de mots | 0,15 |

Le résumé pèse double parce que c'est lui qui porte le sujet du film. Le titre en bigrammes est ce qui rattrape les sagas : *Harry Potter à l'école des sorciers* remonte les six autres épisodes avec 59 à 82 % de correspondance. Le compositeur ne pèse que 0,15 : un même compositeur signe des films trop différents pour être un bon signal de proximité.

Matrice finale : **202 643 × 1 215 504**, 9,3 millions de valeurs non nulles. Une recommandation se calcule en **moins de 0,4 seconde**.

### Un défaut corrigé en cours de route

La version d'origine passait les colonnes chiffrées — notes, votes, année, durée — dans un `MinMaxScaler`. Sur l'année, cela produit une valeur quasi identique pour tous les films : 2001 / 2025 ≈ 0,99.

Sous une similarité cosinus, cette quasi-constante domine le produit scalaire. Comme `cos(A, B) = A·B / (‖A‖ ‖B‖)`, le numérateur devient à peu près constant d'une paire à l'autre, et **le classement se réduit à « quel film a le vecteur de plus petite norme »** — c'est-à-dire les œuvres au résumé et au générique les plus pauvres. Concrètement, *Le Fabuleux Destin d'Amélie Poulain* recommandait des spectacles d'humour allemands.

Les mêmes informations, découpées en tranches nommées (`decennie_2000`, `notoriete_mondiale`, `duree_moins_120`, `note_plus_8`), redeviennent discriminantes : deux films partagent leur décennie et leur format, ou ne les partagent pas.

| Film demandé | Avant | Après |
|---|---|---|
| *Inception* | Documentaires confidentiels | Mad Max: Fury Road, Avengers, Les Gardiens de la Galaxie |
| *Amélie* | Spectacles d'humour | Slumdog Millionaire, Un homme d'exception, Cité de Dieu |
| *Le Parrain* | Documentaires | Le Parrain II, Apocalypse Now, Scarface, Heat |

Un second garde-fou complète la correction : les films réunissant moins de 1 000 votes sont écartés du résultat par défaut, seuil réglable dans l'interface. Leur fiche est si creuse que la moindre caractéristique partagée suffit à les rapprocher de n'importe quoi.

---

## Limites assumées

- **Les titres français viennent de `title.akas` sans distinction de région.** France et Québec s'y mélangent : *The Shawshank Redemption* s'affiche « À l'ombre de Shawshank » plutôt que « Les Évadés ». La recherche accepte les deux ; seul l'affichage est concerné.
- **8 % des films n'ont pas d'affiche** dans le jeu TMDB — l'interface l'indique explicitement plutôt que d'afficher un cadre vide.
- **Les résumés sont en anglais**, tels que TMDB les fournit. Les traduire coûterait un appel d'API par film, soit 202 643 appels — hors de portée ici, et sans effet sur la qualité des recommandations puisque la comparaison se fait entre résumés d'une même langue.
- **Aucune personnalisation.** Le moteur ne connaît que les films, jamais les spectateurs. C'est la contrainte du *cold start*, pas un choix.

---

## Structure du dépôt

```
├── app.py                 # interface Streamlit : 4 modes de recherche, fiches films
├── moteur.py              # chargement, modèle KNN, tris, recherche
├── test_app.py            # rejoue app.py avec un faux Streamlit (8 scénarios)
├── films_01..04.csv.gz    # catalogue : 202 643 films, 4 × 14 Mo
├── requirements.txt
├── .streamlit/
│   └── config.toml        # thème sombre
└── README.md
```

### Pourquoi le catalogue est découpé

`DIM_FILM_LIST_FINAL.csv` pèse 119 Mo — au-delà de la limite de 100 Mo par fichier de GitHub. Le catalogue embarqué est donc réparti en quatre morceaux compressés d'environ 14 Mo, réassemblés au chargement. Les identifiants de personnes y sont **déjà résolus en noms**, ce qui évite de charger un référentiel de 450 Mo au démarrage de l'application.

> ⚠️ Les fichiers sources IMDb et TMDB ne sont pas versionnés (plusieurs centaines de Mo). Ils se téléchargent sur [datasets.imdbws.com](https://datasets.imdbws.com/).

## Tests

```bash
python test_app.py
```

Streamlit n'étant pas installable dans l'environnement de préparation, `test_app.py` remplace le module par un double et rejoue `app.py` de bout en bout sur huit scénarios : recherche par film, par personne, titre partiel, titre inexistant, champ vide. Il ne teste pas le rendu visuel — seulement que rien ne casse.

## Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `SciPy (matrices creuses)` · `Streamlit` · `Jupyter / Anaconda` · `Git`

## Auteur

**Eddy Faucher** — Data Analyst (RNCP 37429)
[LinkedIn](https://www.linkedin.com/in/eddy-f) · [Portfolio](https://pandawildcode.github.io)
