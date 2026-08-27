"""
Harnais de test — rejoue app.py avec un faux module Streamlit.

Streamlit ne peut pas être installé dans l'environnement où ce projet a été
préparé ; ce harnais permet malgré tout de détecter les erreurs Python
(colonne absente, API mal appelée, f-string cassée) sur le chemin complet :
formulaire → moteur → affichage des fiches.

Il ne teste pas le rendu visuel.

    python test_app.py
"""

import sys
import types

JOURNAL = []


class Faux:
    """Objet passe-partout : contexte, itérable, et tout attribut appelable."""

    def __init__(self, nom="st"):
        self._nom = nom

    def __getattr__(self, item):
        def appel(*args, **kwargs):
            JOURNAL.append(f"{self._nom}.{item}")
            return Faux(f"{self._nom}.{item}")
        return appel

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter([Faux(self._nom)] * 6)


class ArretDemande(Exception):
    pass


st = types.ModuleType("streamlit")

REPONSES = {
    "radio": "Film",
    "text_input": "Inception",
    "selectbox": None,
    "form_submit_button": True,
}


def _colonnes(spec, **kwargs):
    n = spec if isinstance(spec, int) else len(spec)
    return [Faux(f"colonne{i}") for i in range(n)]


def _radio(label, options, **kwargs):
    JOURNAL.append(f"st.radio({label!r})")
    return REPONSES["radio"] if REPONSES["radio"] in options else list(options)[0]


def _text_input(label, value="", **kwargs):
    JOURNAL.append(f"st.text_input({label!r})")
    return REPONSES["text_input"]


def _selectbox(label, options, index=0, format_func=lambda x: x, **kwargs):
    JOURNAL.append(f"st.selectbox({label!r})")
    options = list(options)
    for o in options:
        format_func(o)
    return options[index or 0]


def _cache(*dargs, **dkwargs):
    def decorateur(fn):
        memo = {}

        def enveloppe(*a, **k):
            cle = (a, tuple(sorted(k.items())))
            if cle not in memo:
                memo[cle] = fn(*a, **k)
            return memo[cle]

        return enveloppe

    if dargs and callable(dargs[0]):
        return decorateur(dargs[0])
    return decorateur


def _image(url, **kwargs):
    JOURNAL.append(f"st.image({str(url)[:60]})")


def _markdown(corps, **kwargs):
    JOURNAL.append("st.markdown")
    texte = str(corps)
    # Une accolade restée dans le HTML trahit une f-string mal fermée.
    if "{film" in texte or "{r." in texte:
        raise AssertionError("f-string non interpolée dans le HTML : " + texte[:120])


def _stop():
    raise ArretDemande()


st.set_page_config = lambda **k: JOURNAL.append("st.set_page_config")
st.markdown = _markdown
st.columns = _colonnes
st.radio = _radio
st.text_input = _text_input
st.selectbox = _selectbox
st.image = _image
st.cache_data = _cache
st.cache_resource = _cache
st.form_submit_button = lambda *a, **k: REPONSES["form_submit_button"]
st.stop = _stop
def _inconnu(item):
    """Tout ce qui n'est pas explicitement simulé devient une fonction inoffensive."""
    def appel(*args, **kwargs):
        JOURNAL.append(f"st.{item}")
        return Faux(f"st.{item}")
    return appel


st.__getattr__ = _inconnu

sys.modules["streamlit"] = st


def rejouer(mode, requete, tri_index=0):
    REPONSES["radio"] = mode
    REPONSES["text_input"] = requete
    JOURNAL.clear()
    espace = {"__name__": "__main__", "__file__": "app.py"}
    code = open("app.py", encoding="utf-8").read()
    exec(compile(code, "app.py", "exec"), espace)
    fiches = sum(1 for a in JOURNAL if a.startswith("st.image") or "sans-affiche" in a)
    return len(JOURNAL), fiches


if __name__ == "__main__":
    cas = [
        ("Film", "Inception"),
        ("Film", "Les Évadés"),
        ("Film", "Harry Potter"),          # titre partiel → repli saga
        ("Film", "zzzzz-inexistant"),      # doit afficher une erreur, pas planter
        ("Acteur", "Jean Dujardin"),
        ("Réalisateur", "Christopher Nolan"),
        ("Compositeur", "Hans Zimmer"),
        ("Film", ""),                      # champ vide → st.stop()
    ]

    for mode, requete in cas:
        try:
            appels, fiches = rejouer(mode, requete)
            print(f"OK   {mode:13s} « {requete[:24]:24s} » → {appels:>4} appels, {fiches} affiche(s)")
        except ArretDemande:
            print(f"OK   {mode:13s} « {requete[:24]:24s} » → arrêt propre (champ vide)")
        except Exception as erreur:
            print(f"ÉCHEC {mode:13s} « {requete[:24]:24s} » → {type(erreur).__name__}: {erreur}")
            raise
