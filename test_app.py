"""
Harnais de test : rejoue app.py avec un faux module streamlit.

Objectif : détecter les erreurs Python (API mal appelée, colonne absente,
f-string cassée) sans avoir Streamlit installé. Ne teste pas le rendu visuel.
"""

import sys
import types
from contextlib import contextmanager

APPELS = []


class Faux:
    """Objet passe-partout : tout attribut renvoie une fonction qui journalise."""

    def __init__(self, nom="st"):
        self._nom = nom

    def __getattr__(self, item):
        def appel(*args, **kwargs):
            APPELS.append(f"{self._nom}.{item}")
            return Faux(f"{self._nom}.{item}")
        return appel

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter([Faux(self._nom)] * 4)


class ArretDemande(Exception):
    pass


st = types.ModuleType("streamlit")

# Valeurs que les widgets renverront pendant le test.
VALEURS = {
    "text_input": "Inception",
    "selectbox": None,          # remplacé plus bas par un vrai index
    "slider": None,
    "select_slider": None,
}


def _colonnes(spec, **kwargs):
    n = spec if isinstance(spec, int) else len(spec)
    return [Faux(f"col{i}") for i in range(n)]


def _tabs(noms, **kwargs):
    return [Faux(f"tab:{n.strip()}") for n in noms]


def _markdown(corps, **kwargs):
    APPELS.append("st.markdown")
    assert "{" not in str(corps) or "}" not in str(corps) or "wm-" in str(corps) or "<" in str(corps)


def _text_input(label, value="", **kwargs):
    APPELS.append(f"st.text_input({label!r})")
    return VALEURS.get("_text_input_" + label, value or "")


def _selectbox(label, options, format_func=lambda x: x, **kwargs):
    APPELS.append(f"st.selectbox({label!r})")
    options = list(options)
    for o in options[:5]:
        format_func(o)          # on vérifie que le formateur ne casse pas
    return options[0]


def _slider(label, mini=None, maxi=None, valeur=None, *args, **kwargs):
    APPELS.append(f"st.slider({label!r})")
    return valeur if valeur is not None else (mini if mini is not None else 0)


def _select_slider(label, options=None, value=None, **kwargs):
    APPELS.append(f"st.select_slider({label!r})")
    return value if value is not None else list(options)[0]


def _cache(*dargs, **dkwargs):
    def decorateur(fn):
        cache = {}

        def enveloppe(*a, **k):
            cle = (a, tuple(sorted(k.items())))
            if cle not in cache:
                cache[cle] = fn(*a, **k)
            return cache[cle]

        return enveloppe

    if dargs and callable(dargs[0]):
        return decorateur(dargs[0])
    return decorateur


def _dataframe(df, **kwargs):
    APPELS.append(f"st.dataframe({len(df)} lignes, colonnes={list(df.columns)})")


def _bar_chart(data, **kwargs):
    APPELS.append(f"st.bar_chart({len(data)} valeurs)")


def _metric(label, valeur, **kwargs):
    APPELS.append(f"st.metric({label!r} = {valeur!r})")


def _stop():
    raise ArretDemande()


st.set_page_config = lambda **k: APPELS.append("st.set_page_config")
st.markdown = _markdown
st.columns = _colonnes
st.tabs = _tabs
st.text_input = _text_input
st.selectbox = _selectbox
st.slider = _slider
st.select_slider = _select_slider
st.cache_data = _cache
st.cache_resource = _cache
st.dataframe = _dataframe
st.bar_chart = _bar_chart
st.metric = _metric
st.caption = lambda *a, **k: APPELS.append("st.caption")
st.write = lambda *a, **k: APPELS.append("st.write")
st.info = lambda *a, **k: APPELS.append("st.info")
st.warning = lambda *a, **k: APPELS.append("st.warning")
st.stop = _stop

sys.modules["streamlit"] = st

# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    code = open("app.py", encoding="utf-8").read()
    espace = {"__name__": "__main__", "__file__": "app.py"}
    try:
        exec(compile(code, "app.py", "exec"), espace)
    except ArretDemande:
        print("!! app.py a appelé st.stop()")
        raise SystemExit(1)

    print(f"OK — {len(APPELS)} appels Streamlit simulés, aucune exception.\n")
    interessants = [a for a in APPELS if not a.startswith("st.markdown")]
    for a in interessants:
        print("  ", a)
