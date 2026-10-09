from pathlib import Path
import re
 
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # deterministic results
 
# Local place names that appear in most queries for this app and carry no
# language signal -- strip them before detection. Add more as needed.
LOCAL_PLACE_NAMES = [
    r"c[oô]te[\s-]des[\s-]neiges",
    r"notre[\s-]dame[\s-]de[\s-]gr[aâ]ce",
    r"ndg\b",
    r"cdn\b",
]
_PLACE_NAME_RE = re.compile("|".join(LOCAL_PLACE_NAMES), flags=re.IGNORECASE)
 
# Only distinctive, low-ambiguity French words/phrases -- deliberately
# excludes short function words ("de", "des", "du", "la", "le", "un",
# "une", "il", "y", "a") that are too likely to appear inside place names,
# acronyms, or English text by coincidence.
FRENCH_STOPWORDS = {
    "où", "et", "pour", "avec",
    "gratuit", "gratuite", "gratuits", "gratuites",
    "trouver", "comment", "est-ce", "qu'est-ce", "quel", "quelle",
    "près", "proche", "besoin", "j'ai", "avoir",
}
FRENCH_ACCENTS = set("àâäéèêëïîôöùûüÿçœæ")
 
 
def _strip_place_names(text: str) -> str:
    return _PLACE_NAME_RE.sub(" ", text)
 
 
def detect_query_language(query: str) -> str:
    """Returns 'fr' or 'en'. Biased toward short, informal queries."""
    q = (query or "").strip()
    if not q:
        return "en"
 
    # Remove local place names first -- they carry no language signal and
    # were causing false "fr" positives (e.g. "Cote-des-Neiges" contains "des").
    q_clean = _strip_place_names(q)
 
    signals = 0
 
    # Accented characters are a strong, low-ambiguity signal
    if any(ch in FRENCH_ACCENTS for ch in q_clean.lower()):
        signals += 2  # counts as 2 -- accents alone are enough to decide "fr"
 
    # Distinctive stopword overlap (short/ambiguous words already excluded)
    tokens = set(w.strip(".,?!'\"").lower() for w in q_clean.split())
    signals += len(tokens & FRENCH_STOPWORDS)
 
    if signals >= 1:
        return "fr"
 
    # Weak or no heuristic signal -- fall back to langdetect on the
    # place-name-stripped text (avoids the same false-positive risk)
    try:
        lang = detect(q_clean if q_clean.strip() else q)
        return "fr" if lang.startswith("fr") else "en"
    except Exception:
        return "en"  # safe default
 
 
def infer_bundle_language(path: str) -> str:
    """Infer 'en'/'fr' from a path containing an /en/ or /fr/ segment."""
    parts = Path(path).parts
    if "en" in parts:
        return "en"
    if "fr" in parts:
        return "fr"
    return "unknown"
 
 
def select_paths_for_language(paths_by_domain_lang: dict, lang: str, domains=("macommunaute", "211")):
    """
    Helper for config-driven path selection (structure-based, not per-file config).
 
    paths_by_domain_lang: dict like
        {
            "macommunaute": {"en": "path/to/en", "fr": "path/to/fr"},
            "211": {"en": "path/to/en", "fr": "path/to/fr"},
        }
    Returns the list of paths for the requested language across all domains.
    """
    selected = []
    for domain in domains:
        domain_paths = paths_by_domain_lang.get(domain, {})
        if lang in domain_paths:
            selected.append(domain_paths[lang])
        elif "unknown" in domain_paths:
            selected.append(domain_paths["unknown"])
    return selected
 
 
if __name__ == "__main__":
    # Quick sanity check -- run `python lang_utils.py` to verify the fix
    test_cases = [
        ("get free food at cote des neiges", "en"),
        ("where to get free wifi at cote des neiges", "en"),
        ("où trouver de la nourriture gratuite à cote des neiges", "fr"),
        ("wifi gratuit cote des neiges", "fr"),
        ("job search assistance", "en"),
        ("banque alimentaire", "fr"),
        ("free food near notre-dame-de-grace", "en"),
        ("aide alimentaire près de NDG", "fr"),
    ]
    for query, expected in test_cases:
        got = detect_query_language(query)
        status = "OK" if got == expected else "FAIL"
        print(f"[{status}] expected={expected:<3} got={got:<3}  {query!r}")