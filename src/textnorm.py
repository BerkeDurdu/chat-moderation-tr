"""Regex-based normalization of noisy chat text (Area B: classical NLP).

Handles the things the assignment lists for Use Case 4:
  * leetspeak        s1kt1r   -> siktir
  * repeated letters saaaalak -> salak
  * Turkish chars    correct lower-casing of I/İ, optional ASCII folding (ş -> s)
  * abbreviations    orsp cocu -> orospu çocuğu, sktr -> siktir, aq / amk -> amına koyayım
plus user mentions and URLs.
"""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_REPEAT = re.compile(r"(.)\1{2,}")          # 3+ of the same char -> 1
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[a-zçğıöşü]+")
_LETTER = re.compile(r"[A-Za-zçğıöşüÇĞİÖŞÜ]")
# a token that mixes letters with digits/symbols, e.g. "s1kt1r", "4ptal", "$alak"
_LEET_TOKEN = re.compile(r"[A-Za-zçğıöşüÇĞİÖŞÜ0-9@$]+")
# a real number with a Turkish suffix ("100lerce", "2020de", "3lü") is not leetspeak
_NUMBER_SUFFIX = re.compile(r"^(\d{2,}[A-Za-zçğıöşüÇĞİÖŞÜ]*|\d[A-Za-zçğıöşüÇĞİÖŞÜ]{0,2})$")
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_FOLD_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
_USER_TOKEN = "kullanıcı"                   # placeholder that survives tokenization

# ---- abbreviated profanity -------------------------------------------------------------
# Chat users drop vowels ("orsp", "sktr", "pzvnk") or use fixed shorthands ("aq", "oç").
# The models were trained on tweets where these forms are rare, so they are expanded back to
# the full word before any model sees the text.
_PROFANITY = ["orospu", "çocuğu", "siktir", "sikerim", "sikeyim", "sikik", "sikim", "amına",
              "koyayım", "koyarım", "amcık", "pezevenk", "şerefsiz", "yavşak", "yarrak", "kahpe",
              "gavat", "dangalak", "gerizekalı", "salak", "aptal", "kaltak", "puşt", "piç",
              "götveren", "ibne", "ananı", "ananın", "bacını", "avradını", "haysiyetsiz",
              "namussuz", "terbiyesiz", "şerefsizler", "orospular"]
_ABBREV = {          # keys are lower-case; ASCII forms only where no real word collides
    "oç": "orospu çocuğu", "oc": "orospu çocuğu", "oçlar": "orospu çocukları",
    "oclar": "orospu çocukları",           # not "öç" (revenge) or "öcü" (bogeyman)
    "cocu": "çocuğu", "çocu": "çocuğu", "cocuu": "çocuğu", "çocuu": "çocuğu", "cocugu": "çocuğu",
    "aq": "amına koyayım", "amk": "amına koyayım", "amq": "amına koyayım",
    "aminakoyim": "amına koyayım", "amınakoyim": "amına koyayım", "aminakoyayim": "amına koyayım",
    "amınakoyayım": "amına koyayım", "mk": "amına koyayım", "amina": "amına",
    "sg": "siktir git", "siktirgit": "siktir git", "sktrgt": "siktir git",
    "pic": "piç", "pust": "puşt", "got": "göt", "ibine": "ibne", "yarak": "yarrak",
}
# dotted / spaced shorthands: "o.ç", "o ç", "a.q", "a.m.k"
_DOTTED = [(re.compile(r"(?<!\w)[oO](\s*[.\-*]\s*[çÇcC]|\s*[çÇ])(?!\w)"), "orospu çocuğu"),
           (re.compile(r"(?<!\w)[aA]\s*[.\-*]\s*[mM]?\s*[.\-*]?\s*[kKqQ](?!\w)"), "amına koyayım")]
_ALPHA_TOKEN = re.compile(r"[A-Za-zçğıöşüÇĞİÖŞÜ]+")


def turkish_lower(text: str) -> str:
    """Python's lower() turns 'I' into 'i'; in Turkish it must become 'ı'."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def _deleet(match: re.Match) -> str:
    tok = match.group(0)
    if not _LETTER.search(tok) or _NUMBER_SUFFIX.match(tok):
        return tok                          # plain numbers stay as they are
    return tok.translate(_LEET_MAP)


def fold(text: str) -> str:
    """ASCII-fold Turkish letters so that 'şerefsiz' and 'serefsiz' match."""
    return text.translate(_FOLD_MAP)


def _skeleton(word: str) -> str:
    """Consonants only, doubled letters merged: 'orospu' -> 'rsp', 'yarrak' -> 'yrk'."""
    out = [c for c in fold(word) if c not in "aeiou"]
    return "".join(c for i, c in enumerate(out) if i == 0 or c != out[i - 1])


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


# consonant skeleton -> full word, only for skeletons long enough to be unambiguous
_SKELETONS = {}
for _w in _PROFANITY:
    if len(_skeleton(_w)) >= 3:
        _SKELETONS.setdefault(_skeleton(_w), _w)


def expand_abbreviation(token: str) -> str | None:
    """Full profanity for an abbreviated token, or None.

    A token counts as an abbreviation of a word when it is a known shorthand, or when it is
    shorter than the word, starts with the same letter, keeps the same consonants in the same
    order and only drops vowels ('orsp' -> 'orospu', 'srfsz' -> 'şerefsiz').
    """
    low = turkish_lower(token)
    if low in _ABBREV:
        return _ABBREV[low]
    key = fold(low)
    if len(key) < 3:
        return None
    word = _SKELETONS.get(_skeleton(key))
    if word and len(key) < len(word) and key[0] == fold(word)[0] and _is_subsequence(key, fold(word)):
        return word
    return None


def _expand(match: re.Match) -> str:
    return expand_abbreviation(match.group(0)) or match.group(0)


def normalize(text: str, lower: bool = True) -> str:
    """Clean one chat message. `lower=False` keeps the case (used for cased BERTurk)."""
    text = _URL.sub(" ", text)
    text = _USER.sub(f" {_USER_TOKEN} ", text)
    text = _LEET_TOKEN.sub(_deleet, text)
    if lower:
        text = turkish_lower(text)
    text = _REPEAT.sub(r"\1", text)
    for pattern, full in _DOTTED:
        text = pattern.sub(full, text)
    text = _ALPHA_TOKEN.sub(_expand, text)
    return _SPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Normalize, then split into lower-case word tokens."""
    return _WORD.findall(normalize(text))
