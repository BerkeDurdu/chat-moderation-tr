"""Regex-based normalization of noisy chat text (Area B: classical NLP).

Handles the things the assignment lists for Use Case 4:
  * leetspeak        s1kt1r   -> siktir
  * repeated letters saaaalak -> salak
  * Turkish chars    correct lower-casing of I/İ, optional ASCII folding (ş -> s)
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


def turkish_lower(text: str) -> str:
    """Python's lower() turns 'I' into 'i'; in Turkish it must become 'ı'."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def _deleet(match: re.Match) -> str:
    tok = match.group(0)
    if not _LETTER.search(tok) or _NUMBER_SUFFIX.match(tok):
        return tok                          # plain numbers stay as they are
    return tok.translate(_LEET_MAP)


def normalize(text: str, lower: bool = True) -> str:
    """Clean one chat message. `lower=False` keeps the case (used for cased BERTurk)."""
    text = _URL.sub(" ", text)
    text = _USER.sub(f" {_USER_TOKEN} ", text)
    text = _LEET_TOKEN.sub(_deleet, text)
    if lower:
        text = turkish_lower(text)
    text = _REPEAT.sub(r"\1", text)
    return _SPACE.sub(" ", text).strip()


def fold(text: str) -> str:
    """ASCII-fold Turkish letters so that 'şerefsiz' and 'serefsiz' match."""
    return text.translate(_FOLD_MAP)


def tokenize(text: str) -> list[str]:
    """Normalize, then split into lower-case word tokens."""
    return _WORD.findall(normalize(text))
