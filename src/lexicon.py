"""Rule baseline: an offensive-word lexicon matched with minimum edit distance (Area B).

The lexicon is not hand-written: it is mined from the training split as the words that
appear almost only in offensive messages. A message is flagged when one of its tokens is
in the lexicon, or is within edit distance 1 of a lexicon word (catches typos / evasions).
"""
from collections import Counter
from functools import lru_cache

from .textnorm import fold, tokenize


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert / delete / substitute, cost 1) by dynamic programming."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,                 # deletion
                           cur[j - 1] + 1,              # insertion
                           prev[j - 1] + (ca != cb)))   # substitution
        prev = cur
    return prev[-1]


def tokens(text: str) -> list[str]:
    return [fold(t) for t in tokenize(text)]


def count_words(texts, labels) -> tuple[Counter, Counter]:
    """In how many messages / offensive messages each word occurs."""
    total, off = Counter(), Counter()
    for text, label in zip(texts, labels):
        for tok in set(tokens(text)):
            total[tok] += 1
            off[tok] += label
    return total, off


def mine_lexicon(total, off, min_count: int, min_ratio: float, min_len: int = 4) -> list[str]:
    """Words seen in >= min_count messages, of which >= min_ratio are offensive."""
    return sorted(w for w, n in total.items()
                  if len(w) >= min_len and n >= min_count and off[w] / n >= min_ratio)


def _deletions(word: str) -> set[str]:
    """The word itself plus every string obtained by deleting one character."""
    return {word} | {word[:i] + word[i + 1:] for i in range(len(word))}


class LexiconMatcher:
    """Exact or edit-distance-1 lookup in the lexicon.

    Comparing every token with every lexicon word is too slow for live chat, so candidates
    are first found through a deletion index (two words within distance 1 always share a
    string in their one-deletion neighbourhoods) and then verified with edit_distance().
    """

    def __init__(self, words, fuzzy_min_len: int = 6):
        self.words = set(words)
        self.fuzzy_min_len = fuzzy_min_len
        self.index: dict[str, list[str]] = {}
        for w in self.words:
            if len(w) >= fuzzy_min_len:
                for d in _deletions(w):
                    self.index.setdefault(d, []).append(w)
        self._match = lru_cache(maxsize=200_000)(self._match_uncached)

    def _match_uncached(self, tok: str):
        """Return (lexicon word, distance) or None."""
        if tok in self.words:
            return tok, 0
        if len(tok) >= self.fuzzy_min_len:
            for d in _deletions(tok):
                for w in self.index.get(d, ()):
                    if edit_distance(tok, w) <= 1:
                        return w, 1
        return None

    def matches(self, text: str) -> list[tuple[str, str, int]]:
        """[(token, matched lexicon word, distance), ...] for one message."""
        out = []
        for tok in tokens(text):
            hit = self._match(tok)
            if hit:
                out.append((tok, hit[0], hit[1]))
        return out

    def score(self, text: str) -> float:
        """Pseudo-probability of 'offensive': 0.95 exact hit, 0.75 fuzzy hit, 0.10 no hit."""
        hits = self.matches(text)
        if not hits:
            return 0.10
        return 0.95 if min(d for _, _, d in hits) == 0 else 0.75
