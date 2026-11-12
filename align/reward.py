"""The programmatic reward used to build preference pairs and train DPO.

Design choice, stated up front: the *training* reward ("proxy") below never looks at
fluency. It only checks the three things we can verify mechanically -- does the story
contain the required keyword, does it end in the requested tone, is it close to the
target length, and is it not degenerate (repeating itself). Fluency is deliberately
withheld and measured only at evaluation time, as "true quality", under the frozen
reference model's own log-likelihood. That gap between proxy and true quality is the
whole point of this repo: DPO will happily push the policy to satisfy the proxy, and
``align.evaluate`` measures what that does to the part of quality nobody was rewarding.

Every sub-score is clipped to [0, 1] before weighting so no single term can dominate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HAPPY_LEXICON = {
    "happy",
    "happily",
    "smiled",
    "smile",
    "smiling",
    "laughed",
    "laughing",
    "joy",
    "joyful",
    "delighted",
    "cheerful",
    "glad",
    "hugged",
    "celebrated",
    "grinned",
}
SAD_LEXICON = {
    "sad",
    "sadly",
    "cried",
    "crying",
    "tears",
    "lonely",
    "sorry",
    "upset",
    "unhappy",
    "miserable",
    "wept",
    "sighed",
    "frowned",
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def keyword_score(text: str, keyword: str) -> float:
    """1.0 if the keyword (or a simple plural/-ed/-ing variant) appears, else 0.0."""
    kw = keyword.lower()
    variants = {kw, kw + "s", kw + "es", kw + "ed", kw + "d", kw + "ing"}
    if kw.endswith("e"):
        variants.add(kw[:-1] + "ing")
    if kw.endswith("y") and len(kw) > 1 and kw[-2] not in "aeiou":
        variants.add(kw[:-1] + "ies")  # puppy -> puppies
    words = set(_words(text))
    return 1.0 if words & variants else 0.0


def ending_score(text: str, tone: str, tail_sentences: int = 2) -> float:
    """Score the requested emotional tone using the lexicon of the *last* sentences only.

    A story that is sad throughout but "happy" only in its very last clause should
    score well for tone=happy -- that is in fact what we are asking for.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    tail = " ".join(sentences[-tail_sentences:]) if sentences else text
    words = set(_words(tail))
    happy_hits = len(words & HAPPY_LEXICON)
    sad_hits = len(words & SAD_LEXICON)
    if tone == "happy":
        if happy_hits == 0 and sad_hits == 0:
            return 0.0
        return happy_hits / max(happy_hits + sad_hits, 1)
    if tone == "sad":
        if happy_hits == 0 and sad_hits == 0:
            return 0.0
        return sad_hits / max(happy_hits + sad_hits, 1)
    raise ValueError(f"unknown tone: {tone!r}")


def length_score(text: str, target: int = 60, tolerance: int = 25) -> float:
    """1.0 at the target word count, falling off linearly to 0.0 past `tolerance` words away."""
    n = len(_words(text))
    diff = abs(n - target)
    return max(0.0, 1.0 - diff / tolerance)


def diversity_score(text: str, n: int = 3) -> float:
    """Distinct n-gram ratio: punishes the loops small LMs fall into ("and then and then...")."""
    words = _words(text)
    if len(words) < n:
        return 1.0 if words else 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


@dataclass(frozen=True)
class RewardWeights:
    keyword: float = 0.35
    ending: float = 0.25
    length: float = 0.20
    diversity: float = 0.20


DEFAULT_WEIGHTS = RewardWeights()


@dataclass(frozen=True)
class RewardBreakdown:
    keyword: float
    ending: float
    length: float
    diversity: float
    total: float


def score(text: str, keyword: str, tone: str, weights: RewardWeights = DEFAULT_WEIGHTS) -> RewardBreakdown:
    """The proxy reward used for preference labelling and DPO training. Fluency-blind, on purpose."""
    k = keyword_score(text, keyword)
    e = ending_score(text, tone)
    len_s = length_score(text)
    d = diversity_score(text)
    total = weights.keyword * k + weights.ending * e + weights.length * len_s + weights.diversity * d
    return RewardBreakdown(keyword=k, ending=e, length=len_s, diversity=d, total=total)
