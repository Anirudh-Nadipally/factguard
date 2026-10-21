"""Pull numeric claims out of free text.

A claim is a number as the text states it, with the context needed to judge it:
its unit (%, currency, multiplier), the precision it was written at, and the
sentence it sits in. Nothing here calls a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
    "twice": 2, "doubled": 2, "tripled": 3, "halved": 0.5,
}
# number words that are usually part of a fixed phrase, not a count
_WORD_PHRASE = re.compile(
    r"^\s*-?\s*(point(er)?s?|digit|double|time|way|fold)\b|^\s*-\s*(double|triple)", re.IGNORECASE)

_SCALE = {
    "k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "t": 1e12, "tn": 1e12, "trillion": 1e12, "lakh": 1e5, "lakhs": 1e5,
    "crore": 1e7, "crores": 1e7,
}

_CURRENCY = r"(?:\$|USD\s?|US\$|€|£|₹|Rs\.?\s?|INR\s?)"

_NUM_RE = re.compile(
    rf"""
    (?P<cur>{_CURRENCY})?
    (?P<sign>[-+−])?
    (?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    (?:\s?(?P<scale>k|K|M|mn|bn|B|tn|T|thousand|million|billion|trillion|lakhs?|crores?)\b)?
    (?P<unit>\s?%|\s?percent\b|\s?pp\b|\s?percentage\ points?\b|\s?x\b|\s?×)?
    """,
    re.VERBOSE,
)

_TENS = "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
_UNITS = "one|two|three|four|five|six|seven|eight|nine"
# "seventy-five" / "seventy five" is one number, tried before the single words
_WORD_RE = re.compile(r"\b((?:" + _TENS + r")[\s-](?:" + _UNITS + r")|" + "|".join(_WORDS) + r")\b", re.IGNORECASE)

_YEAR_RE = re.compile(r"^(19|20)\d\d$")
_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Claim:
    text: str          # the number as written, e.g. "$1.2M" or "38%"
    value: float       # parsed value, scale applied (38% -> 38.0, unit="%")
    unit: str          # "", "%", "pp", "x", "cur"
    decimals: int      # digits after the point as written (precision)
    scale: float       # 1, or 1e3/1e6/1e5... when written as "1.2M", "3 lakh"(implies rounding)
    start: int
    end: int
    sentence: str
    from_word: bool = False


def _sentence_at(text: str, pos: int) -> str:
    start = 0
    for m in _SENT_RE.finditer(text):
        if m.start() >= pos:
            return text[start:m.start()].strip()
        start = m.end()
    return text[start:].strip()


def _is_noise(text: str, m: re.Match) -> bool:
    """Numbers that are formatting or names rather than claims."""
    s, e = m.start(), m.end()
    prev = text[s - 1] if s else " "
    prev2 = text[s - 2] if s > 1 else " "
    nxt = text[e] if e < len(text) else " "
    nxt2 = text[e + 1] if e + 1 < len(text) else " "
    # list markers: "1. ", "2) " at the start of a line
    line_start = text.rfind("\n", 0, s) + 1
    if text[line_start:s].strip() == "" and nxt in ".)" and nxt2 in " \t":
        return True
    # glued to a word or tag: H100, gpt4o, v2, #12, @3, 4th
    if prev.isalnum() or prev in "#@_":
        return True
    if nxt.isalpha() and not (m.group("scale") or m.group("unit")):
        return True
    # dates, times, versions: 2026-10-20, 20/10/2026, 10:30, 3.1.4
    around = text[max(0, s - 11):e + 11]
    if re.search(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}", around) and (prev in "-/" or nxt in "-/"):
        return True
    if (prev == ":" and prev2.isdigit()) or (nxt == ":" and nxt2.isdigit()):
        return True
    if (prev == "." and prev2.isdigit()) or (nxt == "." and nxt2.isdigit()):
        return True
    # "version 2.1", "Python 3.12", "release 4": named versions, not facts
    if not (m.group("cur") or m.group("unit") or m.group("scale")) and \
            _VERSION_WORD_RE.search(text[max(0, s - 16):s]):
        return True
    return False


_VERSION_WORD_RE = re.compile(
    r"\b(?:version|ver\.?|release|python|node(?:\.js)?|ios|android|macos|windows|ubuntu|java|php|ruby|go)\s+$",
    re.IGNORECASE)


def extract_claims(text: str, *, ignore_years: bool = True, words: bool = True) -> list[Claim]:
    claims: list[Claim] = []
    for m in _NUM_RE.finditer(text):
        if m.group("sign") and m.start() > 0 and text[m.start() - 1].isalnum():
            # "113-107": the dash separates two numbers; it is not a minus sign
            m = _NUM_RE.match(text, m.start() + 1)
            if m is None:
                continue
        raw = m.group(0).strip()
        if not raw or _is_noise(text, m):
            continue
        num = m.group("num").replace(",", "")
        if ignore_years and not m.group("cur") and not m.group("unit") and _YEAR_RE.match(num):
            continue
        value = float(num)
        decimals = len(num.split(".")[1]) if "." in num else 0
        scale = m.group("scale")
        if scale:
            value *= _SCALE[scale.lower()]
        if m.group("sign") in ("-", "−"):
            value = -value
        unit_raw = (m.group("unit") or "").strip().lower()
        if unit_raw in ("%", "percent"):
            unit = "%"
        elif unit_raw.startswith("p"):
            unit = "pp"
        elif unit_raw in ("x", "×"):
            unit = "x"
        elif m.group("cur"):
            unit = "cur"
        else:
            unit = ""
        claims.append(Claim(raw, value, unit, decimals, _SCALE[scale.lower()] if scale else 1.0,
                            m.start(), m.end(),
                            _sentence_at(text, m.start())))
    if words:
        taken = [(c.start, c.end) for c in claims]
        for m in _WORD_RE.finditer(text):
            if any(a <= m.start() < b for a, b in taken):
                continue
            w = re.sub(r"[\s-]+", " ", m.group(1).lower())
            val = sum(_WORDS[p] for p in w.split())
            if w == "one":  # "one of", "no one" are almost never claims
                continue
            if _WORD_PHRASE.match(text[m.end():m.end() + 12]) or re.search(r"-\s*$", text[max(0, m.start() - 3):m.start()]):  # three-pointer, double-digit, double-double
                continue
            unit = "x" if w in ("twice", "doubled", "tripled", "halved") else ""
            claims.append(Claim(m.group(1), float(val), unit, 0, 1.0,
                                m.start(), m.end(), _sentence_at(text, m.start()), True))
    claims.sort(key=lambda c: c.start)
    return claims
