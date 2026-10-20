"""Check every number in a piece of text against the data it was written from."""
from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from .extract import Claim, extract_claims
from .scope import Scoper

_HEDGES = re.compile(
    r"(about|around|roughly|approximately|approx\.?|nearly|almost|close to|over|more than|"
    r"under|less than|at least|~|≈)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Fact:
    path: str
    value: float


@dataclass
class Verdict:
    claim: str
    value: float
    unit: str
    status: str                 # "supported" | "derived" | "contradicted" | "unsupported"
    evidence: str = ""          # path or formula that supports it
    hint: str = ""              # when flagged: the fact the text most likely meant (path=value)
    sentence: str = ""
    low_signal: bool = False    # small bare integer; often a count or list size
    start: int = 0              # character span of the number in the text
    end: int = 0


@dataclass
class Report:
    verdicts: list[Verdict] = field(default_factory=list)

    @property
    def flagged(self) -> list[Verdict]:
        """Claims the data contradicts or does not contain."""
        return [v for v in self.verdicts if v.status in ("contradicted", "unsupported")]

    @property
    def contradicted(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.status == "contradicted"]

    @property
    def unsupported(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.status == "unsupported"]

    @property
    def score(self) -> float:
        """Share of numeric claims the data supports (1.0 when there are none)."""
        if not self.verdicts:
            return 1.0
        return 1 - len(self.flagged) / len(self.verdicts)

    @property
    def ok(self) -> bool:
        return not self.flagged

    def to_dict(self) -> dict:
        return {"score": round(self.score, 4), "claims": len(self.verdicts),
                "contradicted": len(self.contradicted), "unsupported": len(self.unsupported),
                "verdicts": [asdict(v) for v in self.verdicts]}

    def __str__(self) -> str:
        lines = [f"factguard: {len(self.verdicts) - len(self.flagged)}/{len(self.verdicts)} "
                 f"numeric claims grounded (score {self.score:.2f})"]
        for v in self.verdicts:
            mark = {"supported": "ok ", "derived": "ok*", "contradicted": "XX ", "unsupported": "?? "}[v.status]
            line = f"  {mark} {v.claim:<12} {v.evidence or ''}"
            if v.status == "contradicted":
                line += f"  <- data says {v.hint}"
            elif v.status == "unsupported":
                line += "  <- not in data" + (f"; did it mean {v.hint}?" if v.hint else "")
            lines.append(line.rstrip())
        return "\n".join(lines)


# ---------------------------------------------------------------- facts

def _num(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and math.isfinite(x):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "").rstrip("%").lstrip("$₹€£")
        try:
            v = float(s)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def flatten(data: Any, prefix: str = "") -> list[Fact]:
    """Every numeric leaf, plus the length of every list, with its JSON path."""
    out: list[Fact] = []
    if isinstance(data, dict):
        for k, v in data.items():
            out += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(data, list):
        out.append(Fact(f"len({prefix or '$'})", float(len(data))))
        for i, v in enumerate(data):
            out += flatten(v, f"{prefix}[{i}]")
    else:
        n = _num(data)
        if n is not None:
            out.append(Fact(prefix or "$", n))
    return out


# ---------------------------------------------------------------- matching

def _tolerance(c: Claim, hedged: bool, rel_tol: float) -> float:
    """How far a data value may sit from the claim and still be the same number.

    A number written as 4.7 covers 4.65-4.75; "$1.2M" covers 1.15M-1.25M.
    Hedges ("about", "nearly", "over") widen it to 10%.
    """
    tol = 0.5 * 10 ** (-c.decimals) * c.scale
    tol = max(tol, abs(c.value) * rel_tol)
    if hedged:
        tol = max(tol, abs(c.value) * 0.10)
    return tol + 1e-9


def _forms(c: Claim, tol: float) -> list[tuple[float, float]]:
    """(value, tolerance) pairs the data could store this claim as."""
    out = [(c.value, tol)]
    if c.unit == "%":
        out.append((c.value / 100, tol / 100))   # 38% stored as 0.38
    return out


def _tokens(s: str) -> set[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    toks = {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 2 and not t.isdigit()}
    # crude stemming so "orders" meets "order_count", "ratings" meets "rating"
    return toks | {t[:-1] for t in toks if t.endswith("s")} | {t[:-3] for t in toks if t.endswith("ing")}


def _path_tokens(path: str) -> set[str]:
    return _tokens(re.sub(r"\[\d+\]|len\(|\)", " ", path))


_CUES = re.compile(
    r"\b(more|less|fewer|than|combined|total|totals|together|sum|difference|gap|margin|ahead|behind|"
    r"lead|led by|up|down|increase[sd]?|decrease[sd]?|grew|grow|fell|fall|rose|rise|drop(ped)?|ratio|"
    r"times|per|average[sd]?|averaging|avg|mean|gain(ed)?|los[st]|change[sd]?|share|of all|outscor\w*|"
    r"double[sd]?|triple[sd]?|half|halved|x)\b",
    re.IGNORECASE)


def _parent(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else "$"


def _leaf(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path.rsplit(".", 1)[-1])


def _comparable(a: str, b: str) -> bool:
    """Arithmetic only between fields of one record (revenue / orders) or the same
    field across records (players[0].points - players[1].points)."""
    return _parent(a) == _parent(b) or _leaf(a) == _leaf(b)


_FILLER = {"the", "a", "an", "in", "on", "of", "and", "his", "her", "their", "its", "for", "to", "at", "s"}


# which derivations can stand behind each kind of claim
_KINDS = {
    "": {"agg", "sum", "diff", "ratio", "pct"},
    "cur": {"agg", "sum", "diff"},
    "%": {"pct", "diff", "agg"},
    "pp": {"diff"},
    "x": {"ratio"},
}


def _cue_context(text: str, c: Claim) -> str:
    """Words around a number plus the number itself; a multiplier is its own cue."""
    ctx = f"{_window(text, c, 8, 6)} {c.text}"
    return ctx + " times" if c.unit in ("x", "pp") else ctx


def _window(text: str, c: Claim, before: int = 6, after: int = 3) -> str:
    """The few words around a number: what it is most likely about."""
    left = re.split(r"[.!?\n;]", text[:c.start])[-1].split()[-before:]
    right = re.split(r"[.!?\n;]", text[c.end:])[0].split()[:after]
    return " ".join(left + right)


class Checker:
    """Holds the data once; checks any number of texts against it."""

    def __init__(self, data: Any, *, derived: bool = True, rel_tol: float = 0.0,
                 ignore_years: bool = True, words: bool = True,
                 ignore_small_ints: bool = False, derive_from: int = 12,
                 aliases: dict[str, list[str]] | None = None, scoped: bool = True,
                 pairs: dict[str, tuple[str, str]] | None = None, strict_rows: bool = False):
        # strict_rows: once a sentence names a row, a value found only on another row is a
        # contradiction. Right for product or store tables; noisy in dense multi-entity prose
        # (it cost 7 points of precision on the INLG train set), so it is opt-in.
        self.strict_rows = strict_rows
        self.facts = flatten(data)
        self.scoper = Scoper(data, aliases) if scoped else None
        self.aliases = aliases or {}
        # "7-15 FG" / "7 of 15 shooting": word -> (field for the first number, field for the second)
        self.pairs = {k.lower(): v for k, v in (pairs or {}).items()}
        self.derived = derived
        self.rel_tol = rel_tol
        self.ignore_years = ignore_years
        self.words = words
        self.ignore_small_ints = ignore_small_ints
        self.derive_from = derive_from
        self._ptok = {}
        for f in self.facts:
            leaf = re.sub(r"\[\d+\]$", "", f.path).split(".")[-1]
            self._ptok[f.path] = _path_tokens(f.path) | {w for a in self.aliases.get(leaf, []) for w in _tokens(a)}
        self._groups: dict[str, list[Fact]] = {}
        for f in self.facts:
            key = re.sub(r"\[\d+\]", "[*]", f.path)
            if key != f.path:
                self._groups.setdefault(key, []).append(f)

    def _relevant(self, sentence: str) -> list[Fact]:
        """Facts whose names the sentence mentions, best first.

        Derived values are only tried among these, so a stray number cannot be
        "explained" by some arbitrary pair out of hundreds of fields.
        """
        words = _tokens(sentence)
        scored = [(len(self._ptok[f.path] & words), f) for f in self.facts]
        hits = [f for n, f in sorted(scored, key=lambda t: -t[0]) if n > 0]
        if not hits and len(self.facts) <= 6:
            return list(self.facts)
        return hits[: self.derive_from]

    def _derivations(self, facts: list[Fact], context: str) -> Iterable[tuple[float, str, str]]:
        words = _tokens(context)
        # aggregates over a list the text names come first: they are the likeliest reading
        for key, fs in self._groups.items():
            if len(fs) > 1 and _path_tokens(key) & words:
                vals = [f.value for f in fs]
                yield sum(vals), f"sum({key})", "agg"
                yield sum(vals) / len(vals), f"mean({key})", "agg"
                yield max(vals), f"max({key})", "agg"
                yield min(vals), f"min({key})", "agg"
        if not _CUES.search(context):
            return  # no "more than", "up", "total"...: the text is not doing arithmetic
        plain = [f for f in facts if not f.path.startswith("len(")]
        for a, b in itertools.permutations(plain, 2):
            if not _comparable(a.path, b.path):
                continue
            x, y = a.value, b.value
            if a.path < b.path:
                yield x + y, f"{a.path} + {b.path}", "sum"
            yield abs(x - y), f"|{a.path} - {b.path}|", "diff"
            if y:
                yield x / y, f"{a.path} / {b.path}", "ratio"
                yield x / y * 100, f"{a.path} / {b.path} * 100", "pct"
                yield abs(x - y) / abs(y) * 100, f"|{a.path} - {b.path}| / {b.path} * 100", "pct"

    def _ranked(self, window: str) -> list[Fact]:
        """All facts, the ones the nearby words name first (better evidence paths)."""
        near = self._relevant(window) if _tokens(window) else []
        seen = {f.path for f in near}
        return near + [f for f in self.facts if f.path not in seen]

    def _hint(self, window: str, sentence: str) -> str:
        for ctx in (window, sentence):
            rel = [f for f in self._relevant(ctx) if not f.path.startswith("len(")]
            if rel and _path_tokens(rel[0].path) & _tokens(ctx):
                return f"{rel[0].path}={rel[0].value:g}"
        return ""

    def check_claim(self, c: Claim, text: str) -> Verdict | None:
        low = c.unit == "" and c.decimals == 0 and c.scale == 1 and abs(c.value) <= 10
        if low and self.ignore_small_ints:
            return None
        hedged = bool(_HEDGES.search(text[max(0, c.start - 20):c.start]))
        forms = _forms(c, _tolerance(c, hedged, self.rel_tol))
        v = Verdict(c.text, c.value, c.unit, "unsupported", sentence=c.sentence, low_signal=low,
                    start=c.start, end=c.end)
        window = _window(text, c)
        if self.scoper is not None and c.unit not in ("x", "pp"):
            sv = self._scoped(c, text, forms, v)
            if sv is not None:
                return sv
        # a multiplier ("3x") or a point change ("+2 pp") is never a raw field
        direct = c.unit not in ("x", "pp")
        for target, tol in forms if direct else []:
            for f in self._ranked(window):
                if c.unit == "%" and f.path.startswith("len("):
                    continue
                if abs(f.value - target) <= tol:
                    v.status, v.evidence = "supported", f.path
                    return v
        if self.derived:
            rel = self._relevant(c.sentence)
            for target, tol in forms:
                for val, formula, kind in self._derivations(rel, _cue_context(text, c)):
                    if kind not in _KINDS[c.unit]:
                        continue
                    if abs(abs(val) - abs(target)) <= tol:
                        v.status, v.evidence = "derived", formula
                        return v
        v.hint = self._hint(window, c.sentence)
        return v

    def _entities(self, c: Claim, text: str) -> list:
        """Records the number could be about, likeliest first: names before it in its
        sentence (nearest first, skipping objects like "for Cleveland"), then names after
        it, then names in the previous sentence (for "he", "they")."""
        bounds = [m.end() for m in re.finditer(r"[.!?](?=\s)|\n", text)]
        s_start = max([b for b in bounds if b <= c.start], default=0)
        s_end = min([b for b in bounds if b > c.start], default=len(text))
        prev_start = max([b for b in bounds if b < s_start], default=0)
        ms = self.scoper.mentions(text[prev_start:s_end])
        ms = [(a + prev_start, b + prev_start, r) for a, b, r in ms]

        def _object(m):
            return re.search(r"\b(for|against|over|to|than|vs\.?|versus|beat|beating|defeated|of|led|"
                             r"helped|joined|past|at|into|in|on|paced|leads|leading)\s+(the\s+)?$",
                             text[max(0, m[0] - 16):m[0]], re.IGNORECASE)
        before = [m for m in ms if s_start <= m[0] and m[1] <= c.start]
        subj = [m[2] for m in reversed(before) if not _object(m)]
        objs = [m[2] for m in reversed(before) if _object(m)]
        after = [m[2] for m in ms if m[0] >= c.end]
        prev = [m[2] for m in reversed(ms) if m[1] <= s_start]
        out, seen = [], set()
        for r in subj + after + prev + objs:
            if id(r) not in seen:
                seen.add(id(r))
                out.append(r)
        return out

    @staticmethod
    def _field_words(text: str, c: Claim) -> tuple[list[str], list[str]]:
        after: list[str] = []
        for tok in re.findall(r"[A-Za-z][A-Za-z'’]*|\d+|[^\sA-Za-z\d]", text[c.end:c.end + 60]):
            if tok[0].isdigit() or tok in ",.;:()!?" or tok.lower() in ("and", "but", "while", "whereas"):
                break
            if tok[0].isalpha() and tok.lower() not in _FILLER:
                after.append(tok)
                if len(after) == 3:
                    break
        # only the word right before the number ("points: 38", "rating 4.7"), never further back
        m = re.search(r"([A-Za-z][A-Za-z'’]*)\s*[:=]?\s*$", text[max(0, c.start - 25):c.start])
        before = [m.group(1)] if m else []
        # "revenue was 42,000", "rating slipped to 3.5": a field word a few back in the same clause
        clause = re.split(r"[,.;:()!?]|\d", text[max(0, c.start - 45):c.start])[-1]
        for w in reversed(re.findall(r"[A-Za-z][A-Za-z'’]*", clause)[-4:]):
            if w not in before and w.lower() not in _FILLER:
                before.append(w)
        return after, before

    def _pair_field(self, c: Claim, text: str) -> tuple[str, str] | None:
        """For "7 - 15 FG" or "7-of-15 shooting": (word, which side) when a pair word follows."""
        if not self.pairs:
            return None
        nxt = re.match(r"\s*(?:-|of|for|/|–)\s*(?:-\s*)?(?:(?:of|for)\s*-?\s*)?(\d+)\s+(?:from\s+)?(?:the\s+)?([A-Za-z0-9]+)", text[c.end:c.end + 40])
        if nxt and nxt.group(2).lower() in self.pairs:
            return nxt.group(2).lower(), "a"
        prv = re.search(r"(\d+)\s*(?:-|of|for|/|–)\s*(?:(?:of|for)\s*-?\s*)?$", text[max(0, c.start - 20):c.start])
        if prv:
            m = re.match(r"\s+(?:from\s+)?(?:the\s+)?([A-Za-z0-9]+)", text[c.end:c.end + 30])
            if m and m.group(1).lower() in self.pairs:
                return m.group(1).lower(), "b"
        return None

    def _scoped(self, c: Claim, text: str, forms, v: Verdict) -> Verdict | None:
        cands = self._entities(c, text)
        if not cands:
            return None
        after, before = self._field_words(text, c)
        pair = self._pair_field(c, text)
        rec, key = None, None
        for r in cands:  # the first record that has the field the words name
            if pair:
                k = self.pairs[pair[0]][0 if pair[1] == "a" else 1]
                k = k if k in r.fields else None
            else:
                k = self.scoper.resolve_field(r, after, c.unit) if after else \
                    self.scoper.resolve_field(r, before, c.unit)
            if k is not None:
                rec, key = r, k
                break
        if key is None:
            rec = cands[0]
            if pair or (after and any(self.scoper.resolve_field(r, after, c.unit) for r in self.scoper.records)):
                return None  # the field exists, just not on anyone named here: let the loose check decide
            for k, (path, val) in rec.fields.items():
                if any(abs(val - t) <= tol for t, tol in forms):
                    v.status, v.evidence = "supported", path
                    return v
            # a row is named but the value sits only on another row: "Breeze Fan costs $49.99"
            # when $49.99 is the Lamp's price. Never borrow a value across rows.
            if self.strict_rows and not v.low_signal:
                mine = {p for p, _ in rec.fields.values()}
                hits = [f for f in self.facts if f.path not in mine and not f.path.startswith("len(")
                        and any(abs(f.value - t) <= tol for t, tol in forms)]
                if len(hits) == 1:
                    v.status, v.hint = "contradicted", f"{rec.path} has no such value; the stated value is {hits[0].path}"
                    return v
            return None
        path, val = rec.fields[key]
        if any(abs(val - t) <= tol for t, tol in forms):
            v.status, v.evidence = "supported", path
            return v
        # "27 more points than Horford": a difference or ratio of that field
        if self.derived:
            same = [f for f in self.facts if _leaf(f.path) == key or _parent(f.path) == _parent(path)]
            for t, tol in forms:
                for dv, formula, kind in self._derivations(same, _cue_context(text, c)):
                    if kind in _KINDS[c.unit] and abs(abs(dv) - abs(t)) <= tol and path in formula:
                        v.status, v.evidence = "derived", formula
                        return v
        v.status, v.hint = "contradicted", f"{path}={val:g}"
        # wrong row: the same field on another record holds the stated value
        # (only when exactly one other record matches, so "four rebounds" among 26 players stays quiet)
        others = [f for f in self.facts if _leaf(f.path) == key and f.path != path
                  and any(abs(f.value - t) <= tol for t, tol in forms)]
        if len(others) == 1:
            v.hint += f"; the stated value is {others[0].path}"
        return v

    def _article_claims(self, text: str, taken: list[Claim]) -> list[Claim]:
        """'a steal', 'an assist', 'a pair of steals': counts written as articles.
        Only for words you named as field aliases, so "a game" is never a claim."""
        words = {w for ws in self.aliases.values() for w in ws}
        if not words:
            return []
        out = []
        spans = [(c.start, c.end) for c in taken]
        for m in re.finditer(r"\b(an?|a pair of|a couple of)\s+([A-Za-z]+)", text, re.IGNORECASE):
            w = m.group(2).lower()
            if w not in words and w.rstrip("s") not in words:
                continue
            if any(a <= m.start() < b for a, b in spans):
                continue
            val = 2.0 if "pair" in m.group(1).lower() or "couple" in m.group(1).lower() else 1.0
            from .extract import _sentence_at
            out.append(Claim(m.group(1), val, "", 0, 1.0, m.start(), m.start() + len(m.group(1)),
                             _sentence_at(text, m.start()), True))
        return out

    def check(self, text: str) -> Report:
        rep = Report()
        claims = extract_claims(text, ignore_years=self.ignore_years, words=self.words)
        claims = sorted(claims + self._article_claims(text, claims), key=lambda c: c.start)
        for c in claims:
            verdict = self.check_claim(c, text)
            if verdict is not None:
                rep.verdicts.append(verdict)
        return rep


def check(text: str, data: Any, **kw) -> Report:
    """Check every number in `text` against `data` (a dict, list, or JSON string)."""
    if isinstance(data, str):
        data = json.loads(data)
    return Checker(data, **kw).check(text)


def assert_grounded(text: str, data: Any, *, min_score: float = 1.0, **kw) -> Report:
    """For tests: raise AssertionError listing every claim the data does not support."""
    rep = check(text, data, **kw)
    if rep.score < min_score:
        raise AssertionError(str(rep))
    return rep
