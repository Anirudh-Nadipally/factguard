"""Work out what a number is about: which record (entity) and which field.

"Isaiah Thomas scored 38 points" should be checked against Isaiah Thomas's
points, not against any field anywhere that happens to equal 38. A loose
check passes a number that exists somewhere in the data; a scoped check
catches a number that is real but attached to the wrong thing.

Records are the dicts in the data. A record's names are its short string
fields ("Isaiah Thomas", "Celtics"); a multi-word name is also known by its
last word ("Thomas") when that word points at one record only. A field's
names are the words in its key plus any aliases you give
(aliases={"PTS": ["points", "point"]}).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_STOP = {"the", "and", "for", "with", "from", "this", "that", "team", "city", "name", "n/a", "none",
         "null", "true", "false", "yes"}
_WORD = re.compile(r"[A-Za-z][A-Za-z'’.]*|\d+(?:\.\d+)?%?")


def _num(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace(",", "").rstrip("%"))
        except ValueError:
            return None
    return None


def key_words(key: str) -> set[str]:
    key = re.sub(r"([a-z])([A-Z])", r"\1 \2", key)
    return {w for w in re.split(r"[^a-z0-9]+", key.lower()) if w}


@dataclass
class Record:
    path: str
    names: list[str]
    fields: dict[str, tuple[str, float]] = field(default_factory=dict)  # key -> (path, value)


@dataclass
class Scope:
    record: Record | None
    field_key: str | None


class Scoper:
    def __init__(self, data: Any, aliases: dict[str, list[str]] | None = None,
                 name_keys: list[str] | None = None):
        self.records: list[Record] = []
        self._walk(data, "")
        self.name_keys = set(name_keys or [])
        self.aliases: dict[str, set[str]] = {}
        for k, words in (aliases or {}).items():
            self.aliases[k] = {w.lower() for w in words}
        self._index: dict[str, list[Record]] = {}
        for r in self.records:
            for n in r.names:
                self._index.setdefault(n.lower(), []).append(r)
        # last word of a multi-word name, if it names one record only
        last: dict[str, list[Record]] = {}
        for r in self.records:
            for n in r.names:
                parts = n.split()
                if len(parts) > 1 and len(parts[-1]) > 2 and parts[-1].lower() not in _STOP:
                    last.setdefault(parts[-1].lower(), []).append(r)
        for w, rs in last.items():
            if len({id(r) for r in rs}) == 1 and w not in self._index:
                self._index[w] = rs[:1]
        # longest names first so "Los Angeles Lakers" wins over "Lakers"
        self._names = sorted(self._index, key=len, reverse=True)
        self._name_re = re.compile(
            r"(?<![A-Za-z])(" + "|".join(re.escape(n) for n in self._names) + r")(?![A-Za-z])",
            re.IGNORECASE) if self._names else None

    def _walk(self, node: Any, path: str) -> None:
        if isinstance(node, dict):
            names, fields = [], {}
            for k, v in node.items():
                p = f"{path}.{k}" if path else str(k)
                n = _num(v)
                if n is not None:
                    fields[str(k)] = (p, n)
                elif isinstance(v, str):
                    s = v.strip()
                    if 3 <= len(s) <= 60 and s.lower() not in _STOP and not s.isdigit():
                        names.append(s)
                elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                    names += [x for x in v if 3 <= len(x) <= 60]  # e.g. "aliases": [...]
            if names and fields:
                self.records.append(Record(path or "$", names, fields))
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    self._walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                self._walk(v, f"{path}[{i}]")

    # -- entity -----------------------------------------------------------

    def mentions(self, text: str) -> list[tuple[int, int, Record]]:
        if not self._name_re:
            return []
        out = []
        for m in self._name_re.finditer(text):
            rs = self._index.get(m.group(1).lower(), [])
            if len({id(r) for r in rs}) == 1:
                out.append((m.start(), m.end(), rs[0]))
        return out

    # -- field ------------------------------------------------------------

    def field_words(self, key: str) -> set[str]:
        return key_words(key) | self.aliases.get(key, set())

    def vocab(self) -> set[str]:
        if not hasattr(self, "_vocab"):
            self._vocab = {w for r in self.records for k in r.fields for w in self.field_words(k)}
        return self._vocab

    @staticmethod
    def is_rate(key: str) -> bool:
        return bool(key_words(key) & {"pct", "percent", "percentage", "rate", "ratio", "share"})

    def resolve_field(self, rec: Record, words: list[str], unit: str = "") -> str | None:
        """The field of `rec` named by the words right after (or before) the number.
        A rate field (fg_pct, conversion_rate) only takes a number written as a percent."""
        if words:
            first = words[0].lower().strip(".,;:()'’")
            vocab = self.vocab()
            if (first in vocab or first.rstrip("s") in vocab) and not any(
                    first in self.field_words(k) or first.rstrip("s") in self.field_words(k) for k in rec.fields):
                return None  # "three minutes": the noun is a field this record does not have
        best, best_score = None, 0.0
        for key in rec.fields:
            if self.is_rate(key) != (unit == "%") and not (unit == "%" and not any(self.is_rate(k) for k in rec.fields)):
                continue
            fw = self.field_words(key)
            score = 0.0
            for rank, w in enumerate(words):
                w = w.lower().strip(".,;:()'’")
                if w in fw or (w.endswith("s") and w[:-1] in fw):
                    score += 10 - rank
            if score:
                score -= 0.01 * len(fw)  # prefer the plainest field: PTS over PTS_QTR1
                if score > best_score:
                    best, best_score = key, score
        return best
