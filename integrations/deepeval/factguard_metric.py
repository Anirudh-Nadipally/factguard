"""DeepEval metric: share of numeric claims in actual_output that the source JSON supports.

    from factguard_metric import FactGuardMetric
    metric = FactGuardMetric(data=source_json, threshold=1.0)
    assert_test(LLMTestCase(input=q, actual_output=answer), [metric])

Follows DeepEval's custom-metric interface (BaseMetric: __init__ sets threshold,
measure/a_measure set score, success and reason, is_successful, __name__):
https://deepeval.com/docs/metrics-custom

deepeval is imported lazily: importing this module (or factguard) never imports
deepeval; the class is built the first time FactGuardMetric is accessed.

Where the source data comes from (first match wins):
  1. data=...                          object, JSON string, or pathlib.Path to a JSON file
  2. test_case.metadata["factguard_source"]   (additional_metadata on older deepeval)
  3. test_case.context, when it is one string that parses as JSON
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["FactGuardMetric", "make_metric_class"]

METADATA_KEY = "factguard_source"
_cls = None


def _load(value: Any) -> Any:
    if isinstance(value, Path):
        return json.loads(value.read_text())
    if isinstance(value, str):
        return json.loads(value)
    return value


def _source_from(test_case: Any) -> Any:
    for attr in ("metadata", "additional_metadata"):
        md = getattr(test_case, attr, None)
        if isinstance(md, dict) and md.get(METADATA_KEY) is not None:
            return _load(md[METADATA_KEY])
    ctx = getattr(test_case, "context", None)
    if isinstance(ctx, list) and len(ctx) == 1 and isinstance(ctx[0], str):
        try:
            return json.loads(ctx[0])
        except ValueError:
            pass
    raise ValueError("FactGuardMetric: no source data. Pass data=..., set "
                     f"metadata['{METADATA_KEY}'], or put the JSON as the only item of context.")


def make_metric_class():
    """Build (once) and return the FactGuardMetric class. Imports deepeval here."""
    global _cls
    if _cls is not None:
        return _cls

    from deepeval.metrics import BaseMetric  # lazy: only when the metric is used
    from factguard import Checker

    class FactGuardMetric(BaseMetric):
        """Deterministic: no LLM, no API key. score = grounded numeric claims / all numeric claims."""

        def __init__(self, data: Any = None, threshold: float = 1.0, *, aliases: dict | None = None,
                     strict_rows: bool = False, derived: bool = True, strict_mode: bool = False,
                     include_reason: bool = True, async_mode: bool = True):
            self.data = _load(data) if data is not None else None
            self.threshold = 1.0 if strict_mode else threshold
            self.aliases = aliases
            self.strict_rows = strict_rows
            self.derived = derived
            self.strict_mode = strict_mode
            self.include_reason = include_reason
            self.async_mode = async_mode
            self.evaluation_model = None   # no LLM involved
            self.report = None

        def measure(self, test_case, *args, **kwargs) -> float:
            try:
                data = self.data if self.data is not None else _source_from(test_case)
                checker = Checker(data, aliases=self.aliases, derived=self.derived,
                                  strict_rows=self.strict_rows)
                self.report = checker.check(test_case.actual_output or "")
                self.score = self.report.score
                if self.strict_mode and self.score < self.threshold:
                    self.score = 0.0
                self.success = self.score >= self.threshold
                if self.include_reason:
                    self.reason = self._reason()
                return self.score
            except Exception as e:
                self.error = str(e)
                raise

        async def a_measure(self, test_case, *args, **kwargs) -> float:
            return self.measure(test_case, *args, **kwargs)  # pure CPU, nothing to await

        def is_successful(self) -> bool:
            if self.error is not None:
                self.success = False
            else:
                try:
                    self.success = self.score >= self.threshold
                except TypeError:
                    self.success = False
            return self.success

        def _reason(self) -> str:
            r = self.report
            head = f"{len(r.verdicts) - len(r.flagged)}/{len(r.verdicts)} numeric claims grounded"
            parts = []
            for v in r.flagged:
                if v.status == "contradicted":
                    parts.append(f"{v.claim}: data says {v.hint}")
                else:
                    parts.append(f"{v.claim}: not in data")
            return head + ("; " + "; ".join(parts) if parts else "")

        @property
        def __name__(self):
            return "FactGuard Numeric Grounding"

    _cls = FactGuardMetric
    return _cls


def __getattr__(name: str):  # PEP 562: build the class on first access
    if name == "FactGuardMetric":
        return make_metric_class()
    raise AttributeError(name)
