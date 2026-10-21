"""promptfoo python assertion: check every number in the LLM output against source JSON.

promptfoo calls get_assert(output, context) and accepts a bool, a float, or a
GradingResult dict ({"pass", "score", "reason", ...}).
Docs: https://www.promptfoo.dev/docs/configuration/expected-outputs/python/

Where the source data comes from (first match wins):
  1. config.source       inline object, or a JSON string
  2. config.source_file  path to a JSON file
  3. vars[<config.source_var or "source">]  object or JSON string (a var loaded
     with file://data.json arrives as the file's text)
  4. vars["source_file"] path to a JSON file

Config options (all optional):
  fail_under   pass threshold on the grounded share, default 1.0 (any flag fails)
  strict_rows  bool, flag values that belong to a different row
  derived      bool, accept one-step arithmetic (default true)
  aliases      {field: [words the text uses for it]}

Requires `pip install factguard` (or the git URL) in the Python promptfoo runs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from factguard import Checker


def _as_data(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise ValueError(f"source must be a JSON object, array or string, got {type(value).__name__}")


def _source(context: Dict[str, Any]) -> Any:
    config = context.get("config") or {}
    vars_ = context.get("vars") or {}
    if config.get("source") is not None:
        return _as_data(config["source"])
    if config.get("source_file"):
        return json.loads(Path(config["source_file"]).read_text())
    var = config.get("source_var", "source")
    if vars_.get(var) is not None:
        return _as_data(vars_[var])
    if vars_.get("source_file"):
        return json.loads(Path(str(vars_["source_file"])).read_text())
    raise ValueError("factguard: no source data. Set config.source, config.source_file, "
                     "vars.source or vars.source_file.")


def _bool(v: Any) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes")


def get_assert(output: str, context: Dict[str, Any]) -> Union[bool, float, Dict[str, Any]]:
    config = context.get("config") or {}
    try:
        data = _source(context)
    except Exception as e:  # report a setup problem as a failed assertion, not a crash
        return {"pass": False, "score": 0.0, "reason": f"factguard: could not load source data: {e}"}

    fail_under = float(config.get("fail_under", 1.0))
    checker = Checker(data,
                      aliases=config.get("aliases"),
                      derived=_bool(config.get("derived", True)),
                      strict_rows=_bool(config.get("strict_rows", False)))
    report = checker.check(output if isinstance(output, str) else json.dumps(output))

    grounded = len(report.verdicts) - len(report.flagged)
    head = (f"factguard: {grounded}/{len(report.verdicts)} numeric claims grounded "
            f"(score {report.score:.2f}, fail_under {fail_under:g})")
    details = []
    for v in report.flagged:
        if v.status == "contradicted":
            details.append(f"{v.claim}: data says {v.hint}")
        else:
            details.append(f"{v.claim}: not in data" + (f" (nearest {v.hint})" if v.hint else ""))
    return {
        "pass": report.score >= fail_under,
        "score": round(report.score, 4),
        "reason": head + ("; " + "; ".join(details) if details else ""),
        "namedScores": {"factguard": round(report.score, 4)},
    }
