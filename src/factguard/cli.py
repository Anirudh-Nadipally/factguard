"""factguard check --data facts.json --text report.txt"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import Checker


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="factguard",
                                description="Check every number in a text against the JSON it was written from.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="check one text")
    c.add_argument("--data", required=True, help="JSON file the text was written from")
    c.add_argument("--text", default="-", help="text file to check (default: stdin)")
    c.add_argument("--aliases", help="JSON file: {field: [words the text uses for it]}")
    c.add_argument("--json", action="store_true", help="print the report as JSON")
    c.add_argument("--fail-under", type=float, default=None,
                   help="exit 1 if the grounded share is below this (e.g. 1.0 in CI)")
    c.add_argument("--strict-rows", action="store_true",
                   help="flag a value that belongs to a different row than the one the sentence names")
    c.add_argument("--no-derived", action="store_true", help="do not accept sums, differences, ratios")
    a = p.parse_args(argv)

    data = json.loads(Path(a.data).read_text())
    text = sys.stdin.read() if a.text == "-" else Path(a.text).read_text()
    aliases = json.loads(Path(a.aliases).read_text()) if a.aliases else None
    report = Checker(data, aliases=aliases, derived=not a.no_derived, strict_rows=a.strict_rows).check(text)
    print(json.dumps(report.to_dict(), indent=2) if a.json else report)
    if a.fail_under is not None and report.score < a.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
