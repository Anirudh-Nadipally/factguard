"""Score factguard on the INLG 2021 Shared Task on Evaluating Accuracy.

Data: https://github.com/ehudreiter/accuracySharedTask (Thomson & Reiter).
Neural systems wrote basketball game summaries from box scores; human experts
marked every factual error token by token. We run factguard on each summary
against its box score and compare the numbers it flags with the gold NUMBER
errors. No LLM is called.

    git clone --depth 1 https://github.com/ehudreiter/accuracySharedTask /tmp/ast
    python benchmarks/inlg2021/run.py /tmp/ast            # dev (60 texts) and test (30 texts)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from factguard import Checker

ALIASES = {
    "PTS": ["points", "point", "pts"],
    "REB": ["rebounds", "rebound", "boards", "board"],
    "AST": ["assists", "assist", "dimes"],
    "STL": ["steals", "steal"],
    "BLK": ["blocks", "block", "rejections"],
    "TO": ["turnovers", "turnover"],
    "TOV": ["turnovers", "turnover"],
    "MIN": ["minutes", "minute"],
    "FG3M": ["three", "threes", "pointers", "triples"],
    "FTM": ["free"],
    "PTS_QTR1": ["points", "point", "first", "1st", "quarter", "period"],
    "PTS_QTR2": ["points", "point", "second", "2nd", "quarter", "period"],
    "PTS_QTR3": ["points", "point", "third", "3rd", "quarter", "period"],
    "PTS_QTR4": ["points", "point", "fourth", "4th", "quarter", "period"],
    "WINS": ["wins", "victories"],
    "LOSSES": ["losses", "defeats"],
}


PAIRS = {
    "fg": ("FGM", "FGA"), "shooting": ("FGM", "FGA"), "field": ("FGM", "FGA"),
    "3pt": ("FG3M", "FG3A"), "three": ("FG3M", "FG3A"), "long": ("FG3M", "FG3A"), "beyond": ("FG3M", "FG3A"),
    "ft": ("FTM", "FTA"), "free": ("FTM", "FTA"), "charity": ("FTM", "FTA"),
}


def game_data(g: dict) -> dict:
    """Box score (columnar in the source) as records: two teams, then players."""
    teams = []
    for side in ("home", "vis"):
        line = g[f"{side}_line"]
        rec = {"name": line["TEAM-NAME"], "city": line["TEAM-CITY"],
               "full_name": f'{line["TEAM-CITY"]} {line["TEAM-NAME"]}'}
        for k, v in line.items():
            if k not in ("TEAM-NAME", "TEAM-CITY"):
                rec[k.replace("TEAM-", "")] = v
        teams.append(rec)
    bs = g["box_score"]
    players = []
    for idx, name in bs["PLAYER_NAME"].items():
        rec = {"name": name}
        for col, vals in bs.items():
            if col in ("PLAYER_NAME", "FIRST_NAME", "SECOND_NAME", "TEAM_CITY", "START_POSITION"):
                continue
            v = vals.get(idx)
            if v not in (None, "N/A"):
                rec[col] = v
        players.append(rec)
    return {"teams": teams, "players": players}


def token_spans(text: str) -> list[tuple[int, int]]:
    """Char span of each token; the shared task numbers tokens from 1 on single spaces."""
    spans, pos = [], 0
    for tok in text.split(" "):
        spans.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    return spans


def tokens_of(span: tuple[int, int], spans: list[tuple[int, int]]) -> set[int]:
    a, b = span
    return {i + 1 for i, (s, e) in enumerate(spans) if s < b and a < e}


def run(split_dir: Path, games_jsonl: Path, gsml: Path, prefix: str) -> dict:
    games = {}
    for line in games_jsonl.open():
        g = json.loads(line)
        games[g["shared_task_text_id"]] = g
    gold: dict[str, list[dict]] = {}
    for row in csv.DictReader(gsml.open()):
        gold.setdefault(row["TEXT_ID"].replace(".txt", ""), []).append(row)

    res = {"texts": 0, "claims": 0, "gold_number": 0, "found": {"all": 0, "contradicted": 0},
           "flags": {"all": 0, "contradicted": 0}, "flag_hits_number": {"all": 0, "contradicted": 0},
           "flag_hits_any": {"all": 0, "contradicted": 0}}
    examples = []
    for path in sorted((split_dir / "texts").glob(f"{prefix}*.txt")):
        tid = path.stem
        if tid not in games:
            continue
        text = path.read_text().strip()
        spans = token_spans(text)
        checker = Checker(game_data(games[tid]), aliases=ALIASES, pairs=PAIRS)
        report = checker.check(text)
        res["texts"] += 1
        res["claims"] += len(report.verdicts)

        g_rows = gold.get(tid, [])
        g_num = [set(range(int(r["DOC_TOKEN_START"]), int(r["DOC_TOKEN_END"]) + 1))
                 for r in g_rows if r["TYPE"] == "NUMBER"]
        g_any = [set(range(int(r["DOC_TOKEN_START"]), int(r["DOC_TOKEN_END"]) + 1)) for r in g_rows]
        res["gold_number"] += len(g_num)

        flagged = {"all": [], "contradicted": []}
        for v in report.verdicts:
            if v.status in ("contradicted", "unsupported"):
                toks = tokens_of((v.start, v.end), spans)
                flagged["all"].append(toks)
                if v.status == "contradicted":
                    flagged["contradicted"].append(toks)
                if len(examples) < 12 and v.status == "contradicted":
                    hit = any(toks & gt for gt in g_num)
                    examples.append(f"{tid} {'HIT ' if hit else 'MISS'} '{v.claim}' in: {v.sentence[:110]} -> {v.hint}")
        for mode in ("all", "contradicted"):
            fl = flagged[mode]
            res["flags"][mode] += len(fl)
            res["flag_hits_number"][mode] += sum(1 for t in fl if any(t & gt for gt in g_num))
            res["flag_hits_any"][mode] += sum(1 for t in fl if any(t & gt for gt in g_any))
            res["found"][mode] += sum(1 for gt in g_num if any(t & gt for t in fl))
    out = {"texts": res["texts"], "numeric_claims": res["claims"], "gold_number_errors": res["gold_number"]}
    for mode in ("all", "contradicted"):
        f, n = res["flags"][mode], res["gold_number"]
        out[mode] = {
            "flags": f,
            "recall": round(res["found"][mode] / n, 3) if n else None,
            "precision_number": round(res["flag_hits_number"][mode] / f, 3) if f else None,
            "precision_any_error": round(res["flag_hits_any"][mode] / f, 3) if f else None,
        }
    out["examples"] = examples
    return out


def main(root: str) -> None:
    r = Path(root)
    dev = run(r, r / "shared_task.jsonl", r / "gsml.csv", "S")
    test = run(r / "test_set", r / "test_set" / "shared_task.jsonl", r / "test_set" / "gsml.csv", "T")
    print(json.dumps({"dev": dev, "test": test}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ast")
