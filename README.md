# factguard

**Catch an LLM inventing or misattributing numbers when it writes from your data.**
Deterministic, no LLM judge, no API key, no dependencies. Runs in milliseconds, so it fits in CI and in the request path.

```python
from factguard import check

report = check(llm_summary, source_json)
report.ok          # False
print(report)
```

```
factguard: 8/10 numeric claims grounded (score 0.80)   # examples/, abridged
  ok  42           store.orders
  ok  4.7          store.rating
  ok  3.8%         store.conversion_rate
  ok  $1.2M        store.revenue_usd
  ok  $12.50       products[0].price
  XX  15             <- data says products[1].units_sold=12
  ok* $10          mean(products[*].price)
  ??  25%            <- not in data; did it mean store.orders=42?
```

`ok` = the number is in the data. `ok*` = it can be computed from the data (sum, mean, difference, ratio, % change), and the text signals the arithmetic ("more than", "average", "doubled"). `XX` = it's attached to a field that holds a different value. `??` = nothing in the data supports it.

## Why

Products that turn data into prose (analytics summaries, financial briefs, store reports, sports recaps, dashboards with an "explain" button) fail in one specific way: the model writes a number that isn't in the data, or puts a real number on the wrong thing. Most faithfulness checks ask another LLM to judge. That's slow, costs money, gives a different answer each run, and is hard to put in CI.

factguard checks numbers the way a careful editor would:

1. **Extract** every numeric claim: digits, currency, %, multipliers (`3x`), scales (`$1.2M`, `5 lakh`), number words (`six`), and counts written as articles (`a steal`). It skips dates, times, versions, list markers and model names.
2. **Scope** each number: which record it's about (the nearest name before it, skipping objects like "for the Celtics", with pronouns carried from the previous sentence) and which field (the noun after it, matched through field names and your aliases).
3. **Verify** the number at the precision it was written: `4.7` covers 4.65 to 4.75, `$1.2M` covers 1.15M to 1.25M, and a hedge ("about", "nearly") widens it to 10%. Percentages also match stored fractions.
4. **Derive** only when the text signals arithmetic, and only between related fields, so a stray number can't be "explained" by an arbitrary pair out of hundreds.

## Benchmark: INLG 2021 Accuracy Shared Task

The [Shared Task on Evaluating Accuracy](https://github.com/ehudreiter/accuracySharedTask) (Thomson & Reiter) has 90 basketball summaries written by three neural data-to-text systems. Trained annotators marked every factual error token by token. Every tuning decision came from errors on the 60 training texts; the 30 test texts were only scored, never inspected. The checker sees only the game's box score.

Incorrect **numbers**, test set (30 texts, 224 gold number errors), mistake-level:

| System | Approach | Recall | Precision |
|---|---|---|---|
| Laval University | human + tools (hybrid) | 0.862 | 0.881 |
| Charles-UPF | trained language model | 0.777 | 0.750 |
| NIJL | rules + trained classifier | 0.696 | 0.419 |
| **factguard 0.1** | **rules, no model, no training** | **0.406** | **0.611** |
| Laval University | rules + classifiers (metric only) | 0.442 | 0.351 |
| Eurecom | claim verification | 0.205 | 0.329 |

Published rows are from the shared task results paper ([Thomson & Reiter 2021](https://arxiv.org/abs/2108.05644), Tables 4-8). Training set: recall 0.445, precision 0.781 (60 texts, 474 number errors).

What this says honestly: a zero-model checker gets about 6 in 10 of its flags right and finds about 4 in 10 number errors. That's useful as a cheap first gate in CI, and it explains every flag. A trained LM checker still finds more. Most of the misses are numbers the system computed (half-time scores from quarter scores), numbers about prior games that aren't in the box score, and rows where the model got the player's name wrong rather than the stat.

Caveats: my scoring counts a flag as correct when it overlaps a gold NUMBER error token. The shared task's own script matches the category too, which is equivalent here because factguard only flags numbers. The sports aliases and the made-attempted pairs ("7-15 FG") are config in `benchmarks/inlg2021/run.py`, not built into the library.

```bash
git clone --depth 1 https://github.com/ehudreiter/accuracySharedTask /tmp/ast
python benchmarks/inlg2021/run.py /tmp/ast
```

CI re-runs this and fails if the numbers drift from `benchmarks/inlg2021/results.json`.

## Use

```bash
pip install git+https://github.com/Anirudh-Nadipally/factguard
factguard check --data facts.json --text summary.txt
factguard check --data facts.json --text summary.txt --fail-under 1.0   # exit 1 on any flag
```

In tests:

```python
from factguard import assert_grounded

def test_weekly_report_has_no_invented_numbers():
    data = load_fixture("week_42.json")
    assert_grounded(generate_report(data), data)   # AssertionError lists each bad number
```

When your field names don't match the words in the text, give aliases. When a pair like "7-15" maps to two fields, give pairs:

```python
from factguard import Checker

checker = Checker(box_score,
                  aliases={"PTS": ["points"], "AST": ["assists", "assist"]},
                  pairs={"fg": ("FGM", "FGA")})
report = checker.check(summary)
report.flagged        # contradicted + unsupported, each with .hint = "path=value"
report.to_dict()      # JSON for logs and dashboards
```

## Limits

- It checks numbers, not names or words. "LeBron scored 30" passes when someone else scored 30 and the check scopes to that player's record.
- Entity and field scoping is heuristic, so it's English-first and works best when the data uses readable names.
- Computed claims are accepted only for one step of arithmetic (a sum, mean, difference, ratio or % change).
- No claims about anything outside the data you pass. "Third straight win" is unsupported unless the data has it.

MIT licensed.
