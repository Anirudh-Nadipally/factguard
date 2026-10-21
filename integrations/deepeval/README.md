# factguard for DeepEval

A [custom DeepEval metric](https://deepeval.com/docs/metrics-custom) (`BaseMetric`) that scores `actual_output` by the share of its numeric claims the source JSON supports. No LLM, no API key, same answer every run.

```bash
pip install deepeval git+https://github.com/Anirudh-Nadipally/factguard
# copy factguard_metric.py into your test folder
```

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from factguard_metric import FactGuardMetric

def test_weekly_report():
    data = load_fixture("week_42.json")
    case = LLMTestCase(input="Summarise the week", actual_output=generate_report(data))
    assert_test(case, [FactGuardMetric(data=data, threshold=1.0)])
```

Source data, first match wins: `data=` (object, JSON string, or `pathlib.Path`), `test_case.metadata["factguard_source"]`, or `test_case.context` when it is a single JSON string. Options: `threshold` (default 1.0), `strict_rows`, `derived`, `aliases`, `strict_mode` (score 0 below threshold), `include_reason`.

`metric.reason` lists each bad number with the value the data holds; `metric.report` is the full factguard `Report`.

deepeval is imported only when `FactGuardMetric` is first accessed, so factguard itself stays dependency-free.

Run the example: `deepeval test run test_factguard_metric.py` (or `pytest`).
