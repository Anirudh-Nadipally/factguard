"""Example: `pip install deepeval factguard`, then `deepeval test run test_factguard_metric.py`
(plain `pytest` works too). Runs offline: the metric calls no LLM."""
import json
from pathlib import Path

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from factguard_metric import FactGuardMetric

STORE = json.loads((Path(__file__).parents[2] / "examples" / "store.json").read_text())


def test_summary_numbers_are_grounded():
    answer = "Acme Prints closed the month with 42 orders and a 4.7 rating. Conversion sits at 3.8%."
    assert_test(LLMTestCase(input="Summarise the month", actual_output=answer),
                [FactGuardMetric(data=STORE, threshold=1.0)])
