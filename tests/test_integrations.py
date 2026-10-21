"""Integrations: promptfoo assertion (mock context) and the DeepEval metric."""
import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STORE_FILE = ROOT / "examples" / "store.json"
STORE = json.loads(STORE_FILE.read_text())
REPORT = (ROOT / "examples" / "report.txt").read_text()
GOOD = "Acme Prints closed the month with 42 orders and a 4.7 rating. Conversion sits at 3.8%."


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pf = _load("factguard_assert", "integrations/promptfoo/factguard_assert.py")


def ctx(vars=None, config=None):
    # Shape of promptfoo's AssertionValueFunctionContext (the keys this assertion reads)
    return {"prompt": "Summarise", "vars": vars or {}, "test": {}, "config": config, "provider": None,
            "providerResponse": None, "logProbs": None, "trace": None, "metadata": None}


def test_promptfoo_passes_grounded_text_from_json_string_var():
    r = pf.get_assert(GOOD, ctx(vars={"source": json.dumps(STORE)}))
    assert r["pass"] is True and r["score"] == 1.0 and r["namedScores"] == {"factguard": 1.0}


def test_promptfoo_fails_and_explains_bad_numbers():
    r = pf.get_assert(REPORT, ctx(vars={"source": STORE}, config={"fail_under": 1.0}))
    assert r["pass"] is False and 0 < r["score"] < 1
    assert "15: data says products[1].units_sold=12" in r["reason"]
    assert "25%: not in data" in r["reason"]


def test_promptfoo_threshold_and_source_file_config():
    r = pf.get_assert(REPORT, ctx(config={"source_file": str(STORE_FILE), "fail_under": "0.5",
                                          "strict_rows": "true"}))
    assert r["pass"] is True


def test_promptfoo_missing_source_is_a_failed_assertion_not_a_crash():
    r = pf.get_assert(GOOD, ctx())
    assert r["pass"] is False and "no source data" in r["reason"]


def test_deepeval_integration_is_lazy():
    code = ("import sys; sys.path.insert(0, %r); import factguard, factguard_metric; "
            "assert 'deepeval' not in sys.modules" % str(ROOT / "integrations" / "deepeval"))
    subprocess.run([sys.executable, "-c", code], check=True)


def test_deepeval_metric():
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase
    sys.path.insert(0, str(ROOT / "integrations" / "deepeval"))
    from factguard_metric import FactGuardMetric

    m = FactGuardMetric(data=STORE, threshold=1.0)
    assert m.measure(LLMTestCase(input="q", actual_output=GOOD)) == 1.0 and m.is_successful()

    m = FactGuardMetric(data=STORE_FILE, threshold=1.0)   # Path source
    score = asyncio.run(m.a_measure(LLMTestCase(input="q", actual_output=REPORT)))
    assert score < 1.0 and not m.is_successful() and "15: data says" in m.reason

    m = FactGuardMetric(threshold=0.5)                   # source from metadata
    m.measure(LLMTestCase(input="q", actual_output=REPORT, metadata={"factguard_source": STORE}))
    assert m.is_successful()

    m = FactGuardMetric()                                # source from context
    assert m.measure(LLMTestCase(input="q", actual_output=GOOD, context=[json.dumps(STORE)])) == 1.0

    m = FactGuardMetric(threshold=0.5, strict_mode=True)
    assert m.measure(LLMTestCase(input="q", actual_output=REPORT, context=[json.dumps(STORE)])) == 0.0

    with pytest.raises(ValueError):
        FactGuardMetric().measure(LLMTestCase(input="q", actual_output=GOOD))
    assert m.__name__ == "FactGuard Numeric Grounding"
