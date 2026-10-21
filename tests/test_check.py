import json
import subprocess
import sys
from pathlib import Path

import pytest

from factguard import Checker, assert_grounded, check

STORE = {
    "store": {"name": "Acme", "orders": 42, "prev_orders": 21, "rating": 4.68,
              "conversion_rate": 0.038, "revenue": 1234567},
    "items": [{"name": "Holo Sticker", "price": 12.5, "units": 30},
              {"name": "Magnet", "price": 7.5, "units": 12}],
}
PLAYERS = {
    "players": [{"name": "Isaiah Thomas", "points": 38, "assists": 4, "rebounds": 2},
                {"name": "Al Horford", "points": 11, "assists": 8, "rebounds": 8}],
    "team": {"name": "Celtics", "points": 113, "wins": 32, "losses": 18},
}


def status(report, claim):
    return next(v.status for v in report.verdicts if v.claim == claim)


def test_exact_and_rounded_numbers_are_supported():
    r = check("Acme has 42 orders and a 4.7 rating.", STORE)
    assert r.ok and r.score == 1.0


def test_percent_matches_a_stored_fraction():
    assert status(check("Conversion is 3.8%.", STORE), "3.8%") == "supported"


def test_scaled_number_uses_written_precision():
    assert status(check("Revenue is $1.2M.", STORE), "$1.2M") == "supported"
    assert status(check("Revenue is $1.5M.", STORE), "$1.5M") != "supported"


def test_hedge_widens_tolerance():
    assert status(check("Revenue is nearly $1.3M.", STORE), "$1.3M") == "supported"
    assert status(check("Revenue is $1.3M.", STORE), "$1.3M") != "supported"


def test_invented_number_is_flagged():
    r = check("Orders grew 25% week over week.", STORE)
    assert not r.ok and r.flagged[0].claim == "25%"


def test_misattributed_number_is_contradicted():
    # 6 is nowhere near Thomas's assists (4); a loose check would pass it via other fields
    r = check("Isaiah Thomas scored 38 points with 8 assists.", PLAYERS)
    v = next(v for v in r.verdicts if v.claim == "8")
    assert v.status == "contradicted" and "assists=4" in v.hint


def test_pronoun_carries_entity_from_previous_sentence():
    r = check("Al Horford scored 11 points. He also had 3 rebounds.", PLAYERS)
    assert status(r, "3") == "contradicted"


def test_object_of_preposition_is_not_the_subject():
    r = check("Isaiah Thomas scored 38 points against the Celtics.", PLAYERS)
    assert status(r, "38") == "supported"


def test_derived_values_need_an_arithmetic_cue():
    assert status(check("Thomas scored 27 more points than Horford.", PLAYERS), "27") == "derived"
    assert status(check("Average item price is $10.", STORE), "$10") == "derived"
    assert status(check("Orders doubled from 21.", STORE), "doubled") == "derived"
    # same value, no cue: arithmetic must not explain it away
    assert check("Isaiah Thomas had 27 points.", PLAYERS).flagged


def test_multiplier_only_matches_ratios():
    assert status(check("Orders are 2x prev orders.", STORE), "2x") == "derived"
    assert status(check("Orders are 3x prev orders.", STORE), "3x") == "unsupported"


def test_list_length_supports_counts():
    assert status(check("Acme sells two items.", STORE), "two") == "supported"


def test_aliases_and_articles():
    data = {"players": [{"name": "Al Horford", "STL": 0, "BLK": 1}]}
    c = Checker(data, aliases={"STL": ["steals", "steal"], "BLK": ["blocks", "block"]})
    r = c.check("Al Horford added a steal and a block.")
    assert [(v.claim, v.status) for v in r.verdicts] == [("a", "contradicted"), ("a", "supported")]


def test_assert_grounded_raises_with_report():
    with pytest.raises(AssertionError, match="25%"):
        assert_grounded("Orders grew 25%.", STORE)
    assert_grounded("Acme has 42 orders.", STORE)


def test_report_json_roundtrip():
    d = check("Acme has 42 orders and 99 returns.", STORE).to_dict()
    json.dumps(d)
    assert d["claims"] == 2 and d["unsupported"] + d["contradicted"] == 1


def test_cli_fail_under(tmp_path: Path):
    data = tmp_path / "d.json"
    data.write_text(json.dumps(STORE))
    txt = tmp_path / "t.txt"
    txt.write_text("Acme has 42 orders and 99 returns.")
    cmd = [sys.executable, "-m", "factguard.cli", "check", "--data", str(data), "--text", str(txt)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    assert subprocess.run(cmd + ["--fail-under", "1.0"], capture_output=True).returncode == 1


def test_wrong_row_is_flagged_and_named():
    from factguard import check
    data = {"stores": [{"name": "Mumbai", "revenue": 42000, "rating": 4.6},
                       {"name": "Delhi", "revenue": 38500, "rating": 4.1},
                       {"name": "Pune", "revenue": 12000, "rating": 3.8}]}
    r = check("Mumbai's revenue was 42,000. Delhi's revenue was 42,000. "
              "Pune's rating slipped to 3.5.", data)
    bad = [v for v in r.verdicts if v.status == "contradicted"]
    assert [v.claim for v in bad] == ["42,000", "3.5"]
    assert "stores[1].revenue=38500" in bad[0].hint and "stores[0].revenue" in bad[0].hint
    assert bad[1].hint.startswith("stores[2].rating")


def test_strict_rows_and_compound_number_words():
    data = {"products": [{"name": "Breeze Fan", "price": 89.0, "stock": 75},
                         {"name": "Lamp", "price": 49.99, "stock": 12}]}
    text = "Breeze Fan costs $49.99. The Lamp has seventy-five in stock."
    assert status(check(text, data), "$49.99") == "supported"          # default: loose
    r = Checker(data, strict_rows=True).check(text)
    assert status(r, "$49.99") == "contradicted"
    assert status(r, "seventy-five") == "contradicted"
    assert next(v.value for v in r.verdicts if v.claim == "seventy-five") == 75


def test_earlier_field_word_does_not_steal_a_same_row_value():
    data = {"products": [{"name": "Holo Sticker", "price": 12.5, "units_sold": 30}]}
    r = check("Holo Sticker sold 30 units at $12.50.", data)
    assert status(r, "$12.50") == "supported"
    assert status(r, "30") == "supported"
