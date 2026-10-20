from factguard import extract_claims


def texts(t):
    return [c.text for c in extract_claims(t)]


def test_units_and_scales():
    cs = extract_claims("Revenue hit $1.2M, conversion 3.8%, 3x growth, Rs 5 lakh, 1,200 users.")
    got = {c.text: (c.value, c.unit) for c in cs}
    assert got["$1.2M"] == (1.2e6, "cur")
    assert got["3.8%"] == (3.8, "%")
    assert got["3x"] == (3.0, "x")
    assert got["Rs 5 lakh"] == (5e5, "cur")
    assert got["1,200"] == (1200.0, "")


def test_skips_formatting_numbers():
    t = "Released 2026-10-20 at 10:30 on v3.1.4 using gpt4o and an H100.\n1. First step\n2) Second"
    assert texts(t) == []


def test_ranges_are_two_claims_not_a_negative():
    cs = extract_claims("They won 113-107 and fell -5% after.")
    assert [(c.text, c.value) for c in cs] == [("113", 113.0), ("107", 107.0), ("-5%", -5.0)]


def test_number_words_but_not_fixed_phrases():
    t = "He had six assists, a three-point play and reached double figures; one of them."
    assert texts(t) == ["six"]


def test_years_ignored_by_default():
    assert texts("In 2025 sales were 40.") == ["40"]
    assert "2025" in [c.text for c in extract_claims("In 2025 sales were 40.", ignore_years=False)]
