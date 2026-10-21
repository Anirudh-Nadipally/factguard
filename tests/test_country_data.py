from factguard import check


def test_trillion_scale_grounds():
    data = {"countries": [{"country": "United States", "gdp_usd": 27811500000000}]}
    r = check("The United States economy was $27.8 trillion in 2023.", data)
    assert r.score == 1.0, r


def test_trillion_wrong_value_flagged():
    data = {"countries": [{"country": "United States", "gdp_usd": 27811500000000}]}
    r = check("The United States economy was $31.2 trillion in 2023.", data)
    assert r.score < 1.0, r


def test_two_letter_country_name_scopes():
    data = {"countries": [
        {"country": "United States", "alt": "US", "gdp_per_capita_usd": 82587},
        {"country": "India", "gdp_per_capita_usd": 2434},
    ]}
    r = check("India trails far behind. US GDP per capita was $82,587.", data)
    assert r.score == 1.0, r


def test_object_in_same_sentence_beats_previous_sentence():
    data = {"countries": [
        {"country": "United States", "alt": "US", "unemployment_pct": 3.6},
        {"country": "United Kingdom", "alt": "UK", "unemployment_pct": 4.0},
    ]}
    r = check("The US rate was 3.6%. In the UK, unemployment was 4.0%.", data)
    assert r.score == 1.0, r
