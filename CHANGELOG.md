# Changelog

## Unreleased
- Named versions (`version 2.1`, `Python 3.12`, `release 4`) are no longer read as numeric claims. Benchmark unchanged.
- Published to PyPI as `factguard-check` (import name stays `factguard`); the name `factguard` belongs to another project.
- Fix: a field word earlier in the clause no longer overrides an exact match on another field of the same record ("sold 30 units at $12.50" flagged the price). Benchmark unchanged.
- GitHub Action (`action.yml`): `uses: Anirudh-Nadipally/factguard@v0.2` with `text-file`, `source-file`, `fail-under`, `strict-rows`; outputs `score` and `flagged`, writes a job summary. Dogfooded in CI on `examples/`.
- promptfoo python assertion (`integrations/promptfoo`) and DeepEval `BaseMetric` (`integrations/deepeval`, deepeval imported lazily). Core stays zero-dependency.

## 0.2.0 (2026-10-21)
- Wrong-row detection: a correct number attached to the wrong entity is flagged, and the report names the row the value actually belongs to (reported by a user on X).
- Compound number words are read as one number (`seventy-five` -> 75).
- Opt-in `strict_rows` / `--strict-rows`: once a sentence names a row, a value found only on another row is contradicted even without a metric word. Off by default. On INLG 2021 test: off R 0.406 / P 0.611, on R 0.433 / P 0.497.
- Benchmark results regenerated; CI checks for drift on every push.

## 0.1.0 (2026-10-20)
- First release: deterministic numeric-claim checker for LLM text against source JSON, entity and field scoping, tolerance for written precision, hedges, gated derivations, aliases, CLI with `--fail-under`, `assert_grounded`, INLG 2021 benchmark.
