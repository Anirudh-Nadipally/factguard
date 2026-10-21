# factguard for promptfoo

A [python assertion](https://www.promptfoo.dev/docs/configuration/expected-outputs/python/) that checks every number in the model output against the JSON the output was written from. Deterministic, no grader model.

```bash
pip install git+https://github.com/Anirudh-Nadipally/factguard   # into the python promptfoo runs
# copy factguard_assert.py next to your promptfooconfig.yaml
```

```yaml
tests:
  - vars:
      source: file://data/week_42.json     # the data the model writes from
    assert:
      - type: python
        value: file://factguard_assert.py
        config:
          fail_under: 1.0                  # default 1.0: any invented or misattributed number fails
          strict_rows: false
```

Returns a GradingResult: `pass`, `score` (share of numeric claims grounded), a `reason` listing each bad number with the value the data actually holds, and a `factguard` named score for the promptfoo UI.

Source data, first match wins: `config.source` (inline object or JSON string), `config.source_file` (path, relative to where you run promptfoo), `vars.source` (or the var named by `config.source_var`), `vars.source_file`. Other config: `aliases` (`{field: [words]}`), `derived` (default true).

If promptfoo's `python` isn't the one with factguard installed, set `PROMPTFOO_PYTHON=/path/to/python`.

`promptfooconfig.yaml` here is a runnable example using the offline `echo` provider: one grounded summary that passes, and `examples/report.txt` (wrong numbers) under `not-python`, so it passes because factguard fails it.

```bash
cd integrations/promptfoo && npx promptfoo@latest eval
```
