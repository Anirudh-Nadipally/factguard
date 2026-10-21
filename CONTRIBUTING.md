# Contributing

Bug reports with a real (scrubbed) report plus its source data are the most useful thing you can send.

1. Open an issue with the text, the JSON, what factguard said, and what it should have said.
2. For a fix: add a failing test in `tests/`, make it pass, and run `python benchmarks/inlg2021/run.py /path/to/accuracySharedTask`. If benchmark numbers move, say so in the PR and update `benchmarks/inlg2021/results.json`.
3. Keep it zero-dependency.

Run tests: `pip install -e . pytest && pytest -q`
