# Contributing

Thanks for your interest! This is primarily a portfolio project, but issues and
pull requests are welcome.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Before opening a pull request

```bash
ruff check .          # lint
pytest -q             # tests (network-free)
streamlit run app.py  # manual smoke test
```

Please keep the test suite green and network-free: tests must rely on the
committed snapshots under `market_data/`, `datasets/` and `models/`, never on a
live fetch. If you change a data schema, update the relevant snapshot builder in
`scripts/` and the tests together.

## Code style

- British/neutral English in prose; consistent naming in code.
- Keep preprocessing **inside** the pipeline to avoid leakage.
- Add a short docstring to every public function.
- Run `ruff check .` before pushing; CI runs lint + tests + a Docker build.

## Regenerating artifacts

| What | Command |
| --- | --- |
| US market snapshots | `python scripts/build_market_snapshot.py` |
| International snapshots | `python scripts/build_international_snapshot.py` |
| Real dataset (Ames) | `python scripts/fetch_real_dataset.py` |
| Real-data benchmark | `python scripts/benchmark_real_data.py` |
| README charts | `python scripts/build_images.py` |
| GitHub Pages site | `python scripts/build_site.py` |

## Reporting issues

Open a GitHub issue with steps to reproduce, expected vs. actual behaviour, and
your Python version. For data-source problems, note the source and date.
