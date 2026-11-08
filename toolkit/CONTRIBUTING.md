# Contributing

Contributions that make MemCut easier to run, extend or check are welcome: adapters for further memory systems,
new facts built with the overlay tool, fixes, and clearer documentation.

## Before you open a pull request

```bash
python -m pytest -c pytest.ini tests -q        # no key, network or model weights needed
```

- **A new memory system**: follow [docs/ADDING_A_SYSTEM.md](docs/ADDING_A_SYSTEM.md), starting from
  `examples/keyword_bank.py`, and add an offline test like `tests/test_example_bank.py`.
- **New facts**: follow [docs/EXTENDING_DATA.md](docs/EXTENDING_DATA.md); send the overlay, not a modified episode,
  and say which reader-model checks you ran.
- **A change to shared protocol code** (`memcut/source_anchored_v2/`, scoring, the candidate pool, the readouts):
  explain the effect on results. Such a change gives every entry point a new code identity, and earlier PLANs are
  not resumed under it.

## Rules the code relies on

- API keys come from environment variables only. Never put a key into an argument, a PLAN, a test or a committed
  file.
- Tests use synthetic data and recorded replies; they make no paid call.
- Do not commit upstream checkouts, model weights, memory banks, replies or results (see `.gitignore`).
- Upstream systems are cloned at pinned commits by `tools/setup_upstreams.py`; record new ones in `NOTICE` with
  their licenses.

## Reporting a problem

Include the entry point, the PLAN's `code` identity, the stage, and the error text. `turnrag`, `sessionrag`, `source` and
`simplemem` have a `doctor` command that checks the installation without a model call.
