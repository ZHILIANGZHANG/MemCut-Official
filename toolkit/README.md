# MemCut evaluation toolkit

Reference evaluation code for **MemCut**, a benchmark for memory unlearning in long-term agent memory: a user asks
the assistant to forget one fact, and the question is whether the fact stays derivable from what the memory still
holds. The data bundle (`memcut_v1.zip`: 15 episodes, 43,402 messages, 456 deletion requests, plus the case-study
episode EP08A, with a data card, schema and validator) is released with this toolkit, which runs the protocol on it.

## Start here

| I want to | Read |
|---|---|
| see it work in five minutes, with no key | [Quick check](#quick-check-no-model-calls) below, then `tests/test_example_bank.py` |
| run a system on MemCut | [docs/USAGE.md](docs/USAGE.md) |
| reproduce a table or figure of the paper | [docs/REPRODUCE.md](docs/REPRODUCE.md) |
| evaluate my own memory system | [docs/ADDING_A_SYSTEM.md](docs/ADDING_A_SYSTEM.md), starting from [examples/keyword_bank.py](examples/keyword_bank.py) |
| add my own facts to an episode | [docs/EXTENDING_DATA.md](docs/EXTENDING_DATA.md) and `tools/extend_episode.py` |
| contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Documents

| Topic | Document |
|---|---|
| Install, import the bundle, prepare and run an experiment | [docs/USAGE.md](docs/USAGE.md) |
| Paper results and the commands that produce them | [docs/REPRODUCE.md](docs/REPRODUCE.md) |
| The adapter contract for a new memory system | [docs/ADDING_A_SYSTEM.md](docs/ADDING_A_SYSTEM.md) |
| Adding facts with overlays; the construction guide | [docs/EXTENDING_DATA.md](docs/EXTENDING_DATA.md) |
| Layout of the package, scripts, data and outputs | [docs/STRUCTURE.md](docs/STRUCTURE.md) |
| Deletion budget, 2/2 scoring, entry points and their variants | [docs/PROTOCOL.md](docs/PROTOCOL.md) |
| The full protocol (implementation spec v4) | [PROTOCOL.md](PROTOCOL.md) |
| The six systems' adapters and how they differ from their LoCoMo configurations | [docs/integrations/](docs/integrations/) |
| What was verified for the release | [docs/VERIFICATION.md](docs/VERIFICATION.md) |

## What it does

For every claim of an episode: build the system's memory from the dialogue, delete the disclosure turns (S0), let the
system run its own cleanup on the deletion request (S1: delete the top retrieved source messages; S2: one model call
picks which to delete), then read out **native** (the system's own retrieval and answer) and **residual** (an answer
from everything the evidence turns still derive). A target counts as recovered when the reader picks it under both
display orders (2/2). `aligned-retention` then asks unrelated questions the full memory answered, after the deletion.

Systems included, each through its own upstream code at a pinned commit: the five of the paper, **Turn RAG**
(raw-turn RAG), **Session RAG**, **MemoryBank**, **Nemori** and **LightMem**, and an additional adapter for
**SimpleMem**. All of them run through one entry point, `run.py aligned`, with the same failure handling and the
same ledger of calls and tokens. Turn RAG and Session RAG take either retrieval encoder, `e5-small-v2` (default) or
`all-MiniLM-L6-v2` (`--embedding`).

## Quick check (no model calls)

Python 3.11 on Linux or macOS:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-dev.txt
export TIKTOKEN_CACHE_DIR="$PWD/runtime/tiktoken"
python tools/download_assets.py coverage      # fixed tokenizer files, downloaded once
python -m pytest -c pytest.ini tests -q
python run.py --help
```

The tests use small synthetic data and recorded replies; they need no API key, upstream checkout or model weights.
In an offline environment, place the tokenizer files in `TIKTOKEN_CACHE_DIR` beforehand (`download_assets.py` names
them).

## Running the protocol

1. Import the data bundle (only manifest-listed files, hashes checked):
   `python tools/import_dataset.py /path/to/memcut_v1.zip --output data/bundle`
2. Create one environment per system family and fetch its upstream at the pinned commit
   (`tools/setup_upstreams.py <system>`); see `docs/USAGE.md` for the dependency files.
3. Point the toolkit at any OpenAI-compatible endpoint. The adapters under `memcut/model_adapters/` and the presets in
   `memcut/session_rag/transport.py` name a few providers as defaults; they are presets only. API keys are read
   from environment variables and never written to disk.
4. `python run.py aligned prepare ...`, `python run.py aligned run ...`, `python run.py aligned-retention --run DIR`.
   Each run directory records its plan, the protocol document's hash, every call and its cost, and resumes after
   an interruption. Native retrieval and each system's own QA settings are kept; the deletion pool of 20 is not
   imposed as a common QA k.

## Retention: `aligned-retention`

The `aligned` entry point does not measure retention itself. After the answer stage:

```bash
python run.py aligned-retention --run DIR                                  # default: S2@10, 5 probes per claim
python run.py aligned-retention --run DIR --conditions S2@10 S1@10 S0
```

Probe rule: the other planned claims of the same episode whose evidence and disclosure turns are disjoint from this
claim's, in a fixed hash order, the first five whose native readout on the full memory is correct 2/2; after the
deletion they are read out 2/2 through the system's own retrieval and answer stage, and an unavailable answer does
not score. Output: `RETENTION.json` and `retention_aligned/<claim>/<condition>.json`. Requests go through the run
directory's receipts, so a rerun sends only what is missing.

## Release scope

Source, configs, tests and docs only. The data bundle is imported on demand; models, virtual environments, upstream
checkouts, databases, replies, results and keys are never committed. Every PLAN records the code identity it was
prepared with, and a PLAN is never resumed under a different one. Default parameters and scoring variants are
whatever the PLAN each entry point writes says they are.

## Scope and provenance

The toolkit holds reusable S0/S1/S2 source and synthetic tests; no experiment outputs, model weights, databases,
API replies or credentials. It clones the systems under test at setup rather than vendoring them; see `NOTICE` for
their repositories and licenses.

## Citation

```bibtex
@misc{zhang2026memcut,
  title         = {MemCut: Every Secret Leaves a Trail in the Long-Term Memory of LLM Agents},
  author        = {Zhang, Cheng and Cao, Xuanning and Zheng, Zhicheng and Xiong, Hui},
  year          = {2026},
  eprint        = {2610.21405},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.21405}
}
```

`CITATION.cff` carries the same reference.

## License

MIT (see `LICENSE`). The data bundle is CC BY 4.0.
