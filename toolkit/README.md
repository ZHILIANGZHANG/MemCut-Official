# MemCut evaluation toolkit

Reference evaluation code for **MemCut**, a benchmark for user-requested forgetting in the long-term memory of LLM
agents: a user asks the assistant to forget one fact, and the question is whether the fact is still recoverable from
what the memory holds. It runs the benchmark's protocol on the data bundle in `../dataset/`.

## What it does

For every claim of an episode: build the system's memory from the dialogue, delete the disclosure turns (**S0**),
then delete ten further source turns chosen from a 20-turn candidate pool by the system's own retrieval ranking
(**S1**) or by the write model scoring each candidate's relevance to the request (**S2**). After each deletion it
reads out **native** recovery (the system's own retrieval and context) and **residual** recovery (everything the
evidence and disclosure turns still derive), both answered by the same model; a claim is recovered when the target
is chosen under both fixed display orders (2/2). The hidden-risk share is (residual − native) / residual.

The five systems run through their own upstream code at pinned commits: **Turn RAG**, **Session RAG**,
**MemoryBank**, **Nemori** and **LightMem**, all from one entry point, `run.py aligned`, with the same failure
handling and the same ledger of calls and tokens. Turn RAG and Session RAG take either retrieval encoder,
`e5-small-v2` (default) or `all-MiniLM-L6-v2`.

## Quick check (no model calls)

Python 3.11 on Linux or macOS:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-dev.txt
export TIKTOKEN_CACHE_DIR="$PWD/runtime/tiktoken"
python tools/download_assets.py coverage      # fixed tokenizer files, downloaded once
python -m pytest -c pytest.ini tests -q
```

The tests use small synthetic data and recorded replies; they need no API key, upstream checkout or model weights.

## Running a system

```bash
python tools/import_dataset.py ../dataset/memcut_v1.zip --output data/bundle
python run.py aligned prepare --system turnrag --output outputs/turnrag-ep01 \
  --source data/bundle/data/EP01/public/EP01.json \
  --contracts data/bundle/data/EP01/author/CLAIM_CONTRACTS.json \
  --annotations data/bundle/data/EP01/author/ANNOTATIONS.json \
  --model YOUR_MODEL_ID --endpoint https://YOUR_PROVIDER/v1
python run.py aligned run --run outputs/turnrag-ep01        # calls the model API; key from MEMCUT_MODEL_API_KEY
```

Each system needs its own environment and upstream checkout; see [docs/USAGE.md](docs/USAGE.md).

## Documents

| Topic | Document |
|---|---|
| Install, import the bundle, prepare and run | [docs/USAGE.md](docs/USAGE.md) |
| The five systems: what they store, provenance, retrieval depths | [docs/SYSTEMS.md](docs/SYSTEMS.md) |
| Scoring rules in one page | [docs/PROTOCOL.md](docs/PROTOCOL.md) |
| The full specification | [PROTOCOL.md](PROTOCOL.md) |
| Evaluating your own memory system | [docs/ADDING_A_SYSTEM.md](docs/ADDING_A_SYSTEM.md), starting from [examples/keyword_bank.py](examples/keyword_bank.py) |
| Adding facts to an episode | [docs/EXTENDING_DATA.md](docs/EXTENDING_DATA.md) |
| Layout of the package | [docs/STRUCTURE.md](docs/STRUCTURE.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Scope

Source, configs, tests and docs only. The systems under test are cloned at setup rather than vendored (see
`NOTICE`); models, virtual environments, upstream checkouts, memory banks, replies, results and keys are never
committed. Every PLAN records the code identity it was prepared with and is never resumed under a different one.

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

## License

MIT (see `LICENSE`). The data bundle is CC BY 4.0.
