<p align="center">
  <img src="assets/memcut_logo.png" alt="MemCut: LLM Memory Forgetting Benchmark" width="420">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.21405"><img src="https://img.shields.io/badge/arXiv-2610.21405-b31b1b.svg" alt="arXiv"></a>
  <a href="dataset/LICENSE"><img src="https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg" alt="Data license"></a>
  <a href="toolkit/LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue.svg" alt="Code license"></a>
</p>

# MemCut: Every Secret Leaves a Trail in the Long-Term Memory of LLM Agents

MemCut is a benchmark for **user-requested forgetting** in the long-term memory of LLM agents. A user asks the
assistant to forget one fact. The question is whether the fact is still recoverable from what the memory holds,
including from ordinary messages that never state it.

- **456 secrets** in 15 long, multi-session conversations (43,402 messages), each with its disclosure, annotated
  evidence messages, look-alike distractors and a five-option readout question.
- **Two recovery metrics**: *native* recovery through the system's own retrieval, and *residual* recovery from every
  surviving object derived from the evidence.
- **Reference deletion baselines** (S0: delete the disclosure; S1: retrieval-ranked cleanup; S2: LLM-guided cleanup)
  for Turn RAG, Session RAG, MemoryBank, Nemori, LightMem and SimpleMem, through their own upstream code.

## What is in this repository

```
dataset/
  memcut_v1.zip          the data bundle: dialogues, claims, annotations, minimal sufficient sets, validator
  README.md              the data card (also inside the zip)
  SCHEMA.md  LICENSE  validate.py
  case_study/            the overlay that defines the case-study episode EP08A
toolkit/                 the evaluation toolkit: protocol, adapters for six systems, tests, docs
assets/                  logo
```

## Quick start

Check the data (Python 3, no dependencies):

```bash
python3 dataset/validate.py dataset/memcut_v1.zip
```

Install the toolkit and run its offline tests (no API key, no model weights):

```bash
cd toolkit
python3.11 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-dev.txt
export TIKTOKEN_CACHE_DIR="$PWD/runtime/tiktoken"
python tools/download_assets.py coverage
python -m pytest -c pytest.ini tests -q
python tools/import_dataset.py ../dataset/memcut_v1.zip --output data/bundle
```

Then follow the toolkit's guides:

| I want to | Read |
|---|---|
| run a memory system on MemCut | [toolkit/docs/USAGE.md](toolkit/docs/USAGE.md) |
| reproduce a table or figure of the paper | [toolkit/docs/REPRODUCE.md](toolkit/docs/REPRODUCE.md) |
| evaluate my own memory system | [toolkit/docs/ADDING_A_SYSTEM.md](toolkit/docs/ADDING_A_SYSTEM.md) |
| add my own facts to an episode | [toolkit/docs/EXTENDING_DATA.md](toolkit/docs/EXTENDING_DATA.md) |
| read the full protocol | [toolkit/PROTOCOL.md](toolkit/PROTOCOL.md) |
| understand the data format | [dataset/README.md](dataset/README.md), [dataset/SCHEMA.md](dataset/SCHEMA.md) |

## Citation

```bibtex
@misc{memcut2026,
  title         = {MemCut: Every Secret Leaves a Trail in the Long-Term Memory of LLM Agents},
  year          = {2026},
  eprint        = {2610.21405},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.21405}
}
```

## License

The data in `dataset/` is released under CC BY 4.0 ([dataset/LICENSE](dataset/LICENSE)). The code in `toolkit/` and
`dataset/validate.py` are released under the MIT License ([LICENSE](LICENSE)). The memory systems under test are
fetched from their own repositories at setup and keep their own licenses ([toolkit/NOTICE](toolkit/NOTICE)).
