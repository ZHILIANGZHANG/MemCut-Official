<p align="center">
  <img src="assets/memcut_logo.png" alt="MemCut" width="260">
</p>

<h3 align="center">Every Secret Leaves a Trail in the Long-Term Memory of LLM Agents</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2610.21405"><img src="https://img.shields.io/badge/arXiv-2610.21405-b31b1b.svg" alt="arXiv"></a>
  <a href="dataset/LICENSE"><img src="https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg" alt="Data: CC BY 4.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue.svg" alt="Code: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11-3776ab.svg" alt="Python 3.11">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.21405">Paper</a> ·
  <a href="dataset/README.md">Dataset</a> ·
  <a href="toolkit/README.md">Toolkit</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="#citation">Citation</a>
</p>

---

**MemCut** is a benchmark for user-requested forgetting in the long-term memory of LLM agents. A user asks the
assistant to forget one fact; MemCut measures whether that fact can still be recovered from what the memory holds,
including from ordinary messages that never state it.

| Episodes | Sessions | Messages | Protected facts | Derivation types |
|:---:|:---:|:---:|:---:|:---:|
| 15 | 490 | 43,402 | 456 | 4 |

Each fact comes with its disclosure, the evidence messages it can be derived from, its minimal sufficient evidence
sets and minimum cuts, look-alike distractors and a five-option readout question. The four derivation types are factual composition, relational linking, aggregation
and temporal state.

## What MemCut measures

- **Native recovery**: the system answers through its own retrieval, as a user of the assistant would see it.
- **Residual recovery**: a reader answers from every surviving object derived from the fact's evidence, i.e. what the
  memory still supports.
- **Hidden-risk share**: (residual − native) / residual, the part of what the memory still supports that its own
  retrieval does not surface.

Deletion runs through three reference baselines: **S0** deletes the disclosure, **S1** adds retrieval-ranked cleanup
and **S2** adds LLM-guided cleanup over the same candidate pool. The toolkit runs the five memory designs,
**Turn RAG**, **Session RAG**, **MemoryBank**, **Nemori** and **LightMem**, each through its own upstream code. Turn RAG
and Session RAG work with either `e5-small-v2` or `all-MiniLM-L6-v2` as the retrieval encoder.

## Repository layout

```
MemCut-Official/
├── dataset/
│   ├── memcut_v1.zip      dialogues, facts, annotations, minimal sufficient sets
│   ├── README.md          data card
│   ├── SCHEMA.md          field-by-field format
│   ├── validate.py        integrity check for the bundle
│   └── case_study/        overlay for the case-study episode
├── toolkit/               evaluation code, system adapters, tests and guides
└── assets/
```

## Quick start

**1. Check the data** (Python 3, no dependencies)

```bash
python3 dataset/validate.py dataset/memcut_v1.zip
```

**2. Install the toolkit and run its offline tests** (no API key, no model weights)

```bash
cd toolkit
python3.11 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-dev.txt
export TIKTOKEN_CACHE_DIR="$PWD/runtime/tiktoken"
python tools/download_assets.py coverage
python -m pytest -c pytest.ini tests -q
```

**3. Import the data and run a system**

```bash
python tools/import_dataset.py ../dataset/memcut_v1.zip --output data/bundle
python run.py aligned prepare --system turnrag --output outputs/turnrag-ep01 \
  --source data/bundle/data/EP01/public/EP01.json \
  --contracts data/bundle/data/EP01/author/CLAIM_CONTRACTS.json \
  --annotations data/bundle/data/EP01/author/ANNOTATIONS.json
python run.py aligned run --run outputs/turnrag-ep01        # calls the model API; key from the environment
```

Each system runs in its own environment with its own upstream code and assets; see
[toolkit/docs/USAGE.md](toolkit/docs/USAGE.md) for the setup of each one.

## Documentation

| I want to | Guide |
|---|---|
| Install and run a memory system | [toolkit/docs/USAGE.md](toolkit/docs/USAGE.md) |
| Evaluate my own memory system | [toolkit/docs/ADDING_A_SYSTEM.md](toolkit/docs/ADDING_A_SYSTEM.md) |
| Add my own facts to an episode | [toolkit/docs/EXTENDING_DATA.md](toolkit/docs/EXTENDING_DATA.md) |
| Read the full protocol | [toolkit/PROTOCOL.md](toolkit/PROTOCOL.md) |
| Understand the data format | [dataset/README.md](dataset/README.md) · [dataset/SCHEMA.md](dataset/SCHEMA.md) |
| Contribute | [toolkit/CONTRIBUTING.md](toolkit/CONTRIBUTING.md) |

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

Data in `dataset/` is released under [CC BY 4.0](dataset/LICENSE). Code in `toolkit/` and `dataset/validate.py` is
released under the [MIT License](LICENSE). The memory systems under test are fetched from their own repositories
at setup and keep their own licenses ([toolkit/NOTICE](toolkit/NOTICE)).
