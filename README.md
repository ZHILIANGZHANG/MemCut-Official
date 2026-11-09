<p align="center">
  <img src="assets/memcut_logo.png" alt="MemCut" width="260">
</p>

<h3 align="center">Can your agent's memory really forget?</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2610.21405"><img src="https://img.shields.io/badge/arXiv-2610.21405-b31b1b.svg" alt="arXiv"></a>
  <a href="dataset/LICENSE"><img src="https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg" alt="Data: CC BY 4.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-blue.svg" alt="Code: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11-3776ab.svg" alt="Python 3.11">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#evaluate-your-own-memory-system">Your own system</a> ·
  <a href="#use-the-data-directly">Use the data directly</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="#citation">Citation</a>
</p>

---

**MemCut** tests whether an LLM agent's long-term memory can forget a fact when the user asks it to. Deleting the
message that states a secret is easy; MemCut checks whether the secret can still be **pieced together** from the
ordinary messages the memory keeps: a booking, a filing, a dinner invitation.

| Episodes | Messages | Protected facts | Evidence per fact | Minimal sufficient sets |
|:---:|:---:|:---:|:---:|:---:|
| 15 | 43,402 | 456 | 6–10 messages | 7,148 |

Each fact comes with the user's deletion request, the message that discloses it, the evidence it can be derived
from, look-alike distractors and a five-option question. Four kinds of derivation: factual composition, relational
linking, aggregation and temporal state.

## How a system is scored

```mermaid
flowchart LR
    A[Dialogue] --> B[Build the memory]
    B --> C["Delete<br/>S0: the disclosure<br/>S1 / S2: + 10 more turns"]
    C --> D["Native readout<br/>(the system's own retrieval)"]
    C --> E["Residual readout<br/>(everything still derived<br/>from the evidence)"]
    D --> F[Recovered?]
    E --> F
```

1. **Build** the system's memory from the full dialogue.
2. **Delete** for each fact on its own branch: **S0** removes the disclosure; **S1** also removes the 10 turns the
   system's own retrieval ranks highest for the request; **S2** lets the model choose those 10 from the same 20
   candidates. Deleting a turn removes everything the memory derived from it.
3. **Read out** twice. **Native**: the system answers the question through its own retrieval. **Residual**: the
   model answers from every surviving memory object derived from the evidence.
4. **Score**: a fact counts as recovered only if the right option is chosen under both display orders.

Lower recovery is better forgetting. The gap between the two readouts, (residual − native) / residual, is the
**hidden-risk share**: what the memory still supports but its own retrieval does not show.

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

**3. Evaluate a built-in system on one episode**

Turn RAG is the quickest: it builds its memory with a local encoder and no LLM calls.

```bash
python -m pip install -r requirements/encoder.txt
python tools/download_assets.py encoder
python tools/import_dataset.py ../dataset/memcut_v1.zip --output data/bundle

EP=data/bundle/data/EP01
python run.py aligned prepare --system turnrag --output outputs/turnrag-ep01 \
  --source $EP/public/EP01.json \
  --contracts $EP/author/CLAIM_CONTRACTS.json \
  --annotations $EP/author/ANNOTATIONS.json \
  --model YOUR_MODEL --endpoint https://YOUR_PROVIDER/v1

export MEMCUT_MODEL_API_KEY=...                       # read from the environment only
python run.py aligned run --run outputs/turnrag-ep01  # build → delete → answer → report; resumable
```

Any OpenAI-compatible endpoint works; add `--compatibility deepseek` or `mimo` for those providers. The built-in
systems are `turnrag`, `sessionrag`, `memorybank`, `nemori` and `lightmem`; each runs its own upstream code at a
pinned commit, in its own environment ([toolkit/docs/USAGE.md](toolkit/docs/USAGE.md)).

**4. Read the result**

`outputs/turnrag-ep01/SUMMARY.json` has one entry per readout and condition, for example (numbers illustrative):

```json
"residual-S2@10": {"denominator": 28, "completed_claims": 28, "rate": 0.39, "ci95": [0.21, 0.57]}
```

| Key | Meaning |
|---|---|
| `native-Full`, `residual-Full` | recovery before any deletion: what the memory holds |
| `native-S0`, `residual-S0` | after deleting the disclosure |
| `native-S1@10`, `residual-S1@10` | after deleting 10 more turns ranked by the system's retrieval |
| `native-S2@10`, `residual-S2@10` | after deleting 10 more turns chosen by the model |

Per-fact details (what was deleted, the candidates and S2's scores, every answer) are under
`outputs/turnrag-ep01/claims/`. Run every episode, `EP01`–`EP15`, for the full benchmark.

## Evaluate your own memory system

Your system needs four things: build its memory from the dialogue while recording which turns each stored object
came from, retrieve, delete objects, and list what it stores. The candidate pool, S0/S1/S2, both readouts and the
scoring are shared code.

1. Copy [`toolkit/examples/keyword_bank.py`](toolkit/examples/keyword_bank.py), a complete memory system in about
   100 lines, and replace its storage and retrieval with yours.
2. Check it offline with a test like [`toolkit/tests/test_example_bank.py`](toolkit/tests/test_example_bank.py).
3. Register it with the `aligned` entry point and run it as above.

The step-by-step guide is [toolkit/docs/ADDING_A_SYSTEM.md](toolkit/docs/ADDING_A_SYSTEM.md).

## Use the data directly

If you have your own pipeline, read the bundle without the toolkit. Your system may read only the `public/` dialogue;
everything under `author/` is for scoring.

```python
import json, zipfile

bundle = zipfile.ZipFile("dataset/memcut_v1.zip")
manifest = json.loads(bundle.read("MANIFEST.json"))
for ep in manifest["episodes"]:
    if ep.get("set") == "case_study":
        continue                                            # the 15 main-set episodes
    read = lambda kind: json.loads(bundle.read(ep["files"][kind]["path"]))
    dialogue, contracts, notes = read("source"), read("contracts"), read("annotations")
    turns = [t for s in dialogue["sessions"] for t in s["turns"]]   # what your memory ingests
    for claim in contracts["claims"]:
        request = claim["deletion_request"]                 # what the user asks your system to forget
        disclosure = claim["direct_record_ids"]             # the turns that state the fact outright
        question = claim["question"]                        # ask it with the five choices...
        choices = {c["id"]: c["text"] for c in claim["choices"]}
        orders = [r["choice_ids_in_display_order"] for r in claim["rotations"][:2]]   # ...in these two orders
        target = claim["target_choice_id"]                  # recovered = target chosen under both orders
```

`notes["carriers"]` lists each fact's evidence turns and `notes["distractors"]` its look-alikes; every episode also
carries the minimal sufficient evidence sets and minimum cuts (`SUPPORT_PATHS.json`,
`PATH_COMBINATION_AUDIT.json`). The format is in [dataset/SCHEMA.md](dataset/SCHEMA.md). You can also add your own
facts to an episode with [toolkit/docs/EXTENDING_DATA.md](toolkit/docs/EXTENDING_DATA.md).

## What's in this repository

```
dataset/   memcut_v1.zip (the benchmark), its data card, schema, license and validator
toolkit/   the evaluation code: five built-in systems, the S0/S1/S2 pipeline, tests and guides
assets/    logo
```

## Documentation

| I want to | Guide |
|---|---|
| Install and run the built-in systems | [toolkit/docs/USAGE.md](toolkit/docs/USAGE.md) |
| Know what each built-in system stores and how deep it retrieves | [toolkit/docs/SYSTEMS.md](toolkit/docs/SYSTEMS.md) |
| Evaluate my own memory system | [toolkit/docs/ADDING_A_SYSTEM.md](toolkit/docs/ADDING_A_SYSTEM.md) |
| Add my own facts to an episode | [toolkit/docs/EXTENDING_DATA.md](toolkit/docs/EXTENDING_DATA.md) |
| Read the scoring rules | [toolkit/docs/PROTOCOL.md](toolkit/docs/PROTOCOL.md) · full spec: [toolkit/PROTOCOL.md](toolkit/PROTOCOL.md) |
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
released under the [MIT License](LICENSE). The memory systems under test are fetched from their own repositories at
setup and keep their own licenses ([toolkit/NOTICE](toolkit/NOTICE)).
