# Install and run

Paths are relative to this directory (`memcut_eval/`); run the commands from here. Python 3.11 on Linux or macOS
(the run lock uses `fcntl`); this is not a native Windows setup.

## 1. One environment per system family

`requirements-dev.txt` is enough to check the source. To run a system, pick its dependency file below. Do not
install every system into one environment: MemoryBank's old LangChain / Pydantic 1 stack conflicts with the others.

| System | Dependency file | Upstream preparation |
|---|---|---|
| Raw-turn / Session RAG | `requirements/encoder.txt` | none |
| MemoryBank | `requirements/memorybank.txt` | `python tools/setup_upstreams.py memorybank` |
| Nemori | `requirements/source-systems.txt` | `python tools/setup_upstreams.py nemori` |
| LightMem | `requirements/source-systems.txt` | `python tools/setup_upstreams.py lightmem` |
| SimpleMem | `requirements/simplemem.txt` | `python tools/setup_upstreams.py simplemem` |

For example:

```bash
python3.11 -m venv .venv-encoder
. .venv-encoder/bin/activate
python -m pip install -r requirements/encoder.txt
python tools/download_assets.py encoder
export TIKTOKEN_CACHE_DIR="$PWD/runtime/tiktoken"
python tools/download_assets.py coverage
# LightMem only: python tools/download_assets.py lightmem-compressor
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=0
python run.py turnrag doctor
```

`setup_upstreams.py` only clones source at the pinned commit; `download_assets.py` only downloads public weights and
tokenizer files at the pinned revision. Neither makes a model call. Download before setting the offline variables;
if they are already set, unset them in the shell used for downloading.

The dependency files record the versions used on the reference machine; they are **not a complete cross-platform
lockfile**. The release was verified with the zero-API functional tests, the data import and the CLI, not by
reinstalling every native environment or running a full episode. If an upstream's system packages or optional
features differ on your machine, check that upstream's requirements and run the chosen system's `doctor`; do not
swap libraries under an existing PLAN. Session RAG's diagnostic counts need exactly tiktoken 0.12.0; the tiktoken
version in MemoryBank's environment is not suitable for the Session RAG tests.

## 2. Import the data bundle

```bash
python tools/import_dataset.py /path/to/memcut_v1.zip --output data/bundle
```

The importer refuses an existing destination, extracts only the files the bundle's `MANIFEST.json` lists, checks
their sizes and hashes, and ignores every other archive member. The data is not added to Git.

The importer prints the main-set episodes and, apart, the case-study episodes (EP08A: the paper's case study, EP08
plus the Arden Advisory fact). A main-set run takes its episodes from the manifest entries without a `set` field.

`data/bundle/MANIFEST.json` is the entry point for file paths. EP01, for example:

```bash
EP_DIR=data/bundle/data/EP01
python run.py turnrag prepare \
  --output outputs/turnrag-ep01-example \
  --source "$EP_DIR/public/EP01.json" \
  --contracts "$EP_DIR/author/CLAIM_CONTRACTS.json" \
  --annotations "$EP_DIR/author/ANNOTATIONS.json" \
  --answer-model YOUR_MODEL_ID \
  --endpoint https://YOUR_PROVIDER/v1
```

This only writes a new PLAN; the model id and endpoint are placeholders. The output directory must not exist. Take
the other episodes' paths from the manifest. The fixed parameters, input hashes, execution policy and model
configuration are all stored in the PLAN.

## 3. Run the stages explicitly

With a real endpoint and model configured, provide `MEMCUT_MODEL_API_KEY` in the shell environment. If the answerer
or the selector has its own endpoint, provide `MEMCUT_RECOVERY_API_KEY` / `MEMCUT_SELECTOR_API_KEY` as the PLAN's
`providers.*.key_env` names them. The GLM entry point reads `MEMCUT_GLM_API_KEY`; the Qwen entry point can map
`DASHSCOPE_API_KEY`. Never put a key into an argument, the source, a PLAN or a committed file; the toolkit does not
read key files from the machine.

```bash
# Raw-turn RAG builds its memory with local encoding only, no LLM
python run.py turnrag build --run outputs/turnrag-ep01-example
# S2 ranking and answering use the configured API; run the following only for a real experiment
python run.py turnrag delete --run outputs/turnrag-ep01-example
python run.py turnrag answer --run outputs/turnrag-ep01-example
python run.py turnrag report --run outputs/turnrag-ep01-example
```

That is the staged use of the older Raw-turn RAG entry point, for its own plans; new experiments use the `aligned`
entry point of section 4 (`--system turnrag`), which has uniform failure handling. MemoryBank, Nemori, LightMem and
SimpleMem also call the LLM while building their memory. Fill in the arguments the chosen entry point's
`prepare --help` lists, and keep using the same entry point and the same PLAN; do not switch execution policies
mid-run.

`python run.py <entry>` or `--help` only prints help. There is no background monitor and no paid run by default.
`aligned run` is an explicit, one-shot chain of stages. Stopping leaves the state readable; the next `run` resumes
from the checkpoint. A reply with a complete identity is reused; a dispatch whose outcome is unknown (timeout,
connection lost after sending) is never treated as a cache miss and re-paid until `aligned resolve --retry` has
archived it explicitly.

## 4. The `aligned` entry point: four memory systems and two RAG references

The default model is Qwen3.8-Flash for every role, thinking off. Install the upstream and the assets in the matching
environment first, then create a new PLAN:

```bash
python run.py aligned prepare --system lightmem \
  --output outputs/lightmem-aligned-ep01 \
  --source "$EP_DIR/public/EP01.json" \
  --contracts "$EP_DIR/author/CLAIM_CONTRACTS.json" \
  --annotations "$EP_DIR/author/ANNOTATIONS.json"
# only the explicit run below calls the API; the key comes from the environment
python run.py aligned run --run outputs/lightmem-aligned-ep01
python run.py aligned status --run outputs/lightmem-aligned-ep01
```

Systems: `memorybank`, `nemori`, `lightmem`, `simplemem`, and the two RAG references `turnrag` and `sessionrag`
(their build is local encoding only; `--native-eval-k` sets how many items the native readout reads, default 16;
Session RAG measures, after S0/S1 and before any paid request, how much of the episode its native context covers,
and stops above 25% unless `prepare` was given `--accept-coverage` or a smaller k). `run` goes build → delete →
answer → report, with no pilot and no separate audit stage; single stages can still be run. MemoryBank keeps QA
k=10 and early stopping on a valid wrong answer; Nemori has its native merge and prediction correction, flushes the
tail batch as v0.1.1 does, and its native prompt's semantic lines carry times; LightMem's effective compression
ratio is 0.6; SimpleMem accepts its native default fields and, like upstream, returns no lexical result for a
keyword Tantivy rejects. The answer cap is 2048 tokens for all six systems. Older entry points and PLANs are
unchanged; do not resume an old bank through another entry point.

Since v2 a run no longer stalls on these:

- a truncated, refused or ill-labelled answer: that readout is written as `<family>.unavailable.json` and the
  stage continues; `SUMMARY.json`'s `available_cohort` lists the claims available under every condition and their
  rates;
- 429, 5xx or a request that never left: back-off and retry within the same dispatch (2, 4, 8, 16, 32 s), unbilled;
- a content refusal: saved as that request's result and not re-sent on rerun; an answer request is recorded
  unavailable;
- any other 4xx (key, balance, parameters): stop; after the fix, the next `run` archives that receipt and re-sends;
- a dispatch with unknown outcome (timeout, etc.): stop and keep the receipt. `python run.py aligned resolve --run
  DIR` lists them; with `--retry` they are archived and re-sent on the next `run`.

Nemori's environment needs spaCy's `en_core_web_sm` (`python -m spacy download en_core_web_sm`); without it the
build fails before starting instead of silently degrading to vector-only retrieval. `full/DIAGNOSTICS.json` records
the source turns that left no surviving memory in Full and the claims they touch.

Retention is not measured inside the `aligned` entry point. After the answer stage:

```bash
python run.py aligned-retention --run outputs/lightmem-aligned-ep01                          # default: S2@10, 5 probes per claim
python run.py aligned-retention --run outputs/lightmem-aligned-ep01 --conditions S2@10 S1@10 S0
```

Probe rule (the protocol's retention): the other planned claims of the same episode whose evidence and disclosure
turns are disjoint from this claim's, in a fixed hash order, the first five whose native readout on Full is correct
2/2 (so the PLAN must contain all of the episode's claims, the default); after the deletion they are read out 2/2
through the system's own retrieval and answer stage, and an unavailable answer does not score. Output:
`RETENTION.json` and `retention_aligned/<claim>/<condition>.json`. Requests go through the run directory's
receipts, so a rerun sends only what is missing; the paid lock and the pending-dispatch check are shared with the
`aligned` entry point.

The paid lock defaults to `outputs/MEMCUT_PAID.lock` in this checkout. To share it with another paid job on
the same machine, set `MEMCUT_PAID_LOCK` to that job's lock file. The entry point has no resource waiting or hard
cost ceiling; use a separate environment per job and avoid running paid tasks from several directories at once.
Per-system differences, their reasons, failure behaviour and the verification boundary are in
[the alignment notes](integrations/MEMCUT_LOCOMO_ALIGNMENT_20261001.md).

## 5. Checks before committing

```bash
python -m pytest -c pytest.ini tests -q
git status --short --untracked-files=all
```

Keep `.gitignore`; commit only this directory's source. Upstream source is fetched under its own license; never
commit downloaded checkouts, weights, banks, replies or results.
