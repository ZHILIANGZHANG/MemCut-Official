# Install and run

Paths are relative to this directory (`toolkit/`); run the commands from here. Python 3.11 on Linux or macOS (the
run lock uses `fcntl`); this is not a native Windows setup.

## 1. One environment per system family

`requirements-dev.txt` is enough to check the source. To run a system, pick its dependency file below. Do not
install every system into one environment: MemoryBank's old LangChain / Pydantic 1 stack conflicts with the others.

| System | Dependency file | Upstream preparation |
|---|---|---|
| Turn RAG / Session RAG | `requirements/encoder.txt` | none |
| MemoryBank | `requirements/memorybank.txt` | `python tools/setup_upstreams.py memorybank` |
| Nemori | `requirements/source-systems.txt` | `python tools/setup_upstreams.py nemori` |
| LightMem | `requirements/source-systems.txt` | `python tools/setup_upstreams.py lightmem` |

For example:

```bash
python3.11 -m venv .venv-encoder
. .venv-encoder/bin/activate
python -m pip install -r requirements/encoder.txt
python tools/download_assets.py encoder        # both retrieval encoders; --embedding NAME for one
export TIKTOKEN_CACHE_DIR="$PWD/runtime/tiktoken"
python tools/download_assets.py coverage
# LightMem only: python tools/download_assets.py lightmem-compressor
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=0
```

Turn RAG and Session RAG take either of two retrieval encoders, registered side by side in
`configs/embeddings.json` and chosen at `prepare` with `--embedding`; the choice is frozen in the PLAN:

| `--embedding` | Model | Input convention |
|---|---|---|
| `e5-small-v2` (default) | `intfloat/e5-small-v2`, 384 dimensions, 512 tokens | `query: ` before queries, `passage: ` before stored text |
| `all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, 256 tokens | plain text |

`download_assets.py encoder` prints the commit of each encoder it fetched. `all-MiniLM-L6-v2` is pinned to a
commit; `e5-small-v2` uses the commit in your local Hugging Face cache, and every cached vector is keyed by it; write
that commit into `configs/embeddings.json` to pin it.

`setup_upstreams.py` only clones source at the pinned commit; `download_assets.py` only downloads public weights and
tokenizer files at the pinned revision. Neither makes a model call. Download before setting the offline variables.
Nemori's environment also needs spaCy's `en_core_web_sm` (`python -m spacy download en_core_web_sm`).

The dependency files record the versions used on the reference machine; they are not a complete cross-platform
lockfile. If an upstream's system packages differ on your machine, check that upstream's requirements; do not swap
libraries under an existing PLAN. Session RAG's token counts need exactly tiktoken 0.12.0.

## 2. Import the data bundle

```bash
python tools/import_dataset.py ../dataset/memcut_v1.zip --output data/bundle
```

The importer refuses an existing destination, extracts only the files the bundle's `MANIFEST.json` lists, and checks
their sizes and hashes. It lists the 15 main-set episodes and, apart, the case-study episode EP08A. The data is not
added to Git.

## 3. Run a system

One entry point runs every system: `run.py aligned`. `prepare` freezes a PLAN without any model call; `run` goes
build → delete → answer → report and resumes from its checkpoint after an interruption.

```bash
EP_DIR=data/bundle/data/EP01
python run.py aligned prepare --system lightmem \
  --output outputs/lightmem-ep01 \
  --source "$EP_DIR/public/EP01.json" \
  --contracts "$EP_DIR/author/CLAIM_CONTRACTS.json" \
  --annotations "$EP_DIR/author/ANNOTATIONS.json" \
  --model YOUR_MODEL_ID --endpoint https://YOUR_PROVIDER/v1
python run.py aligned run --run outputs/lightmem-ep01        # calls the model API
python run.py aligned status --run outputs/lightmem-ep01
```

| `prepare` option | Meaning |
|---|---|
| `--system` | `turnrag`, `sessionrag`, `memorybank`, `nemori` or `lightmem` |
| `--model`, `--endpoint` | any OpenAI-compatible endpoint; the same model writes the memory, ranks S2 and answers both readouts (default Qwen3.8-Flash) |
| `--compatibility` | `openai` (default), or `deepseek` / `mimo`, which send those providers' request with thinking off |
| `--claims` | a subset of the episode's claims (default: all) |
| `--embedding` | Turn RAG and Session RAG: the retrieval encoder (default `e5-small-v2`) |
| `--native-eval-k` | Turn RAG and Session RAG: items the native readout reads (default 16) |
| `--accept-coverage` | Session RAG: keep a native depth whose context shows more than a quarter of the episode |

The PLAN holds the conditions Full, S0, S1@10 and S2@10, the input hashes, the models and the code identity. A PLAN is
never resumed under a different code identity; prepare a new one instead.

**Keys.** The key comes from `MEMCUT_MODEL_API_KEY` (or `DASHSCOPE_API_KEY` for the default Qwen endpoint); a local
endpoint needs none. Never put a key into an argument, a PLAN or a committed file.

**Failures.** A run does not stall on one bad reply:

- a truncated, refused or ill-labelled answer: that readout is written as `<family>.unavailable.json` and the stage
  continues; `SUMMARY.json`'s `available_cohort` lists the claims available under every condition;
- 429, 5xx or a request that never left: back-off and retry within the same dispatch, unbilled;
- any other 4xx (key, balance, parameters): stop; after the fix, the next `run` re-sends;
- a dispatch with unknown outcome (timeout): stop and keep the receipt; `python run.py aligned resolve --run DIR`
  lists them, and `--retry` re-sends them on the next `run`.

**Outputs.** `SUMMARY.json` holds native and residual recovery per condition (2/2 over the two display orders)
and the available cohort; each claim's directory keeps its deletion receipts and readouts. Replies are cached per request, so rerunning the same PLAN costs
nothing. The paid lock defaults to `outputs/MEMCUT_PAID.lock`; set `MEMCUT_PAID_LOCK` to share it with another job.

## 4. Checks before committing

```bash
python -m pytest -c pytest.ini tests -q
```

Never commit downloaded checkouts, weights, memory banks, replies or results.
