# Release verification

What was checked when the toolkit was assembled for release. These are source-level checks, not new experimental
results and not an independent audit of experiments.

## Source assembly (2026-10-01)

- In a fresh Python 3.11 virtual environment with only `requirements-dev.txt` installed and the pinned tokenizer
  files downloaded and verified: `pytest` **114 passed / 3 skipped**.
- Skipped: the Raw-turn test that needs a real local encoder, the test that needs a native MemoryBank environment,
  and an evidence test that depends on unreleased historical outputs. None of them is counted as passed.
- The nine command entry points print their help from any working directory; without a stage argument nothing
  starts.
- Importing the real 15-episode bundle succeeded: 15 episodes, 60 data files; `prepare` on the imported EP01 gave
  28 claims, QA k=16, 0 API requests.
- The synthetic import tests cover a hash mismatch, a path outside the destination, a duplicate episode and a
  refusal to overwrite an existing directory.
- The 95 source files selected from the working tree hash the same as when extracted; the changes made in the
  release copy are listed in `SOURCE_MANIFEST.json`. The extracted LightMem rendering function is AST-identical to
  the original.
- Local Markdown links resolve and Python files parse; no absolute local paths and no common key patterns were
  found. The directory contains no data, replies, banks, models or virtual environments.

The four native environments were not reinstalled and re-verified, and no new paid formation / deletion / answer
experiment was run. Each native system still needs its pinned upstream and model assets prepared and verified in
its own environment as described in `USAGE.md`.

## Four-system alignment and difference notes

- Default tests in the release copy: **120 passed / 7 skipped**, with an existing Python 3.11 native environment
  and the pinned tiktoken 0.12.0 cache; no fresh environment this time. Beyond the three earlier skips, four new
  native integration cases are skipped by default.
- With `MEMCUT_ALIGNED_NATIVE_TESTS=1`, the new adapter files: **9 passed**, using the installed SimpleMem /
  LanceDB, Nemori / Chroma and pinned upstreams; LLM replies and embeddings remain fixtures; two upstream
  deprecation warnings are left as they are.
- The new adapters', source-scoring and early-stop tests from the working tree: **21 passed**. No old experiment or
  full prefix audit was rerun.
- The default configuration tests read the release configs directly and simulate the upstream file checks; they
  verify PLAN / runtime overrides and do not claim a native environment is installed. The native tests separately
  verify real stores and upstream interfaces.
- Entry points went from nine to ten: `aligned` also prints only help from another directory with no arguments or
  `--help`.
- Test fixtures ship with the release and no longer import unreleased historical tests or audit modules;
  downloaded upstream source, tokenizer files and test runtime stay ignored by Git.

Zero paid API calls and zero real-model experiments in this round. Differences and experimental limits:
[the alignment notes](integrations/MEMCUT_LOCOMO_ALIGNMENT_20261001.md).

## aligned v2 (same day)

- Default tests: **155 passed / 9 skipped** (tiktoken cache as above). `tests/test_aligned_v2.py` adds 37 cases
  (2 native ones skipped by default): EP14/15-style trial names in the three systems' `prepare`, the Qwen endpoint
  checked at `prepare`, an interrupted `prepare` leaves no directory, the transport policy (retry / refusal /
  archive / resolve), unavailable readouts and the available cohort, MemoryBank early stopping, SimpleMem planning
  reply defaults, LightMem tail records, formation diagnostics. The older aligned tests use a temporary paid lock.
- Native cases (`MEMCUT_ALIGNED_NATIVE_TESTS=1`; replies and embeddings are fixtures, model assets are stand-ins):
  Nemori environment (chromadb 1.0.20, spaCy en_core_web_sm, pinned v0.1.1 upstream) 5 passed, including the tail
  flush, semantic times and buffer checkpoint recovery; SimpleMem environment (LanceDB 0.25.3, Tantivy 0.25.1,
  pinned db80b6a upstream) 15 passed, including rejected keywords, single-threaded FTS and an end-to-end run.
- LightMem probe (real torch / llmlingua with the pinned upstream, LLMLingua-2 with random weights, a bag-of-words
  stand-in for embeddings): the turns the tail probe records as dropped match an independent observation; a full
  run with one truncated answer completes all four stages, that readout is recorded unavailable and is not re-sent
  on rerun.
- Still not done: the four real full-episode paid runs, and an aligned end-to-end run in the native MemoryBank
  environment (its unavailable readouts and early stopping are covered by unit tests).

## Two RAG references in the aligned entry point (same day)

- Default tests: **162 passed / 9 skipped**. `tests/test_aligned_rag.py` adds 7 cases (synthetic embeddings and
  tokenizer, scripted replies): Raw-turn RAG completes the four stages and a rerun sends nothing; one truncated
  answer is recorded unavailable while the run completes; Session RAG stops before any paid request when its
  coverage exceeds 25% (S0/S1 saved, `COVERAGE.json` written); with coverage accepted it completes and writes the
  coverage into `SUMMARY.json`; RAG-only options are refused for other systems.
- The older `turnrag` and `sessionrag` entry points and their tests are unchanged.

## Note on offline test runs

`memcut/session_rag/text.py` needs the pinned `cl100k_base` tokenizer file in `TIKTOKEN_CACHE_DIR`
(`python tools/download_assets.py coverage`). Without network access that download fails and the Session RAG tests
(15 of them) report "Pinned cl100k_base cache is absent or changed"; place the file in the cache directory first.
