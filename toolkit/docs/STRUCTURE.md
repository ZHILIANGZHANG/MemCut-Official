# Layout

```text
memcut_eval/
├── run.py                         # the one explicit command entry point; no default experiment
├── memcut/
│   ├── aligned_s012/              # aligned entry point: four memory systems (Nemori mechanics / tail batch /
│   │                              #   rendering, LightMem parameters / tail records, SimpleMem compatibility),
│   │                              #   two RAG references (rag.py), unavailable readouts, transport policy
│   ├── native_memory/             # native object interfaces, MemoryBank / LightMem dependencies
│   ├── source_anchored_v2/        # sources, candidate pool, deletion, native / residual scoring
│   ├── source_anchored_systems/   # Nemori / LightMem and the formation runtime
│   ├── simplemem_s012/            # SimpleMem S0/S1/S2 adapter
│   ├── raw_turn_rag/              # raw-turn reference
│   ├── session_rag/               # whole-session retrieval, exact-turn deletion
│   ├── model_adapters/            # GLM / Qwen request configuration
│   ├── execution_optimization/    # shared CLI core, receipts and concurrency
│   ├── answer_execution/          # early stop on a valid wrong answer
│   └── efficient_execution/       # condition reuse and execution policies
├── tools/                         # thin entry points, upstream / asset download, data import,
│                                  #   extend_episode.py (add a fact to an episode from an overlay)
├── examples/                      # keyword_bank.py: a complete minimal memory system to copy
├── configs/                       # upstream commits, native settings, model revisions
├── requirements/                  # per-environment dependency files
├── tests/                         # synthetic fixtures and zero-API functional tests
├── docs/                          # install, reproduce, adding a system, adding data, protocol
└── CITATION.cff  CONTRIBUTING.md
```

The runtime directories `data/`, `runtime/`, `outputs/` and `.venv*` are ignored by Git. Import the data bundle to
`data/bundle/` (it keeps the bundle's `data/...` paths and manifest). Upstream source lives under
`runtime/upstream/<system>/`; the tools check out the pinned commit and never update an existing checkout.

The shared CLI implementation lives in `memcut/execution_optimization/cli_core.py`; `tools/optimized_s012.py` is a
compatibility entry point. MemoryBank's early-stop entry point acts on the actual core module, so a cross-module
patch cannot silently miss. LightMem's formation text renderer is `formation_runtime/render.py` and is no longer
imported from a measurement script.

This is a source-directory layout, not an installable wheel. The internal ROOT locates the configs through this
layout, so do not copy only `memcut/` into site-packages, and do not move or rewrite the directory after a PLAN
has been frozen. Entry points may be called by absolute path; input and output arguments resolve against the
caller's working directory.

When maintaining it, keep the pinned upstream commits and file hashes in the configs. A change to an algorithm, a
prompt, the scoring or a dependency should produce a new experiment identity. Test fixtures must not reference a
researcher's absolute paths or private outputs.

New experiments use `run.py aligned`: an explicit run completes build, delete, answer and report with no extra
audit period. The older entry points remain. Per-system differences from the systems' LoCoMo configurations, and
their reasons, are in [the alignment notes](integrations/MEMCUT_LOCOMO_ALIGNMENT_20261001.md).
