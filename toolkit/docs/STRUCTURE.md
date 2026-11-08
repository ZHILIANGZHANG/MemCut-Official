# Layout

```text
toolkit/
├── run.py                         # the command entry point: run.py aligned ...
├── memcut/
│   ├── aligned_s012/              # the aligned entry: system registry, runtime, readouts, transport policy
│   ├── source_anchored_v2/        # shared core: candidate pool, S2 selector, deletion, native / residual scoring;
│   │                              #   the MemoryBank runner
│   ├── source_anchored_systems/   # Nemori and LightMem adapters and their formation runtime
│   ├── raw_turn_rag/              # Turn RAG and the retrieval encoders
│   ├── session_rag/               # Session RAG: whole-session retrieval, exact-turn deletion
│   ├── native_memory/             # object contract, persistence, MemoryBank and LightMem native layers
│   ├── model_adapters/            # request conventions of the Qwen and GLM endpoints, local endpoints
│   ├── answer_execution/          # MemoryBank's stop after a valid wrong first answer
│   ├── execution_optimization/    # request receipts
│   └── efficient_execution/       # request identity and retry policy for answers and S2
├── tools/                         # memcut_s012.py (the aligned entry), data import, extend_episode.py,
│                                  #   upstream and asset download
├── examples/                      # keyword_bank.py: a complete minimal memory system to copy
├── configs/                       # upstream commits, native settings, retrieval encoders
├── requirements/                  # per-environment dependency files
├── tests/                         # synthetic fixtures and zero-API tests
├── docs/                          # usage, systems, protocol summary, reproduction, extension guides
├── PROTOCOL.md                    # the full specification
└── CITATION.cff  CONTRIBUTING.md  LICENSE  NOTICE
```

The runtime directories `data/`, `runtime/`, `outputs/` and `.venv*` are ignored by Git. Import the data bundle to
`data/bundle/` (it keeps the bundle's `data/...` paths and manifest). Upstream source lives under
`runtime/upstream/<system>/`; the tools check out the pinned commit and never update an existing checkout.

This is a source-directory layout, not an installable wheel. The internal ROOT locates the configs through this
layout, so do not copy only `memcut/` into site-packages, and do not move the directory after preparing a PLAN.
Entry points may be called by absolute path; input and output arguments resolve against the caller's working
directory.

A change to an algorithm, a prompt, the scoring or a dependency gives a new code identity, and a PLAN prepared under
another identity is not resumed.
