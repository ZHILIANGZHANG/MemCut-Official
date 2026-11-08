# Reproducing the paper

Where each result of the MemCut paper comes from, and the command that produces it. Commands are run from this
directory after `tools/import_dataset.py` (see [USAGE.md](USAGE.md)); `$EP` stands for an imported episode
directory such as `data/bundle/data/EP02`, and `FILES` for its three inputs:

```bash
FILES="--source $EP/public/$(basename $EP).json --contracts $EP/author/CLAIM_CONTRACTS.json \
       --annotations $EP/author/ANNOTATIONS.json"
```

Every paid stage reads its key from the environment (USAGE.md, section 3). `prepare` never calls a model.

| Paper | Setting | How |
|---|---|---|
| Main table: native and residual recovery after S0, S1, S2 | five systems (Turn RAG, Session RAG, MemoryBank, Nemori, LightMem) × four write models; pool 20, budget 10; reader Qwen3.8-Flash; EP01–EP15 | `python run.py aligned prepare --system <system> --output <dir> $FILES --model <write model> --endpoint <url>`, then `aligned run --run <dir>` |
| Retention | the same runs | `python run.py aligned-retention --run <dir>` |
| Pool sweep (Figure 3a) | pool 10, 20, 30, 40, 50, 60 at budget 10; Turn RAG, MemoryBank, LightMem; DeepSeek and Qwen; EP01–EP05 | one PLAN per pool size with the staged entry points: `turnrag prepare ... --candidate-objects N --candidate-turns N --k 10`, `source prepare --system lightmem ... --candidate-objects N --candidate-turns N --k 10`; each pool is ranked afresh |
| Budget sweep (Figure 3b) | budget 2, 4, 6, 10, 15, 20 at pool 20 | one PLAN per system with `--k 2 4 6 10 15 20`; the budgets delete prefixes of one ranking |
| Case study (Figure 4, Appendix E) | Turn RAG, S2, pools 10 to 60, on the case-study episode EP08A | as the pool sweep, on `data/bundle/data/EP08A`; the similarity ranks of the evidence come from the `build` stage's retrieval (`turnrag trace --run <dir> --query "<deletion request>" --top-k 200`) |
| Dataset statistics (Tables 2, 5, 6) | the bundle | the manifest, the annotations and every episode's minimal-set files; no model call |

Notes:

- MemoryBank's staged entry point takes the budget list `--k` but not the pool size; the pool sweep for it needs the
  same options added to its `prepare`.
- The `aligned` entry point fixes the main setting. Sweeps use the staged entry points (`turnrag`, `sessionrag`,
  `source`, `memorybank`), which take the pool and budget options; see each one's `prepare --help`.
- Write models are selected with `--model` / `--endpoint` (OpenAI-compatible) and, for DeepSeek and MiMo, the
  `--compatibility` presets; the `glm` and `qwen` entry points carry their providers' request settings.
- Replies are cached per request, so rerunning an identical PLAN costs nothing; a changed setting is a new PLAN,
  never a resumed one.
