# Reproducing the paper

Commands are run from this directory after `tools/import_dataset.py` (see [USAGE.md](USAGE.md)). `$EP` stands for
an imported episode directory such as `data/bundle/data/EP02`:

```bash
FILES="--source $EP/public/$(basename $EP).json --contracts $EP/author/CLAIM_CONTRACTS.json \
       --annotations $EP/author/ANNOTATIONS.json"
```

| Paper | How |
|---|---|
| Native and residual recovery after S0, S1 and S2, for the five systems and four write models on EP01–EP15 | for each episode, system and write model: `python run.py aligned prepare --system <system> --output <dir> $FILES --model <model> --endpoint <url> [--compatibility deepseek\|mimo]`, then `python run.py aligned run --run <dir>`; the write model also ranks S2 and answers both readouts. Session RAG reads 3 sessions in the paper: add `--native-eval-k 3` |
| Dataset statistics | from the bundle alone: the manifest, the annotations and every episode's minimal-set files; no model call |

Each `run` writes `SUMMARY.json` with the recovery rates per condition. Replies are cached per request, so rerunning
the same PLAN costs nothing; a changed setting is a new PLAN, never a resumed one.
