# Scoring rules and entry points

Common rules: every claim branches independently from the same Full memory; S0 deletes the disclosure turns D whole,
with their source cascade; S1 / S2 map the native candidate objects retrieved after S0 back to source turns, by
default at most 20 objects, 20 distinct turns, and 10 additional turns deleted. S1 takes the native ranking, S2 the
selector's; Direct is a separate condition. Derived memory objects are deleted whole and never regenerated; Raw-turn
and Session RAG are handled according to their own raw-text storage semantics.

**native** answers from the system's own retrieval; **residual** answers from what the surviving evidence and
disclosure sources still derive. Recovery requires the correct choice under both fixed display orders (2/2). An
invalid, missing or truncated answer is not counted as wrong; report the Full-condition denominator and the missing
cases together.

| `run.py` system | Use and caveats |
|---|---|
| `memorybank` | basic MemoryBank lifecycle; its historical native k constant is not the main experiment's k=10 |
| `memorybank-efficient` | MemoryBank main experiment with early stopping; `prepare` still takes `--system memorybank` and freezes k=10 explicitly |
| `source` | Nemori / LightMem basic lifecycle; `--system` selects the system |
| `simplemem` | SimpleMem standalone S0/S1/S2; not the LoCoMo adapter |
| `turnrag` | one raw turn per object, formation without an LLM, default QA k=16 |
| `sessionrag` | raw-session retrieval, chunks only for indexing; exact-turn deletion then local re-encoding |
| `efficient` | execution-optimised variant of four systems: MemoryBank / Nemori / LightMem / Raw-turn RAG; explicitly freezes early stopping on a valid wrong trial 1 and a QA cap of 1024; one re-send after a connection error or 5xx, at most two after an invalid answer (`efficient_execution/recovery.py`), whether the first was billed is unknown |
| `glm` / `qwen` | entry points with the corresponding model request configuration; `glm`'s execution base class differs from `qwen`'s, read each one's help and PLAN |

`efficient` is an explicit experimental variant, not an unconditional replacement for the main entry points.
`memorybank-efficient` enables early stopping for MemoryBank only: when trial 1 is valid and wrong, trial 2 is not
run, and the skipped trial is not a real observation. A correct trial 1 still needs a correct trial 2. Such results
cannot serve a diagnostic that needs two executed answers.

HTTP identities, configuration checks, checkpoints and cost accounting in the code are execution logic; this release
adds no separate audit stage or run gate. Installation checks and fixture tests do not mean a real full-episode
experiment passed.

The full definitions are in [PROTOCOL.md](../PROTOCOL.md). Its historical dates and trial descriptions are part of
the protocol text, not evidence that experiments were run for this release.

## The aligned entry point

`run.py aligned` keeps the deletion and scoring protocol above and applies mechanism / configuration changes only to
new PLANs; the four memory systems and the two references Raw-turn / Session RAG all run from it
(`--system turnrag|sessionrag`) with the same failure handling and answer cap. Since v2 an invalid answer is
recorded as an unavailable readout and the stage continues (it no longer interrupts the whole answer stage), the
answer cap is 2048 for the four systems, and the tie rule and the S2 selector's single scoring pass are described
in the alignment notes, section 2. The formation, version, QA k, temperature, context and recovery differences
between the systems' MemCut and LoCoMo configurations are in
[the alignment notes](integrations/MEMCUT_LOCOMO_ALIGNMENT_20261001.md). This is not an equal-budget LoCoMo
reproduction and requires no extra audit stage.
