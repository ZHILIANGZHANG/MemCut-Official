# Adding a memory system

MemCut asks one thing of a memory system: after a deletion, can the forgotten fact still be recovered from what the
memory holds? To answer it for your system, the toolkit needs to build its memory, ask it to retrieve, delete
source turns from it, and see what it still stores. Everything else (the candidate pool, S0/S1/S2, the S2 selector,
both readouts, 2/2 scoring, caching and cost accounting) is shared code you do not write.

The worked example is [`examples/keyword_bank.py`](../examples/keyword_bank.py): a complete system in about 100
lines with no model and no third-party package. [`tests/test_example_bank.py`](../tests/test_example_bank.py) runs
it through formation, the S1/S2 candidate pool, closure deletion and both readout contexts. Copy both files, then
replace the storage and retrieval with your own.

## 1. The adapter: a `SourceBank` subclass

`memcut.source_anchored_systems.base.SourceBank` owns provenance (which turns each object came from), snapshots,
source-closure deletion and the residual view. Your subclass owns the native store.

| Method | What it must do |
|---|---|
| `form()` | Build the full memory from `self.turns` (in order). For every object you write, record its source turns with `self.lineage.input([turn ids])` inside `self.observe_write(...)`. End with `progress['stage'] = 'FULL'` and `freeze_updates(True)`. |
| `async search(query, *, top_k)` | Your system's own retrieval. Return a `SearchResult` whose `ranked['retrieved_document']` lists the objects in rank order, and whose `costs['native_seed_hits']` gives `{address, rank, distance}` per object (the pool reads `-distance` as the score). |
| `system_render(result)` | The text your system puts in front of its reader for a retrieval result; the native readout answers from it. |
| `native_objects()` | Every stored object as `{id, native_id, layer, kind, text, value}`; `kind` is `atomic` or `aggregate`. The residual readout and the deletion accounting read this, never `search`. |
| `erase_native(obj_id)` | Physically remove one object (text, vectors, index entries). |
| `dump_native()` / `load_native(value)` | Serialize and restore the native store; branches and snapshots use them. |
| `regenerate_groups(...)` | Only for systems that rewrite merged text (summaries, profiles). The main protocol deletes derived objects whole and never regenerates, so most adapters leave it out. |

Rules the shared code checks, and why:

- **Provenance must be exact.** An object derived from turns t3 and t4 is deleted when either is deleted (closure),
  and the other turn is reported as collateral. If your system merges content, record every contributing turn.
- **Retrieval must not change the memory.** `native_context` and the pool builder compare `state_identity()` before
  and after a search and stop on a difference. Recall counters or access timestamps belong outside the hashed store,
  or must be undone.
- **The object address is `json.dumps([system, layer, native_id])`.** Build your object ids the same way
  (`MemoryObject.address`), so that search results and stored objects refer to the same thing.
- **No gold labels.** Your adapter sees the dialogue only; it never reads `author/` files.

## 2. Check it offline

Write the equivalent of `tests/test_example_bank.py` for your bank: provenance of every object, a pool from
`build_pool`, a deletion receipt from `delete_turns`, and `contexts(...)` for both readouts. These run without a
key. Then run the whole suite:

```bash
python -m pytest -c pytest.ini tests -q
```

## 3. Wire it into a runner

A runner turns a bank into the lifecycle `prepare → build → delete → answer → report`. The smallest one to copy
is [`memcut/raw_turn_rag/runner.py`](../memcut/raw_turn_rag/runner.py): replace `RawTurnBank` and its `Factory`
with yours and keep the stage functions, which call the shared candidate, deletion and evaluation code.

Then register it with the `aligned` entry point: add the system name to `SYSTEMS` in
`memcut/aligned_s012/__init__.py` and its runner module to `runner()` in `memcut/aligned_s012/runtime.py`, and run
it with `python run.py aligned prepare --system <name> ...`. Editing that package changes the `aligned` code
identity for every system: do it before you prepare PLANs, because a run prepared under the old identity is not
resumed under the new one.

## 4. Report it

Run the main setting (pool 20, budget 10) on the 15 main-set episodes, and report native and residual recovery for
S0, S1 and S2. State the write model and the system's retrieval depth. If your system needs an upstream checkout, pin its commit in
`tools/setup_upstreams.py` and list it in `NOTICE`.
