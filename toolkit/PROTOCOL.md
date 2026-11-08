# MemCut Deletion and Evaluation Protocol (Implementation Spec v4: fine-grained sources, derived objects deleted whole, 2/2 evaluation)

This document is written for implementers. Goal: run one and the same "source-anchored" deletion and two leakage readouts on MemoryBank, Nemori, LightMem and the SimpleMem text version (plus the optional [Raw-turn RAG control](docs/integrations/RAW_TURN_RAG.md)). The boundary between black-box diffing and same-call source extension is stated separately for each; the integration interface is in Section 7.

The concrete MemoryBank implementation and its mapping onto native storage are in [MemoryBank S0/S1/S2](docs/integrations/MEMORYBANK_S012.md); the standalone Nemori and LightMem integrations are in [Lightweight configuration, diffing, deletion and evaluation](docs/integrations/NEMORI_LIGHTMEM_S012.md). The budget is counted in **source turns**: S0 deletes all derived objects of D whole; S1/S2 additionally select 10 distinct turns and delete all of their derived objects whole. Nothing is regenerated during deletion. Verification of newly integrated local native components is recorded separately from pilots with real models; implementation-test scores are never reported as recovery-rate results.

LightMem currently uses [same-call source labelling](docs/integrations/LIGHTMEM_FINE_SOURCES.md): the fact-extraction reply carries `source_ids`, and the consolidation reply carries the `source_memory_ids` that actually contributed; the program parses them and maintains the bidirectional table. **No extra model call is made for attribution**, and all turns of a batch are not automatically counted as sources of every fact. This explicitly modifies LightMem's generation prompt and output fields; it is an integration adaptation and is not called an unmodified black box. The existing source rules of MemoryBank and Nemori are unchanged. An older batch-associated store cannot be renamed and passed off as a fine-grained store under this version.

The [SimpleMem text version](docs/integrations/SIMPLEMEM_S012.md) was added later: a fixed native 40-turn window with overlap 2 on the single-pass bulk build path; the existing extraction reply additionally returns `source_ids/source_memory_ids`, sources are bound by actual contribution, and no attribution request is added. It uses the same source closure, independent Direct budget, S0/S1/S2 and two-readout 2/2 rules; the merge order of the native multi-path retrieval, the final 20-object cut, the time reference and local index maintenance are in the integration document. Older systems, older formation identities and frozen results are unchanged. The SimpleMem integration covers the main experiment; the appendix residual+ and retention are designed and frozen separately, and unimplemented appendices are not presented as tested.

---

## 0. Overview (read this first)

1. **The anchor is the turn.** The conversation is the shared substrate of all systems; every unit, budget and evaluation is defined on turns.
2. **The provenance table is the core data structure.** It records which turns each memory object came from (many-to-many) and is updated in step with deletions and rewrites.
3. **Three stages, three units: retrieval by object, budget by turn, deletion by closure.**
   - Retrieval: call the system's native search, take the system's own objects, top-N (N=20)
   - Budget: map objects back to turns, deduplicate and sort, take the first k turns (k=10)
   - Deletion: for each selected turn, delete all objects it derived, whole (atomic and aggregate alike are deleted whole; nothing is regenerated)
4. **Two leakage readouts.** native = run the question stem through the system's retrieval; residual = feed directly the objects still derived from the evidence/disclosure turns.
5. **Every claim is evaluated on an independent snapshot.** Each formation identity builds the store once, the snapshot is saved to disk, and no claim rebuilds the store.

---

## 1. Data structures

### 1.1 Input
```
Turn        = {turn_id: str, session_id, speaker, timestamp, text}
Episode     = {episode_id, turns: [Turn]}            # one complete conversation scenario
Claim       = {
  claim_id, episode_id,
  deletion_request: str,                              # the user's natural-language deletion request
  disclosure_turns: [turn_id],                        # D_c, usually 1–2 turns
  evidence_turns:   [turn_id],                        # E_c, ≤10 turns
  sufficient_sets:  [[turn_id]],                      # sufficient combinations, for analysis; invisible to deletion strategies
  distractor_turns: [turn_id],                        # X_c
  category: "fact_assembly" | "relation_chain" | "aggregation" | "temporal_state",
  mcq: {stem: str, options: [str x4], answer_idx: int}
}
```

### 1.2 Memory object (unified system-side view)
```
MemObject = {
  obj_id: str,            # unique id within the system; a rewrite does not change the id
  obj_type: str,          # system-defined type name, see table 2.3
  kind: "atomic" | "aggregate",
  text: str,              # the text as stored by the system, verbatim
  meta: dict              # timestamps etc.
}
```

### 1.3 Provenance table
```
src:  obj_id  -> set[turn_id]        # which turns an object came from
der:  turn_id -> set[obj_id]         # which objects a turn derived (reverse index of src, must be kept in sync)
history: [ {op: "create"|"modify"|"delete"|"rewrite", obj_id, turns, step} ]
touched: turn_id -> set[obj_id]      # objects that ever had this turn as a source (including rewritten, excluding deleted); derived from history
```
Invariant: `t ∈ src[o]  ⇔  o ∈ der[t]`. Assert immediately after every write operation.

---

## 2. Store construction and provenance-table construction

### 2.1 Build rules
- Each (system, write_model, episode) is built exactly once; write model at temperature 0, cache on
- Ingest in the system's natural unit (see 2.3), one unit at a time
- After ingestion, run all of the system's offline processes (LightMem offline consolidation, MemoryBank personality update), then `freeze_updates(True)`
- Persist: `snapshot()` saves the store files + provenance table + freeze marker

### 2.2 Provenance-table construction: snapshot diffing (black-box, generic)
Call `list_all()` once before and once after feeding each unit U (a set of turns), obtaining `{obj_id: hash(text)}`:
```
before = {o.obj_id: hash(o.text) for o in list_all()}
ingest(U)
after  = {o.obj_id: hash(o.text) for o in list_all()}

for oid in after - before:            src[oid] = set(U);        # create
for oid in after & before:
    if after[oid] != before[oid]:     src[oid] |= set(U);       # modify → multiple sources
for oid in before - after:            log delete/merge; remove from src/der
update der in sync; assert the invariant
```
Diffing makes no LLM calls. If a system natively carries a source field (Nemori's episode), read it directly and take the union with the diff result.

Implementation note: the sources here must be **the union of the input turns that actually took part in generation and the source turns of the parent objects**; the current turn that externally triggers a write must not be treated as the source of a buffered batch. MemoryBank derives the daily-field sources from the native actual input and then lets the overall fields inherit from the daily fields; the provenance table makes no additional LLM calls.

**LightMem fine-grained special case (current implementation).** Diffing is still responsible for discovering created, modified and deleted native objects, but the fine-grained sources are returned by the existing generation call in the same reply; the diff batch U is not used to fill `src` directly:

1. During native extraction, each fact returns `source_id` and a non-empty `source_ids`. The former keeps its native use for time/speaker metadata; the latter lists all input message numbers the fact actually used, including the messages needed for pronoun resolution, subject completion and cross-turn synthesis. The program maps **the numbers printed in that actual input** to the original single-speaker turns, then binds them to the object IDs created by the native converter.
2. During native offline consolidation, input objects only receive local `[memory N]` numbers; the same update reply gives `source_memory_ids`. The sources of the new text equal the union of the sources of the **actual input versions** of these contributing objects; candidates that were not adopted are not added. The update replaces the current `src` with the new sources and does not mechanically keep the old sources of discarded text; historical sources remain in `history/touched`.
3. The complete input batch and all candidate parent versions are kept separately for checking; they must not be used in reverse to enlarge the current `src/der`. Missing sources, out-of-range numbers and unattributed writes fail outright; there is no silent fallback to whole-batch association, and no extra model call to patch labels.
4. `src[obj]` and `der[turn]` are updated bidirectionally and automatically with the storage diff and are persisted with checkpoints/Full. A retrieved object can be traced back directly to its source turns; given a turn, all currently derived objects can be enumerated. This table is used for the candidate pool, the deletion closure and residual; the original turn text is used only for research-side tracing and must not, on that account, be given to deletion strategies or recovery answering.
5. Facts and source labels are persisted as one and the same reply; after an interruption, replaying the same request reads the cache directly. Consolidation input uses a stable ordering and local numbering, so native object UUID rebuilds do not cause duplicate model requests. Source tracing adds a small number of prompt/field tokens; **the number of added attribution requests is 0**, which is not a claim of zero added tokens.

These are **content sources declared by the model**, not proven semantic causal relations. The program verifies that the numbers are valid, the inputs do exist, and the object binding and the bidirectional table are consistent; whether a real model under- or mislabels must be checked separately by sampling. A fact formed jointly by several turns still has several sources, and deleting it whole may still affect other turns.

MemoryBank implementation: before and after each write, the native storage fields are enumerated independently; object changes are discovered by diffing ID/text and field-value hashes, then bound to the actual input sources. Enumeration is not filtered by the provenance table, and index text that may not yet have been refreshed is not used in place of the stored text. Ordinary ingestion modifications take the union of sources; the current pure deletion removes deleted objects from the live source table and the historical association table. Source-replacement events of the old regeneration variant are only used to understand historical records. See [MemoryBank diffing notes](docs/integrations/MEMORYBANK_S012.md#how-the-diff-is-executed).

### 2.3 Object types per system (verify against the code at integration time and fill in this table)

| System | obj_type | kind | src source | Natural ingestion unit | Notes |
|---|---|---|---|---|---|
| MemoryBank | raw_entry (native QA pair) | atomic | = user turn and assistant turn | original two-speaker conversation pair | budget still counted per single turn; collateral effect of deleting the whole pair recorded separately |
| | daily_summary | aggregate | diff: turns of that day | | |
| | personality_daily | aggregate | diff: turns of that day | | |
| | personality_global | aggregate | all turns | | deleted whole when any of its source turns is selected |
| | overall_history | aggregate | union of the source turns of each day's event summary | | stored natively, not shown by the native QA template; still included in the closure and in residual |
| Nemori | buffer_message | atomic | = that turn | episode segments cut by the system itself | flushed before Full; original messages then live inside episodes, no separate permanent raw layer |
| | episode_narrative | atomic | native: turns of that segment | | a segment contains several turns but is treated as atomic (rewriting the whole segment is a rewrite) |
| | semantic_fact | aggregate | diff: union of the turns of the source segments | | |
| LightMem | sensory_entry / stm_buffer_message | atomic | diff + actual original turn | single turn | native compression/segmentation buffer; fully flushed before Full |
| | stm_topic_summary | — | no independent persistent object in the current flat lightweight configuration | | topic segments and extraction batch boundaries are still recorded |
| | ltm_entry | aggregate | diff + same-call fact `source_ids`; at consolidation only inherits the sources of the contributing parent versions pointed to by `source_memory_ids` | | native extraction at ingestion; one offline consolidation run at the end of the build; complete input batch stored separately for checking |
| SimpleMem | memory_entry | aggregate | diff + contributing source numbers of the same-call fact; covers the fact text and all structured fields | native sliding window 40 / overlap 2, single bulk path | one LanceDB row per object; the three indexes are retrieval views of the same object; local MiniLM 384, no independent global rewrite; frozen after the whole batch tail is processed |
| Raw RAG | turn | atomic | = that turn | single turn | control |

Raw-turn RAG is integrated: each original user/assistant turn is stored as one object, encoded with local MiniLM, with no write LLM during the build; exact cosine retrieval, default answer k16, frozen separately from the deletion-candidate top20. Each selected turn corresponds to exactly one object to delete; text and vector are removed together. The original text in its residual comes from the surviving objects themselves; a deleted turn must not be restored from the research-side source. Ranking and answer models can share a compatible Full; the encoder, length limit, tie order and reuse conditions are in [Raw-turn RAG integration](docs/integrations/RAW_TURN_RAG.md).

**kind determination rule** (for filling in the table): src is always a single turn or a single contiguous segment → atomic; src spans segments or is global → aggregate.

MemoryBank fixes kind by the native object type as in the table above: pair is atomic, daily/overall summaries and personality are aggregate; a summary is not reclassified as atomic because its sources happen to be contiguous or because only one source remains after a rewrite.

Nemori/LightMem also fix kind by the actual representation: episode is atomic, semantic/ltm are aggregate; a generated fact is not reclassified as atomic merely because the sources of one input happen to be contiguous or only one remains. The table above is the current fixed lightweight configuration; other upstream configurations require a separate check of the storage layer.

SimpleMem's generated facts are likewise fixed as aggregate and are still deleted whole. All windows of its single bulk pass share the pre-build context (empty); when resuming from a checkpoint, the outputs of already-committed windows are not additionally pushed into later prompts. If it is extended to a serial prior-context memory path, the input versions of the contributing parent objects must be recorded; the bulk formation identity must not be reused for it.

---

## 3. Deletion primitive

### 3.1 Definition of `delete_turn(t)` (current main experiment: pure deletion)
```
for o in list(der[t]):
    delete(o.obj_id)                         # atomic / aggregate alike are deleted whole
    remove from the live-object mappings of src/der/ever/touched
record the selected source turn; assert der[t] is empty
```
- When several turns are selected, their object IDs are merged first and each object is deleted only once; the storage and index deletion interfaces of the corresponding system are called.
- Deletion makes no LLM calls, passes no text of the remaining turns, and regenerates nothing; S2 candidate ranking and the two-readout answering still call the LLM.
- An object with several turns is deleted whole and the collateral effect is recorded separately; the budget is still the number of selected source turns, not the number of objects, and collateral turns are not added to the budget.
- Independent branches are opened only from a Full that has had nothing deleted; S0 deletes the derived closure of D whole, S1/S2 share the candidate pool rebuilt on top of S0 and each deletes independently.
- Results of the discontinued "delete atomic + regenerate aggregate" variant are kept as a historical version and cannot be used as results of this version; where the formation identity is fully compatible, Full can be reused directly, but deletion and evaluation need a new plan. The LightMem fine-grained sources changed the formation prompt and the source rules, so its older batch-associated Full is not compatible.

### 3.2 Appendix variants (not used in the main experiment)
- **Historical regeneration variant**: atomic objects deleted whole; aggregate objects regenerated from the remaining sources. Only used to understand the frozen results of earlier versions; not used in the current main experiment.
- **System-native deletion**: only the system's own delete API is called, with no closure.

---

Nemori candidate-size comparison: under the same Full, whole-deletion rule, B10 and two-readout 2/2, N_OBJ=N_TURN=20 and 40 are frozen separately; the native QA retrieval k is unchanged. 40 means the native retrieval parameter and the candidate-turn cap are enlarged together, not that candidates are padded by repetition. The default remains 20; each run's PLAN.protocol.n_obj/n_turn records the override value explicitly. The candidate-size comparison only compares new results with the same deletion semantics.

## 4. Candidate pool and deletion strategies

### 4.1 Constants
```
N_OBJ   = 20     # number of objects taken from native retrieval (retrieval budget)
N_TURN  = 20     # cap on the number of turns in the candidate pool
K       = 10     # main-table deletion budget (number of additionally deleted turns)
K_LIST  = [1,2,3,5,7,10,15,20]   # residual curve
```

### 4.2 Candidate-pool construction (shared by S1 and S2)
```
objs = search(claim.deletion_request, N_OBJ)          # the system's native objects
cand = {}                                              # turn_id -> {score, exposed_objs}
for o in objs:
    for t in src[o.obj_id]:
        if t in claim.disclosure_turns: continue       # D_c already handled by S0
        cand[t].score = max(cand[t].score, o.score)
        cand[t].exposed_objs.append(o)                 # system-exposed view: text of the retrieved objects under this turn
pool = top N_TURN of cand by score
```
Notes:
- Tracing an aggregate object back brings in several turns; they all enter cand, but only as **candidates**; whether they are deleted is decided by the ranking and k. An aggregate object is deleted whole when any of its source turns is selected
- Strategies can only see the text of `exposed_objs`; they cannot see the turn text, E_c, X_c, sufficient_sets or the answer

### 4.3 Strategies

| Strategy | Deletion set | Ranking |
|---|---|---|
| S0 | D_c | — |
| S1 | D_c ∪ first k of pool | by cand.score |
| S2 (LLM-guided cleanup) | D_c ∪ first k of pool | the LLM scores each turn in the pool 0–10 for relevance to the deletion request; input = deletion_request + the stored text the system exposes for that turn; score and reason are recorded |

There are only these three strategies. S1 and S2 each rank once; different k take prefixes.

S2 (LLM-guided cleanup) is positioned as a diagnostic deletion baseline: under the same memory system, the same model, the same candidate pool and deletion budget, its two-readout recovery rates are compared with the native ranking S1 to observe the difference the selection strategy makes. It is not required to eliminate all residue, nor to beat S1 on both readouts at once; both readouts are reported as they are.

### 4.4 Execution flow per claim
```
restore(snapshot)
freeze_updates(True)
for t in D_c: delete_turn(t)                 # S0
record eval("S0")
pool = build_pool()
order = rank(pool, strategy)                 # S1 or S2
for i, k in enumerate(K_LIST):
    for t in order[prev_k:k]: delete_turn(t) # prefix grows, no rollback needed
    record eval(f"{strategy}@{k}")
discard working copy
```

---

## 5. Evaluation

### 5.1 Scoring rule (shared by both readouts)
- The recovery model is fixed, temperature 0
- Two option permutations (two fixed permutations); both correct → recovered = 1; a valid answer that is wrong in either → 0
- Use the trial1 and trial2 fixed in advance in the source contract; when an older contract contains trial3, the source file is kept, but new runs neither request nor count the third trial, and permutations are never picked by whether the answer was right
- The native/residual of Full, S0, S1, S2, as well as residual+ and the full-evidence / no-memory controls, all use the same 2/2 rule; missing, malformed and truncated answers are marked incomplete, not counted as 0
- Leakage rate = proportion of claims with recovered; claim-level bootstrap 95% CI

Earlier versions of the protocol used a 3/3 rule; the current rule is 2/2. Existing results frozen under 3/3 keep their original label; new plans carry an independent evaluation version. The definitions of store construction, deletion and context do not change with the number of permutations, so Full/checkpoints with a compatible formation identity and the replies to identical requests can be reused without rebuilding the store.

**MemoryBank execution extension: early stop on a valid wrong answer.** New MemoryBank main experiments use `tools/memorybank_efficient_s012.py` and freeze `two-trial-valid-failure-short-circuit-v1` separately in `PLAN.answer_execution`. trial1 is always run first: if it yields a valid and wrong option, 2/2 can no longer hold, so the claim is recorded as not recovered and trial2 is explicitly recorded as not executed; if trial1 is correct, the fixed trial2 must be run, and only two valid and correct answers count as recovered. A timeout, format error or truncation of any executed request is still incomplete, not 0, and does not trigger an automatic paid retry.

This extension keeps the two-readout recovery judgement, the main denominator, the shared Full-residual conditional set and all inputs; "stopped by plan after a valid error" and "abnormal missing" are recorded separately, and fabricating a second answer or passing a stop off as an empty residual is forbidden. Stopped questions have no second-trial stability diagnostic data. It applies only to new MemoryBank plans that explicitly adopt the strategy; other systems and historical full two-trial plans keep their original identity. Implementation and offline equivalence verification: [MemoryBank execution savings](docs/engineering/MEMORYBANK_EXECUTION_SAVINGS.md).

### 5.2 Native readout
```
ctx_objs = search(claim.mcq.stem, N_EVAL)     # the stem is the query, options excluded; N_EVAL = the system's default top_n, fixed across strategies
context  = system_render(ctx_objs)            # the system's own format as shown to the agent
for perm in PERMS:
    answer = recovery_model(context, stem, permute(options, perm))
recovered_native = all correct
```
Meaning: whether the secret leaks under an ordinary question.

### 5.3 Residual readout
**What to feed**: the objects still derived from the evidence turns and disclosure turns after deletion.
```
R = set()
for t in claim.evidence_turns + claim.disclosure_turns:
    R |= der[t]                               # objects still derived at present
    R |= touched[t]                           # generic historical association; objects deleted whole have already left this set
objs = [get_object(oid) for oid in R]         # only objects that are still alive
objs.sort(by first write time)
context = "\n".join(o.text for o in objs)     # object text verbatim; turn text is not added
if not objs: recovered_residual = 0; count "evidence_derived_empty"
else: the two permutations of 5.2, both correct
```
Rules:
- Feed object text, not turn text (the original text may no longer be in the system)
- Deletion under this version produces no rewritten objects. After S0, the deleted derived objects of D do not come back via touched; when all E/D derived objects have been deleted, the residual input is empty and is recorded as 0 as a structural empty set. touched is kept for ingestion-history tracing and for explaining historical versions; it cannot bring deleted objects back.
- Do not feed objects derived from distractor turns
Meaning: whether the remaining memory can in principle still support recovery, with retrieval factored out.

### 5.4 Residual+ (optional sensitivity check)
```
extra = ∪_{t ∈ E_c} search(turn_text(t), 3)   # retrieve with the original evidence text
R_plus = R ∪ extra
```
residual+ more than 5 points above residual → the provenance table has a gap; go back and check the diffing.

### 5.5 Controls and denominators
| Condition | Procedure |
|---|---|
| Full evidence | feed the original text of E_c ∪ D_c directly (data check, ≈100%) |
| Encoding rate | residual before deletion (Full row) |
| Native before deletion | native before deletion |
| No memory | only the stem and options |

- Unconditional leakage rate: denominator = all claims (main table)
- Conditional leakage rate: denominator = claims whose Full-residual was correct on both permutations; all strategies and both readouts share this one fixed set

### 5.6 Utility Retention
- Before deletion, run the retention question bank twice and take the stably correct set Q₊
- UR@k = proportion of Q₊ still answered correctly after deletion (native readout)
- Attribution of flipped questions: whether their supporting objects (der of the distractor turns) were deleted/rewritten → caused by deletion vs noise

---

## 6. Output and logs (the analysis experiments depend on these)

For each (system, write_model, claim, strategy, k) record:
```
deleted_turns, rewritten_objs, deleted_objs
pool: [{turn_id, score, exposed_obj_ids, llm_score, llm_reason}]
native: {retrieved_obj_ids, context, answers[2], recovered}
residual: {R_obj_ids, context, answers[2], recovered}
residual_plus (if run)
ur: {q_ids_correct_before, q_ids_correct_after}
```
Attribution analysis needs: for each turn in E_c, whether it is in pool, its llm_score, whether it is in order[:k], and whether der[t] is empty after deletion.

---

## 7. System integration interface

```
list_all()                       -> [MemObject]
search(query, top_n)             -> [MemObject + score]
delete(obj_id)                   -> None
regenerate(obj_id, src_turn_ids) -> None      # extension for the historical variant; not called on the current pure-deletion path
snapshot() / restore(snap_id)    -> store files + provenance table, saved and loaded as a whole
freeze_updates(flag)             -> disable all writes triggered by reads (heat, forgetting curve, predict-calibrate, offline consolidation)
system_render(objs)              -> str        # the context format the system shows to the agent
```
`trace(obj_id)=src[obj_id]` and `reverse_trace(turn_id)=der[turn_id]` are provided by the provenance table and do not depend on the system.

System-specific notes:
- **MemoryBank**: personality_global is also deleted whole; if the forgetting curve affects retrieval weights, freeze it
- **Nemori**: confirm the in-memory/SQLite backend; confirm whether search triggers writes and freeze them if so; record episode segment boundaries together with the native source field
- **LightMem**: offline consolidation runs only once, at the end of the build; record topic segment boundaries; confirm whether the sensory layer is traceable

---

## 8. Acceptance checks for a new integration (pass all of these on one system and one episode first)
1. Provenance spot check: 30 random objects, src confirmed by hand; no invariant assertion fails at any point
2. Whole-deletion correctness: all objects in the derived closure disappear, the text and sources of the other objects are unchanged; the number of LLM calls during deletion is 0; storage and index are consistent.
3. Encoding rate: Full-row residual ≥ 70%, otherwise find out which kind of reasoning is being lost
4. residual vs residual+ differ by ≤ 5 points
5. S0 row: report the actual native/residual change; whole deletion may delete E information as collateral, so do not assume residual does not drop.
6. Prefix growth gives the same results as a full rerun (verify on 5 sampled claims)

---

## 9. Cost (one system, one write model, 456 claims)
- Store construction: normal ingestion cost, zero extra for diffing
- Per claim: executing S0/S1/S2 deletions makes 0 LLM calls; S2 ranks at most 20 candidate turns; each condition makes 2 answers per readout (a structurally empty residual makes no model request).
- 8 k points ≈ 456 × (20 + 8×3 + 8×6) ≈ 42k calls
