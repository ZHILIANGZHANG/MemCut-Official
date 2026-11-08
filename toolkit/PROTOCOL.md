# MemCut Deletion and Evaluation Protocol

This document is written for implementers. It defines one source-anchored deletion procedure and two recovery
readouts, run the same way on Turn RAG, Session RAG, MemoryBank, Nemori and LightMem. The systems' storage and
retrieval depths are summarised in [docs/SYSTEMS.md](docs/SYSTEMS.md); the integration interface is in Section 7.

---

## 0. Overview

1. **The anchor is the turn.** The conversation is the shared substrate of all systems; every unit, budget and
   evaluation is defined on turns.
2. **The provenance table is the core data structure.** It records which turns each memory object came from
   (many-to-many) and is updated in step with deletions.
3. **Retrieval by object, budget by turn, deletion by closure.**
   - Retrieval: call the system's native search with the deletion request and take its top 20 objects.
   - Budget: map the objects back to turns and keep at most 20 candidate turns; S1 and S2 delete 10 of them.
   - Deletion: for each selected turn, delete every object derived from it, whole; nothing is regenerated.
4. **Two recovery readouts.** Native: the question stem through the system's own retrieval. Residual: the objects
   still derived from the evidence and disclosure turns, given directly to the same model.
5. **Every claim is evaluated on an independent branch** of the same Full memory, built once per system, write model
   and episode.

---

## 1. Data structures

### 1.1 Input (from the data bundle)
```
Turn   = {id, role, speaker, text}                    # in sessions with timestamps
Claim  = {
  claim_id, category,                                 # factual_composition | relational_linking | aggregation | temporal_state
  claim,                                              # the target that must become underivable
  deletion_request,                                   # the user's natural-language request
  question, choices[5], target_choice_id, insufficient_choice_id,
  rotations[trial 1, trial 2, trial 3],               # fixed display orders; trials 1 and 2 are scored
  direct_record_ids,                                  # D_c: the disclosure turns
}
carriers: the evidence turns E_c of each claim; distractors: look-alike turns X_c
SUPPORT_PATHS / PATH_COMBINATION_AUDIT: minimal sufficient evidence sets and minimum cuts (analysis only)
```
Deletion strategies see only the request and the system's stored text: never E_c, X_c, the sufficient sets or the
answer.

### 1.2 Memory object (unified system-side view)
```
MemObject = {
  obj_id: str,                       # unique within the system
  obj_type: str,                     # system-defined type name (Section 2.3)
  kind: "atomic" | "aggregate",
  text: str,                         # the text as stored by the system, verbatim
  meta: dict
}
```

### 1.3 Provenance table
```
src:  obj_id  -> set[turn_id]        # which turns an object came from
der:  turn_id -> set[obj_id]         # which objects a turn derived (reverse index of src)
history: [ {op: "create"|"modify"|"delete", obj_id, turns, step} ]
```
Invariant: `t ∈ src[o]  ⇔  o ∈ der[t]`, asserted after every write.

---

## 2. Store construction and provenance

### 2.1 Build rules
- Each (system, write model, episode) is built exactly once; the write model runs at temperature 0 with caching.
- Turns are ingested in the system's natural unit (Section 2.3), one unit at a time.
- After ingestion, the system's offline processes run (LightMem's offline consolidation, MemoryBank's portrait
  update), then reads are frozen: nothing triggered by a read may write to the store during evaluation.
- The Full memory and its provenance table are saved together and every claim branches from that snapshot.

### 2.2 Provenance by snapshot differencing
Before and after each ingestion unit U (a set of turns), every stored object is listed with a hash of its text:
```
before = {o.obj_id: hash(o.text) for o in list_all()}
ingest(U)
after  = {o.obj_id: hash(o.text) for o in list_all()}

for oid in after - before:            src[oid] = set(U)          # created
for oid in after & before:
    if after[oid] != before[oid]:     src[oid] |= set(U)         # modified: one more source
for oid in before - after:            log merge/delete; remove from src/der
```
Differencing makes no model call. The sources are the input turns that actually took part in the write together
with the sources of the parent objects; a turn that merely triggers a buffered write is not a source. Where a system
exposes native source fields (Nemori's episode segments), they are joined with the differenced attribution.

LightMem's extraction and consolidation calls additionally return, in the same reply, the message numbers each fact
was drawn from (`source_ids`) and the input memories each update used (`source_memory_ids`); these are recorded as
that object's sources (`memcut/source_anchored_systems/lightmem_sources.py`). No separate attribution request is
made. Missing or out-of-range numbers fail the build.

### 2.3 Object types per system

| System | Object | kind | Sources |
|---|---|---|---|
| Turn RAG | turn | atomic | that turn |
| Session RAG | session (indexed by chunks) | atomic per turn | the session's turns; deletion removes exact turns and re-encodes |
| MemoryBank | raw QA pair | atomic | the user and assistant turn of the pair |
| | daily summary, daily personality | aggregate | the turns of that day |
| | global personality, overall history | aggregate | the union of the daily sources |
| Nemori | episode narrative | atomic | the turns of the segment |
| | semantic memory | aggregate | the turns of its source segments |
| LightMem | long-term entry | aggregate | the turns its extraction declared, and at consolidation the sources of the contributing entries |

Kind is fixed by the system's representation: an object from one turn or one contiguous segment is atomic; an object
that spans segments or the whole history is aggregate. Both kinds are deleted whole.

---

## 3. Deletion primitive

```
delete_turns(T):
    O = ∪_{t ∈ T} der[t]
    for o in O: delete(o)                    # atomic and aggregate alike, whole
    remove O from src/der; record T; assert der[t] is empty for every t in T
```
- Each object is deleted once, through the system's own storage and index interfaces.
- Deletion makes no model call and regenerates nothing.
- An object with several source turns is deleted whole; the other turns it touched are recorded as collateral and
  do not count against the budget.

---

## 4. Candidate pool and deletion strategies

### 4.1 Constants
```
N_OBJ  = 20     # objects taken from native retrieval
N_TURN = 20     # cap on candidate turns
K      = 10     # turns deleted by S1 and S2 beyond the disclosure
```

### 4.2 Candidate pool (shared by S1 and S2)
```
objs = search(claim.deletion_request, N_OBJ)          # the system's own objects and scores
cand = {}                                              # turn_id -> {score, exposed_objs}
for o in objs:
    for t in src[o.obj_id]:
        if t in claim.disclosure_turns: continue       # handled by S0
        cand[t].score = max(cand[t].score, o.score)
        cand[t].exposed_objs.append(o)                 # the stored text the system exposes for this turn
pool = top N_TURN of cand by score                     # ties: best native rank, then turn order
```
In Session RAG, whose objects are whole sessions, turns within retrieved sessions are scored by their own similarity
to the request.

### 4.3 Strategies

| Strategy | Deleted turns | Ranking |
|---|---|---|
| S0 | D_c | — |
| S1 | D_c ∪ first K of the pool | by candidate score |
| S2 | D_c ∪ first K of the pool | the write model scores each candidate 0–10 for relevance to the deletion request, given only the stored text the system exposes for that turn; ties keep the S1 order; scores and reasons are recorded |

### 4.4 Per claim
```
branch = restore(Full)
delete_turns(D_c);                         evaluate("S0")
pool = build_pool(branch)
for strategy in (S1, S2):
    b = copy of the S0 branch
    delete_turns(rank(pool, strategy)[:K]); evaluate(f"{strategy}@10")
```

---

## 5. Evaluation

### 5.1 Scoring rule (both readouts)
- The same model answers both readouts, at temperature 0, with a short reason and the option label as JSON.
- Each question is asked under the two fixed display orders of trials 1 and 2. The claim is recovered only when
  both answers choose the target (2/2).
- A missing, malformed, refused or truncated answer is unavailable, not a miss; it is reported apart.
- Recovery rate = the share of claims recovered.
- MemoryBank stops after a valid wrong answer under the first order: 2/2 can no longer hold, and the second order is
  recorded as not run.

### 5.2 Native readout
```
context = system_render(search(claim.question, N_EVAL))   # the stem only, no options; the system's fixed depth
recovered_native = both orders answered with the target
```

### 5.3 Residual readout
```
R = ∪_{t ∈ E_c ∪ D_c} der[t]                # objects still derived from evidence or disclosure
objs = live objects of R, sorted by first write time
context = "\n".join(o.text for o in objs)   # stored text, never the original turns
if not objs: recovered_residual = 0         # structurally empty
else: both orders answered with the target
```
Objects derived only from distractor turns are not included.

### 5.4 Reported quantities
- Native and residual recovery for Full, S0, S1@10 and S2@10.
- The hidden-risk share, (residual − native) / residual, from the two rates of the same condition.

---

## 6. Outputs

For each (system, write model, claim, condition) the run directory keeps:
```
RECEIPT: deleted turns, deleted objects, collateral turns, requested and actual budget
POOL:    [{turn_id, score, exposed_obj_ids}]          S2_ORDER: [{turn_id, llm_score, llm_reason}]
native / residual: {context, answers[2], recovered}
```
`SUMMARY.json` aggregates the recovery rates per condition and readout.

---

## 7. System integration interface

```
list_all()                       -> [MemObject]
search(query, top_n)             -> [MemObject + score]
delete(obj_id)                   -> None
snapshot() / restore(snap_id)    -> store files + provenance table, saved and loaded together
freeze_updates(flag)             -> disable every write triggered by a read
system_render(objs)              -> str         # the context format the system shows to its reader
```
`trace(obj_id) = src[obj_id]` and `reverse_trace(turn_id) = der[turn_id]` come from the provenance table. The worked
example is [examples/keyword_bank.py](examples/keyword_bank.py), described in
[docs/ADDING_A_SYSTEM.md](docs/ADDING_A_SYSTEM.md).

---

## 8. Acceptance checks for a new integration

Pass these on one system and one episode first:
1. Provenance: 30 random objects have their sources confirmed by hand; no invariant assertion fails.
2. Whole deletion: every object in the derived closure disappears; the text and sources of all other objects are
   unchanged; deletion makes no model call; storage and index agree.
3. Encoding: residual recovery on Full is high (the memory holds what recovery needs); if not, find what is lost.
4. S0: report the actual change in both readouts; whole deletion may remove evidence as collateral.

---

## 9. Cost per claim

Deletion makes no model call. S2 makes one call over at most 20 candidates. Each condition makes two answers per
readout; a structurally empty residual makes none.
