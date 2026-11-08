# MemCut

A benchmark for **memory unlearning** in long-term agent memory: a user asks the assistant to forget one thing, and
the question is whether the thing stays derivable from what the memory still holds.

Release `memcut-public-v1`, created 2026-11-08, content id `5ba89f004ddd`. Main set: 15 episodes, 490 sessions,
43,402 messages, 456 deletion requests. One case-study episode, EP08A, is described below and counted apart.

## What is in an episode

Each episode is a long user–assistant history (26–40 sessions, about 2,000–3,700 messages) in a fictional
setting, written by the benchmark authors. Each of its claims is a fact the user later asks the assistant to forget:

- **The target** (`claim`): the sentence that must become underivable.
- **The disclosure** (`direct_record_ids`): the one or two messages that state the target outright. Every evaluated
  system has them deleted first (step S0), so the benchmark measures what happens *after* the obvious record is gone.
- **The evidence** (`carriers`): 6–10 ordinary messages, spread over 5–10 sessions, from which the target can still
  be derived once the disclosure is gone. No single message and no single session suffices.
- **The distractors**: near-miss messages that look relevant and are not evidence.
- **The readout**: a five-option multiple-choice question with one option stating the target and one
  "insufficient information" option, asked under two display orders.

Four categories, about equally many each: `factual_composition`, `relational_linking`, `aggregation`,
`temporal_state`.

## Protocol in one paragraph

Build the memory from the full dialogue (the `public/` side only). For each claim, start from the full memory,
delete the disclosure (S0), then let the system perform its own deletion on the request (S1/S2: its own cleanup).
Read out twice: **native**, the system's own retrieval and answering on the question; **residual**, the question
answered from everything the evidence turns still derive (what a reader could reconstruct). The target counts as
recovered only when the reader picks it under both display orders (2/2). Report the share of claims whose target is
still recovered (lower is better), and retention: questions about unrelated facts the full memory answered that are
still answered after the deletion (higher is better). The authors' reference code is the MemCut evaluation toolkit,
released with this data.

## Files

```
MANIFEST.json            sizes and SHA-256 of every file; registry_sha256 is the bundle's content id
README.md  LICENSE  SCHEMA.md  validate.py
data/EPxx/public/EPxx.json           the dialogue (the only thing a memory system may read)
data/EPxx/author/CLAIM_CONTRACTS.json  targets, requests, questions, options, display orders
data/EPxx/author/ANNOTATIONS.json      evidence carriers, disclosures, distractors (record ids + text hashes)
data/EPxx/author/TOKEN_COUNTS.json     token counts of the dialogue
data/EP01/author/SUPPORT_PATHS.json, PATH_COMBINATION_AUDIT.json   minimal sufficient evidence sets (EP01, EP08A)
data/EP08A/...                       the case-study episode (same layout)
```

`python3 validate.py memcut_v1.zip` checks the manifest, the schema and every annotated record against the dialogue.
The field-by-field description is in `SCHEMA.md`.

## How the data was made

Every decision about content was made by the benchmark authors, against a written construction guide: the scenario
and cast, then each fact with its deletion request, then where the disclosure falls and which turns carry evidence,
then distractors that share surface features with the evidence, then the question and its options. A language model
phrased each designed turn as dialogue; the authors checked every turn against its design and sent deviations back
to be phrased again (the paper's Appendix A gives the checks). The evidence annotations and minimal sufficient sets
are the authors' judgement, not the output of a model run.

## The case-study episode EP08A

The paper's case study (Figure 4, Appendix E) follows one relation-chain fact: Arden Advisory, the firm Linden hired,
belongs to Nina Shah, the wife of Linden's CFO, Ravi Shah. EP08A is the case-study data, which is why it is listed
apart from the main set: it is EP08 with 48 messages appended at the ends of existing sessions, every EP08 message
keeps its id, text and position, and it is annotated for this one fact (`c33`) only; the 32 EP08 facts are
evaluated on EP08.

The fact rests on four links, each carried by two evidence messages: hiring (`c33-e1`, `c33-e2`), ownership
(`c33-e3`, `c33-e4`), marriage (`c33-e5`, `c33-e6`) and the CFO's identity (`c33-e2`, `c33-e7`). Recovery needs one
message from every link, so the minimal cuts are {e1, e2}, {e3, e4}, {e5, e6} and {e2, e7}; `SUPPORT_PATHS.json` and
`PATH_COMBINATION_AUDIT.json` list the eight minimal sufficient sets and the cuts. The disclosure is in session S28,
after all evidence, in a session without evidence. Twenty distractors pair with evidence messages; four of them are
existing EP08 messages.

EP08A is built from a declarative overlay by the toolkit's `tools/extend_episode.py`, the same tool anyone can use
to add a fact to an episode.

## Using it honestly

The answers and the evidence ship with the data, so there is no hidden test set. When you report numbers on MemCut:

- build the memory from `public/` only; nothing under `author/` may reach the system under test;
- say which episodes you used for development; the authors' own work used EP01 for development and EP02–EP15 as a
  test set run once after the configuration was frozen;
- report both readouts and retention, not one of them;
- report the main set (EP01–EP15) and the case-study episode separately.

The timeline is fictional (session timestamps run through 2026); nothing in the data refers to real persons,
companies or events.

## License and citation

Data: CC BY 4.0 (see `LICENSE`). Please cite the MemCut paper:

```bibtex
@misc{memcut2026,
  title         = {MemCut: Every Secret Leaves a Trail in the Long-Term Memory of LLM Agents},
  year          = {2026},
  eprint        = {2610.21405},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.21405}
}
```
