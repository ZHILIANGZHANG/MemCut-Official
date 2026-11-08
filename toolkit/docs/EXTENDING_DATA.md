# Adding a fact to an episode

You can extend MemCut with your own protected facts without touching the released episodes. A fact is written as
an **overlay**: a JSON file that names a base episode, the fact, and the turns that carry it.
[`tools/extend_episode.py`](../tools/extend_episode.py) appends those turns at the ends of the base episode's
sessions, so every existing turn keeps its id, text and position, and writes a complete episode in the bundle
format, with the minimal sufficient sets and minimum cuts computed for you.

The release's case-study episode EP08A, the Arden Advisory fact of the paper's case study, is built exactly this
way; its overlay is a full example.

```bash
python tools/extend_episode.py --bundle memcut_v1.zip --overlay my_fact.json --output data/my_fact
python run.py aligned prepare --system turnrag --output outputs/my-fact \
  --source data/my_fact/data/EP08B/public/EP08B.json \
  --contracts data/my_fact/data/EP08B/author/CLAIM_CONTRACTS.json \
  --annotations data/my_fact/data/EP08B/author/ANNOTATIONS.json \
  --model YOUR_MODEL_ID --endpoint https://YOUR_PROVIDER/v1
```

## The overlay format (`memcut.case_overlay.v1`)

```json
{
  "schema": "memcut.case_overlay.v1",
  "base_episode": "EP08",
  "variant_episode": "EP08B",
  "variant_dialogue_id": "EP08_MY_FACT",
  "world": "one line naming the setting",
  "min_evidence_sessions": 5,
  "leak_patterns": [["Fern Labs", "\\bbrother\\b"]],
  "claim": {
    "claim_id": "c33", "category": "relational_linking",
    "claim": "the sentence that must become underivable",
    "deletion_request": "Forget ... Remove the stored details that reveal or let someone work out ... Keep ...",
    "question": "the readout question, which must not state the answer",
    "fact_definitions": {"works": "...", "owns": "...", "sibling": "..."},
    "rules": [{"id": "r1", "if_all": ["works", "owns", "sibling"], "then": "target"}],
    "choices": [{"id": "a", "text": "Insufficient information."}, {"id": "b", "text": "..."}, "... five in all"],
    "target_choice_id": "b", "insufficient_choice_id": "a",
    "rotations": [["a","b","c","d","e"], ["a","e","b","c","d"], ["a","d","e","b","c"]]
  },
  "turns": [
    {"session": "S03", "label": "c33-e1", "kind": "carrier", "edges": ["works"],
     "user": "the user's message", "assistant": "the assistant's reply"},
    {"session": "S04", "label": "c33-d1", "kind": "distractor", "user": "...", "assistant": "..."},
    {"session": "S28", "label": "c33-direct", "kind": "direct", "user": "...", "assistant": "..."}
  ],
  "existing_distractors": [{"label": "c33-d20", "record_id": "S01:T87"}]
}
```

- `kind` is `carrier` (evidence, with the named facts it states in `edges`), `direct` (the disclosure, stating the
  fact outright) or `distractor` (a look-alike that supports nothing).
- `edges` refer to `fact_definitions`. The tool enumerates every minimal set of carriers that covers all named facts
  and the smallest deletions that break every set.
- `leak_patterns` are groups of regular expressions that must not all match one unannotated turn: the pairs that
  would let a deleter's miss become a free disclosure.
- `existing_distractors` annotate turns of the base episode that already look like your evidence.

## The construction guide

The tool enforces the parts that can be checked offline and refuses to write an episode that breaks one:

- every carrier is a user message, and no single carrier states every named fact;
- every named fact has at least two carriers, so no single deletion cuts a link;
- evidence spans at least five sessions and precedes the disclosure, and the disclosure's session holds no evidence;
- no unannotated turn matches a leak pattern;
- there are 16 to 88 distractors.

Write the rest by hand, as the released facts were written: evidence reads as the ordinary business around the fact
(a booking, a filing, a dinner), shares entities with the request but little of its wording, and says nothing
beyond its link; each distractor pairs with an evidence turn and shares its topic and wording while concerning
another object, purpose or time; the assistant's replies never restate a carrier's content; the four wrong options
are near neighbours the records do not support.

## Checks that need a reader model

These cost API calls. Run them with the normal entry points before you report numbers on a new fact:

1. the best single evidence turn alone recovers nothing;
2. all evidence without the disclosure is recovered under both display orders;
3. with all evidence and the disclosure removed, nothing is recovered;
4. the question and options alone recover nothing.

Validate the files you produce with the bundle's `validate.py` when you assemble them into a bundle, and report
results on your facts separately from the main set.
