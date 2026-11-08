# MemCut bundle schema

Format id `memcut-15ep-data-bundle-v1`. One directory per episode, `data/EPxx/`, with a `public/` and an `author/`
side. The `public/` side is what a memory system may read; the `author/` side is evaluation metadata and must never
be shown to the system under test.

## `MANIFEST.json`

| field | meaning |
|---|---|
| `format` | `memcut-15ep-data-bundle-v1` |
| `release`, `created_date` | release name and date |
| `registry_sha256` | the bundle's content id: SHA-256 over every file's SHA-256 in episode order |
| `totals` | the main set only (EP01–EP15) |
| `case_study_totals` | the case-study episodes, when present |
| `episodes[]` | `id`, `world` (a one-line fictional setting), `sessions`, `turns`, `claims`, `files{kind: {path, sha256, bytes}}` with kinds `source`, `contracts`, `annotations`, `tokens`, `support_paths`, `path_audit`; a case-study episode also has `set: "case_study"` and `base_episode` (an entry without `set` is in the main set) |

## `public/EPxx.json` (the dialogue)

```
{ "schema": "memcut.long_episode.source.v1", "id", "language": "en", "timezone", "subject", "source_unit": "one_message_turn",
  "sessions": [ { "id": "S01", "timestamp": ISO-8601, "turns": [ { "id": "S01:T01", "role": "user" | "assistant", "speaker", "text" } ] } ] }
```

A turn is one message. Turn ids are `<session>:<turn>` and unique within the episode. Timestamps are the fictional
timeline's session times; relative dates in the text ("last Tuesday") resolve against them.

## `author/CLAIM_CONTRACTS.json`

`schema`, `source_sha256` (SHA-256 of the dialogue file), `scoring` (the readout rule), `claims[]`:

| field | meaning |
|---|---|
| `claim_id` | `c1`, `c2`, ... |
| `category` | `factual_composition`, `relational_linking`, `aggregation`, `temporal_state` |
| `claim` | the target: the sentence the deletion must make underivable |
| `deletion_request` | the user's request, as a memory system receives it |
| `question`, `neutral_question` | the readout question (identical unless the authors kept a separate neutral wording) |
| `protected_propositions` | the propositions the target decomposes into |
| `fact_definitions`, `rules` | the named facts the evidence carries and the derivation rules that combine them (when the authors wrote them down) |
| `choices[]` | five options `{id, text}`; exactly one states the target, one is the insufficient-information option |
| `target_choice_id`, `insufficient_choice_id` | which option is which |
| `direct_record_ids` | the disclosure turns: records that state the target outright (deleted before any system's own cleanup, step S0) |
| `rotations[]` | three display orders `{trial: 1|2|3, choice_ids_in_display_order, display_to_choice}`; the protocol reads under trials 1 and 2 |

## `author/ANNOTATIONS.json`

`schema`, `source_sha256`, and three lists of records, each `{key, claim_id, record_id, role, text_sha256}`:

- `carriers`: the evidence turns from which the target can still be derived after the disclosure is gone; a carrier may add `edges` or `supported_facts` (the fact ids of `fact_definitions` it supports), `derived_from` and `parent_record_id` (the earlier record it restates).
- `direct`: the disclosure turns (the same ids as `direct_record_ids`).
- `distractors`: near-miss turns that look relevant and are not evidence; a deletion that removes them is collateral damage.

`text_sha256` is the SHA-256 of the record's `text` in the dialogue; `validate.py` checks it.

## `author/TOKEN_COUNTS.json`

Token counts of the dialogue text under `cl100k_base` and `o200k_base`, and message counts.

## `author/SUPPORT_PATHS.json`, `author/PATH_COMBINATION_AUDIT.json`

Every episode carries both. `SUPPORT_PATHS.json` lists, per claim, the minimal sufficient evidence sets as
`witnesses` (`keys`: carrier keys, `record_ids`: their turns): each set yields the target, and no member can be
dropped. `PATH_COMBINATION_AUDIT.json` repeats the sets as `minimal_source_combinations` and adds the minimum cut:
its size (`minimum_hitting_set_size_in_audited_family`), one cut (`one_minimum_hitting_set`) and every cut of that
size (`all_minimum_hitting_sets`), plus `source_key_to_record_id`. A cut is per claim: the fewest evidence turns
whose deletion leaves no set intact. The readout protocol does not read these files; analyses of deletion
mechanisms do.
