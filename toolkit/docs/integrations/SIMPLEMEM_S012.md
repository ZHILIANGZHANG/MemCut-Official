# SimpleMem S012

This module is the MemCut deletion-experiment adapter, not the earlier LoCoMo experiment. It pins the upstream in `configs/simplemem_s012_v1.json` and the native bulk window 40 / overlap 2; source fields are returned in the same extraction call, forming a new, separate bank. Native planning, multi-path retrieval, reflection and the time reference are kept, as defined by the frozen configuration and the runner's final object-selection rule.

Deletion removes whole objects over the source-turn closure and maintains the local indexes; the main experiment's native and residual readouts are both scored 2/2. This entry does not pass off the unimplemented residual+ or plain retention appendices as measured results.

Use the separate `requirements/simplemem.txt` environment, run `python tools/setup_upstreams.py simplemem` and the required pinned asset downloads, then `python run.py simplemem doctor`. Full usage is in the [installation notes](../USAGE.md).

## Retrieval and index rules

The four points named in the protocol notes, as recorded from the code:

- **Multi-path merge order:** the retriever follows the serial order of upstream `HybridRetriever`: the planned semantic queries run in turn (locally parallel, but merged in query order), then keyword retrieval, then structured retrieval, and finally at most 2 rounds of reflection; the merge only appends entries not yet seen, so an existing prefix is never displaced by later results (`Retriever` in `simplemem_s012/native.py`).
- **Final 20-object cut:** QA uses the complete native path with no extra truncation; the deletion candidate pool requests semantic 20, and once the merged ordered prefix holds 20 objects the remaining paths are skipped (`prefix_stop` records where it stopped).
- **Time reference:** dateparser's `RELATIVE_BASE` is the time of the last turn of the source conversation, not the run day; on a parse exception there is no time range, as upstream (aligned v2).
- **Local index maintenance:** after batched commits, one native Tantivy `en_stem` full-text index is built for Full; because Tantivy stores row positions, the index is rebuilt once per deletion transaction after whole-object deletion. aligned v2 rebuilds with a single-threaded writer, so the same data gives the same BM25 tie order every time; a keyword query rejected by Tantivy (e.g. one containing a possessive) gives no lexical result and records `query_rejected`, as upstream `VectorStore.keyword_search` does.

