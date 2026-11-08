# The five systems

Every system runs through `python run.py aligned --system <name>`, with its own upstream code at the commit pinned in
`configs/`. Each claim branches independently from the same Full memory; S0 removes the disclosure turns, S1/S2
remove up to ten further source turns from a 20-turn candidate pool. Deleting a turn removes every object derived
from it, atomic or aggregate, and nothing is regenerated. Native recovery asks the system's own retrieval and
renders its own context; residual recovery reads every surviving object derived from the evidence or disclosure
turns, in order of first write.

| `--system` | What is stored | Native readout |
|---|---|---|
| `turnrag` | one object per original turn, text unchanged; local encoding, no LLM | 16 turns |
| `sessionrag` | one object per session, indexed by chunk embeddings; deletion removes the exact turns and re-encodes locally | 16 sessions by default (`--native-eval-k`) |
| `memorybank` | raw QA pairs, daily summaries and the user portrait (English FAISS path; forgetting off) | top 10 entries |
| `nemori` | topical episodes rewritten as narratives, and semantic memories distilled from them | top 10 episodic and top 20 semantic memories, with the original messages of the top two episodes |
| `lightmem` | compressed, topic-segmented short-term entries consolidated offline into long-term entries (compression rate 0.6) | top 60 entries |

## Retrieval encoders (Turn RAG, Session RAG)

`--embedding e5-small-v2` (default) or `--embedding all-MiniLM-L6-v2`; both are registered in
`configs/embeddings.json`, and e5 receives its `query: ` / `passage: ` prefixes. See [USAGE.md](USAGE.md).

## Provenance

Deletion and the residual readout need, for every stored object, the turns it derives from. The two RAG systems
store turns verbatim, so this holds by construction. For MemoryBank and Nemori it is recovered by listing every
stored object before and after each ingestion unit and attributing new and changed objects to that unit; where a
system exposes native source fields (Nemori's segments), they are joined with that attribution. LightMem's
extraction and update calls return the message numbers each memory derives from (`memcut/source_anchored_systems/
lightmem_sources.py`), and those sources are recorded. Updates triggered by reading (forgetting curves, portrait
refreshes, offline consolidation) are frozen after ingestion, so evaluation never changes the store.

## MemoryBank

Native QA reads the top 10 entries rather than upstream's default of two, so that its retrieval depth is
comparable to the other systems. When the first of the two display orders is answered validly and wrongly, the
second is not sent: the target cannot be recovered 2/2.

## Nemori and LightMem

Nemori runs with native episode merging and prediction correction, and flushes the final partial batch. LightMem
flushes at the last turn as upstream does; turns whose segments upstream then replaces are recorded in the run's
diagnostics.
