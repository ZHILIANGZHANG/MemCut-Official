# Raw-turn RAG

Each original single-role turn is one object, stored verbatim; formation consists only of local MiniLM encoding, with no LLM call. The default QA k is 16, with exact cosine retrieval over normalized vectors; the deletion candidate top-20 is separate from QA k.

S0/S1/S2 delete the objects of the corresponding source turns; S2 ranking and answering still involve model calls. The native context uses the complete retrieved turns; the residual context uses the surviving E/D turns; both follow the two fixed permutations and the rule that missing readouts are listed separately.

Entry: `python run.py turnrag`. Examples for new experiments are in the [usage notes](../USAGE.md).
