# Session RAG portable S012 protocol

Session retrieval uses complete session text with chunk embeddings; deletion removes exact source turns and rebuilds local representations. It does not regenerate text with an LLM. Native answer context and the residual source-closure context are evaluated separately. Each claim branches independently from Full; Direct removal is separate from the additional S1/S2 turn budget.

The settings are frozen in each new PLAN: source/contract/annotation hashes, session configuration, candidate object/turn caps, extra-turn budgets, models, coverage diagnostic and answer execution. Do not change these settings to resume an existing run.

The shared scoring protocol requires both fixed permutations correct. Missing, malformed or incomplete trials are not scored as incorrect answers. Early-stop execution is allowed only when explicitly selected by that plan; a valid wrong first trial proves the conjunction false but is not a measured second trial.

See ../../PROTOCOL.md and ../PROTOCOL.md. This source release creates new identities and cannot resume frozen local runs.
