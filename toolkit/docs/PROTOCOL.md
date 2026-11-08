# Scoring rules

Every claim branches independently from the same Full memory, built from the episode's dialogue.

- **S0** deletes the disclosure turns and every object derived from them.
- **S1** sends the deletion request to the system's own retrieval for its top 20 objects, maps them to their source
  turns (the disclosure excluded), scores each turn by the best score of any object derived from it, keeps the 20
  highest-scoring turns and deletes the ten ranked highest.
- **S2** gives the same 20 candidates to the write model, which scores each one from 0 to 10 for relevance to the
  request, seeing only the stored text the system exposes for that turn, and deletes the ten scored highest.

Deleting a turn removes every object derived from it, atomic or aggregate; nothing is regenerated. The budget is
counted in source turns.

**Native** recovery sends the question stem, without its options, to the system's retrieval at its fixed depth,
renders the result in the system's own context format and asks the model to answer. **Residual** recovery gives the
same model every surviving object derived from an evidence or disclosure turn, in order of first write; a claim with
no surviving object counts as not recovered. A claim is recovered when the target is chosen under both fixed display
orders (2/2). An invalid, missing or truncated answer is reported as unavailable, never as a miss.

The full specification is [PROTOCOL.md](../PROTOCOL.md); the systems' storage and depths are in
[SYSTEMS.md](SYSTEMS.md).
