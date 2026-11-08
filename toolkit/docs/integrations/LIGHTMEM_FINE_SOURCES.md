# LightMem fine-grained sources

The extraction output includes `source_ids`, and the update output includes the `source_memory_ids` that actually contributed; the program parses them and maintains a bidirectional source table. Attribution reuses the existing generation calls, adds no separate attribution request, and does not attach every turn of the batch to every object by default.

This modifies the native generation prompts/fields and is a disclosed integration adaptation; it cannot be called a completely unmodified black box. A multi-source memory is deleted whole when any of its sources falls into the deletion closure; the remaining text is not stitched together to generate a replacement. Old batch-level source banks cannot be used directly as this version's fine-grained banks.

In the release copy, the pure function that renders formation text is in `memcut/source_anchored_systems/formation_runtime/render.py`; its reverse dependency on historical statistics scripts has been removed.
