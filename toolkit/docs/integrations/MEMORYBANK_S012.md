# MemoryBank S012

This release pins the English FAISS SiliconFriend path; the configuration is in `configs/native_memory_profiles.json`, and forgetting is not enabled. The base adapter keeps the historical native constants; the new main experiment explicitly freezes QA k=10 through `memorybank-efficient` or the corresponding model/optimization entry. QA k is separate from the deletion candidate top-20.

## How the diff is executed

Before and after ingestion, the native QA pairs, daily summaries, personality and global fields are enumerated and compared by stable ID/content, maintaining the many-to-many relation from source turns to derived objects. Deletion removes whole objects along the source closure and does not regenerate from the remaining original text; a QA pair may cause cascading deletions, which must appear in the statistics.

`native` still goes through native FAISS, adjacency expansion and rendering; `residual` independently enumerates the objects whose evidence sources survive. Stored fields that the native template does not display must not be hidden from the residual enumeration either.

Use the [separate environment](../USAGE.md); see the [early-stop policy](../engineering/MEMORYBANK_EXECUTION_SAVINGS.md).
