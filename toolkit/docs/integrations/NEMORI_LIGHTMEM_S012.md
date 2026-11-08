# Nemori / LightMem S012

The source code is pinned in `configs/source_anchored_systems_v1.json`; use `python run.py source` and select with `--system nemori|lightmem`. Formation, flush and native retrieval are determined by the pinned upstream and the adapter; these are not the separate baseline implementations used in the LoCoMo baseline runs.

Nemori maintains the source closure over the native segmented narratives and the derived semantic facts; the direct semantic extraction setting from the configuration is used here. LightMem uses native compression/segmentation/extraction and the final offline update, returning fine-grained sources within the existing generation calls. Both branch independently from Full, delete derived objects whole, and do not regenerate on deletion.

Detailed definitions are in the [main protocol](../../PROTOCOL.md) and the [LightMem source notes](LIGHTMEM_FINE_SOURCES.md). Installation and execution are in the [usage notes](../USAGE.md).
