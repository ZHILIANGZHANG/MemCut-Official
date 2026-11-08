"""Versioned MemCut adapters sharing native memory mechanisms with LoCoMo."""

VERSION = 'locomo-aligned-s012-v2'
RAG = ('turnrag', 'sessionrag')            # the two retrieval controls (aligned_s012/rag.py)
SYSTEMS = ('memorybank', 'nemori', 'lightmem') + RAG
