"""Native object deletion. Optional upstream dependencies are imported lazily."""

from .contract import MemoryObject, SearchResult, DeletionReceipt, NativeAdapter

__all__ = ["MemoryObject", "SearchResult", "DeletionReceipt", "NativeAdapter"]
