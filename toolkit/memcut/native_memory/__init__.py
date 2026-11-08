"""Native object deletion. Optional upstream dependencies are imported lazily."""

from .contract import MemoryObject, SearchResult, DeletionReceipt, NativeAdapter
from .cleanup import CleanupSession

__all__ = ["MemoryObject", "SearchResult", "DeletionReceipt", "NativeAdapter", "CleanupSession"]
