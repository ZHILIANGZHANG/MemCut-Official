"""Visibility, object identity and deletion accounting shared across systems.

The auditor can enumerate storage. The cleaner receives only SearchResult.
Native IDs never become embedding text. No gold annotations enter this package.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def token_count(text: str) -> int:
    import tiktoken
    return len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))


@dataclass(frozen=True)
class MemoryObject:
    system: str
    layer: str
    native_id: str
    text: str
    version: str
    visibility: str = "searchable"  # searchable, resident, or stored_only

    @property
    def address(self) -> str:
        # JSON avoids ambiguous delimiters in provider IDs.
        return json.dumps([self.system, self.layer, self.native_id], separators=(",", ":"))

    def public(self) -> dict:
        return {"address": self.address, "layer": self.layer, "text": self.text,
                "version": self.version, "visibility": self.visibility}


@dataclass
class SearchResult:
    query: str
    # Ordered within each layer. Resident objects have no retrieval rank.
    ranked: dict[str, list[MemoryObject]] = field(default_factory=dict)
    resident: list[MemoryObject] = field(default_factory=list)
    native_context: Any = None
    costs: dict = field(default_factory=dict)

    def visible(self) -> list[MemoryObject]:
        """Distinct addresses; use occurrences() to preserve multiple text views."""
        return list({o.address: o for o in self.occurrences()}.values())

    def occurrences(self) -> list[MemoryObject]:
        return [o for objects in [*self.ranked.values(), self.resident] for o in objects]


@dataclass
class DeletionReceipt:
    address: str
    version_before: str
    operation: str
    removed_text: str
    removed_tokens: int
    maintenance: dict
    verified: bool
    seconds: float


class NativeAdapter(ABC):
    system: str

    @abstractmethod
    async def audit_objects(self) -> list[MemoryObject]:
        """Auditor-only current objects. Never use this to rank cleaner candidates."""

    @abstractmethod
    async def search(self, query: str) -> SearchResult:
        """One explicitly named native retrieval operation; keep internal expansion."""

    @abstractmethod
    async def _erase(self, obj: MemoryObject) -> dict:
        """Remove/empty the selected object and maintain native references/indexes."""

    async def get(self, address: str) -> MemoryObject | None:
        return next((o for o in await self.audit_objects() if o.address == address), None)

    async def delete(self, obj: MemoryObject) -> DeletionReceipt:
        current = await self.get(obj.address)
        if current is None or current.version != obj.version:
            raise ValueError("Missing or changed object; refresh visibility before deleting")
        removed_tokens = token_count(current.text)
        start = time.monotonic()
        maintenance = await self._erase(current)
        after = await self.get(obj.address)
        verified = after is None or not after.text
        if not verified:
            raise RuntimeError("Native erase returned but selected text is still active")
        return DeletionReceipt(current.address, current.version,
                               "empty_container" if after else "delete_object",
                               current.text, removed_tokens, maintenance, True,
                               time.monotonic() - start)

    async def audit_digest(self) -> str:
        return digest(sorted((o.address, o.version) for o in await self.audit_objects()))


def mutation_record(before: list[MemoryObject], after: list[MemoryObject],
                    trigger_sources: list[str], read_dependencies: list[dict],
                    coverage: str) -> dict:
    old, new = ({o.address: o for o in items} for items in (before, after))
    return {"trigger_sources": trigger_sources,
            "observed_read_dependencies": read_dependencies,
            "provenance_coverage": coverage,
            "semantic_support": "not inferred from dependency edges",
            "created": [asdict(o) for a, o in new.items() if a not in old],
            "updated": [{"before": asdict(old[a]), "after": asdict(o)}
                        for a, o in new.items() if a in old and o.version != old[a].version],
            "removed": [asdict(o) for a, o in old.items() if a not in new]}
