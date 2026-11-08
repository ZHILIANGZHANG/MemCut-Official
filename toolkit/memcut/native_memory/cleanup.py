"""Bounded, model-agnostic diagnostic selection over native visibility only."""
from dataclasses import asdict
from .contract import NativeAdapter, MemoryObject, token_count


SELECTION_PROMPT = """You are helping honor a user's request to remove information from an assistant's memory.
Select the visible stored objects whose removal would best reduce retention or inference of the requested information.
Records may contain unrelated or misleadingly similar information. Use only the request and visible text.
Return JSON with selected_addresses (at most the supplied total budget) and next_query (a plain-language retrieval query).
Do not invent addresses. This is a diagnostic procedure; you have no evidence labels, answer options or hidden history.
"""


class CleanupSession:
    """One claim/condition on an independent writable Full/Direct branch.

    R rounds build visibility on unchanged Direct state. apply() deletes at most
    B distinct objects once. R=3,B=10 cannot silently become thirty deletions.
    Native controls must name a layer; no fabricated cross-layer native ranking.
    """
    def __init__(self, adapter: NativeAdapter, request: str, budget: int = 10, record_step=None):
        if budget < 0:
            raise ValueError("Negative budget")
        self.adapter, self.request, self.budget = adapter, request, budget
        self.seen: dict[str, MemoryObject] = {}
        self.views: dict[str, list[str]] = {}
        self.searches = []
        self.applied = False
        self.baseline_digest = None
        self.record_step = record_step
        self.partial_receipts = []

    async def retrieve(self, query: str):
        if self.applied:
            raise RuntimeError("Start another independent branch for a different condition")
        current = await self.adapter.audit_digest()
        if self.baseline_digest is None:
            self.baseline_digest = current
        elif current != self.baseline_digest:
            raise RuntimeError("Direct state changed between candidate-pool rounds")
        result = await self.adapter.search(query)
        for obj in result.occurrences():
            previous = self.seen.get(obj.address)
            if previous is not None and previous.version != obj.version:
                # A changed object invalidates its prior views; don't carry old
                # semantic content into selection of its new version.
                self.views[obj.address] = []
            self.seen[obj.address] = obj
            views = self.views.setdefault(obj.address, [])
            if obj.text not in views:
                views.append(obj.text)
        self.searches.append(result)
        # Some native searches update recall metadata. Track that actual state.
        self.baseline_digest = await self.adapter.audit_digest()
        return result

    def selector_payload(self) -> dict:
        return {"request": self.request, "total_object_budget": self.budget,
                "visible_objects": [{**o.public(), "text": self.views[o.address][0],
                                     "observed_native_views": self.views[o.address]}
                                    for o in self.seen.values()]}

    def native_selection(self, layer: str) -> list[str]:
        if not any(layer in r.ranked for r in self.searches):
            raise ValueError("Unknown ranked native layer; resident objects have no native rank")
        seen = {}
        for result in self.searches:
            for obj in result.ranked.get(layer, []):
                seen.setdefault(obj.address, obj)
        return list(seen)[:self.budget]

    async def apply(self, selected_addresses: list[str]) -> dict:
        if self.applied:
            raise RuntimeError("Cleanup already applied")
        selected = list(dict.fromkeys(selected_addresses))
        if len(selected) > self.budget or any(a not in self.seen for a in selected):
            raise ValueError("Selection exceeds budget or contains an unseen address")
        if await self.adapter.audit_digest() != self.baseline_digest:
            raise RuntimeError("State changed after retrieval; use an isolated branch")
        # Validate every object before the first mutation.
        validated = {}
        for address in selected:
            current = await self.adapter.get(address)
            if current is None or current.version != self.seen[address].version:
                raise ValueError("Stale candidate")
            validated[address] = current
        all_objects = await self.adapter.audit_objects()
        denominator = sum(token_count(o.text) for o in all_objects
                          if o.visibility != "stored_only")
        self.applied = True  # Failure remains a partial run; never replay blindly.
        receipts = []
        for index, address in enumerate(selected):
            # Earlier erasures may remap only structural references on later notes.
            current = await self.adapter.get(address)
            if current is None or current.text != validated[address].text:
                raise RuntimeError("Earlier native maintenance changed another selected object's text")
            if self.record_step:
                self.record_step(index, "intent", {"address": address, "version": current.version,
                                 "stored_text": current.text, "tokens": token_count(current.text)})
            receipt = await self.adapter.delete(current)
            receipts.append(asdict(receipt))
            self.partial_receipts = list(receipts)
            if self.record_step:
                self.record_step(index, "receipt", asdict(receipt))
        removed = sum(r["removed_tokens"] for r in receipts)
        return {"status": "complete", "retrieval_rounds": len(self.searches),
                "total_object_budget": self.budget, "touched_objects": len(receipts),
                "action_count": len(receipts), "removed_tokens": removed,
                "pre_cleanup_active_tokens": denominator,
                "removed_token_fraction": removed / denominator if denominator else 0,
                "tokenizer": "o200k_base", "receipts": receipts,
                "search_costs": [r.costs for r in self.searches],
                "claim_deletion_success": "must be measured separately by model readout"}
