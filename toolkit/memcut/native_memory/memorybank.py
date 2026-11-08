"""Native English FAISS MemoryBank with historical pairs and summary fields.

Reference: upstream SiliconFriend ChatGLM/BELLE path (k=2, chunk_size=200,
MiniLM-L6, forgetting disabled by upstream default). The LLM is replaceable;
this is not the different ChatGPT/LlamaIndex path.
"""
import copy
import hashlib
import json
import pickle
import tempfile
import uuid
from pathlib import Path
from .contract import MemoryObject, NativeAdapter, SearchResult, digest, mutation_record


class TracedText(str):
    """Track concatenation without changing a character of the native context."""
    def __new__(cls, value, ids=()):
        obj = super().__new__(cls, value)
        obj.ids = tuple(ids)
        return obj

    def __add__(self, other):
        return TracedText(str(self) + str(other), self.ids + getattr(other, "ids", ()))

    def strip(self, *args):
        return TracedText(super().strip(*args), self.ids)

    def replace(self, *args):
        return TracedText(super().replace(*args), self.ids)


class MemoryBankAdapter(NativeAdapter):
    system = "memorybank"
    # Pinned SiliconFriend CLI DataArguments overrides the class's k=3 with 2.
    native_answer_top_k = 2

    def __init__(self, native_module, prompt_module, qa, user_id, checkpoint):
        self.native, self.prompts, self.qa = native_module, prompt_module, qa
        self.user_id, self.checkpoint = user_id, Path(checkpoint)
        self.memory, self.pair_ids, self.doc_addresses = {}, {}, {}
        self.store = None
        self._checkpoint_hash = None

    def fields(self):
        user = self.memory[self.user_id]
        for date, pairs in user.get("history", {}).items():
            for i, pair in enumerate(pairs):
                ident = self.pair_ids[date][i]
                yield "history", ident, json.dumps(pair, ensure_ascii=False), pair, "searchable"
        for layer in ("summary", "personality"):
            for date, value in user.get(layer, {}).items():
                text = value.get("content", "") if isinstance(value, dict) else value
                visibility = "searchable" if layer == "summary" else "stored_only"
                yield layer, date, text, value, visibility
        # The official template actually interpolates personality, but NOT
        # history_summary, despite constructing/passing it to str.format().
        for layer in ("overall_history", "overall_personality"):
            value = user.get(layer, "")
            yield layer, self.user_id, value, value, ("resident" if layer == "overall_personality" else "stored_only")

    async def audit_objects(self):
        # A native FAISS document is the canonical text of a searchable object:
        # the original loader adds date/speaker prefixes and renders summary
        # dictionaries. Use it for both visibility and deleted-token accounting.
        documents = {pair: str(self.store.docstore.search(ident).page_content)
                     for ident, pair in self.doc_addresses.items()}
        objects = []
        for layer, ident, text, value, visibility in self.fields():
            text = documents.get((layer, ident), text)
            if visibility == "searchable" and (layer, ident) not in documents:
                visibility = "stored_only"
            objects.append(MemoryObject(self.system, layer, ident, text,
                                        digest({"stored": value, "rendered": text}), visibility))
        return objects

    async def initialize(self, memory):
        if self.checkpoint.exists():
            raise FileExistsError("Existing bank; restore explicitly")
        self.memory = copy.deepcopy(memory)
        if set(self.memory) != {self.user_id}:
            raise ValueError("Use a dedicated single-user bank for independent claim forks")
        # Native eval assumes these keys even for a newly formed, empty bank.
        self.memory[self.user_id].setdefault("overall_history", "")
        self.memory[self.user_id].setdefault("overall_personality", "")
        self.pair_ids = {date: [str(uuid.uuid4()) for _ in pairs]
                         for date, pairs in self.memory[self.user_id].get("history", {}).items()}
        self._build_index()
        self.save()

    def _build_index(self):
        from langchain.vectorstores import FAISS
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.json"
            path.write_text(json.dumps(self.memory, ensure_ascii=False))
            docs = self.native.load_memory_file(str(path), self.user_id, "en")
        addresses = []
        user = self.memory[self.user_id]
        for date, pairs in user.get("history", {}).items():
            addresses.extend(("history", ident) for ident in self.pair_ids[date])
            if date in user.get("summary", {}):
                addresses.append(("summary", date))
        if len(docs) != len(addresses):
            raise RuntimeError("Native loader/document address mismatch")
        if not docs:
            self.store, self.doc_addresses = None, {}
            return
        self.store = FAISS.from_documents(docs, self.qa.embeddings)
        self.store.chunk_size = self.qa.chunk_size
        self.doc_addresses = {self.store.index_to_docstore_id[i]: pair for i, pair in enumerate(addresses)}

    async def search(self, query, *, top_k=None):
        """Run native retrieval, optionally overriding k for this call only.

        k controls FAISS seed hits. Native adjacency can add objects, and native
        date/group presentation determines the returned order. A benchmark
        candidate cap belongs to its caller; it must not truncate native QA's
        context or silently change the bank's saved/default retrieval settings.
        """
        import types
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("MemoryBank top_k override must be a positive integer")
        effective_k = self.qa.top_k if top_k is None else top_k
        objects = {(o.layer, o.native_id): o for o in await self.audit_objects()}
        observed_groups = []
        seed_hits, native_ks = [], []
        # Native search concatenates Document.page_content in place. Run its
        # presentation code on private documents while sharing the read-only index.
        view = copy.copy(self.store) if self.store is not None else None
        if view is not None:
            original_index = self.store.index
            doc_addresses = self.doc_addresses

            class TracedIndex:
                # Record the one actual FAISS call, without an extra query or
                # changing native vectors, distances, order or adjacency.
                def __getattr__(self, name):
                    return getattr(original_index, name)

                def search(self, embedding, k):
                    scores, indices = original_index.search(embedding, k)
                    native_ks.append(k)
                    for rank, (row, score) in enumerate(zip(indices[0], scores[0]), 1):
                        if row != -1:
                            ident = view.index_to_docstore_id[int(row)]
                            seed_hits.append({"rank": rank,
                                              "address": objects[doc_addresses[ident]].address,
                                              "distance": float(score)})
                    return scores, indices

            view.index = TracedIndex()
            view.docstore = copy.deepcopy(self.store.docstore)
            for ident, pair in self.doc_addresses.items():
                doc = view.docstore.search(ident)
                doc.page_content = TracedText(doc.page_content, [ident])
            upstream = self.native.similarity_search_with_score_by_vector
            def traced(store, embedding, k=4):
                if store.index.ntotal == 0:
                    return []  # upstream seperate_list([]) crashes; empty search is empty.
                result = upstream(store, embedding, k)
                for doc, score in result:
                    if not isinstance(doc.page_content, TracedText):
                        raise RuntimeError("Native context lineage was lost during concatenation")
                    observed_groups.append((doc.metadata["source"], doc.page_content.ids))
                return result
            view.similarity_search_with_score_by_vector = types.MethodType(traced, view)
        p = self.prompts
        # Use the actual native evaluation builder; empty-history QA is frozen.
        proxy = copy.copy(self.qa)
        proxy.top_k = effective_k
        if view is None or view.index.ntotal == 0:
            proxy.search_memory = lambda *args: ([], "")
        prompt, retrieved_text = p.build_prompt_with_search_memory_chatglm_eval(
            [], query, self.memory[self.user_id], self.user_id, view, proxy,
            p.generate_meta_prompt_dict_chatglm_belle_eval()["en"], "[|User|]", "[|AI|]", "SiliconFriend", "en")
        visible = []
        # search_memory sorts grouped documents by date before presenting them.
        observed = [ident for _, ids in sorted(observed_groups, key=lambda item: item[0])
                    for ident in ids]
        for ident in dict.fromkeys(observed):
            pair = self.doc_addresses[ident]
            obj = objects[pair]
            # Show exactly native document text, not an expanded original source.
            doc_text = self.store.docstore.search(ident).page_content
            visible.append(MemoryObject(obj.system, obj.layer, obj.native_id,
                                        str(doc_text), obj.version, obj.visibility))
        # Grouped order is native display order, not an invented FAISS score rank.
        resident = [o for o in objects.values() if o.visibility == "resident" and o.text]
        return SearchResult(query, {"retrieved_document": visible}, resident,
                            {"prompt": prompt, "retrieved_text": retrieved_text},
                            {"native_retrieval_operations": 1,
                             "embedding_search_calls": len(native_ks),
                             "requested_top_k": effective_k,
                             "configured_top_k": self.qa.top_k,
                             "native_top_k": effective_k,
                             "native_search_k": native_ks,
                             "native_seed_hit_count": len(seed_hits),
                             "native_seed_hits": seed_hits,
                             "native_returned_groups": len(observed_groups),
                             "native_chunk_size_characters": self.qa.chunk_size,
                             "display_order": "native date/group order", "returned_objects": len(visible)})

    async def answer_context(self, question):
        """Build the actual native QA prompt with its reference k=2 retrieval.

        The caller sends native_context['prompt'] to the chosen answer model.
        Cleanup's k=20 does not become the QA k, including when reading an older
        diagnostic bank whose persisted retrieval override was set to 20.
        """
        return await self.search(question, top_k=self.native_answer_top_k)

    async def _erase(self, obj):
        # Stage JSON and the native FAISS/index maps together. The pinned 2023
        # LangChain wrapper has no delete API; FAISS itself provides remove_ids.
        # Copying preserves all surviving vectors, without re-embedding content.
        memory, pair_ids = copy.deepcopy(self.memory), copy.deepcopy(self.pair_ids)
        addresses = dict(self.doc_addresses)
        store = self.store
        user = memory[self.user_id]
        if obj.layer == "history":
            date, index = next((date, ids.index(obj.native_id)) for date, ids in pair_ids.items() if obj.native_id in ids)
            del user["history"][date][index]
            del pair_ids[date][index]
        elif obj.layer in ("summary", "personality"):
            del user[obj.layer][obj.native_id]
        elif obj.layer in ("overall_history", "overall_personality"):
            user[obj.layer] = ""  # required keys remain valid for native prompt builder
        else:
            raise ValueError("Unsupported MemoryBank layer")
        removed_ids = [ident for ident, pair in self.doc_addresses.items() if pair == (obj.layer, obj.native_id)]
        if removed_ids:
            import faiss
            import numpy as np
            store = copy.copy(self.store)
            store.index = faiss.clone_index(self.store.index)
            store.docstore = copy.deepcopy(self.store.docstore)
            removed = set(removed_ids)
            rows = np.array([row for row, ident in store.index_to_docstore_id.items()
                             if ident in removed], dtype=np.int64)
            if store.index.remove_ids(rows) != len(removed_ids):
                raise RuntimeError("Native FAISS deletion count mismatch")
            survivors = [ident for _, ident in sorted(store.index_to_docstore_id.items())
                         if ident not in removed]
            store.index_to_docstore_id = dict(enumerate(survivors))
            for ident in removed_ids:
                del store.docstore._dict[ident]
                del addresses[ident]
        previous = self.memory, self.pair_ids, self.doc_addresses, self.store
        self.memory, self.pair_ids, self.doc_addresses, self.store = memory, pair_ids, addresses, store
        try:
            self.save()
        except BaseException:
            self.memory, self.pair_ids, self.doc_addresses, self.store = previous
            raise
        return {"json_object_removed_or_emptied": True, "faiss_rows_removed": len(removed_ids),
                "deletion_primitive": "native FAISS remove_ids + JSON/docstore/address maintenance",
                "summaries_regenerated": False, "forgetting": "upstream reference default disabled"}

    def save(self):
        import faiss
        import fcntl
        state = {"memory": self.memory, "pair_ids": self.pair_ids, "doc_addresses": self.doc_addresses,
                 "format_version": 1, "user_id": self.user_id,
                 "retrieval_config": self.retrieval_config(),
                 "index": faiss.serialize_index(self.store.index) if self.store is not None else None,
                 "docstore": self.store.docstore if self.store is not None else None,
                 "index_ids": self.store.index_to_docstore_id if self.store is not None else None}
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(state)
        with self.checkpoint.with_suffix(self.checkpoint.suffix + ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            actual_hash = (hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
                           if self.checkpoint.exists() else None)
            if actual_hash != self._checkpoint_hash:
                raise RuntimeError("MemoryBank checkpoint changed outside this adapter; restore or use an independent fork")
            temp = self.checkpoint.with_suffix(self.checkpoint.suffix + ".tmp")
            temp.write_bytes(payload)
            temp.replace(self.checkpoint)
            self._checkpoint_hash = hashlib.sha256(payload).hexdigest()

    def restore(self):
        import faiss
        from langchain.vectorstores import FAISS
        payload = self.checkpoint.read_bytes()
        state = pickle.loads(payload)
        expected = self.retrieval_config()
        if (state.get("format_version") != 1 or state.get("user_id") != self.user_id
                or state.get("retrieval_config") != expected):
            raise ValueError("MemoryBank checkpoint identity/configuration mismatch")
        self.memory, self.pair_ids, self.doc_addresses = (state[k] for k in ("memory", "pair_ids", "doc_addresses"))
        self.store = None
        if state["index"] is not None:
            self.store = FAISS(self.qa.embeddings.embed_query, faiss.deserialize_index(state["index"]),
                               state["docstore"], state["index_ids"])
            self.store.chunk_size = self.qa.chunk_size
        self._checkpoint_hash = hashlib.sha256(payload).hexdigest()

    def retrieval_config(self):
        return {"top_k": self.qa.top_k, "chunk_size": self.qa.chunk_size,
                "language": self.qa.language,
                "embedding_model": getattr(self.qa.embeddings, "model_name", None),
                "embedding_model_kwargs": getattr(self.qa.embeddings, "model_kwargs", {}),
                "embedding_encode_kwargs": getattr(self.qa.embeddings, "encode_kwargs", {}),
                "prompt": "upstream chatglm_belle_eval; empty recent history",
                "forgetting": False}

    async def summarize(self, summary_module, source_ids):
        before = await self.audit_objects()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.json"
            path.write_text(json.dumps(self.memory, ensure_ascii=False))
            self.memory = summary_module.summarize_memory(str(path), name=self.user_id, language="en")
        # Formation rebuilds the native index once; deletion never calls this.
        self._build_index()
        self.save()
        after = await self.audit_objects()
        record = mutation_record(before, after, source_ids,
                                 [{"address": o.address, "version": o.version} for o in before],
                                 "conservative summary input set, not clause-level semantic provenance")
        # Native generation retries and ultimately returns an empty string on
        # exhaustion. Preserve that behavior, but never hide empty derived state
        # behind a successful JSON save in an experiment's formation report.
        empty = []
        user = self.memory[self.user_id]
        for layer in ("summary", "personality"):
            for date, value in user.get(layer, {}).items():
                content = value.get("content", "") if isinstance(value, dict) else value
                if not content:
                    empty.append({"layer": layer, "native_id": date})
        for layer in ("overall_history", "overall_personality"):
            if not user.get(layer):
                empty.append({"layer": layer, "native_id": self.user_id})
        record["native_empty_summary_fields"] = empty
        record["summary_generation_status"] = "contains_empty_fields" if empty else "nonempty_outputs"
        record["summary_generation_status_scope"] = "output presence, not semantic correctness"
        return record


UPSTREAM_COMMIT = "cf61c4196e4cfdb0f2b7a0316249fa40312dc3a9"
UPSTREAM_FILES = (
    "memory_bank/memory_retrieval/local_doc_qa.py",
    "memory_bank/memory_retrieval/configs/model_config.py",
    "memory_bank/memory_retrieval/textsplitter/__init__.py",
    "memory_bank/memory_retrieval/textsplitter/chinese_text_splitter.py",
    "memory_bank/summarize_memory.py", "utils/prompt_utils.py", "utils/sys_args.py",
    "SiliconFriend-ChatGLM-BELLE/cli_demo.py",
)


def load_native_modules(upstream):
    """Load the verified FAISS reference; never import the locally patched forget path.

    Imports require MemoryBank's own dependency environment. Checking only files
    used by this profile permits unrelated work in the upstream checkout.
    """
    import importlib.util
    import subprocess
    import sys

    root = Path(upstream).resolve()
    for relative in UPSTREAM_FILES:
        original = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{UPSTREAM_COMMIT}:{relative}"])
        if original != (root / relative).read_bytes():
            raise ValueError(f"MemoryBank reference source changed: {relative}")
    package_root = root / "memory_bank"
    for name, module in list(sys.modules.items()):
        if name == "memory_retrieval" or name.startswith("memory_retrieval."):
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(package_root):
                raise RuntimeError("Another memory_retrieval package is loaded; use an isolated runtime")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    native = load("_memcut_memorybank_native_qa", UPSTREAM_FILES[0])
    prompts = load("_memcut_memorybank_native_prompts", "utils/prompt_utils.py")
    summaries = load("_memcut_memorybank_native_summaries", "memory_bank/summarize_memory.py")
    return native, prompts, summaries


def make_memorybank(upstream, user_id, checkpoint, *, top_k=None):
    """Official English FAISS settings; only the later summary/answer LLM is swappable.

    No summarization or model API is performed here. Native init_cfg selects
    the upstream device (CUDA, MPS or CPU) and MiniLM-L6 embedding defaults.
    The CLI DataArguments overrides the class default k=3 with k=2.
    """
    if top_k is not None and (type(top_k) is not int or top_k < 1):
        raise ValueError("MemoryBank top_k override must be a positive integer")
    native, prompts, summaries = load_native_modules(upstream)
    qa = native.LocalMemoryRetrieval()
    qa.init_cfg(embedding_model=native.EMBEDDING_MODEL_EN, top_k=2 if top_k is None else top_k, language="en")
    return MemoryBankAdapter(native, prompts, qa, user_id, checkpoint), summaries


def bind_summary_model(summary_module, model, chat_completion):
    """Replace transport/model while retaining native prompts, parameters and retry logic.

    chat_completion is a synchronous callable accepting OpenAI-style keyword
    arguments and returning the native dictionary shape (choices/message/content).
    An OpenAI SDK client can be bridged with response.model_dump(). No credentials
    are stored by this adapter. Calls are only made by summarize(), explicitly.
    """
    from types import SimpleNamespace
    config = copy.deepcopy(summary_module.chatgpt_config)
    config["model"] = model
    summary_module.openai = SimpleNamespace(
        api_key=None, ChatCompletion=SimpleNamespace(create=chat_completion))
    summary_module.llm_client = summary_module.LLMClientSimple(config)
