"""Secret-free, pinned native-system configuration recipes (no runtime imports).

These are explicit reference variants, not claims of paper score reproduction.
Factories must verify the selected upstream files before importing native code and
record both the resolved recipe and any compatibility deviations in the run.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs/native_memory_profiles.json"


class ProfileError(ValueError):
    """A reference profile or requested model substitution is invalid."""


def _registry(registry_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != 1 or not isinstance(value.get("profiles"), dict):
        raise ProfileError("Unsupported native profile registry schema")
    return value


def list_profiles(registry_path: str | Path | None = None) -> dict[str, str]:
    """Return profile names and reference descriptions without loading models."""
    return {name: value["reference"] for name, value in _registry(registry_path)["profiles"].items()}


def load_profile(name: str, registry_path: str | Path | None = None) -> dict[str, Any]:
    """Read a detached recipe; its native code has not yet been verified."""
    profiles = _registry(registry_path)["profiles"]
    if name not in profiles:
        raise ProfileError("Unknown native memory profile")
    result = deepcopy(profiles[name])
    upstream = result.get("upstream", {})
    if not re.fullmatch(r"[0-9a-f]{40}", upstream.get("commit", "")):
        raise ProfileError("Profile has no valid pinned upstream commit")
    if upstream.get("source_kind") not in {"git", "archive"}:
        raise ProfileError("Profile has no supported upstream source kind")
    hashes = upstream.get("upstream_file_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ProfileError("Profile has no pinned upstream file hashes")
    for filename, digest in hashes.items():
        if (not isinstance(filename, str) or Path(filename).is_absolute()
                or ".." in Path(filename).parts or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ProfileError("Invalid pinned upstream file entry")
    if not isinstance(result.get("config"), dict):
        raise ProfileError("Profile has no configuration recipe")
    result["profile_id"] = name
    result["profile_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def verify_profile(
    name: str,
    repository: str | Path | None = None,
    *,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify relevant native source bytes, plus Git HEAD for a Git checkout.

    Deliberately does not fail for unrelated local changes or untracked caches.
    An archive proves selected file identity, not an entire Git worktree identity.
    """
    profile = load_profile(name, registry_path)
    upstream = profile["upstream"]
    directory = Path(repository) if repository is not None else PROJECT_ROOT / upstream["repository"]
    directory = directory.resolve()
    errors: list[str] = []
    checked: dict[str, str | None] = {}
    commit_verified = False
    observed_commit = None
    if upstream["source_kind"] == "git":
        if not (directory / ".git").exists():
            errors.append("Expected a Git checkout at the selected repository root")
        else:
            process = subprocess.run(
                ["git", "-C", str(directory), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False, timeout=10,
            )
            observed_commit = process.stdout.strip() if process.returncode == 0 else None
            commit_verified = observed_commit == upstream["commit"]
            if not commit_verified:
                errors.append("Native upstream HEAD differs from the pinned commit")
    for filename, expected in upstream["upstream_file_hashes"].items():
        candidate = (directory / filename).resolve()
        if not candidate.is_relative_to(directory):
            errors.append(f"Native source path escapes repository: {filename}")
            checked[filename] = None
            continue
        if not candidate.is_file():
            errors.append(f"Missing native source file: {filename}")
            checked[filename] = None
            continue
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        checked[filename] = actual
        if actual != expected:
            errors.append(f"Native source differs from the pinned file: {filename}")
    return {
        "profile_id": name,
        "profile_sha256": profile["profile_sha256"],
        "repository": str(directory),
        "expected_commit": upstream["commit"],
        "observed_commit": observed_commit,
        "commit_verified": commit_verified,
        "verification_scope": "pinned_selected_files_and_git_head" if upstream["source_kind"] == "git" else "pinned_selected_archive_files",
        "checked_files": checked,
        "ok": not errors,
        "errors": errors,
    }


def _model_name(value: str) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}", value)
            or value.startswith(("sk-", "tp-"))):
        raise ProfileError("Supply a model identifier, never a credential")
    return value


def _endpoint(value: str) -> str:
    if not isinstance(value, str):
        raise ProfileError("Model endpoint must be an HTTP(S) URL without credentials")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or re.search(r"[\s\\]", value)):
        raise ProfileError("Model endpoint must be an HTTP(S) URL without credentials, query or fragment")
    return value.rstrip("/")


def resolve_profile(
    name: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    context_window: int | None = None,
    registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Substitute the backbone transport only; embeddings and retrieval stay fixed.

    No API-key argument is accepted. Inject credentials into a live native runtime,
    never this serializable recipe. This function does not verify or start a bank.
    """
    profile = load_profile(name, registry_path)
    config = profile["config"]
    overrides = []
    system = profile["system"]
    if model is not None:
        model = _model_name(model)
        if system == "mem0":
            model_fields = [(config["memory_config"]["llm"]["config"], "model")]
        elif system == "amem":
            model_fields = [(config["constructor"], "llm_model")]
        elif system == "memorybank":
            model_fields = [(config["summary_request"], "model"), (config["answer_reference"], "model")]
        elif system == "letta":
            model_fields = [(config["llm_config"], "model")]
        elif system in {"memoryos", "hela"}:
            model_fields = [(config, "model")]
            if system == "hela":
                model_fields.append((config["environment"], "HEBBIAN_MODEL"))
        else:
            raise ProfileError("Profile system has no model substitution recipe")
        for target, field in model_fields:
            before = target[field]
            target[field] = model
            overrides.append({"field": field, "from": before, "to": model, "reason": "backbone_model_substitution"})
    if base_url is not None:
        base_url = _endpoint(base_url)
        if system == "mem0":
            config["memory_config"]["llm"]["config"]["openai_base_url"] = base_url
        elif system == "amem":
            config["constructor"]["api_base"] = base_url
        elif system == "memorybank":
            config["llm_transport"] = {"base_url": base_url}
        elif system == "letta":
            config["llm_config"]["model_endpoint"] = base_url
        elif system in {"memoryos", "hela"}:
            config["chat_endpoint"] = base_url
        overrides.append({"field": "chat_endpoint", "to": base_url, "reason": "backbone_transport_substitution"})
    if context_window is not None:
        if system != "letta" or isinstance(context_window, bool) or not isinstance(context_window, int) or context_window < 4096:
            raise ProfileError("Explicit context-window override is supported only for Letta, with at least 4096 tokens")
        before = config["llm_config"]["context_window"]
        config["llm_config"]["context_window"] = context_window
        overrides.append({"field": "context_window", "from": before, "to": context_window, "reason": "explicit_provider_context_limit_override"})
    profile["overrides"] = overrides
    profile["resolved_sha256"] = hashlib.sha256(
        json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return profile
