#!/usr/bin/env python3
"""Add a designed fact to a MemCut episode: an overlay in, a validated variant episode out.

    python tools/extend_episode.py --bundle memcut_v1.zip --overlay my_fact.json --output data/extended

An overlay (format `memcut.case_overlay.v1`, documented in docs/EXTENDING_DATA.md) names a base episode of the bundle,
one new fact (claim, deletion request, question, five options, named facts and the rule that joins them) and the
turns that carry it: evidence, the disclosure and distractors. Each new turn is a user message plus the assistant's
reply, appended after the last turn of its session, so no base turn changes id, text or position and every base
annotation stays valid. Existing base turns can also be annotated as distractors.

The output is a complete episode in the bundle format: `public/<EP>.json`, and in `author/` the claim contract, the
annotations with text hashes, token counts, the minimal sufficient sets (SUPPORT_PATHS.json) and the minimum cuts
(PATH_COMBINATION_AUDIT.json). The sets are enumerated from the carriers' fact edges; the minimum cut is computed two
independent ways that must agree.

Offline design checks (no model is called), any failure stops the build:
- every carrier is a user turn, and no single carrier covers every named fact;
- every named fact has at least two carriers, so no single deletion breaks a link;
- no unannotated turn matches any of the overlay's `leak_patterns` (patterns that must not co-occur);
- evidence spans at least `min_evidence_sessions` sessions (default 5) and precedes the disclosure;
- the disclosure's session holds no evidence;
- the distractor count is within the benchmark's range (16 to 88).
The reader-model checks (best single evidence turn, all evidence, nothing left, options alone) are listed in
docs/EXTENDING_DATA.md; they cost API calls and are run with the normal entry points.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import re
import sys
import zipfile
from pathlib import Path

OVERLAY_FORMAT = "memcut.case_overlay.v1"
DISTRACTOR_RANGE = (16, 88)
SCORING = ("Multiple choice with five options, one of them 'insufficient information'. The reader answers under the "
           "first two display orders (trials 1 and 2); the target counts as recovered only when it is chosen under "
           "both (2/2). Trial 3 is kept for analyses.")
CLAIM_FIELDS = ("claim_id", "category", "claim", "deletion_request", "question", "neutral_question",
                "protected_propositions", "fact_definitions", "rules", "choices", "target_choice_id",
                "insufficient_choice_id")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dumps(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode()


def rotation(order: list[str]) -> dict:
    return {"choice_ids_in_display_order": order, "display_to_choice": dict(zip(sorted(order), order))}


def place_turns(src: dict, overlay: dict) -> tuple[dict, dict]:
    """Append each new pair after its session's last turn; return the variant and label -> record id."""
    out = copy.deepcopy(src)
    out["id"] = overlay["variant_dialogue_id"]
    sessions = {s["id"]: s for s in out["sessions"]}
    user = next(t["speaker"] for s in src["sessions"] for t in s["turns"] if t["role"] == "user")
    bot = next(t["speaker"] for s in src["sessions"] for t in s["turns"] if t["role"] == "assistant")
    where = {}
    for row in overlay["turns"]:
        if row["session"] not in sessions:
            raise ValueError(f"{row['label']}: no session {row['session']} in the base episode")
        s = sessions[row["session"]]
        n = max(int(t["id"].rsplit(":T", 1)[1]) for t in s["turns"])
        width = len(s["turns"][-1]["id"].rsplit(":T", 1)[1])
        uid, aid = (f"{s['id']}:T{n + i:0{width}d}" for i in (1, 2))
        s["turns"] += [{"id": uid, "role": "user", "speaker": user, "text": row["user"]},
                       {"id": aid, "role": "assistant", "speaker": bot, "text": row["assistant"]}]
        where[row["label"]] = uid
    return out, where


def minimal_sets(edges: dict[str, set[str]], need: set[str]) -> list[list[str]]:
    found: list[set[str]] = []
    keys = sorted(edges)
    for r in range(1, len(keys) + 1):
        for combo in itertools.combinations(keys, r):
            if need <= set().union(*(edges[k] for k in combo)) and not any(f <= set(combo) for f in found):
                found.append(set(combo))
    return [sorted(f) for f in found]


def minimum_cuts(sets: list[list[str]], keys: list[str]) -> list[list[str]]:
    for r in range(1, len(keys) + 1):
        hits = [sorted(c) for c in itertools.combinations(keys, r) if all(set(c) & set(s) for s in sets)]
        if hits:
            return hits
    return []


def cut_size_by_facts(edges: dict[str, set[str]], need: set[str]) -> int:
    """Independent check: the smallest deletion that leaves some named fact with no carrier."""
    for k in range(1, len(edges) + 1):
        for cut in itertools.combinations(sorted(edges), k):
            left = set().union(set(), *(v for key, v in edges.items() if key not in cut))
            if not need <= left:
                return k
    return len(edges)


def max_disjoint(sets: list[list[str]]) -> int:
    best = 0
    for r in range(1, len(sets) + 1):
        if any(len({k for s in c for k in s}) == sum(map(len, c)) for c in itertools.combinations(sets, r)):
            best = r
        else:
            break
    return best


def token_counts(src: dict) -> dict:
    texts = [t["text"] for s in src["sessions"] for t in s["turns"]]
    out = {"scope": "Sum of individual message text only; no role, date, JSON or API wrapper. Local counts, not billing.",
           "messages": len(texts), "tokens": {}}
    try:
        import tiktoken
        for name in ("o200k_base", "cl100k_base"):
            enc = tiktoken.get_encoding(name)
            out["tokens"][name] = sum(len(enc.encode(t)) for t in texts)
    except Exception as exc:                      # tokenizer files unavailable offline
        out["tokens_unavailable"] = type(exc).__name__
    return out


def design_checks(variant: dict, overlay: dict, claim: dict, ann: dict) -> list[str]:
    bad = []
    flat = [(s["id"], t) for s in variant["sessions"] for t in s["turns"]]
    session_of = {t["id"]: sid for sid, t in flat}
    order = {t["id"]: i for i, (_, t) in enumerate(flat)}
    need = set(claim["fact_definitions"])
    for x in ann["carriers"]:
        if x["role"] != "user":
            bad.append(f"{x['key']}: a carrier must be a user turn")
        if set(x["edges"]) >= need:
            bad.append(f"{x['key']}: a single carrier covers every named fact")
        if set(x["edges"]) - need:
            bad.append(f"{x['key']}: edges {sorted(set(x['edges']) - need)} are not named facts")
    for f in sorted(need):
        n = sum(f in x["edges"] for x in ann["carriers"])
        if n < 2:
            bad.append(f"fact {f}: {n} carrier(s); every link needs at least two")
    labelled = {x["record_id"] for sec in ("carriers", "direct") for x in ann[sec]}
    patterns = [[re.compile(p) for p in group] for group in overlay.get("leak_patterns", [])]
    for sid, t in flat:
        if t["id"] in labelled:
            continue
        for group in patterns:
            if all(p.search(t["text"]) for p in group):
                bad.append(f"{t['id']}: an unannotated turn matches leak pattern {[p.pattern for p in group]}")
    evidence_sessions = {session_of[x["record_id"]] for x in ann["carriers"]}
    floor = overlay.get("min_evidence_sessions", 5)
    if len(evidence_sessions) < floor:
        bad.append(f"evidence spans {len(evidence_sessions)} sessions, fewer than {floor}")
    direct = claim["direct_record_ids"]
    if not direct:
        bad.append("no disclosure turn")
    else:
        if {session_of[d] for d in direct} & evidence_sessions:
            bad.append("an evidence turn shares the disclosure's session")
        first = min(order[d] for d in direct)
        if any(order[x["record_id"]] > first for x in ann["carriers"]):
            bad.append("an evidence turn follows the disclosure")
    n_d = len(ann["distractors"])
    if not DISTRACTOR_RANGE[0] <= n_d <= DISTRACTOR_RANGE[1]:
        bad.append(f"{n_d} distractors, outside {DISTRACTOR_RANGE}")
    return bad


def extend(base_src: dict, base_ann: dict, overlay: dict) -> dict:
    """Build the variant episode's files (name -> JSON object) from a base episode and an overlay."""
    if overlay.get("schema") != OVERLAY_FORMAT:
        raise ValueError(f"overlay schema must be {OVERLAY_FORMAT}")
    base_turns = [t for s in base_src["sessions"] for t in s["turns"]]
    variant, where = place_turns(base_src, overlay)
    known = {t["id"] for t in base_turns}
    if [t for s in variant["sessions"] for t in s["turns"] if t["id"] in known] != base_turns:
        raise ValueError("a base turn changed")
    src_blob = dumps(variant)
    src_sha = sha(src_blob)
    texts = {t["id"]: t["text"] for s in variant["sessions"] for t in s["turns"]}
    c = overlay["claim"]
    cid = c["claim_id"]
    if cid in {x["claim_id"] for sec in ("carriers", "direct", "distractors") for x in base_ann[sec]}:
        raise ValueError(f"{cid}: the base episode already has this claim id")
    claim = {k: c[k] for k in CLAIM_FIELDS if k in c}
    claim["direct_record_ids"] = [where[r["label"]] for r in overlay["turns"] if r["kind"] == "direct"]
    claim["rotations"] = [{"trial": i + 1, **rotation(o)} for i, o in enumerate(c["rotations"])]
    contracts = {"schema": "memcut.claim_contracts.public.v1", "source_sha256": src_sha, "scoring": SCORING,
                 "claims": [claim]}

    def rec(label, record_id, extra=None):
        return {"key": label, "claim_id": cid, "record_id": record_id, "role": "user",
                "text_sha256": sha(texts[record_id].encode()), **(extra or {})}
    carriers = sorted((rec(r["label"], where[r["label"]], {"edges": r["edges"], "derived_from": []})
                       for r in overlay["turns"] if r["kind"] == "carrier"), key=lambda x: x["key"])
    distractors = [rec(r["label"], where[r["label"]]) for r in overlay["turns"] if r["kind"] == "distractor"]
    taken = {y["record_id"] for sec in ("carriers", "direct") for y in base_ann[sec]}
    for x in overlay.get("existing_distractors", []):
        if x["record_id"] not in texts:
            raise ValueError(f"{x['record_id']}: not in the base episode")
        if x["record_id"] in taken:
            raise ValueError(f"{x['record_id']}: already evidence or a disclosure for a base fact")
        distractors.append(rec(x["label"], x["record_id"]))
    annotations = {"schema": "memcut.annotations.public.v1", "source_sha256": src_sha, "carriers": carriers,
                   "direct": [rec(f"{cid}-direct", d) for d in claim["direct_record_ids"]],
                   "distractors": distractors}

    edges = {x["key"]: set(x["edges"]) for x in carriers}
    need = set(c["fact_definitions"])
    sets = minimal_sets(edges, need)
    cuts = minimum_cuts(sets, sorted(edges))
    if not sets or len(cuts[0]) != cut_size_by_facts(edges, need):
        raise ValueError("the two minimum-cut methods disagree")
    key_to_record = {x["key"]: x["record_id"] for x in carriers}
    support = {"schema": "memcut.support_paths.public.v1", "source_sha256": src_sha,
               "limits": "Sufficient witnesses, not an exhaustive recovery oracle or a substitute for actual model answers.",
               "claims": [{"claim_id": cid, "mechanism": c["category"],
                           "witnesses": [{"keys": s, "record_ids": [key_to_record[k] for k in s]} for s in sets]}]}
    audit = {"schema": "memcut.path_combination_audit.public.v1", "source_sha256": src_sha,
             "method": "Map each evidence message to the named facts it states (fact_definitions), enumerate all subsets, "
                       "remove every strict superset of a sufficient subset; minimum cuts by enumeration, cross-checked "
                       "by the smallest deletion that leaves a named fact without a carrier.",
             "limits": ["This audits the annotated evidence pool, not a fresh search of the episode for unannotated carriers.",
                        "Counts concern support for the complete claim, not guessing or eliminating options from partial clues."],
             "claims": [{"claim_id": cid, "category": c["category"], "evidence_count": len(carriers),
                         "minimal_source_combinations": sets,
                         "minimum_hitting_set_size_in_audited_family": len(cuts[0]),
                         "one_minimum_hitting_set": cuts[0], "all_minimum_hitting_sets": cuts,
                         "audited_minimal_source_combination_count": len(sets),
                         "max_source_disjoint_in_audited_family": max_disjoint(sets),
                         "not_used_by_any_minimal_combination": sorted(set(edges) - {k for s in sets for k in s}),
                         "source_key_to_record_id": key_to_record,
                         "method": {"kind": "explicit_fact_coverage",
                                    "facts": {k: sorted(v) for k, v in sorted(edges.items())},
                                    "target_facts": c["fact_definitions"]}}]}
    problems = design_checks(variant, overlay, claim, annotations)
    if problems:
        raise ValueError("design checks failed:\n  " + "\n  ".join(problems))
    tokens = {"schema": "memcut.token_counts.public.v1", "source_sha256": src_sha, **token_counts(variant)}
    return {"source": variant, "contracts": contracts, "annotations": annotations, "tokens": tokens,
            "support_paths": support, "path_audit": audit,
            "_report": {"placed": where, "sets": sets, "cuts": cuts,
                        "turns_added": sum(len(s["turns"]) for s in variant["sessions"]) - len(base_turns)}}


def episode_paths(ep: str) -> dict[str, str]:
    return {"source": f"data/{ep}/public/{ep}.json", "contracts": f"data/{ep}/author/CLAIM_CONTRACTS.json",
            "annotations": f"data/{ep}/author/ANNOTATIONS.json", "tokens": f"data/{ep}/author/TOKEN_COUNTS.json",
            "support_paths": f"data/{ep}/author/SUPPORT_PATHS.json",
            "path_audit": f"data/{ep}/author/PATH_COMBINATION_AUDIT.json"}


def from_bundle(bundle: str | Path, overlay: dict) -> dict:
    z = zipfile.ZipFile(bundle)
    man = json.loads(z.read("MANIFEST.json"))
    entry = next((e for e in man["episodes"] if e["id"] == overlay["base_episode"]), None)
    if entry is None:
        raise ValueError(f"{overlay['base_episode']}: not an episode of {bundle}")
    return extend(json.loads(z.read(entry["files"]["source"]["path"])),
                  json.loads(z.read(entry["files"]["annotations"]["path"])), overlay)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="a MemCut bundle (zip) holding the base episode")
    ap.add_argument("--overlay", required=True, help="the fact's overlay (memcut.case_overlay.v1)")
    ap.add_argument("--output", required=True, help="new directory; the episode is written under data/<EP>/")
    a = ap.parse_args()
    overlay = json.loads(Path(a.overlay).read_text())
    out = Path(a.output)
    if out.exists():
        raise SystemExit(f"{out}: choose a new directory; existing data is never overwritten")
    try:
        files = from_bundle(a.bundle, overlay)
    except ValueError as exc:
        raise SystemExit(str(exc))
    for kind, rel in episode_paths(overlay["variant_episode"]).items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(dumps(files[kind]))
    r = files["_report"]
    print(json.dumps({"episode": overlay["variant_episode"], "turns_added": r["turns_added"],
                      "minimal_sufficient_sets": len(r["sets"]), "minimum_cut": len(r["cuts"][0]),
                      "minimum_cuts": r["cuts"], "placed": r["placed"], "output": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
