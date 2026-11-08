#!/usr/bin/env python3
"""Validate a MemCut public data bundle without unpacking it.

    python3 validate.py memcut_v1.zip

Checks: MANIFEST.json names every file with its size and SHA-256 and the content id matches; every dialogue has the
public schema (sessions with id, timestamp, turns with id, role, speaker, text; turn ids unique); every claim has
five options including the insufficient-information option, a case-study episode names its main-set base episode, a target that is not that option, three display orders
that are permutations of the options, a deletion request, a question and at least one evidence carrier; every
annotated record (carrier, direct, distractor) exists in the dialogue and its text hash matches; no file outside the
dialogue contains CJK text. Exit code 0 when clean; the problems are printed otherwise.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile

CJK = re.compile("[%s-%s]" % (chr(0x4E00), chr(0x9FFF)))   # a CJK ideograph
FORMAT = "memcut-15ep-data-bundle-v1"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def validate(path: str) -> list[str]:
    z = zipfile.ZipFile(path)
    names = set(z.namelist())
    bad: list[str] = []
    man = json.loads(z.read("MANIFEST.json"))
    if man.get("format") != FORMAT:
        bad.append(f"MANIFEST format {man.get('format')!r}, expected {FORMAT!r}")
    hashes = []
    listed = {"MANIFEST.json"}
    for e in man.get("episodes", []):
        ep = e["id"]
        for kind, f in e["files"].items():
            listed.add(f["path"])
            if f["path"] not in names:
                bad.append(f"{ep}: {f['path']} missing")
                continue
            blob = z.read(f["path"])
            if sha(blob) != f["sha256"] or len(blob) != f["bytes"]:
                bad.append(f"{ep}: {f['path']} hash or size differs from the manifest")
            hashes.append(f["sha256"])
            if kind != "source" and CJK.search(blob.decode("utf-8", "ignore")):
                bad.append(f"{ep}: {f['path']} contains CJK text")
        if "source" not in e["files"] or "contracts" not in e["files"] or "annotations" not in e["files"]:
            bad.append(f"{ep}: source, contracts and annotations are required")
            continue
        src = json.loads(z.read(e["files"]["source"]["path"]))
        turns: dict[str, str] = {}
        for s in src.get("sessions", []):
            for k in ("id", "timestamp", "turns"):
                if k not in s:
                    bad.append(f"{ep}: session without {k}")
            for t in s.get("turns", []):
                if set(t) != {"id", "role", "speaker", "text"}:
                    bad.append(f"{ep}: turn {t.get('id')} has keys {sorted(t)}")
                if t["id"] in turns:
                    bad.append(f"{ep}: duplicate turn id {t['id']}")
                turns[t["id"]] = t["text"]
        if e.get("set", "main") not in ("main", "case_study"):
            bad.append(f"{ep}: unknown set {e.get('set')!r}")
        if e.get("set") == "case_study" and e.get("base_episode") not in {x["id"] for x in man["episodes"]
                                                                          if x.get("set", "main") == "main"}:
            bad.append(f"{ep}: a case-study episode must name its main-set base episode")
        if e.get("turns") != len(turns) or e.get("sessions") != len(src.get("sessions", [])):
            bad.append(f"{ep}: manifest counts differ from the dialogue")
        src_sha = sha(z.read(e["files"]["source"]["path"]))
        con = json.loads(z.read(e["files"]["contracts"]["path"]))
        ann = json.loads(z.read(e["files"]["annotations"]["path"]))
        for d, nm in ((con, "contracts"), (ann, "annotations")):
            if d.get("source_sha256") != src_sha:
                bad.append(f"{ep}: {nm} source_sha256 does not match the dialogue")
        carriers = {}
        for sec in ("carriers", "direct", "distractors"):
            for x in ann.get(sec, []):
                if x["record_id"] not in turns:
                    bad.append(f"{ep} {sec} {x.get('key')}: record {x['record_id']} not in the dialogue")
                elif x.get("text_sha256") and sha(turns[x["record_id"]].encode()) != x["text_sha256"]:
                    bad.append(f"{ep} {sec} {x.get('key')}: text hash differs")
                if sec == "carriers":
                    carriers.setdefault(x["claim_id"], []).append(x["key"])
        if e.get("claims") != len(con.get("claims", [])):
            bad.append(f"{ep}: manifest claim count differs")
        for c in con.get("claims", []):
            cid = c.get("claim_id", "?")
            ids = [x["id"] for x in c.get("choices", [])]
            if len(ids) != 5 or len(set(ids)) != 5:
                bad.append(f"{ep} {cid}: five distinct options required")
            if c.get("target_choice_id") not in ids or c.get("insufficient_choice_id") not in ids:
                bad.append(f"{ep} {cid}: target or insufficient option not among the options")
            if c.get("target_choice_id") == c.get("insufficient_choice_id"):
                bad.append(f"{ep} {cid}: the target is the insufficient option")
            rots = c.get("rotations", [])
            if [r.get("trial") for r in rots] != [1, 2, 3]:
                bad.append(f"{ep} {cid}: three display orders (trials 1, 2, 3) required")
            for r in rots:
                order = r.get("choice_ids_in_display_order", [])
                d2c = r.get("display_to_choice", {})
                if sorted(order) != sorted(ids) or list(d2c.values()) != list(order) or len(d2c) != len(ids):
                    bad.append(f"{ep} {cid}: trial {r.get('trial')} is not a permutation of the options")
            for k in ("claim", "deletion_request", "question", "category"):
                if not c.get(k):
                    bad.append(f"{ep} {cid}: {k} missing")
            if not carriers.get(cid):
                bad.append(f"{ep} {cid}: no evidence carriers")
            for t in c.get("direct_record_ids", []):
                if t not in turns:
                    bad.append(f"{ep} {cid}: direct record {t} not in the dialogue")
    if sha("".join(hashes).encode()) != man.get("registry_sha256"):
        bad.append("registry_sha256 (content id) does not match the files")
    for name in names:
        if name.startswith("data/") and name not in listed:
            bad.append(f"{name}: in the archive but not in the manifest")
        if not name.startswith("data/") and name not in ("README.md", "LICENSE", "SCHEMA.md", "validate.py", "MANIFEST.json"):
            bad.append(f"{name}: unexpected file")
    for name in ("README.md", "SCHEMA.md", "LICENSE", "validate.py"):
        if name not in names:
            bad.append(f"{name} missing")
    return bad


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    bad = validate(sys.argv[1])
    man = json.loads(zipfile.ZipFile(sys.argv[1]).read("MANIFEST.json"))
    t = man.get("totals", {})
    if bad:
        print("\n".join(bad))
        print(f"\n{len(bad)} problem(s)")
        return 1
    cases = [e["id"] for e in man.get("episodes", []) if e.get("set") == "case_study"]
    print(f"ok: {t.get('episodes')} episodes, {t.get('sessions')} sessions, {t.get('turns')} turns, "
          f"{t.get('claims')} claims" + (f"; case-study episodes {', '.join(cases)}" if cases else "")
          + f"; content id {man.get('registry_sha256', '')[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
