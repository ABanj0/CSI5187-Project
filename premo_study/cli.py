"""Explicit, reproducible experiment stages; no paid or GPU work by default."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .context import (GitHubClient, RULES_VERSION, _row_checksum, _verified_snapshot,
                      canonical, coverage_report, retrieve_contexts, write_immutable)
from .data import audit_dataset, load_dataset
from .discovery import discover_backends
from .evaluation import evaluate, human_agreement
from .pilot import select_pilot
from .provenance import runtime_metadata, verify_manifest

SOURCE_SHA256 = "06c604e0daaa5964de30026019e6dbe3ab29b4849b38ea92e7ed6335f46e7fa6"
DEFAULT_DATASET = "../PRemo/sentiment-dataset/data/dataset.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_records(path, expected_sha256=SOURCE_SHA256):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("Dataset fingerprint mismatch; review source provenance before changing the pin")
    return load_dataset(path), actual


def checked_contexts(path, records):
    path = Path(path)
    payload = read_json(path)
    if path.name != "acquisition-" + hashlib.sha256(canonical(payload)).hexdigest() + ".json":
        raise ValueError("Acquisition manifest checksum/name mismatch")
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError("Expected an acquisition manifest containing records")
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate context IDs")
    indexed = {r["id"]: r for r in records}
    if set(by_id) - set(indexed):
        raise ValueError("Context contains unknown dataset IDs")
    basis = [indexed[row["id"]] for row in rows]
    if payload.get("coverage") != coverage_report(basis, rows):
        raise ValueError("Acquisition coverage mismatch")
    usable = {}
    for identifier, row in by_id.items():
        if not row.get("eligible"):
            continue
        source = indexed[identifier]
        if row.get("status") != "available" or row.get("message_url") != source["message_url"] or row.get("target_text_sha256") != hashlib.sha256(source["raw_message"].encode()).hexdigest():
            raise ValueError("Context eligibility/provenance mismatch")
        if row.get("rules_version") != RULES_VERSION or row.get("artifact_sha256") != _row_checksum(row):
            raise ValueError("Context rules/checksum mismatch")
        if read_json(path.parent / "records" / (identifier + ".json")) != row:
            raise ValueError("Manifest context differs from immutable record")
        references = row.get("snapshot_references")
        if not isinstance(references, list) or not references:
            raise ValueError("Eligible context lacks immutable source snapshots")
        for reference in references:
            snapshot = (path.parent / reference).resolve()
            if Path(reference).is_absolute() or not snapshot.is_relative_to((path.parent / "snapshots").resolve()):
                raise ValueError("Snapshot reference escapes acquisition")
            _verified_snapshot(snapshot)
        usable[identifier] = row["context"]
    return usable


def checked_selection(path, records, source_hash):
    selection = read_json(path)
    ids = selection.get("ids")
    if selection.get("dataset_sha256") != source_hash or not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError("Selection provenance or IDs invalid")
    expected = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if selection.get("selection_sha256") != expected:
        raise ValueError("Selection checksum mismatch")
    indexed = {r["id"]: r for r in records}
    if set(ids) - set(indexed):
        raise ValueError("Selection contains unknown IDs")
    return [indexed[i] for i in ids]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Validate original data and recompute consensus diagnostics")
    audit.add_argument("--out", default="reports/dataset_audit.json")
    prepare = commands.add_parser("prepare", help="Write lossless normalized records and human agreement")
    prepare.add_argument("--out", default="artifacts/prepared")
    discover = commands.add_parser("discover", help="Read-only existing inference-service discovery")
    discover.add_argument("--out", default="artifacts/backend-discovery.json")
    retrieve = commands.add_parser("retrieve", help="Retrieve GitHub context; never fabricates missing text")
    retrieve.add_argument("--out", required=True, help="Named acquisition directory; new directory to refresh")
    retrieve.add_argument("--selection", help="Frozen selection manifest; fetch only its validated IDs")
    retrieve.add_argument("--max-consecutive-failures", type=int, default=3)
    pilot = commands.add_parser("pilot", help="Create exploratory candidate or context-eligible pilot manifest")
    pilot.add_argument("--contexts", help="Acquisition manifest; if absent, select candidates only")
    pilot.add_argument("--size", type=int, default=75)
    pilot.add_argument("--sensitive-size", type=int, default=25)
    pilot.add_argument("--seed", type=int, default=5187)
    pilot.add_argument("--out", required=True)
    inference = commands.add_parser("infer", help="Run paired, resumable inference after coverage validation")
    inference.add_argument("--backend", choices=("ollama",), required=True)
    inference.add_argument("--model", help="Exact local model identifier; defaults qwen3.5:9b")
    inference.add_argument("--contexts", required=True)
    inference.add_argument("--selection", required=True, help="Frozen pilot/full cohort manifest")
    inference.add_argument("--out", required=True)
    inference.add_argument("--batch-size", type=int, default=1)
    inference.add_argument("--num-ctx", type=int, default=32768)
    inference.add_argument("--max-attempts", type=int, default=3)
    progress = commands.add_parser("progress", help="Inspect saved inference artifacts without model calls")
    progress.add_argument("--out", required=True, help="Inference output directory")
    score = commands.add_parser("evaluate", help="Score actual majority outputs; no synthetic predictions")
    score.add_argument("--predictions", required=True, help="Inference result JSON, including provenance")
    score.add_argument("--bootstrap-iterations", type=int, default=2000)
    score.add_argument("--seed", type=int, default=5187)
    score.add_argument("--out", required=True)
    integrity = commands.add_parser("integrity", help="Compare every original dependency file to initial manifest")
    integrity.add_argument("--root", default="../PRemo")
    integrity.add_argument("--manifest", default="reports/premo_integrity_baseline.json")
    integrity.add_argument("--out", default="artifacts/integrity-check.json")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "discover":
            result = discover_backends()
        elif args.command == "progress":
            from .inference import inference_progress
            print(json.dumps(inference_progress(args.out), indent=2))
            return 0
        elif args.command == "integrity":
            result = verify_manifest(args.root, read_json(args.manifest))
        else:
            records, source_hash = checked_records(args.dataset)
            if args.command == "audit":
                result = audit_dataset(args.dataset)
            elif args.command == "prepare":
                destination = Path(args.out)
                write_immutable(destination / "records.json", {"dataset_sha256": source_hash, "records": records})
                result = {"dataset_sha256": source_hash, "records": len(records), "human_agreement": human_agreement(records)}
                write_immutable(destination / "summary.json", result)
                print(json.dumps({"records": len(records), "out": str(destination)}))
                return 0
            elif args.command == "retrieve":
                if args.max_consecutive_failures < 1:
                    raise ValueError("max-consecutive-failures must be positive")
                if args.selection:
                    records = checked_selection(args.selection, records, source_hash)
                result = retrieve_contexts(records, GitHubClient(Path(args.out) / "snapshots"), args.out,
                                           max_consecutive_failures=args.max_consecutive_failures)
                print(json.dumps(result["coverage"], indent=2))
                return 0
            elif args.command == "pilot":
                contexts = checked_contexts(args.contexts, records) if args.contexts else None
                result = select_pilot(records, size=args.size, sensitive_size=args.sensitive_size,
                                      seed=args.seed, eligible_ids=contexts.keys() if contexts is not None else None)
                result["dataset_sha256"] = source_hash
            elif args.command == "infer":
                from .inference import OllamaAdapter, run_experiment
                contexts = checked_contexts(args.contexts, records)
                selected = checked_selection(args.selection, records, source_hash)
                adapter = OllamaAdapter(args.model or "qwen3.5:9b", settings={"num_ctx": args.num_ctx})
                result = run_experiment(selected, contexts, adapter, args.out,
                                        dataset_sha256=source_hash, batch_size=args.batch_size, max_attempts=args.max_attempts)
                print(json.dumps({"requested": len(selected), "eligible": len(result["eligible_ids"]),
                                  "completed_pairs": len(result["complete_ids"]), "out": args.out}))
                return 0
            elif args.command == "evaluate":
                from .inference import PROTOCOL_VERSION, digest
                predictions = read_json(args.predictions)
                if predictions.get("dataset_sha256") != source_hash or not isinstance(predictions.get("predictions"), list):
                    raise ValueError("Inference result fingerprint/predictions missing or mismatched")
                if predictions.get("protocol_version") != PROTOCOL_VERSION or Path(args.predictions).stem != digest(predictions):
                    raise ValueError("Expected an immutable runner result with matching protocol/checksum")
                result = evaluate(records, predictions["predictions"], bootstrap_iterations=args.bootstrap_iterations, seed=args.seed)
                result["provenance"] = {"dataset_sha256": source_hash, "inference_result_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
                                        "inference_result": str(args.predictions), "runtime": {k: v for k, v in runtime_metadata().items() if k != "recorded_at"}}
                result["inference_exclusions"] = predictions.get("exclusions", [])
                result["inference_consistency"] = predictions.get("consistency", [])
        write_immutable(args.out, result)
        print(json.dumps({"out": str(args.out), "command": args.command,
                          "paired_n": result.get("paired_n"), "records": result.get("comments")}, indent=2))
        if args.command == "integrity" and not result["unchanged"]:
            return 1
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
