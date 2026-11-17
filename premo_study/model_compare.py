"""Evaluate Claude and Qwen on exactly the same successful paired comments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .cli import DEFAULT_DATASET, checked_records
from .context import write_immutable
from .evaluation import evaluate
from .inference import CONDITIONS, PROTOCOL_VERSION, digest


def load_result(path, source_hash):
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if (path.stem != digest(result) or result.get("dataset_sha256") != source_hash or
            result.get("protocol_version") != PROTOCOL_VERSION):
        raise ValueError("Inference result checksum or provenance mismatch")
    ids = result.get("complete_ids")
    predictions = result.get("predictions")
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or not isinstance(predictions, list):
        raise ValueError("Invalid complete IDs or prediction list")
    expected = {(identifier, condition) for identifier in ids for condition in CONDITIONS}
    actual = {(row["id"], row["condition"]) for row in predictions}
    if actual != expected or len(predictions) != len(expected):
        raise ValueError("Predictions do not exactly match complete paired IDs")
    return result, hashlib.sha256(path.read_bytes()).hexdigest()


def matched_evaluations(records, qwen, claude, *, iterations=2000, seed=5187):
    qwen_ids, claude_ids = set(qwen["complete_ids"]), set(claude["complete_ids"])
    if not claude_ids or not claude_ids <= qwen_ids:
        raise ValueError("Claude paired cohort must be nonempty and contained in Qwen cohort")
    if qwen.get("adapter", {}).get("backend") != "ollama" or claude.get("adapter", {}).get("backend") != "claude":
        raise ValueError("Expected local Qwen and paid Claude results")
    indexed = {row["id"]: row for row in records}
    if not claude_ids <= set(indexed):
        raise ValueError("Unknown matched dataset IDs")
    matched_records = [indexed[identifier] for identifier in claude["complete_ids"]]
    evaluations = {}
    for model, result in (("qwen", qwen), ("claude", claude)):
        predictions = [row for row in result["predictions"] if row["id"] in claude_ids]
        scored = evaluate(matched_records, predictions, bootstrap_iterations=iterations, seed=seed)
        if scored["paired_n"] != len(claude_ids):
            raise ValueError("Matched model evaluation lost a paired ID")
        evaluations[model] = scored
    return evaluations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-result", required=True)
    parser.add_argument("--claude-result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5187)
    args = parser.parse_args(argv)
    records, source_hash = checked_records(DEFAULT_DATASET)
    qwen, qwen_sha = load_result(args.qwen_result, source_hash)
    claude, claude_sha = load_result(args.claude_result, source_hash)
    evaluations = matched_evaluations(records, qwen, claude,
                                      iterations=args.bootstrap_iterations, seed=args.seed)
    payload = {"dataset_sha256": source_hash, "qwen_result_file_sha256": qwen_sha,
               "claude_result_file_sha256": claude_sha, "matched_ids": claude["complete_ids"],
               "method": "Both model evaluations use the same successful Claude/Qwen paired IDs, pass-2 human consensus, and paired PR-cluster bootstrap; between-model differences are descriptive.",
               "evaluations": evaluations}
    write_immutable(args.out, payload)
    print(json.dumps({"out": args.out, "matched_n": len(claude["complete_ids"]),
                      "qwen_micro_f1": {condition: evaluations["qwen"]["comparisons"]["part2"][condition]["micro_f1"] for condition in CONDITIONS},
                      "claude_micro_f1": {condition: evaluations["claude"]["comparisons"]["part2"][condition]["micro_f1"] for condition in CONDITIONS}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
