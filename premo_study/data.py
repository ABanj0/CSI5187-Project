"""Validate PRemo and derive per-emotion majorities without changing source data."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

EMOTIONS = ("joy", "love", "sadness", "anger", "fear", "positivesurprise", "negativesurprise")
PARTS = ("part1", "part2")
URL_PATTERN = re.compile(r"https://github\.com/([^/]+)/([^/]+)/pull/([1-9][0-9]*)(?:#issuecomment-([1-9][0-9]*))?$")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_dataset(path: str | Path) -> list[dict]:
    """Return validated, lossless records with independent 2-of-3 emotion votes.

    Original fields, including the ``negative_itensity`` typo, remain untouched
    in ``source_record``. An empty consensus is never named neutral. Bare PR
    URLs identify candidate PR-body targets; retrieval must verify the body.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(raw, list) and bool(raw), "Dataset must be a nonempty JSON array")
    records = []
    seen = set()
    for index, item in enumerate(raw):
        prefix = f"Row {index}"
        _require(isinstance(item, dict), f"{prefix}: expected object")
        for field in ("project", "message_url", "raw_message"):
            _require(isinstance(item.get(field), str) and bool(item[field].strip()), f"{prefix}: invalid {field}")
        url = item["message_url"]
        match = URL_PATTERN.fullmatch(url)
        _require(match is not None, f"{prefix}: unsupported target URL")
        _require(url not in seen, f"{prefix}: duplicate URL {url}")
        seen.add(url)
        answers = item.get("individual_answers")
        _require(isinstance(answers, list) and len(answers) == 3, f"{prefix}: expected exactly three annotators")
        _require(all(isinstance(a, dict) for a in answers), f"{prefix}: annotators must be objects")
        _require(Counter(a.get("user_type") for a in answers) == {"comp": 2, "neuro": 1}, f"{prefix}: expected two comp annotators and one neuro annotator")
        consensus, votes, all_empty = {}, {}, {}
        for part in PARTS:
            aggregate = item.get(f"{part}_aggregate")
            _require(isinstance(aggregate, dict), f"{prefix}: missing {part} aggregate")
            _require(aggregate.get("polarity") in ("positive", "negative", "neutral", "undefined"), f"{prefix}: invalid aggregate polarity")
            _require(aggregate.get("agreement_type") in ("all", "neuro_and_comp", "comp_only", "undefined"), f"{prefix}: invalid agreement type")
            confidence = aggregate.get("avg_confidence")
            _require(type(confidence) in (int, float) and 1 <= confidence <= 5, f"{prefix}: invalid aggregate confidence")
            counts = Counter()
            for position, answer in enumerate(answers):
                _require(isinstance(answer, dict) and answer.get("user_type") in ("comp", "neuro"), f"{prefix}: invalid annotator role")
                annotation = answer.get(part)
                _require(isinstance(annotation, dict), f"{prefix}: missing {part} annotation {position}")
                emotions = annotation.get("emotions")
                _require(isinstance(emotions, list) and all(isinstance(e, str) for e in emotions), f"{prefix}: emotions must be strings")
                _require(len(emotions) == len(set(emotions)) and set(emotions) <= set(EMOTIONS), f"{prefix}: unknown or duplicate emotion")
                for field, low, high in (("confidence", 1, 5), ("positive_intensity", 0, 4), ("negative_itensity", 0, 4)):
                    value = annotation.get(field)
                    _require(type(value) is int and low <= value <= high, f"{prefix}: invalid {part}.{field}")
                _require(annotation.get("polarity") in ("positive", "negative", "neutral"), f"{prefix}: invalid annotation polarity")
                _require(isinstance(annotation.get("comments"), str), f"{prefix}: invalid annotation comments")
                counts.update(emotions)
            votes[part] = {emotion: counts[emotion] for emotion in EMOTIONS}
            consensus[part] = [emotion for emotion in EMOTIONS if counts[emotion] >= 2]
            all_empty[part] = not counts
        if "discussion_polarity" in item:
            _require(item["discussion_polarity"] in ("positive", "negative", "neutral"), f"{prefix}: invalid discussion polarity")
        records.append({
            "id": hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
            "source_index": index, "project": item["project"], "message_url": url,
            "raw_message": item["raw_message"], "pr_url": url.split("#")[0],
            "target_kind": "issue_comment" if match.group(4) else "pr_body",
            "source_record": deepcopy(item), "consensus": consensus, "votes": votes,
            "all_annotators_no_emotion": all_empty,
        })
    _require(len({r["id"] for r in records}) == len(records), "Stable identifier collision")
    return records


def audit_dataset(path: str | Path) -> dict:
    """Return reproducible schema, coverage and label diagnostics; raise on invalid data."""
    path = Path(path)
    records = load_dataset(path)
    by_pass = {}
    for part in PARTS:
        annotations = [a[part] for r in records for a in r["source_record"]["individual_answers"]]
        by_pass[part] = {
            "individual_emotion_counts": dict(Counter(e for a in annotations for e in a["emotions"])),
            "consensus_emotion_counts": {e: sum(e in r["consensus"][part] for r in records) for e in EMOTIONS},
            "empty_consensus": sum(not r["consensus"][part] for r in records),
            "all_annotators_no_emotion": sum(r["all_annotators_no_emotion"][part] for r in records),
            "empty_consensus_with_human_emotion": sum(not r["consensus"][part] and not r["all_annotators_no_emotion"][part] for r in records),
            "neutral_polarity_with_emotion": sum(a["polarity"] == "neutral" and bool(a["emotions"]) for a in annotations),
            "unanimous_emotion_sets": sum(len({tuple(sorted(a[part]["emotions"])) for a in r["source_record"]["individual_answers"]}) == 1 for r in records),
            "aggregate_polarity_counts": dict(Counter(r["source_record"][f"{part}_aggregate"]["polarity"] for r in records)),
        }
    warnings = [
        "Aggregate polarity and agreement fields are not emotion ground truth.",
        "Empty majority emotion sets include disagreement and do not confirm neutrality.",
        "Human pass2 allowed preceding and subsequent discussion; a preceding-only experiment differs.",
        "Annotator identities and annotation timestamps are unavailable; array positions are local to each row.",
        "Bare PR URLs are candidate PR-body targets and require live/snapshot identity verification.",
    ]
    expected = {"comments": 1791, "projects": 36}
    actual = {"comments": len(records), "projects": len({r["project"] for r in records})}
    return {
        "schema_version": 1, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **actual, "expected_characteristic_discrepancies": {k: {"expected": v, "actual": actual[k]} for k, v in expected.items() if actual[k] != v},
        "unique_urls": len({r["message_url"] for r in records}),
        "unique_texts": len({r["raw_message"] for r in records}),
        "unique_pull_requests": len({r["pr_url"] for r in records}),
        "target_kinds": dict(Counter(r["target_kind"] for r in records)),
        "project_counts": dict(sorted(Counter(r["project"] for r in records).items())),
        "annotations_per_pass": 3 * len(records), "emotions": list(EMOTIONS),
        "role_counts": dict(Counter(a["user_type"] for r in records for a in r["source_record"]["individual_answers"])),
        "passes": by_pass,
        "consensus_changed_between_passes": sum(r["consensus"]["part1"] != r["consensus"]["part2"] for r in records),
        "discussion_polarity_rows": sum("discussion_polarity" in r["source_record"] for r in records),
        "warnings": warnings,
    }
