"""Seeded, explicitly exploratory pilot candidate selection."""
import hashlib
import json
import random


def select_pilot(records, *, size=75, sensitive_size=25, seed=5187, eligible_ids=None):
    if size < 1 or sensitive_size < 0 or sensitive_size > size:
        raise ValueError("Invalid pilot sizes")
    if eligible_ids is not None:
        eligible_ids = set(eligible_ids)
        pool = [row for row in records if row["id"] in eligible_ids]
    else:
        pool = list(records)
    rng = random.Random(seed)
    sensitive = sorted((row for row in pool if row["consensus"]["part1"] != row["consensus"]["part2"]), key=lambda row: row["id"])
    ordinary = sorted((row for row in pool if row["consensus"]["part1"] == row["consensus"]["part2"]), key=lambda row: row["id"])
    changed_ids = rng.sample([row["id"] for row in sensitive], min(sensitive_size, len(sensitive), size))
    ordinary_ids = rng.sample([row["id"] for row in ordinary], min(size - len(changed_ids), len(ordinary)))
    remaining = [row["id"] for row in sensitive if row["id"] not in changed_ids]
    changed_ids += rng.sample(remaining, min(size - len(changed_ids) - len(ordinary_ids), len(remaining)))
    ids = changed_ids + ordinary_ids
    rng.shuffle(ids)
    return {"seed": seed, "requested_size": size, "selected_size": len(ids), "ids": ids,
            "human_sensitive_count": len(changed_ids), "ordinary_count": len(ordinary_ids),
            "context_eligibility_required": eligible_ids is not None,
            "selection_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "purpose": "Exploratory pilot; intentionally oversamples human-context-sensitive rows. Candidates are not inference results. Freeze actual eligible cohort before either condition."}
