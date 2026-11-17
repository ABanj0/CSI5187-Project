"""Paired multi-label evaluation, with PR-clustered uncertainty and explicit coverage."""
from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import combinations
import math
import random
import re

from .data import EMOTIONS, PARTS

CONDITIONS = ("comment_only", "context")
QUALITATIVE_CODEBOOK = {
    "speaker_confusion": "Emotion attributed to another participant, quoted text, or a bot instead of the target author.",
    "missing_context": "Needed information is absent or was omitted from the retrieved/truncated preceding context.",
    "politeness_convention": "Different interpretation of conventional thanks, apologies, greetings, or routine review courtesy.",
    "sarcasm": "An ironic or sarcastic expression is interpreted literally or assigned the wrong emotional intent.",
    "technical_ambiguity": "Technical terminology, code, or an ambiguous reference changes the emotional interpretation.",
    "human_disagreement": "The three source annotators disagree on at least one disputed emotion; record evidence, not assumed human error.",
    "other": "A documented explanation that does not fit the other categories.",
}


def _divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def binary_kappa(left, right):
    """Cohen's kappa; undefined constant marginals return None rather than 1."""
    if not left or len(left) != len(right):
        return None
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    pa, pb = sum(left) / len(left), sum(right) / len(right)
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def multilabel_metrics(truth, predicted):
    """Seven independent binary decisions; zero-denominator P/R/F1 use zero."""
    if len(truth) != len(predicted):
        raise ValueError("Truth and predictions must have equal length")
    if not truth:
        return {"n": 0, "micro_f1": None, "macro_f1": None, "subset_accuracy": None, "hamming_loss": None, "per_emotion": {}}
    per_emotion = {}
    totals = [0, 0, 0]
    for emotion in EMOTIONS:
        left = [emotion in row for row in truth]
        right = [emotion in row for row in predicted]
        tp = sum(a and b for a, b in zip(left, right))
        fp = sum(not a and b for a, b in zip(left, right))
        fn = sum(a and not b for a, b in zip(left, right))
        tn = len(left) - tp - fp - fn
        for i, value in enumerate((tp, fp, fn)):
            totals[i] += value
        per_emotion[emotion] = {
            "support": sum(left), "predicted_positive": sum(right), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": _divide(tp, tp + fp), "recall": _divide(tp, tp + fn),
            "f1": _divide(2 * tp, 2 * tp + fp + fn), "kappa": binary_kappa(left, right),
        }
    tp, fp, fn = totals
    return {
        "n": len(truth), "micro_precision": _divide(tp, tp + fp), "micro_recall": _divide(tp, tp + fn),
        "micro_f1": _divide(2 * tp, 2 * tp + fp + fn),
        "macro_f1": sum(x["f1"] for x in per_emotion.values()) / len(EMOTIONS),
        "subset_accuracy": sum(set(a) == set(b) for a, b in zip(truth, predicted)) / len(truth),
        "hamming_loss": (fp + fn) / (len(truth) * len(EMOTIONS)),
        "per_emotion": per_emotion,
    }


def human_agreement(records):
    """Pool within-comment role pairs; this does not infer global annotator IDs."""
    report = {}
    for part in PARTS:
        groups = defaultdict(lambda: {e: ([], []) for e in EMOTIONS})
        for record in records:
            answers = record["source_record"]["individual_answers"]
            for first, second in combinations(range(3), 2):
                a, b = answers[first], answers[second]
                role = "+".join(sorted((a["user_type"], b["user_type"])))
                for emotion in EMOTIONS:
                    values = (emotion in a[part]["emotions"], emotion in b[part]["emotions"])
                    for group in ("all_within_comment_pairs", role, f"local_positions_{first}_{second}"):
                        groups[group][emotion][0].append(values[0])
                        groups[group][emotion][1].append(values[1])
        report[part] = {group: {emotion: {"pairs": len(a), "kappa": binary_kappa(a, b), "raw_agreement": _divide(sum(x == y for x, y in zip(a, b)), len(a))} for emotion, (a, b) in rows.items()} for group, rows in sorted(groups.items())}
    return report


def model_individual_agreement(records, lookup):
    """Agreement with each human annotation, pooled by role/local position.

    ``lookup`` maps condition to ID to labels. Callers provide the common paired
    records. Repeated model labels and correlated humans make these descriptive
    agreement summaries, not independent inferential samples.
    """
    result = {}
    for part in PARTS:
        result[part] = {}
        for condition in CONDITIONS:
            groups = defaultdict(lambda: {e: ([], []) for e in EMOTIONS})
            for record in records:
                predicted = lookup[condition][record["id"]]
                for position, answer in enumerate(record["source_record"]["individual_answers"]):
                    for emotion in EMOTIONS:
                        values = (emotion in predicted, emotion in answer[part]["emotions"])
                        for group in ("all_individual_annotations", answer["user_type"], f"local_position_{position}"):
                            groups[group][emotion][0].append(values[0])
                            groups[group][emotion][1].append(values[1])
            result[part][condition] = {
                group: {emotion: {"pairs": len(a), "kappa": binary_kappa(a, b), "raw_agreement": _divide(sum(x == y for x, y in zip(a, b)), len(a))} for emotion, (a, b) in rows.items()}
                for group, rows in sorted(groups.items())
            }
    return result


def exact_mcnemar(before_correct, after_correct):
    """Exact two-sided binomial McNemar test of paired correctness decisions.

    This test assumes independent pairs. PR-cluster dependence and multiple
    labels invalidate an unqualified confirmatory interpretation here.
    """
    if len(before_correct) != len(after_correct):
        raise ValueError("McNemar pairs must have equal length")
    worsened = sum(a and not b for a, b in zip(before_correct, after_correct))
    improved = sum(not a and b for a, b in zip(before_correct, after_correct))
    discordant = improved + worsened
    if not before_correct:
        probability = None
    elif not discordant:
        probability = 1.0
    else:
        probability = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(improved, worsened) + 1)) / (2 ** discordant))
    return {"n": len(before_correct), "comment_only_correct_context_wrong": worsened, "comment_only_wrong_context_correct": improved, "discordant_pairs": discordant, "two_sided_exact_p": probability}


def _mcnemar_comparison(records, lookup, ids, part):
    result = {}
    for label in (*EMOTIONS, "subset_exact_match"):
        correctness = {}
        for condition in CONDITIONS:
            if label == "subset_exact_match":
                correctness[condition] = [set(lookup[condition][i]) == set(records[i]["consensus"][part]) for i in ids]
            else:
                correctness[condition] = [(label in lookup[condition][i]) == (label in records[i]["consensus"][part]) for i in ids]
        result[label] = exact_mcnemar(correctness["comment_only"], correctness["context"])
    return result


def comment_strata(record):
    """Prespecified overlapping descriptive strata; they are not semantic labels."""
    text = record["raw_message"]
    length = len(text)
    strata = ["length_1_80" if length <= 80 else "length_81_280" if length <= 280 else "length_281_plus", record["target_kind"]]
    strata.append("human_consensus_changes" if record["consensus"]["part1"] != record["consensus"]["part2"] else "human_consensus_stable")
    strata.append("empty_part2_consensus" if not record["consensus"]["part2"] else "nonempty_part2_consensus")
    if "```" in text or "~~~" in text:
        strata.append("fenced_code")
    if re.search(r"(?<!`)`[^`\n]+`(?!`)", text):
        strata.append("inline_code")
    if re.search(r"https?://", text):
        strata.append("contains_link")
    return strata


def _comparison(records, lookup, ids, part):
    truth = [records[i]["consensus"][part] for i in ids]
    metrics = {condition: multilabel_metrics(truth, [lookup[condition][i] for i in ids]) for condition in CONDITIONS}
    delta = {name: metrics["context"][name] - metrics["comment_only"][name] if ids else None for name in ("micro_f1", "macro_f1", "subset_accuracy", "hamming_loss")}
    metrics["delta_context_minus_comment_only"] = delta
    metrics["per_emotion_delta_f1"] = {e: metrics["context"]["per_emotion"][e]["f1"] - metrics["comment_only"]["per_emotion"][e]["f1"] for e in EMOTIONS} if ids else {}
    return metrics


def _quantile(values, probability):
    values = sorted(values)
    index = (len(values) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def _bootstrap(records, lookup, ids, part, iterations, seed):
    clusters = defaultdict(list)
    for identifier in ids:
        clusters[records[identifier]["pr_url"]].append(identifier)
    keys = sorted(clusters)
    if not ids or iterations == 0:
        return {"iterations": 0, "clusters": len(keys), "intervals": {}, "arm_intervals": {condition: {} for condition in CONDITIONS}}
    rng = random.Random(seed)
    samples = defaultdict(list)
    arm_samples = {condition: defaultdict(list) for condition in CONDITIONS}
    for _ in range(iterations):
        sampled = [i for key in rng.choices(keys, k=len(keys)) for i in clusters[key]]
        result = _comparison(records, lookup, sampled, part)
        for name, value in result["delta_context_minus_comment_only"].items():
            samples[name].append(value)
        for emotion, value in result["per_emotion_delta_f1"].items():
            samples[f"f1_{emotion}"].append(value)
        for condition in CONDITIONS:
            for name in ("micro_f1", "macro_f1", "subset_accuracy", "hamming_loss"):
                arm_samples[condition][name].append(result[condition][name])
            for emotion in EMOTIONS:
                arm_samples[condition][f"f1_{emotion}"].append(result[condition]["per_emotion"][emotion]["f1"])
    intervals = lambda rows: {name: [_quantile(values, .025), _quantile(values, .975)] for name, values in rows.items()}
    return {"iterations": iterations, "clusters": len(keys), "seed": seed, "method": "paired PR-cluster percentile bootstrap, 95% intervals; descriptive, no multiplicity correction", "intervals": intervals(samples), "arm_intervals": {condition: intervals(rows) for condition, rows in arm_samples.items()}}


def evaluate(records, predictions, bootstrap_iterations=1000, seed=5187):
    """Evaluate aggregate predictions against the same human pass in both arms.

    Prediction rows contain ``id``, ``condition`` and ``emotions``. Explicit
    failures may instead contain ``status: error`` (or failed/invalid/skipped)
    and optional error metadata. Missing/failed rows are reported, never scored
    as empty predictions. Duplicate or unknown IDs/conditions fail closed.
    """
    if type(bootstrap_iterations) is not int or bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be a nonnegative integer")
    indexed = {record["id"]: record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate dataset IDs")
    if not records:
        raise ValueError("Cannot evaluate an empty dataset")
    lookup = {condition: {} for condition in CONDITIONS}
    failures = {condition: {} for condition in CONDITIONS}
    seen = set()
    for row in predictions:
        identifier, condition = row.get("id"), row.get("condition")
        if identifier not in indexed or condition not in CONDITIONS:
            raise ValueError(f"Unknown prediction ID/condition: {identifier!r}/{condition!r}")
        key = (identifier, condition)
        if key in seen:
            raise ValueError(f"Duplicate prediction: {key}")
        seen.add(key)
        if row.get("status") in ("error", "failed", "invalid", "skipped"):
            failures[condition][identifier] = row.get("status")
            continue
        emotions = row.get("emotions")
        if not isinstance(emotions, list) or any(not isinstance(e, str) for e in emotions) or len(emotions) != len(set(emotions)) or not set(emotions) <= set(EMOTIONS):
            raise ValueError(f"Invalid emotion set for {identifier}/{condition}")
        lookup[condition][identifier] = emotions
    paired_ids = sorted(set(lookup["comment_only"]) & set(lookup["context"]))
    coverage = {}
    for condition in CONDITIONS:
        missing = sorted(set(indexed) - set(lookup[condition]) - set(failures[condition]))
        coverage[condition] = {"dataset_n": len(records), "valid_n": len(lookup[condition]), "failed_n": len(failures[condition]), "missing_n": len(missing), "failed_ids": failures[condition], "missing_ids": missing}
    comparisons = {part: _comparison(indexed, lookup, paired_ids, part) for part in PARTS}
    strata = defaultdict(list)
    for identifier in paired_ids:
        for name in comment_strata(indexed[identifier]):
            strata[name].append(identifier)
    errors = defaultdict(list)
    for identifier in paired_ids:
        record = indexed[identifier]
        truth = set(record["consensus"]["part2"])
        isolated, context = (set(lookup[condition][identifier]) for condition in CONDITIONS)
        before, after = len(truth ^ isolated), len(truth ^ context)
        category = "improved" if after < before else "worsened" if after > before else "unchanged_error" if after else "both_correct"
        if category == "both_correct":
            continue
        errors[category].append({"id": identifier, "outcome": category, "manual_category": None, "manual_notes": None, "review_status": "unreviewed", "project": record["project"], "url": record["message_url"], "text_preview": record["raw_message"][:240], "truth": sorted(truth), "comment_only": sorted(isolated), "context": sorted(context), "comment_only_false_positive": sorted(isolated - truth), "comment_only_false_negative": sorted(truth - isolated), "context_false_positive": sorted(context - truth), "context_false_negative": sorted(truth - context)})
    selected = sorted((row for rows in errors.values() for row in rows), key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())[:50]
    sample = {category: [row for row in selected if row["outcome"] == category] for category in sorted(errors)}
    return {
        "schema_version": 1, "primary_reference": "part2", "sensitivity_reference": "part1",
        "coverage": coverage, "paired_n": len(paired_ids), "paired_ids": paired_ids,
        "warning": "Incomplete coverage: paired results apply only to the common successful cohort." if len(paired_ids) < len(records) else None,
        "metric_rules": "Seven binary labels; empty consensus is not neutral; undefined precision/recall/F1 use zero; undefined kappa is null. Delta is context minus comment_only; negative hamming-loss delta favors context.",
        "comparisons": comparisons,
        "uncertainty": {part: _bootstrap(indexed, lookup, paired_ids, part, bootstrap_iterations, seed) for part in PARTS},
        "human_agreement": human_agreement([indexed[i] for i in paired_ids]),
        "human_agreement_full_dataset": human_agreement(records),
        "model_individual_agreement": model_individual_agreement([indexed[i] for i in paired_ids], lookup),
        "descriptive_mcnemar": {
            "limitations": "Unadjusted exact tests assume independent comment pairs and do not account for PR dependence or multiple labels/references. Treat as descriptive; paired PR-cluster bootstrap intervals are primary.",
            "passes": {part: _mcnemar_comparison(indexed, lookup, paired_ids, part) for part in PARTS},
        },
        "strata_definitions": "Overlapping descriptive strata: length <=80,81..280,>=281 Unicode characters; target kind; changed human majority; empty/nonempty pass2 majority; fenced/inline code and HTTP links. Not manual semantic comment types.",
        "strata": {name: _comparison(indexed, lookup, ids, "part2") for name, ids in sorted(strata.items())},
        "error_counts": {category: len(rows) for category, rows in sorted(errors.items())},
        "error_sample_seed": seed, "error_samples": sample,
        "qualitative_review": {"status": "unreviewed", "sample_size": len(selected), "maximum_sample_size": 50, "selection_rule": "First 50 errors ordered by SHA256(seed:id), across all outcomes, against part2 consensus.", "codebook": QUALITATIVE_CODEBOOK, "instructions": "Inspect full target text, immutable context, source votes and predictions. Assign one manual_category from the codebook and document supporting evidence in manual_notes; preserve null values until an actual reviewer inspects each case. Improved/worsened/unchanged_error are numerical outcomes, not semantic explanations."},
    }
