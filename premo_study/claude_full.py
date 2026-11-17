"""Budgeted, resumable Claude comparison on the frozen Qwen complete cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from .cli import DEFAULT_DATASET, checked_contexts, checked_records, checked_selection
from .inference import ClaudeAdapter, InferenceError, canonical, digest, run_experiment, utc_now

INPUT_USD_PER_TOKEN = 0.10 / 1_000_000
OUTPUT_USD_PER_TOKEN = 0.50 / 1_000_000
MAX_USD = 3.0
MAX_CALLS = 6000
CALL_RESERVE_USD = 0.01


class PaidRunPause(RuntimeError):
    """Stop the whole run without consuming a study attempt."""


def qwen_complete_selection(result_path, cohort_path, records, source_hash, output_path):
    result_path = Path(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result_path.name != digest(result) + ".json" or result.get("dataset_sha256") != source_hash:
        raise ValueError("Qwen result checksum or dataset provenance mismatch")
    cohort = checked_selection(cohort_path, records, source_hash)
    cohort_ids = [row["id"] for row in cohort]
    ids = result.get("complete_ids")
    if (not isinstance(ids, list) or len(ids) != 846 or len(ids) != len(set(ids))
            or not set(ids).issubset(cohort_ids) or result.get("adapter", {}).get("backend") != "ollama"):
        raise ValueError("Expected 846 complete local Qwen IDs inside frozen cohort")
    if ids != [identifier for identifier in cohort_ids if identifier in set(ids)]:
        raise ValueError("Qwen complete IDs differ from frozen cohort order")
    payload = {"dataset_sha256": source_hash, "ids": ids,
               "selection_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
               "source_qwen_result_sha256": digest(result), "purpose": "Paid Claude comparison on exactly 846 complete Qwen paired IDs"}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if json.loads(output_path.read_text(encoding="utf-8")) != payload:
            raise ValueError("Frozen Claude selection differs from existing file")
    else:
        with output_path.open("x", encoding="utf-8") as stream:
            stream.write(canonical(payload) + "\n")
    return checked_selection(output_path, records, source_hash)


class BudgetedClaudeAdapter(ClaudeAdapter):
    def __init__(self, output_dir, *, max_usd=MAX_USD, max_calls=MAX_CALLS,
                 min_interval_seconds=2.0, **kwargs):
        super().__init__(**kwargs)
        self.output = Path(output_dir)
        self.max_usd = float(max_usd)
        self.max_calls = int(max_calls)
        self.min_interval_seconds = float(min_interval_seconds)
        if self.max_usd <= 0 or self.max_calls < 1 or self.min_interval_seconds < 0:
            raise ValueError("Invalid budget or pacing limit")
        self._last_call = None
        self._warned_2 = False

    def describe(self):
        return {**super().describe(), "spend_guard_usd": self.max_usd,
                "call_guard": self.max_calls, "minimum_call_interval_seconds": self.min_interval_seconds,
                "price_usd_per_million_input": 0.10, "price_usd_per_million_output": 0.50}

    def status(self):
        calls = sorted((self.output / "paid_calls").glob("*.json"))
        charges = sorted((self.output / "paid_charges").glob("*.json"))
        seen, total = set(), 0.0
        for path in calls:
            row = json.loads(path.read_text(encoding="utf-8"))
            if path.stem != row.get("call_id") or row.get("model") != self.model:
                raise PaidRunPause("Paid call ledger has invalid provenance")
            seen.add(path.stem)
        charged = set()
        for path in charges:
            row = json.loads(path.read_text(encoding="utf-8"))
            if (path.stem not in seen or row.get("call_id") != path.stem or
                    row.get("usd") != round(row.get("input_tokens", -1) * INPUT_USD_PER_TOKEN +
                                          row.get("output_tokens", -1) * OUTPUT_USD_PER_TOKEN, 10)):
                raise PaidRunPause("Paid charge ledger has invalid provenance")
            total += row["usd"]
            charged.add(path.stem)
        return {"calls": len(calls), "charged_calls": len(charges),
                "unreconciled_calls": sorted(seen - charged), "measured_usd": round(total, 10),
                "hard_cap_usd": self.max_usd, "max_calls": self.max_calls}

    def _alert(self, reason):
        status = self.status()
        path = self.output / "PAUSED.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical({"reason": reason, "status": status, "at": utc_now()}) + "\n", encoding="utf-8")
        raise PaidRunPause(f"{reason}; see {path}")

    def generate(self, system, user, schema):
        if not self.allow_paid or not self.api_key:
            raise PaidRunPause("Paid run requires --allow-paid and ANTHROPIC_API_KEY")
        status = self.status()
        if status["unreconciled_calls"]:
            self._alert("A prior API call has unknown billing status; reconcile before resume")
        if status["calls"] >= self.max_calls or status["measured_usd"] + CALL_RESERVE_USD > self.max_usd:
            self._alert("Local request or spending cap reached")
        if self._last_call is not None:
            time.sleep(max(0, self.min_interval_seconds - (time.monotonic() - self._last_call)))
        call_id = f"{status['calls'] + 1:06d}"
        path = self.output / "paid_calls" / f"{call_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # The claim precedes the network call, so an interrupted call is never silently repeated.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(canonical({"call_id": call_id, "model": self.model,
                                    "prompt_sha256": digest({"system": system, "user": user, "schema": schema}),
                                    "started_at": utc_now()}) + "\n")
        self._last_call = time.monotonic()
        try:
            response = super().generate(system, user, schema)
        except InferenceError as exc:
            # A known 429 is not billed; any other exception might be billed.
            if str(exc).startswith("HTTP 429"):
                self._write_charge(call_id, 0, 0, "rate_limited")
                self._alert("Anthropic rate limit reached")
            self._alert("API call failed; billing status may be unknown")
        usage = response.get("metadata", {}).get("usage", {})
        input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
        if (type(input_tokens) is not int or type(output_tokens) is not int or
                input_tokens < 0 or output_tokens < 0):
            self._alert("API response lacks valid billing token counts")
        self._write_charge(call_id, input_tokens, output_tokens, "success")
        current = self.status()
        crossed_2 = current["measured_usd"] >= 2.0 and not self._warned_2
        if current["charged_calls"] % 100 == 0 or crossed_2:
            print(f"Claude usage: {current['charged_calls']} calls, measured ${current['measured_usd']:.4f} / ${self.max_usd:.2f}",
                  file=sys.stderr, flush=True)
        if crossed_2:
            self._warned_2 = True
        if current["measured_usd"] > self.max_usd:
            self._alert("Measured spending exceeded local cap")
        return response

    def _write_charge(self, call_id, input_tokens, output_tokens, status):
        usd = round(input_tokens * INPUT_USD_PER_TOKEN + output_tokens * OUTPUT_USD_PER_TOKEN, 10)
        path = self.output / "paid_charges" / f"{call_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(canonical({"call_id": call_id, "status": status, "input_tokens": input_tokens,
                                    "output_tokens": output_tokens, "usd": usd, "recorded_at": utc_now()}) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-result")
    parser.add_argument("--cohort")
    parser.add_argument("--contexts")
    parser.add_argument("--selection-out")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if args.status:
        print(canonical(BudgetedClaudeAdapter(args.out).status()))
        return 0
    for name in ("qwen_result", "cohort", "contexts", "selection_out"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required unless --status is set")
    records, source_hash = checked_records(DEFAULT_DATASET)
    selected = qwen_complete_selection(args.qwen_result, args.cohort, records, source_hash, args.selection_out)
    contexts = checked_contexts(args.contexts, records)
    if any(row["id"] not in contexts for row in selected):
        raise ValueError("Frozen Claude selection lacks verified context")
    if not args.allow_paid or not os.environ.get("ANTHROPIC_API_KEY"):
        raise PaidRunPause("No paid request sent: --allow-paid and ANTHROPIC_API_KEY required")
    adapter = BudgetedClaudeAdapter(args.out, allow_paid=True)
    result = run_experiment(selected, contexts, adapter, args.out, dataset_sha256=source_hash,
                            batch_size=1, repeats=3, max_attempts=3, retry_delay=0)
    print(canonical({"complete_ids": len(result["complete_ids"]), "failed_jobs": len(result["failed_jobs"]),
                     "billing": adapter.status(), "result_sha256": digest(result)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PaidRunPause as exc:
        print(f"Claude run paused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
