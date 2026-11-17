import json
import tempfile
import unittest
from pathlib import Path

from premo_study.claude_full import BudgetedClaudeAdapter, PaidRunPause, qwen_complete_selection
from premo_study.inference import InferenceError


class ClaudeFullTests(unittest.TestCase):
    def test_budget_ledger_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            def transport(method, url, payload, headers, timeout):
                self.assertEqual(headers["x-api-key"], "fixture-secret")
                return {"model": "claude-haiku-5-5", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": '{"annotations":[]}'}],
                        "usage": {"input_tokens": 1000, "output_tokens": 20}}
            first = BudgetedClaudeAdapter(directory, allow_paid=True, api_key="fixture-secret",
                                          transport=transport, min_interval_seconds=0)
            first.generate("system", "user", {"type": "object", "properties": {}})
            resumed = BudgetedClaudeAdapter(directory, allow_paid=True, api_key="fixture-secret",
                                            transport=transport, min_interval_seconds=0)
            self.assertEqual(resumed.status()["calls"], 1)
            self.assertAlmostEqual(resumed.status()["measured_usd"], 0.00011)
            self.assertNotIn("fixture-secret", (Path(directory) / "paid_calls" / "000001.json").read_text())
            resumed.generate("system", "user", {"type": "object", "properties": {}})
            self.assertEqual(resumed.status()["calls"], 2)

    def test_cap_blocks_network(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = BudgetedClaudeAdapter(directory, allow_paid=True, api_key="fixture-secret",
                                            max_usd=0.005, transport=lambda *args: self.fail("network called"))
            with self.assertRaises(PaidRunPause):
                adapter.generate("system", "user", {})
            self.assertEqual(adapter.status()["calls"], 0)

    def test_rate_limit_pauses_without_inference_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = BudgetedClaudeAdapter(directory, allow_paid=True, api_key="fixture-secret",
                                            transport=lambda *args: (_ for _ in ()).throw(InferenceError("HTTP 429")))
            with self.assertRaisesRegex(PaidRunPause, "rate limit"):
                adapter.generate("system", "user", {})
            self.assertEqual(adapter.status()["charged_calls"], 1)
            self.assertEqual(adapter.status()["measured_usd"], 0)

    def test_unreconciled_claim_blocks_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = BudgetedClaudeAdapter(directory, allow_paid=True, api_key="fixture-secret",
                                            transport=lambda *args: (_ for _ in ()).throw(InferenceError("HTTP 500")))
            with self.assertRaises(PaidRunPause):
                adapter.generate("system", "user", {})
            self.assertEqual(len(adapter.status()["unreconciled_calls"]), 1)
            with self.assertRaisesRegex(PaidRunPause, "unknown billing"):
                adapter.generate("system", "user", {})

    def test_selection_rejects_modified_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.json"
            source.write_text(json.dumps({"complete_ids": ["a"]}))
            with self.assertRaises(ValueError):
                qwen_complete_selection(source, root / "cohort.json", [], "hash", root / "out.json")


if __name__ == "__main__":
    unittest.main()
