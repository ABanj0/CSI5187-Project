import copy
import json
from pathlib import Path
import tempfile
import unittest

from premo_study.inference import (
    ClaudeAdapter, GeminiAdapter, OpenAIAdapter, OllamaAdapter, InferenceError, build_prompt, digest,
    majority_vote, response_schema, run_consistency, run_experiment, validate_response,
)


def records():
    return [{"id": "abc", "project": "project", "message_url": "https://github.com/a/b/pull/1#issuecomment-1",
             "raw_message": "Thanks!", "target_kind": "issue_comment", "source_record": {"SECRET_GOLD": "anger"},
             "consensus": {"part1": ["anger"], "part2": ["fear"]}}]


def contexts():
    return {"abc": {"target": {"body": "Thanks!", "created_at": "2020-01-02T00:00:00Z"},
                    "pr_title": "Fix bug", "pr_description": "Description",
                    "preceding_comments": [{"id": 2, "kind": "issue_comment", "body": "Works now",
                                            "created_at": "2020-01-01T00:00:00Z", "author": "a"}],
                    "rules_version": "preceding-only-v1", "SECRET_GOLD": "sadness"}}


class FakeAdapter:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses

    def describe(self):
        return {"backend": "fake", "model": "test", "settings": {"temperature": 0}}

    def generate(self, system, user, schema):
        self.calls.append((system, user, schema))
        if self.responses:
            result = self.responses.pop(0)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, str):
                return {"text": result, "raw": {"content": result}, "metadata": self.describe()}
        labels = ["joy"]
        rows = [{"id": x["id"], "emotions": labels} for x in json.loads(user)["items"]]
        return {"text": json.dumps({"annotations": rows}), "raw": {"token_count": 20}, "metadata": self.describe()}


class ValidationTests(unittest.TestCase):
    def test_strict_response(self):
        valid = '{"annotations":[{"id":"a","emotions":["joy","sadness"]}]}'
        self.assertEqual(validate_response(valid, ["a"])[0]["emotions"], ["joy", "sadness"])
        invalid = ["```json\n" + valid + "\n```", '{"annotations":[]}',
                   '{"annotations":[{"id":"b","emotions":[]}]}',
                   '{"annotations":[{"id":"a","emotions":["neutral"]}]}',
                   '{"annotations":[{"id":"a","emotions":["joy","joy"]}]}',
                   '{"annotations":[{"id":"a","emotions":[]},{"id":"a","emotions":[]}]}',
                   '{"annotations":[{"id":"a","emotions":[],"rationale":"x"}]}',
                   '{"annotations":[],"annotations":[]}',
                   '{"annotations":[{"id":"a","emotions":NaN}]}']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_response(text, ["a"])

    def test_independent_vote(self):
        self.assertEqual(majority_vote([["joy", "anger"], ["joy", "sadness"], ["sadness"]]), ["joy", "sadness"])
        self.assertEqual(majority_vote([["joy"], ["anger"], ["sadness"]]), [])
        with self.assertRaises(ValueError):
            majority_vote([["joy"], []])
        self.assertEqual(run_consistency([[], [], []])["pairwise_jaccard"], 1)
        self.assertFalse(run_consistency([["joy"], ["joy"], []])["all_runs_identical"])

    def test_no_gold_in_prompt(self):
        for condition in ("comment_only", "context"):
            _, user = build_prompt(records(), contexts(), condition)
            self.assertNotIn("SECRET_GOLD", user)
            self.assertNotIn("consensus", user)
            self.assertNotIn("fear", user)
            self.assertEqual(json.loads(user)["items"][0]["target"]["body"], "Thanks!")

    def test_target_author_identified_in_context_without_annotations(self):
        data = contexts()
        data["abc"]["target"]["author"] = "target-speaker"
        _, user = build_prompt(records(), data, "context")
        item = json.loads(user)["items"][0]
        self.assertEqual(item["context"]["target_author"], "target-speaker")
        self.assertEqual(item["target"]["body"], records()[0]["raw_message"])

    def test_future_or_equal_context_rejected(self):
        for timestamp in ("2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z", "2020-01-01T00:00:00"):
            data = contexts()
            data["abc"]["preceding_comments"][0]["created_at"] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                build_prompt(records(), data, "context")

    def test_context_target_mismatch_rejected(self):
        data = contexts()
        data["abc"]["target"]["body"] = "changed"
        with self.assertRaises(ValueError):
            build_prompt(records(), data, "context")


class RunnerTests(unittest.TestCase):
    def test_resume_and_immutable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter()
            first = run_experiment(records(), contexts(), adapter, directory, retry_delay=0)
            self.assertEqual(len(adapter.calls), 6)
            self.assertEqual(first["complete_ids"], ["abc"])
            self.assertEqual(len(first["predictions"]), 2)
            requests = list((Path(directory) / "requests").glob("*.json"))
            self.assertEqual(len(requests), 6)
            request = json.loads(requests[0].read_text(encoding="utf-8"))
            self.assertIn("system", request)
            self.assertIn("Thanks!", request["user"])
            self.assertIn("schema", request)
            snapshots = {p: p.read_bytes() for p in Path(directory).rglob("*.json")}
            second = run_experiment(records(), contexts(), adapter, directory, retry_delay=0)
            self.assertEqual(first, second)
            self.assertEqual(len(adapter.calls), 6)
            self.assertTrue(all(p.read_bytes() == text for p, text in snapshots.items()))

    def test_changed_context_invalidates_only_context_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter()
            run_experiment(records(), contexts(), adapter, directory, retry_delay=0)
            context = contexts()
            context["abc"]["pr_title"] = "Another title"
            run_experiment(records(), context, adapter, directory, retry_delay=0)
            self.assertEqual(len(adapter.calls), 9)

    def test_failure_is_not_empty_labels_and_attempts_are_bounded_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter([InferenceError("blocked")] + ["invalid"] * 20)
            result = run_experiment(records(), contexts(), adapter, directory, max_attempts=1, retry_delay=0)
            self.assertEqual(result["predictions"], [])
            self.assertEqual(len(result["failed_jobs"]), 6)
            self.assertEqual(len(adapter.calls), 6)
            run_experiment(records(), contexts(), adapter, directory, max_attempts=1, retry_delay=0)
            self.assertEqual(len(adapter.calls), 6)
            attempts = [json.loads(p.read_text(encoding="utf-8")) for p in (Path(directory) / "attempts").glob("*.json")]
            self.assertTrue(all(x["status"] == "failed" for x in attempts))
            self.assertTrue(any("response" in x for x in attempts))

    def test_invalid_json_retries_and_preserves_response(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter(["invalid json"])
            result = run_experiment(records(), contexts(), adapter, directory, max_attempts=2, retry_delay=0)
            self.assertEqual(len(result["predictions"]), 2)
            self.assertEqual(len(adapter.calls), 7)
            attempts = [json.loads(p.read_text(encoding="utf-8")) for p in (Path(directory) / "attempts").glob("*.json")]
            self.assertEqual(sum(x["status"] == "failed" for x in attempts), 1)

    def test_tampered_cache_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter()
            run_experiment(records(), contexts(), adapter, directory, retry_delay=0)
            path = next((Path(directory) / "cache").glob("*.json"))
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj["annotations"][0]["emotions"] = ["anger"]
            path.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_experiment(records(), contexts(), adapter, directory, retry_delay=0)

    def test_missing_context_never_creates_unpaired_comment_only(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter()
            result = run_experiment(records(), {}, adapter, directory, retry_delay=0)
            self.assertEqual(result["predictions"], [])
            self.assertEqual(adapter.calls, [])
            self.assertEqual(result["exclusions"][0]["reason"], "context_unavailable")

    def test_invalid_context_excluded_from_both_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            data = contexts()
            data["abc"]["target"]["body"] = "edited"
            result = run_experiment(records(), data, FakeAdapter(), directory, retry_delay=0)
            self.assertEqual(result["eligible_ids"], [])
            self.assertEqual(result["exclusions"][0]["reason"], "invalid_context")

    def test_recover_attempt_after_cache_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = FakeAdapter()
            result = run_experiment(records(), contexts(), adapter, directory, retry_delay=0)
            next((Path(directory) / "cache").glob("*.json")).unlink()
            restored = run_experiment(records(), contexts(), adapter, directory, retry_delay=0)
            self.assertEqual(result, restored)
            self.assertEqual(len(adapter.calls), 6)

    def test_no_eligible_records_do_not_contact_backend(self):
        class Unavailable:
            def describe(self):
                raise AssertionError("Should never contact backend")
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(records(), {}, Unavailable(), directory)
            self.assertEqual(result["adapter"]["backend"], "not_contacted")

    def test_output_lock_rejects_duplicate_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".inference.lock").write_text("{}", encoding="utf-8")
            with self.assertRaises(InferenceError):
                run_experiment(records(), contexts(), FakeAdapter(), directory)

    def test_mixed_gemini_actual_versions_fail_closed(self):
        count = 0
        def transport(method, url, payload, headers, timeout):
            nonlocal count
            count += 1
            return {"modelVersion": "version-a" if count == 1 else "version-b",
                    "candidates": [{"finishReason": "STOP", "content": {"parts": [
                        {"text": '{"annotations":[{"id":"abc","emotions":[]}]}'}]}}]}
        adapter = GeminiAdapter("gemini-2.5-flash", allow_paid=True, api_key="secret", transport=transport)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(InferenceError, "Mixed actual model versions"):
                run_experiment(records(), contexts(), adapter, directory)
            self.assertFalse((Path(directory) / "results").exists())
            self.assertEqual(count, 2)

    def test_mixed_openai_fingerprints_fail_closed(self):
        count = 0
        def transport(*args):
            nonlocal count
            count += 1
            return {"model": "gpt-6-sol", "status": "completed", "system_fingerprint": "fp-a" if count == 1 else "fp-b",
                    "output": [{"type": "message", "status": "completed", "content": [
                        {"type": "output_text", "text": '{"annotations":[{"id":"abc","emotions":[]}]}' }]}]}
        adapter = OpenAIAdapter(allow_paid=True, api_key="fixture-secret", transport=transport)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(InferenceError, "Mixed actual model versions"):
                run_experiment(records(), contexts(), adapter, directory)
            self.assertFalse((Path(directory) / "results").exists())
            self.assertEqual(count, 2)

    def test_later_edits_and_unknown_rules_excluded(self):
        for update in ("2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z", "2020-01-01T00:00:00"):
            data = contexts()
            data["abc"]["preceding_comments"][0]["updated_at"] = update
            with self.subTest(update=update), self.assertRaises(ValueError):
                build_prompt(records(), data, "context")
        data = contexts()
        data["abc"]["rules_version"] = "unsafe"
        with self.assertRaises(ValueError):
            build_prompt(records(), data, "context")

    def test_resolved_target_kind_overrides_candidate_kind(self):
        row = records()
        row[0]["target_kind"] = "pr_body"
        data = contexts()
        data["abc"]["target"]["kind"] = "issue_comment"
        build_prompt(row, data, "context")
        data["abc"]["target"]["kind"] = "pr_body"
        with self.assertRaises(ValueError):
            build_prompt(row, data, "context")


class AdapterTests(unittest.TestCase):
    def test_openai_approval_gate_and_exact_model(self):
        calls = []
        with self.assertRaises(ValueError):
            OpenAIAdapter("gpt-6.1-sol")
        adapter = OpenAIAdapter(api_key="fixture-secret", transport=lambda *args: calls.append(args))
        with self.assertRaises(InferenceError):
            adapter.generate("system", "user", response_schema(["abc"]))
        self.assertEqual(calls, [])

    def test_openai_responses_schema_identity_and_failures(self):
        calls = []
        valid = {"model": "gpt-6-sol", "status": "completed", "system_fingerprint": "fixture-fp",
                 "output": [{"type": "message", "status": "completed", "content": [
                     {"type": "output_text", "text": '{"annotations":[{"id":"abc","emotions":[]}]}' }]}],
                 "usage": {"input_tokens": 10, "output_tokens": 8}}
        def transport(*args):
            calls.append(args)
            return valid
        adapter = OpenAIAdapter(allow_paid=True, api_key="fixture-secret", transport=transport)
        result = adapter.generate("system", "user", response_schema(["abc"]))
        method, url, payload, headers, _ = calls[0]
        self.assertEqual((method, url), ("POST", "https://api.openai.com/v1/responses"))
        self.assertEqual(headers, {"Authorization": "Bearer fixture-secret"})
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertEqual(payload["temperature"], 0)
        self.assertIs(payload["store"], False)
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertNotIn("minItems", json.dumps(payload["text"]["format"]["schema"]))
        self.assertEqual(result["metadata"]["actual_model_version"], "gpt-6-sol:fixture-fp")
        self.assertNotIn("fixture-secret", json.dumps(result))
        for broken in ({**valid, "status": "incomplete"}, {**valid, "output": None},
                       {**valid, "output": [{"type": "message", "status": "completed", "content": [{"type": "refusal", "refusal": "no"}]}]},
                       {**valid, "model": "different-model"}):
            with self.subTest(broken=broken), self.assertRaises(InferenceError):
                OpenAIAdapter(allow_paid=True, api_key="fixture-secret", transport=lambda *args: broken).generate("system", "user", response_schema(["abc"]))

    def test_claude_approval_gate_and_pinned_model(self):
        calls = []
        adapter = ClaudeAdapter(api_key="fixture-secret", transport=lambda *args: calls.append(args))
        with self.assertRaisesRegex(InferenceError, "explicit approval"):
            adapter.generate("guide", "comment", response_schema(["abc"]))
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            ClaudeAdapter("claude-haiku-latest")

    def test_claude_structured_payload_and_actual_version(self):
        calls = []
        def transport(*args):
            calls.append(args)
            return {"model": "claude-haiku-5-5", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": '{"annotations":[{"id":"abc","emotions":[]}]}'}],
                    "usage": {"input_tokens": 100, "output_tokens": 20}}
        adapter = ClaudeAdapter(allow_paid=True, api_key="fixture-secret", transport=transport)
        result = adapter.generate("guide", "comment", response_schema(["abc"]))
        method, url, payload, headers, _ = calls[0]
        self.assertEqual((method, url), ("POST", "https://api.anthropic.com/v1/messages"))
        self.assertEqual(headers["x-api-key"], "fixture-secret")
        self.assertEqual(payload["model"], "claude-haiku-5-5")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["output_config"]["format"]["type"], "json_schema")
        self.assertNotIn("maxItems", json.dumps(payload["output_config"]["format"]["schema"]))
        self.assertEqual(result["metadata"]["actual_model_version"], "claude-haiku-5-5")
        self.assertEqual(validate_response(result["text"], ["abc"]), [{"id": "abc", "emotions": []}])
        self.assertNotIn("fixture-secret", json.dumps(result))

    def test_gemini_approval_gate_before_transport(self):
        calls = []
        adapter = GeminiAdapter("gemini-2.5-flash", api_key="secret", transport=lambda *args: calls.append(args))
        with self.assertRaises(InferenceError):
            adapter.generate("system", "user", response_schema(["abc"]))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(InferenceError):
            run_experiment(records(), contexts(), adapter, directory)
        self.assertEqual(calls, [])

    def test_gemini_rejects_aliases(self):
        for model in ("gemini-flash-latest", "gemini-2.5-flash-preview", "gemini-3-flash-preview", "models/gemini-2.5-flash", "gemini-2.5-pro"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                GeminiAdapter(model)

    def test_gemini_schema_and_actual_version(self):
        calls = []
        def transport(*args):
            calls.append(args)
            return {"modelVersion": "gemini-2.5-flash", "usageMetadata": {"promptTokenCount": 100},
                    "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": '{"annotations":[{"id":"abc","emotions":[]}]}'}]}}]}
        adapter = GeminiAdapter("gemini-2.5-flash", allow_paid=True, api_key="secret", transport=transport)
        result = adapter.generate("system", "user", response_schema(["abc"]))
        self.assertEqual(result["metadata"]["actual_model_version"], "gemini-2.5-flash")
        method, url, payload, headers, _ = calls[0]
        self.assertEqual(method, "POST")
        self.assertNotIn("secret", url)
        self.assertEqual(headers, {"x-goog-api-key": "secret"})
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(payload["generationConfig"]["thinkingConfig"], {"thinkingBudget": 0})
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "system")
        self.assertNotIn("secret", json.dumps(result))

    def test_gemini_3_flash_has_explicit_low_thinking(self):
        calls = []
        def transport(*args):
            calls.append(args)
            return {"modelVersion": "gemini-3.8-flash", "candidates": [{"finishReason": "STOP",
                    "content": {"parts": [{"text": '{"annotations":[{"id":"abc","emotions":[]}]}' }]}}]}
        adapter = GeminiAdapter("gemini-3.8-flash", allow_paid=True, api_key="fixture-secret", transport=transport)
        adapter.generate("system", "user", response_schema(["abc"]))
        self.assertEqual(calls[0][2]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "low"})
        with self.assertRaises(ValueError):
            GeminiAdapter("gemini-3.8-flash", settings={"thinkingConfig": {"thinkingLevel": "high"}})

    def test_gemini_blocked_truncated_and_missing_version_are_failures(self):
        for raw in ({"promptFeedback": {"blockReason": "SAFETY"}},
                    {"candidates": [{"finishReason": "MAX_TOKENS"}]},
                    {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{}"}]}}]}):
            adapter = GeminiAdapter("gemini-2.5-flash", allow_paid=True, api_key="secret", transport=lambda *args: raw)
            with self.subTest(raw=raw), self.assertRaises(InferenceError) as caught:
                adapter.generate("system", "user", {})
            self.assertEqual(caught.exception.raw, raw)

    def test_ollama_readonly_discovery_and_payload(self):
        calls = []
        def transport(method, url, payload, headers, timeout):
            calls.append((method, url, payload))
            if url.endswith("/api/version"):
                return {"version": "0.9.1"}
            if url.endswith("/api/tags"):
                return {"models": [{"name": "qwen3.5:9b", "digest": "sha256:fixed"}]}
            if url.endswith("/api/show"):
                return {"details": {"quantization_level": "Q4_K_M"}, "model_info": {"qwen.context_length": 32768}}
            return {"done": True, "done_reason": "stop", "message": {"content": "{}"}}
        adapter = OllamaAdapter(transport=transport, settings={"num_ctx": 32768})
        result = adapter.generate("system", "user", {})
        self.assertEqual(result["metadata"]["details"]["quantization_level"], "Q4_K_M")
        self.assertEqual(result["metadata"]["digest"], "sha256:fixed")
        self.assertEqual(calls[-1][2]["options"]["temperature"], 0)
        self.assertFalse(calls[-1][2]["stream"])
        self.assertFalse(calls[-1][2]["think"])
        self.assertFalse(any("pull" in url or "create" in url for _, url, _ in calls))

    def test_ollama_missing_model_no_download(self):
        calls = []
        def transport(method, url, *args):
            calls.append(url)
            return {"models": []} if url.endswith("/api/tags") else {"version": "test"}
        with self.assertRaises(InferenceError):
            OllamaAdapter(transport=transport).describe()
        self.assertEqual(len(calls), 2)

    def test_ollama_context_overflow_before_generation(self):
        def transport(method, url, *args):
            if url.endswith("tags"):
                return {"models": [{"name": "qwen3.5:9b", "digest": "fixed"}]}
            return {}
        adapter = OllamaAdapter(transport=transport)
        with self.assertRaises(InferenceError):
            adapter.generate("system", "x" * 10000, {})


if __name__ == "__main__":
    unittest.main()
