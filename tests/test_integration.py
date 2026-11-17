"""Entirely synthetic, temporary end-to-end research pipeline checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from premo_study.context import GitHubClient, retrieve_contexts
from premo_study.data import load_dataset
from premo_study.evaluation import evaluate
from premo_study.inference import GeminiAdapter, OpenAIAdapter, OllamaAdapter, run_experiment


def fixture_sources():
    result = []
    for number, text in ((10, "TARGET_ALPHA: Thanks, but this is disappointing."), (20, "TARGET_BETA: I appreciate the fix, despite the setback.")):
        answers = []
        for position, labels in enumerate((["joy", "sadness"], ["joy"], ["sadness"])):
            annotation = {"polarity": "neutral", "emotions": labels, "positive_intensity": 1, "negative_itensity": 1, "confidence": 4, "comments": "GOLD_CANARY_DO_NOT_SEND"}
            answers.append({"user_type": "neuro" if position == 0 else "comp", "part1": copy.deepcopy(annotation), "part2": copy.deepcopy(annotation)})
        aggregate = {"polarity": "neutral", "avg_confidence": 4, "agreement_type": "all"}
        result.append({"project": "fixture", "raw_message": text, "message_url": f"https://github.com/fixture/repo/pull/1#issuecomment-{number}", "individual_answers": answers, "part1_aggregate": copy.deepcopy(aggregate), "part2_aggregate": copy.deepcopy(aggregate)})
    return result


def fixture_event(number, body, day):
    return {"id": number, "body": body, "created_at": f"2020-01-{day:02}T00:00:00Z", "updated_at": f"2020-01-{day:02}T00:00:00Z", "user": {"login": "fixture-author"}}


class FixtureGitHub:
    def __init__(self, sources):
        self.calls = []
        self.sources = sources

    def __call__(self, url, headers, timeout):
        self.calls.append(url)
        if url.endswith("/pulls/1"):
            return 200, {}, {"id": 1, "title": "Fixture PR title", "body": "Fixture PR description", "created_at": "2020-01-01T00:00:00Z", "updated_at": "2020-01-01T00:00:00Z", "user": {"login": "opener"}}
        if "/issues/1/comments" in url:
            if "page=2" in url:
                return 200, {}, [fixture_event(20, self.sources[1]["raw_message"], 5), fixture_event(99, "FUTURE_SENTINEL_MUST_NOT_ENTER_PROMPT", 6)]
            return 200, {"Link": '<https://api.github.com/repos/fixture/repo/issues/1/comments?per_page=100&page=2>; rel="next"'}, [fixture_event(2, "PRIOR_CONTEXT_FIXTURE", 2), fixture_event(10, self.sources[0]["raw_message"], 3), fixture_event(11, "SIMULTANEOUS_WITH_ALPHA", 3)]
        if url.endswith("/comments?per_page=100") or url.endswith("/reviews?per_page=100"):
            return 200, {}, []
        raise AssertionError(f"Unexpected fixture GitHub endpoint: {url}")


class FixtureModelTransport:
    def __init__(self, backend):
        self.backend = backend
        self.calls = []
        self.generations = []
        self.repeats = {"comment_only": 0, "context": 0}

    def __call__(self, method, url, payload, headers, timeout):
        self.calls.append((method, url))
        if url.endswith("/api/version"):
            return {"version": "fixture-version"}
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.5:9b", "digest": "fixture-model-digest"}]}
        if url.endswith("/api/show"):
            return {"details": {"quantization_level": "fixture-Q4"}, "model_info": {"fixture.context_length": 65536}}
        if self.backend == "ollama":
            user = payload["messages"][1]["content"]
        elif self.backend == "openai":
            user = payload["input"][1]["content"]
        else:
            user = payload["contents"][0]["parts"][0]["text"]
        decoded = json.loads(user)
        condition = decoded["condition"]
        repeat = self.repeats[condition]
        labels = ((["joy"], ["anger"], ["sadness"]) if condition == "comment_only" else (["joy", "anger"], ["joy", "sadness"], ["sadness"]))[repeat]
        self.repeats[condition] += 1
        self.generations.append(copy.deepcopy(payload))
        text = json.dumps({"annotations": [{"id": item["id"], "emotions": labels} for item in decoded["items"]]})
        if self.backend == "ollama":
            return {"done": True, "done_reason": "stop", "message": {"content": text}, "eval_count": 10}
        if self.backend == "openai":
            return {"model": "gpt-6-sol", "status": "completed", "output": [{"type": "message", "status": "completed",
                    "content": [{"type": "output_text", "text": text}]}], "usage": {"input_tokens": 10, "output_tokens": 10}}
        return {"modelVersion": "fixture-gemini-stable-version", "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}], "usageMetadata": {"promptTokenCount": 10}}


class PipelineIntegrationTests(unittest.TestCase):
    def test_missing_target_excludes_both_arms_and_preserves_paired_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = fixture_sources()
            source_path = root / "synthetic-source.json"
            source_path.write_text(json.dumps(sources), encoding="utf-8")
            records = load_dataset(source_path)
            fixture = FixtureGitHub(sources)
            def missing_second_target(url, headers, timeout):
                status, retained, data = fixture(url, headers, timeout)
                if isinstance(data, list):
                    data = [item for item in data if item["id"] != 20]
                return status, retained, data
            acquisition_dir = root / "acquisition"
            acquisition = retrieve_contexts(records, GitHubClient(acquisition_dir / "snapshots", token="", transport=missing_second_target), acquisition_dir)
            self.assertEqual(acquisition["coverage"]["eligible"], 1)
            self.assertEqual(acquisition["coverage"]["statuses"], {"available": 1, "target_not_found": 1})
            contexts = {row["id"]: row["context"] for row in acquisition["records"] if row["eligible"]}
            transport = FixtureModelTransport("ollama")
            adapter = OllamaAdapter(transport=transport, settings={"num_ctx": 65536})
            result = run_experiment(records, contexts, adapter, root / "inference", batch_size=2, retry_delay=0)
            self.assertEqual(result["complete_ids"], [records[0]["id"]])
            self.assertEqual(len(result["predictions"]), 2)
            self.assertEqual(result["exclusions"], [{"id": records[1]["id"], "reason": "context_unavailable"}])
            self.assertTrue(all(len(json.loads(payload["messages"][1]["content"])["items"]) == 1 for payload in transport.generations))
            scored = evaluate(records, result["predictions"], bootstrap_iterations=0)
            self.assertEqual(scored["paired_n"], 1)
            self.assertEqual(scored["coverage"]["context"]["missing_n"], 1)
            self.assertEqual(scored["coverage"]["comment_only"]["missing_n"], 1)
            self.assertEqual(scored["model_individual_agreement"]["part2"]["context"]["all_individual_annotations"]["joy"]["pairs"], 3)
            self.assertEqual(scored["human_agreement"]["part2"]["all_within_comment_pairs"]["joy"]["pairs"], 3)
            self.assertEqual(scored["human_agreement_full_dataset"]["part2"]["all_within_comment_pairs"]["joy"]["pairs"], 6)

    def test_source_to_paired_evaluation_and_immutable_resume_for_both_adapters(self):
        for backend in ("ollama", "gemini", "openai"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sources = fixture_sources()
                source_path = root / "synthetic-source.json"
                source_path.write_text(json.dumps(sources), encoding="utf-8")
                source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
                records = load_dataset(source_path)
                github = FixtureGitHub(sources)
                acquisition_dir = root / "acquisition"
                client = GitHubClient(acquisition_dir / "snapshots", token="", transport=github)
                acquisition = retrieve_contexts(records, client, acquisition_dir)
                self.assertEqual(acquisition["coverage"]["eligible"], 2)
                self.assertEqual(len(github.calls), 5, "Repeated PR requests must use verified raw cache")
                contexts = {row["id"]: row["context"] for row in acquisition["records"] if row["eligible"]}
                alpha = contexts[records[0]["id"]]
                self.assertEqual([item["body"] for item in alpha["preceding_comments"]], ["PRIOR_CONTEXT_FIXTURE"])
                self.assertNotIn("FUTURE_SENTINEL", json.dumps(contexts))
                self.assertTrue(any("FUTURE_SENTINEL" in path.read_text(encoding="utf-8") for path in (acquisition_dir / "snapshots").rglob("*.json")), "Raw snapshot must preserve actual retrieved later messages")
                for row in acquisition["records"]:
                    self.assertTrue(row["snapshot_references"])
                    self.assertTrue(all(not Path(ref).is_absolute() for ref in row["snapshot_references"]))
                model_transport = FixtureModelTransport(backend)
                adapter = (OllamaAdapter(transport=model_transport, settings={"num_ctx": 65536}) if backend == "ollama"
                           else GeminiAdapter("gemini-2.5-flash", allow_paid=True, api_key="fixture-secret-never-real", transport=model_transport) if backend == "gemini"
                           else OpenAIAdapter(allow_paid=True, api_key="fixture-secret-never-real", transport=model_transport))
                output = root / "inference"
                result = run_experiment(records, contexts, adapter, output, dataset_sha256=source_hash, batch_size=2, retry_delay=0)
                self.assertEqual(len(model_transport.generations), 6)
                self.assertEqual(result["complete_ids"], [record["id"] for record in records])
                self.assertEqual(len(result["predictions"]), 4)
                self.assertEqual(result["repeats"], 3)
                for prediction in result["predictions"]:
                    self.assertEqual(prediction["emotions"], [] if prediction["condition"] == "comment_only" else ["joy", "sadness"])
                prompts = json.dumps(model_transport.generations)
                self.assertNotIn("GOLD_CANARY", prompts)
                self.assertNotIn("FUTURE_SENTINEL", prompts)
                self.assertNotIn("source_record", prompts)
                self.assertNotIn("consensus", prompts)
                scored = evaluate(records, result["predictions"], bootstrap_iterations=10, seed=5)
                self.assertEqual(scored["paired_n"], 2)
                self.assertEqual(scored["comparisons"]["part2"]["comment_only"]["micro_f1"], 0)
                self.assertEqual(scored["comparisons"]["part2"]["context"]["micro_f1"], 1)
                self.assertEqual(scored["uncertainty"]["part2"]["clusters"], 1)
                self.assertEqual(scored["uncertainty"]["part2"]["intervals"]["micro_f1"], [1, 1])
                self.assertTrue(all(row["pairwise_exact_agreement"] == 0 for row in result["consistency"]))
                frozen = {path: path.read_bytes() for path in root.rglob("*.json")}
                def no_contact(*args, **kwargs):
                    raise AssertionError("Resuming frozen fixture artifacts must not generate or retrieve")
                resumed_client = GitHubClient(acquisition_dir / "snapshots", token="", transport=no_contact)
                self.assertEqual(retrieve_contexts(records, resumed_client, acquisition_dir), acquisition)
                adapter.transport = no_contact
                self.assertEqual(run_experiment(records, contexts, adapter, output, dataset_sha256=source_hash, batch_size=2, retry_delay=0), result)
                self.assertTrue(all(path.read_bytes() == value for path, value in frozen.items()))
                self.assertEqual({path for path in root.rglob("*.json")}, set(frozen))
                artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*.json"))
                self.assertNotIn("fixture-secret-never-real", artifact_text)


if __name__ == "__main__":
    unittest.main()
