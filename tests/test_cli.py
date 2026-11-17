"""CLI safety/provenance checks use temporary synthetic data and mocked services."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from premo_study import cli
from premo_study.context import GitHubClient, RetrievalError, _row_checksum, canonical, retrieve_contexts
from premo_study.data import load_dataset
from premo_study.inference import OllamaAdapter, digest, run_experiment
from premo_study.pilot import select_pilot
from test_integration import FixtureGitHub, FixtureModelTransport, fixture_sources


class CliTests(unittest.TestCase):
    def test_progress_reads_artifacts_without_dataset_or_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("requests", "cache", "attempts"):
                (root / folder).mkdir()
            (root / "requests" / "one.json").write_text("{}", encoding="utf-8")
            (root / "cache" / "one.json").write_text("{}", encoding="utf-8")
            (root / "attempts" / "one.json").write_text('{"status":"valid"}', encoding="utf-8")
            (root / "attempts" / "two.json").write_text('{"status":"failed"}', encoding="utf-8")
            (root / "attempts" / "three.json").write_text("{", encoding="utf-8")
            (root / ".inference.lock").write_text("{}", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch.object(cli, "checked_records") as dataset:
                self.assertEqual(cli.main(["progress", "--out", directory]), 0)
            dataset.assert_not_called()
            self.assertEqual(json.loads(output.getvalue()), {
                "output_dir": directory, "active_lock": True, "saved_requests": 1,
                "cached_jobs": 1, "attempts": {"valid": 1, "failed": 1, "unreadable": 1},
                "result_files": 0})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "synthetic-source.json"
        self.sources = fixture_sources()
        self.dataset.write_text(json.dumps(self.sources), encoding="utf-8")
        self.source_hash = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        self.records = load_dataset(self.dataset)

    def acquire(self, suffix="acquisition", transport=None):
        directory = self.root / suffix
        github = GitHubClient(directory / "snapshots", token="", transport=transport or FixtureGitHub(self.sources))
        payload = retrieve_contexts(self.records, github, directory, max_consecutive_failures=1)
        path = directory / ("acquisition-" + hashlib.sha256(canonical(payload)).hexdigest() + ".json")
        self.assertTrue(path.exists())
        return path, payload

    def selection(self):
        payload = select_pilot(self.records, size=2, sensitive_size=0, eligible_ids=[row["id"] for row in self.records])
        payload["dataset_sha256"] = self.source_hash
        path = self.root / "fixture-selection.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path, payload

    def invoke(self, arguments, *, allow_fixture=False):
        output, errors = io.StringIO(), io.StringIO()
        original_checked_records = cli.checked_records
        fixture_reader = lambda path: original_checked_records(path, expected_sha256=self.source_hash)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            if allow_fixture:
                with patch.object(cli, "checked_records", side_effect=fixture_reader):
                    status = cli.main(["--dataset", str(self.dataset), *arguments])
            else:
                status = cli.main(["--dataset", str(self.dataset), *arguments])
        return status, output.getvalue(), errors.getvalue()

    def rewritten_manifest(self, path, payload):
        destination = path.parent / ("acquisition-" + hashlib.sha256(canonical(payload)).hexdigest() + ".json")
        destination.write_bytes(canonical(payload) + b"\n")
        return destination

    def test_source_fingerprint_refuses_synthetic_data_before_any_service(self):
        with patch("premo_study.inference.http_json") as model, patch.object(cli, "GitHubClient") as github:
            status, _, error = self.invoke(["infer", "--backend", "ollama", "--contexts", str(self.root / "absent.json"), "--selection", str(self.root / "absent-selection.json"), "--out", str(self.root / "run")])
        self.assertEqual(status, 1)
        self.assertIn("fingerprint mismatch", error)
        model.assert_not_called()
        github.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_retrieve_selection_bounds_api_calls_and_coverage(self):
        selected = select_pilot(self.records, size=1, sensitive_size=0)
        selected["dataset_sha256"] = self.source_hash
        selection_path = self.root / "one-selection.json"
        selection_path.write_text(json.dumps(selected), encoding="utf-8")
        transport = FixtureGitHub(self.sources)
        output_dir = self.root / "selected-acquisition"
        with patch.object(cli, "GitHubClient", side_effect=lambda path: GitHubClient(path, token="", transport=transport)):
            status, _, error = self.invoke(["retrieve", "--selection", str(selection_path), "--out", str(output_dir)], allow_fixture=True)
        self.assertEqual(status, 0, error)
        manifest = json.loads(next(output_dir.glob("acquisition-*.json")).read_text(encoding="utf-8"))
        self.assertEqual(manifest["coverage"]["total"], 1)
        self.assertEqual([r["id"] for r in manifest["records"]], selected["ids"])
        snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in (output_dir / "snapshots").rglob("*.json")]
        self.assertEqual(len(snapshots), 5)  # fixture PR has a second comment page
        self.assertTrue(all("/repos/" in row["url"] for row in snapshots))

    def test_selection_hash_source_and_unique_known_ids_are_checked(self):
        path, selection = self.selection()
        self.assertEqual([r["id"] for r in cli.checked_selection(path, self.records, self.source_hash)], selection["ids"])
        mutations = [
            lambda p: p.update(dataset_sha256="wrong-source"),
            lambda p: p.update(selection_sha256="wrong-checksum"),
            lambda p: p.update(ids=[selection["ids"][0]] * 2),
            lambda p: p.update(ids=["unknown-id"], selection_sha256=hashlib.sha256(b'["unknown-id"]').hexdigest()),
        ]
        for mutate in mutations:
            payload = copy.deepcopy(selection)
            mutate(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                cli.checked_selection(path, self.records, self.source_hash)

    def test_real_public_acquisition_provenance_is_accepted(self):
        path, payload = self.acquire()
        contexts = cli.checked_contexts(path, self.records)
        self.assertEqual(set(contexts), {r["id"] for r in self.records})
        self.assertEqual(contexts[self.records[0]["id"]], payload["records"][0]["context"])

    def test_bounded_acquisition_coverage_uses_only_retrieved_records(self):
        directory = self.root / "bounded-acquisition"
        github = GitHubClient(directory / "snapshots", token="", transport=FixtureGitHub(self.sources))
        payload = retrieve_contexts(self.records[:1], github, directory)
        path = directory / ("acquisition-" + hashlib.sha256(canonical(payload)).hexdigest() + ".json")
        contexts = cli.checked_contexts(path, self.records)
        self.assertEqual(set(contexts), {self.records[0]["id"]})
        payload["coverage"]["total"] += 1
        changed = self.rewritten_manifest(path, payload)
        with self.assertRaises(ValueError):
            cli.checked_contexts(changed, self.records)

    def test_acquisition_filename_and_payload_checksums_fail_closed(self):
        path, payload = self.acquire()
        arbitrary = path.parent / "arbitrary.json"
        arbitrary.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            cli.checked_contexts(arbitrary, self.records)
        payload["records"][0]["context"]["pr_title"] = "tampered title"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            cli.checked_contexts(path, self.records)

    def test_derived_row_raw_snapshot_rules_and_coverage_provenance(self):
        cases = ("row_checksum", "frozen_row", "raw_snapshot", "rules", "escaped_snapshot", "coverage")
        for case in cases:
            with self.subTest(case=case):
                path, payload = self.acquire(case)
                row = payload["records"][0]
                if case == "row_checksum":
                    row["context"]["pr_title"] = "changed without checksum"
                elif case == "frozen_row":
                    row["context"]["pr_title"] = "changed with checksum but different frozen row"
                    row["artifact_sha256"] = _row_checksum(row)
                elif case == "raw_snapshot":
                    snapshot = path.parent / row["snapshot_references"][0]
                    stored = json.loads(snapshot.read_text(encoding="utf-8"))
                    stored["retrieved_at"] = "2021-01-01T00:00:00Z"
                    snapshot.write_text(json.dumps(stored), encoding="utf-8")
                elif case == "rules":
                    row["rules_version"] = row["context"]["rules_version"] = "unsafe-rule-version"
                    row["artifact_sha256"] = _row_checksum(row)
                    (path.parent / "records" / (row["id"] + ".json")).write_bytes(canonical(row) + b"\n")
                elif case == "escaped_snapshot":
                    row["snapshot_references"] = ["../escaped.json"]
                    row["artifact_sha256"] = _row_checksum(row)
                    (path.parent / "records" / (row["id"] + ".json")).write_bytes(canonical(row) + b"\n")
                else:
                    payload["coverage"]["eligible"] = 0
                changed = self.rewritten_manifest(path, payload)
                with self.assertRaises((ValueError, RetrievalError)):
                    cli.checked_contexts(changed, self.records)

    def test_no_eligible_cohort_does_not_contact_model(self):
        contexts, _ = self.acquire(transport=lambda *args: (404, {}, None))
        selection, _ = self.selection()
        with patch("premo_study.inference.http_json") as model:
            status, _, error = self.invoke(["infer", "--backend", "ollama", "--contexts", str(contexts), "--selection", str(selection), "--out", str(self.root / "inference")], allow_fixture=True)
        self.assertEqual(status, 0, error)
        model.assert_not_called()
        result = json.loads(next((self.root / "inference" / "results").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(result["predictions"], [])
        self.assertEqual(result["adapter"]["backend"], "not_contacted")
        self.assertEqual(len(result["exclusions"]), 2)

    def test_cli_evaluation_rejects_forged_runner_provenance(self):
        result = {"protocol_version": "premo-inference-v1", "dataset_sha256": self.source_hash, "predictions": []}
        path = self.root / "forged-inference.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        status, _, _ = self.invoke(["evaluate", "--predictions", str(path), "--bootstrap-iterations", "0", "--out", str(self.root / "evaluation.json")], allow_fixture=True)
        self.assertEqual(status, 1)
        self.assertFalse((self.root / "evaluation.json").exists())
        result["protocol_version"] = "unknown-protocol"
        path = self.root / (digest(result) + ".json")
        path.write_text(json.dumps(result), encoding="utf-8")
        status, _, _ = self.invoke(["evaluate", "--predictions", str(path), "--bootstrap-iterations", "0", "--out", str(self.root / "evaluation.json")], allow_fixture=True)
        self.assertEqual(status, 1)

    def test_cli_evaluation_of_actual_mock_runner_output_is_idempotent(self):
        manifest, _ = self.acquire()
        contexts = cli.checked_contexts(manifest, self.records)
        adapter = OllamaAdapter(transport=FixtureModelTransport("ollama"), settings={"num_ctx": 65536})
        run_experiment(self.records, contexts, adapter, self.root / "run", dataset_sha256=self.source_hash, batch_size=2, retry_delay=0)
        predictions = next((self.root / "run" / "results").glob("*.json"))
        output = self.root / "evaluation.json"
        arguments = ["evaluate", "--predictions", str(predictions), "--bootstrap-iterations", "5", "--out", str(output)]
        status, _, error = self.invoke(arguments, allow_fixture=True)
        self.assertEqual(status, 0, error)
        frozen = output.read_bytes()
        status, _, error = self.invoke(arguments, allow_fixture=True)
        self.assertEqual(status, 0, error)
        self.assertEqual(output.read_bytes(), frozen)
        self.assertEqual(json.loads(frozen)["paired_n"], 2)


if __name__ == "__main__":
    unittest.main()
