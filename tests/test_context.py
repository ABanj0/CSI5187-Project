import json
import errno
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from premo_study.context import (GitHubClient, RetrievalError, build_context,
                                 coverage_report, parse_url, retrieve_contexts, write_immutable,
                                 _transport, _row_checksum)


def record(body="target", fragment="#issuecomment-10"):
    return {"id": "abc", "message_url": "https://github.com/o/r/pull/1" + fragment,
            "raw_message": body, "project": "r"}


def comment(number, body, created, updated=None):
    return {"id": number, "body": body, "created_at": created,
            "updated_at": updated or created, "user": {"login": "author"}}


PR = {"id": 1, "body": "description", "title": "title", "created_at": "2020-01-01T00:00:00Z",
      "updated_at": "2020-01-01T00:00:00Z"}
TARGET = comment(10, "target", "2020-01-03T00:00:00Z")


class ContextTests(unittest.TestCase):
    def test_strict_order_and_future_exclusion(self):
        items = [comment(14, "FUTURE", "2020-01-04T00:00:00Z"), TARGET,
                 comment(12, "second", "2020-01-02T12:00:00Z"),
                 comment(11, "first", "2020-01-02T00:00:00Z"),
                 comment(13, "TIE", TARGET["created_at"])]
        output = build_context(record(), PR, items)
        self.assertTrue(output["eligible"])
        self.assertEqual([c["body"] for c in output["context"]["preceding_comments"]], ["first", "second"])
        self.assertNotIn("FUTURE", json.dumps(output["context"]))
        self.assertNotIn("TIE", json.dumps(output["context"]))

    def test_edited_text_omitted_and_incomplete_flagged(self):
        edited = comment(2, "FUTURE EDIT", "2020-01-02T00:00:00Z", "2020-01-05T00:00:00Z")
        pr = {**PR, "updated_at": "2020-01-05T00:00:00Z", "title": "FUTURE TITLE"}
        output = build_context(record(), pr, [TARGET, edited])
        self.assertFalse(output["eligible"])
        self.assertEqual(output["status"], "incomplete_historical_context")
        self.assertNotIn("FUTURE", json.dumps(output["context"]))

    def test_target_uses_original_text_after_edit(self):
        target = {**TARGET, "body": "modified"}
        output = build_context(record(), PR, [target])
        self.assertEqual(output["context"]["target"]["body"], "target")
        self.assertIn("target_body_changed_use_original_dataset_text", output["warnings"])

    def test_empty_prior_review_is_not_a_historical_gap(self):
        review = {"id": 20, "body": None, "submitted_at": "2020-01-02T00:00:00Z"}
        output = build_context(record(), PR, [TARGET], reviews=[review])
        self.assertTrue(output["eligible"])
        self.assertEqual(output["excluded_counts"]["empty_body"], 1)
        review["body"] = "This review contains unverifiable text"
        self.assertEqual(build_context(record(), PR, [TARGET], reviews=[review])["status"], "incomplete_historical_context")

    def test_edit_timestamp_ties_are_unverifiable(self):
        edited = comment(2, "SAME SECOND EDIT", "2020-01-02T00:00:00Z", TARGET["created_at"])
        pr = {**PR, "updated_at": TARGET["created_at"], "title": "SAME SECOND TITLE"}
        output = build_context(record(), pr, [TARGET, edited])
        self.assertFalse(output["eligible"])
        self.assertEqual(output["status"], "incomplete_historical_context")
        self.assertNotIn("SAME SECOND", json.dumps(output["context"]))

    def test_pr_body_title_at_target_creation_is_not_verified(self):
        output = build_context(record("description", ""), PR, [])
        self.assertFalse(output["eligible"])
        self.assertIsNone(output["context"]["pr_title"])
        self.assertIsNone(output["context"]["pr_description"])
        self.assertEqual(output["status"], "no_preceding_context")

    def test_bare_pr_target_requires_exact_identity(self):
        output = build_context(record("description", ""), PR, [TARGET])
        self.assertEqual(output["target_kind"], "pr_body")
        self.assertIsNone(output["context"]["pr_description"])
        self.assertEqual(output["context"]["preceding_comments"], [])
        output = build_context(record("description", ""), PR, [comment(4, "description", "2020-01-02T00:00:00Z")])
        self.assertEqual(output["status"], "ambiguous_target")
        self.assertFalse(output["eligible"])

    def test_missing_deleted_target_not_invented(self):
        self.assertEqual(build_context(record(), PR, [])["status"], "target_not_found")
        self.assertEqual(build_context(record("unknown", ""), PR, [])["status"], "target_not_found")

    def test_invalid_target_time_and_timezone(self):
        for date in (None, "invalid", "2020-01-03T00:00:00"):
            self.assertFalse(build_context(record(), PR, [{**TARGET, "created_at": date}])["eligible"])

    def test_url_restrictions(self):
        self.assertEqual(parse_url(record()["message_url"]), ("o", "r", 1, "issuecomment-10"))
        for url in ("https://evil.test/o/r/pull/1", "http://github.com/o/r/pull/1", "https://github.com/o/r/pull/1#bad", "https://github.com/o/../pull/1"):
            with self.assertRaises(ValueError):
                parse_url(url)

    def test_report_denominator_includes_unattempted(self):
        records = [record(), {**record(), "id": "def"}]
        report = coverage_report(records, [{"id": "abc", "status": "retrieval_failed", "eligible": False}])
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["statuses"], {"retrieval_failed": 1, "not_attempted": 1})


class AcquisitionTests(unittest.TestCase):
    @staticmethod
    def complete_transport(url, headers, timeout):
        if url.endswith("/pulls/1"):
            return 200, {}, PR
        if "/issues/1/comments" in url:
            return 200, {}, [TARGET]
        return 200, {}, []

    def test_pagination_and_immutable_cache(self):
        calls = []
        first = "https://api.github.com/repos/o/r/issues/1/comments?per_page=100"
        second = "https://api.github.com/repos/o/r/issues/1/comments?page=2"
        def transport(url, headers, timeout):
            calls.append(url)
            return (200, {"Link": f'<{second}>; rel="next"'}, [{"id": 1}]) if url == first else (200, {}, [{"id": 2}])
        with tempfile.TemporaryDirectory() as directory:
            client = GitHubClient(directory, transport=transport, token="TEST_SECRET")
            self.assertEqual(client.paginate(first), [{"id": 1}, {"id": 2}])
            self.assertEqual(client.paginate(first), [{"id": 1}, {"id": 2}])
            self.assertEqual(len(calls), 2)
            self.assertNotIn("TEST_SECRET", "".join(p.read_text() for p in Path(directory).rglob("*.json")))

    def test_untrusted_pagination_never_sends_token(self):
        def transport(url, headers, timeout):
            return 200, {"link": '<https://evil.test/repos/o/r/a>; rel="next"'}, []
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RetrievalError):
                GitHubClient(directory, transport=transport).paginate("https://api.github.com/repos/o/r/a")

    def test_github_numeric_repository_pagination(self):
        first = "https://api.github.com/repos/o/r/pulls/1/comments?per_page=100"
        second = "https://api.github.com/repositories/123/pulls/1/comments?per_page=100&page=2"
        calls = []
        def transport(url, headers, timeout):
            calls.append(url)
            return (200, {"Link": f'<{second}>; rel="next"'}, [{"id": 1}]) if url == first else (200, {}, [{"id": 2}])
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(GitHubClient(directory, transport=transport).paginate(first), [{"id": 1}, {"id": 2}])
        self.assertEqual(calls, [first, second])

    def test_rate_limit_stops_for_long_wait(self):
        def transport(url, headers, timeout):
            return 429, {"retry-after": "3600"}, None
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RetrievalError) as raised:
                GitHubClient(directory, transport=transport, sleeper=lambda _: self.fail("must not sleep")).get("https://api.github.com/repos/o/r/a")
            self.assertEqual(raised.exception.reason, "rate_limited")
            self.assertEqual(raised.exception.retry_after, 3600)

    def test_bounded_retry_then_success(self):
        calls, waits = [], []
        def transport(url, headers, timeout):
            calls.append(url)
            return (503, {}, None) if len(calls) == 1 else (200, {}, {})
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(GitHubClient(directory, transport=transport, sleeper=waits.append).get("https://api.github.com/repos/o/r/a")[0], {})
            self.assertEqual(waits, [1])

    def test_transient_network_retry_and_permanent_network_stop(self):
        for retryable in (True, False):
            calls, waits = [], []
            def transport(url, headers, timeout):
                calls.append(url)
                if len(calls) == 1:
                    raise RetrievalError("timeout" if retryable else "permission_denied", retryable=retryable)
                return 200, {}, {}
            with self.subTest(retryable=retryable), tempfile.TemporaryDirectory() as directory:
                client = GitHubClient(directory, transport=transport, sleeper=waits.append)
                if retryable:
                    self.assertEqual(client.get("https://api.github.com/repos/o/r/a")[0], {})
                    self.assertEqual(waits, [1])
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(len(client.references), 2)
                else:
                    with self.assertRaises(RetrievalError):
                        client.get("https://api.github.com/repos/o/r/a")
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(waits, [])

    def test_transport_classifies_timeout_and_permission_errors(self):
        for cause, retryable in ((TimeoutError(), True), (PermissionError(errno.EACCES, "denied"), False),
                                 (OSError(errno.ECONNRESET, "reset"), True)):
            with self.subTest(cause=cause), patch("premo_study.context.urlopen", side_effect=URLError(cause)):
                with self.assertRaises(RetrievalError) as raised:
                    _transport("https://api.github.com/repos/o/r/a", {}, 1)
                self.assertIs(raised.exception.retryable, retryable)

    def test_snapshot_header_tampering_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            client = GitHubClient(directory, transport=lambda *args: (200, {"Link": '<https://api.github.com/repos/o/r/a?page=2>; rel="next"'}, []))
            url = "https://api.github.com/repos/o/r/a"
            client.get(url)
            path = next(Path(directory).rglob("*.json"))
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj["headers"] = {}
            path.write_text(json.dumps(obj), encoding="utf-8")
            with self.assertRaisesRegex(RetrievalError, "envelope_checksum"):
                client.get(url)

    def test_cached_derived_artifact_and_source_integrity(self):
        for mode in ("derived", "source", "rules"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                client = GitHubClient(root / "raw", transport=self.complete_transport)
                retrieve_contexts([record()], client, root / "derived")
                path = root / "derived/records/abc.json"
                if mode == "source":
                    row = json.loads(path.read_text(encoding="utf-8"))
                    raw_path = root / "derived" / row["snapshot_references"][0]
                    obj = json.loads(raw_path.read_text(encoding="utf-8"))
                    obj["headers"]["link"] = "corrupted"
                    raw_path.write_text(json.dumps(obj), encoding="utf-8")
                else:
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    if mode == "derived":
                        obj["context"]["pr_title"] = "tampered title"
                    else:
                        obj["rules_version"] = "different-version"
                        obj["artifact_sha256"] = _row_checksum(obj)
                    path.write_text(json.dumps(obj), encoding="utf-8")
                with self.assertRaises((ValueError, RetrievalError)):
                    retrieve_contexts([record()], client, root / "derived")

    def test_portable_relative_snapshot_references_resume_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = GitHubClient(root / "original/raw", transport=self.complete_transport)
            first = retrieve_contexts([record()], client, root / "original/derived")
            self.assertTrue(all(not Path(p).is_absolute() for p in first["records"][0]["snapshot_references"]))
            shutil.copytree(root / "original", root / "moved")
            client = GitHubClient(root / "moved/raw", transport=lambda *args: self.fail("Resume must not call network"))
            second = retrieve_contexts([record()], client, root / "moved/derived")
            self.assertEqual(first, second)

    def test_no_overwrite_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file.json"
            write_immutable(path, {"a": 1})
            write_immutable(path, {"a": 1})
            with self.assertRaises(ValueError):
                write_immutable(path, {"a": 2})
            client = GitHubClient(Path(directory) / "cache", transport=lambda *a: (200, {}, [1]))
            url = "https://api.github.com/repos/o/r/a"
            client.get(url)
            cache = next((Path(directory) / "cache").rglob("*.json"))
            data = json.loads(cache.read_text())
            data["data"] = [2]
            cache.write_text(json.dumps(data))
            with self.assertRaises(RetrievalError):
                client.get(url)

    def test_failure_recovery_and_no_silent_row_loss(self):
        def transport(url, headers, timeout):
            raise RetrievalError("network_denied")
        with tempfile.TemporaryDirectory() as directory:
            records = [record(), {**record(), "id": "def"}]
            client = GitHubClient(Path(directory) / "raw", transport=transport)
            report = retrieve_contexts(records, client, Path(directory) / "derived", max_consecutive_failures=1)
            self.assertEqual(report["coverage"]["statuses"], {"retrieval_failed": 1, "not_attempted_after_failures": 1})
            self.assertEqual(len(report["records"]), 2)
            self.assertTrue(list((Path(directory) / "raw").rglob("*.json")))


if __name__ == "__main__":
    unittest.main()
