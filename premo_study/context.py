"""Immutable GitHub acquisition and conservative, preceding-only reconstruction.

Snapshots may contain later replies; only build_context's filtered product is
permitted in inference. No annotation fields are copied into that product.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

RULES_VERSION = "preceding-only-v1"
API_VERSION = "2022-11-28"


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_immutable(path, value):
    path = Path(path)
    payload = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite immutable artifact: {path}")


def parse_url(url):
    parsed = urlsplit(url)
    match = re.fullmatch(r"/([\w.-]+)/([\w.-]+)/pull/(\d+)/?", parsed.path)
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not match or parsed.query:
        raise ValueError("Unsupported GitHub pull-request URL")
    owner, repo, number = match.groups()
    if owner in {".", ".."} or repo in {".", ".."}:
        raise ValueError("Invalid repository")
    fragment = parsed.fragment
    if fragment and not re.fullmatch(r"(?:issuecomment-|discussion_r|pullrequestreview-)\d+", fragment):
        raise ValueError("Unsupported comment fragment")
    return owner, repo, int(number), fragment


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Missing timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must have a timezone")
    return result


def same_body(left, right):
    # Line endings only: never fuzzy-match or strip meaningful whitespace.
    return isinstance(left, str) and isinstance(right, str) and left.replace("\r\n", "\n") == right.replace("\r\n", "\n")


class RetrievalError(RuntimeError):
    def __init__(self, reason, *, status=None, retry_after=None, retryable=False):
        super().__init__(reason)
        self.reason, self.status, self.retry_after = reason, status, retry_after
        self.retryable = retryable


def _transient_network_error(error):
    cause = error.reason if isinstance(error, URLError) else error
    if isinstance(cause, PermissionError) or getattr(cause, "errno", None) in (errno.EACCES, errno.EPERM) or getattr(cause, "winerror", None) == 10013:
        return False
    if isinstance(cause, TimeoutError):
        return True
    if isinstance(cause, socket.gaierror):
        return cause.errno == socket.EAI_AGAIN
    return isinstance(cause, OSError) and cause.errno in {
        errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED, errno.ECONNREFUSED,
        errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE}


def _verified_snapshot(path, expected_url=None):
    path = Path(path)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if sha256(canonical(stored)) != path.stem:
            raise RetrievalError("snapshot_envelope_checksum_mismatch")
        if expected_url is not None and stored.get("url") != expected_url:
            raise RetrievalError("snapshot_url_mismatch")
        if "data" in stored and sha256(canonical(stored["data"])) != stored.get("data_sha256"):
            raise RetrievalError("snapshot_checksum_mismatch")
    except (OSError, ValueError, TypeError, KeyError):
        raise RetrievalError("snapshot_unreadable_or_corrupt") from None
    return stored


def _row_checksum(row):
    return sha256(canonical({k: v for k, v in row.items() if k != "artifact_sha256"}))


def _transport(url, headers, timeout):
    try:
        with urlopen(Request(url, headers=headers), timeout=timeout) as response:
            return response.status, dict(response.headers), json.loads(response.read())
    except HTTPError as error:
        return error.code, dict(error.headers), None
    except (URLError, OSError) as error:
        raise RetrievalError(f"network_error:{type(error).__name__}", retryable=_transient_network_error(error)) from error
    except ValueError as error:
        raise RetrievalError(f"decode_error:{type(error).__name__}", retryable=True) from error


class GitHubClient:
    def __init__(self, snapshot_dir, *, token=None, transport=None, max_attempts=3,
                 max_wait_seconds=30, timeout=20, sleeper=time.sleep):
        self.directory = Path(snapshot_dir)
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN"))
        self.transport = transport or _transport
        self.max_attempts = max_attempts
        self.max_wait_seconds, self.timeout, self.sleeper = max_wait_seconds, timeout, sleeper
        self.references = []

    def get(self, url):
        parsed = urlsplit(url)
        safe_path = (parsed.path.startswith("/repos/") or
                     re.match(r"^/repositories/[0-9]+/", parsed.path) is not None)
        if parsed.scheme != "https" or parsed.netloc != "api.github.com" or not safe_path or parsed.username:
            raise RetrievalError("untrusted_api_url")
        request_key = sha256(url.encode())
        cache = self.directory / "requests" / request_key
        # A frozen acquisition directory is a deliberate, reusable dataset version.
        if cache.exists():
            for path in sorted(cache.glob("*.json")):
                stored = _verified_snapshot(path, url)
                if stored.get("status") == 200:
                    self.references.append(str(path))
                    return stored["data"], stored["headers"]
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "CSI5187-PRemo-study/0.1",
                   "X-GitHub-Api-Version": API_VERSION}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        for attempt in range(1, self.max_attempts + 1):
            now = datetime.now(timezone.utc).isoformat()
            try:
                status, response_headers, data = self.transport(url, headers, self.timeout)
            except RetrievalError as error:
                stored = {"url": url, "retrieved_at": now, "api_version": API_VERSION,
                          "status": None, "error": error.reason, "attempt": attempt,
                          "retryable": error.retryable}
                path = cache / (sha256(canonical(stored)) + ".json")
                write_immutable(path, stored)
                self.references.append(str(path))
                wait = 2 ** (attempt - 1)
                if not error.retryable or attempt == self.max_attempts or wait > self.max_wait_seconds:
                    raise
                self.sleeper(wait)
                continue
            lowered = {key.lower(): val for key, val in response_headers.items()}
            retained = {k: v for k, v in lowered.items() if k in {
                "link", "etag", "last-modified", "x-github-request-id", "x-ratelimit-limit",
                "x-ratelimit-remaining", "x-ratelimit-reset", "retry-after"}}
            stored = {"url": url, "retrieved_at": now, "api_version": API_VERSION,
                      "status": status, "headers": retained, "data": data,
                      "data_sha256": sha256(canonical(data)), "attempt": attempt}
            path = cache / (sha256(canonical(stored)) + ".json")
            write_immutable(path, stored)
            self.references.append(str(path))
            if status == 200:
                return data, retained
            if status in (404, 410, 451):
                raise RetrievalError("resource_unavailable", status=status)
            rate_limited = status == 429 or (status == 403 and (
                lowered.get("x-ratelimit-remaining") == "0" or "retry-after" in lowered))
            retryable = rate_limited or status >= 500
            wait = 2 ** (attempt - 1)
            if rate_limited:
                try:
                    wait = max(wait, float(lowered.get("retry-after", 0)),
                               float(lowered.get("x-ratelimit-reset", 0)) - time.time() + 1)
                except (ValueError, TypeError):
                    wait = self.max_wait_seconds + 1
            if not retryable or attempt == self.max_attempts or wait > self.max_wait_seconds:
                raise RetrievalError("rate_limited" if rate_limited else "http_error",
                                     status=status, retry_after=wait if retryable else None)
            self.sleeper(wait)
        raise RetrievalError("invalid_attempt_configuration")

    def paginate(self, url):
        items, seen = [], set()
        original = urlsplit(url)
        parts = original.path.split("/", 4)
        canonical_path = (r"/repositories/[0-9]+/" + re.escape(parts[4])
                          if len(parts) == 5 and parts[1] == "repos" else None)
        while url:
            parsed = urlsplit(url)
            same_path = parsed.path == original.path
            github_canonical = (canonical_path is not None and
                                original.scheme == parsed.scheme == "https" and
                                original.netloc == parsed.netloc == "api.github.com" and
                                re.fullmatch(canonical_path, parsed.path) is not None)
            if (parsed.scheme != original.scheme or parsed.netloc != original.netloc or
                    not (same_path or github_canonical) or url in seen):
                raise RetrievalError("invalid_pagination_link")
            seen.add(url)
            data, headers = self.get(url)
            if not isinstance(data, list):
                raise RetrievalError("expected_list_response")
            items.extend(data)
            next_links = re.findall(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
            if len(next_links) > 1:
                raise RetrievalError("multiple_next_links")
            url = next_links[0] if next_links else None
        return items


def _event(raw, kind):
    return {"id": raw["id"], "kind": kind, "body": raw.get("body") or "",
            "created_at": raw.get("submitted_at") if kind == "review" else raw.get("created_at"),
            "updated_at": raw.get("updated_at"),
            "author": (raw.get("user") or {}).get("login"), "html_url": raw.get("html_url")}


def build_context(record, pr, issue_comments, review_comments=(), reviews=(), *, acquisition_complete=True):
    """Resolve target and omit text whose state at target time cannot be verified."""
    _, _, _, fragment = parse_url(record["message_url"])
    events = [_event(item, "issue_comment") for item in issue_comments]
    events += [_event(item, "review_comment") for item in review_comments]
    events += [_event(item, "review") for item in reviews if item.get("submitted_at")]
    base = {"id": record["id"], "message_url": record["message_url"], "rules_version": RULES_VERSION,
            "eligible": False, "warnings": [], "status": "unavailable", "context": None}
    target = None
    if fragment:
        prefix, target_id = re.match(r"(issuecomment-|discussion_r|pullrequestreview-)(\d+)", fragment).groups()
        kind = {"issuecomment-": "issue_comment", "discussion_r": "review_comment", "pullrequestreview-": "review"}[prefix]
        matches = [e for e in events if e["kind"] == kind and e["id"] == int(target_id)]
    else:
        matches = [e for e in events if same_body(e["body"], record["raw_message"])]
        if same_body(pr.get("body"), record["raw_message"]):
            matches.append({"id": pr["id"], "kind": "pr_body", "body": pr.get("body") or "",
                            "created_at": pr.get("created_at"), "updated_at": pr.get("updated_at"),
                            "author": (pr.get("user") or {}).get("login"), "html_url": pr.get("html_url")})
    if len(matches) != 1:
        base["status"] = "ambiguous_target" if len(matches) > 1 else "target_not_found"
        return base
    target = matches[0]
    try:
        cutoff = timestamp(target["created_at"])
    except ValueError:
        base["status"] = "missing_target_timestamp"
        return base
    if not same_body(target["body"], record["raw_message"]):
        base["warnings"].append("target_body_changed_use_original_dataset_text")
    target = {**target, "body": record["raw_message"]}
    preceding, excluded = [], Counter()
    unique = set()
    for event in events:
        event_key = (event["kind"], event["id"])
        if event_key in unique or event_key == (target["kind"], target["id"]):
            continue
        unique.add(event_key)
        # A bodyless approval/review adds no textual context, and GitHub does not
        # expose review updated_at. Its absence must not create a historical gap.
        if not event["body"]:
            excluded["empty_body"] += 1
            continue
        try:
            created = timestamp(event["created_at"])
            # Strict less-than excludes timestamp ties whose true ordering is unknown.
            if created >= cutoff:
                excluded["not_strictly_preceding"] += 1
                continue
            # Reviews have no updated_at; cannot verify historical review-body text.
            if not event.get("updated_at") or timestamp(event["updated_at"]) >= cutoff:
                excluded["edited_or_unverifiable"] += 1
                continue
        except ValueError:
            excluded["invalid_timestamp"] += 1
            continue
        if event["body"]:
            preceding.append(event)
    preceding.sort(key=lambda e: (timestamp(e["created_at"]), e["kind"], e["id"]))
    title, description = None, None
    try:
        pr_verified = timestamp(pr["created_at"]) < cutoff and timestamp(pr["updated_at"]) < cutoff
    except (KeyError, ValueError):
        pr_verified = False
    if pr_verified:
        title = pr.get("title")
        if target["kind"] != "pr_body":
            description = pr.get("body")
    else:
        base["warnings"].append("pr_title_description_historical_state_unverifiable_omitted")
    if target["kind"] == "pr_body":
        # PR-body targets cannot be given subsequent discussion as context.
        preceding = []
    base["context"] = {"pr_title": title, "pr_description": description,
                       "preceding_comments": preceding, "target": target, "rules_version": RULES_VERSION}
    base["excluded_counts"] = dict(excluded)
    base["target_kind"] = target["kind"]
    historical_gaps = excluded["edited_or_unverifiable"] + excluded["invalid_timestamp"]
    if not acquisition_complete:
        base["status"] = "incomplete_acquisition"
    elif historical_gaps:
        base["status"] = "incomplete_historical_context"
    elif not (title or description or preceding):
        base["status"] = "no_preceding_context"
    else:
        base["status"], base["eligible"] = "available", True
    return base


def retrieve_contexts(records, client, output_dir, *, max_consecutive_failures=3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], 0
    for record in records:
        destination = output_dir / "records" / (record["id"] + ".json")
        if destination.exists():
            row = json.loads(destination.read_text(encoding="utf-8"))
            if row.get("artifact_sha256") != _row_checksum(row):
                raise ValueError("Cached context artifact checksum mismatch")
            if row.get("id") != record["id"] or row["message_url"] != record["message_url"] or row.get("target_text_sha256") != sha256(record["raw_message"].encode()):
                raise ValueError("Cached context provenance mismatch; use a new acquisition directory")
            if row.get("rules_version") != RULES_VERSION or (row.get("context") is not None and row["context"].get("rules_version") != RULES_VERSION):
                raise ValueError("Cached context rules version mismatch")
            references = row.get("snapshot_references")
            if not isinstance(references, list) or not references:
                raise ValueError("Cached context lacks source snapshots")
            for reference in references:
                path = (output_dir / reference).resolve()
                if Path(reference).is_absolute() or not path.is_relative_to(client.directory.resolve()):
                    raise ValueError("Cached snapshot reference escapes acquisition directory")
                _verified_snapshot(path)
            rows.append(row)
            continue
        if failures >= max_consecutive_failures:
            rows.append({"id": record["id"], "message_url": record["message_url"], "status": "not_attempted_after_failures", "eligible": False})
            continue
        owner, repo, number, _ = parse_url(record["message_url"])
        root = f"https://api.github.com/repos/{owner}/{repo}"
        client.references = []
        try:
            pr, _ = client.get(f"{root}/pulls/{number}")
            issue = client.paginate(f"{root}/issues/{number}/comments?per_page=100")
            review = client.paginate(f"{root}/pulls/{number}/comments?per_page=100")
            reviews = client.paginate(f"{root}/pulls/{number}/reviews?per_page=100")
            row = build_context(record, pr, issue, review, reviews)
            failures = 0
        except (RetrievalError, KeyError, TypeError, ValueError) as error:
            failures += 1
            row = {"id": record["id"], "message_url": record["message_url"], "status": "retrieval_failed",
                   "eligible": False, "error": getattr(error, "reason", type(error).__name__),
                   "http_status": getattr(error, "status", None), "retry_after": getattr(error, "retry_after", None)}
        row.update({"rules_version": RULES_VERSION, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "target_text_sha256": sha256(record["raw_message"].encode()),
                    "snapshot_references": [Path(os.path.relpath(path, output_dir)).as_posix() for path in client.references]})
        row["artifact_sha256"] = _row_checksum(row)
        # Freeze successful records; failure attempts remain recorded and can be retried.
        if row["status"] != "retrieval_failed":
            write_immutable(destination, row)
        else:
            write_immutable(output_dir / "failures" / (sha256(canonical(row)) + ".json"), row)
        rows.append(row)
    report = coverage_report(records, rows)
    payload = {"records": rows, "coverage": report}
    write_immutable(output_dir / ("acquisition-" + sha256(canonical(payload)) + ".json"), payload)
    return payload


def coverage_report(records, contexts):
    index = {r["id"]: r for r in contexts}
    if len(index) != len(contexts):
        raise ValueError("Duplicate context IDs")
    by_project = {}
    statuses = Counter()
    eligible = []
    for record in records:
        context = index.get(record["id"], {"status": "not_attempted", "eligible": False})
        statuses[context["status"]] += 1
        project = by_project.setdefault(record["project"], {"total": 0, "eligible": 0})
        project["total"] += 1
        if context.get("eligible") is True:
            eligible.append(record["id"])
            project["eligible"] += 1
    return {"rules_version": RULES_VERSION, "total": len(records), "eligible": len(eligible),
            "eligible_ids": eligible, "statuses": dict(statuses), "by_project": by_project,
            "interpretation": "Unattempted and failed rows are not evidence of absent discussion. Both conditions must use the identical eligible cohort."}
