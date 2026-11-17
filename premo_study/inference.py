"""Serial, resumable inference. No model download, training or service mutation.

Adapters accept an injectable transport(method, url, payload, headers, timeout) for
offline tests. Transport returns the decoded JSON response. Failures are never labels.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.resources import files
from itertools import combinations
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

EMOTIONS = ("joy", "love", "sadness", "anger", "fear", "positivesurprise", "negativesurprise")
CONDITIONS = ("comment_only", "context")
PROTOCOL_VERSION = "premo-inference-v1"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InferenceError(RuntimeError):
    def __init__(self, message: str, *, raw: Any = None, retryable: bool = False):
        super().__init__(message)
        self.raw = raw
        self.retryable = retryable


def http_json(method, url, payload, headers, timeout):
    body = None if payload is None else canonical(payload).encode("utf-8")
    request = Request(url, data=body, method=method, headers={"Content-Type": "application/json", **headers})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        # Never include request URLs or headers: they may contain credentials.
        raise InferenceError(f"HTTP {exc.code}", raw=raw, retryable=exc.code in (408, 429, 500, 502, 503, 504)) from None
    except (URLError, TimeoutError, OSError):
        raise InferenceError("Network request failed", retryable=True) from None
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        raise InferenceError("Backend returned invalid JSON", raw=raw, retryable=True) from None
    if not isinstance(result, dict):
        raise InferenceError("Backend returned a non-object", raw=result)
    return result


def response_schema(ids):
    return {"type": "object", "properties": {"annotations": {"type": "array", "minItems": len(ids), "maxItems": len(ids),
            "items": {"type": "object", "properties": {"id": {"type": "string", "enum": list(ids)},
            "emotions": {"type": "array", "items": {"type": "string", "enum": list(EMOTIONS)}, "maxItems": 7}},
            "required": ["id", "emotions"], "additionalProperties": False}}},
            "required": ["annotations"], "additionalProperties": False}


def _strict_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON property: {key}")
        obj[key] = value
    return obj


def validate_response(text, expected_ids):
    """Reject malformed JSON, duplicate fields, unknown/missing ids and labels."""
    if not isinstance(text, str):
        raise ValueError("Response must be a JSON string")
    obj = json.loads(text, object_pairs_hook=_strict_object,
                     parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Invalid JSON constant")))
    if not isinstance(obj, dict) or set(obj) != {"annotations"} or not isinstance(obj["annotations"], list):
        raise ValueError("Expected an object containing only annotations")
    expected_ids = list(expected_ids)
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Expected ids must be unique")
    by_id = {}
    for row in obj["annotations"]:
        if not isinstance(row, dict) or set(row) != {"id", "emotions"}:
            raise ValueError("Each annotation must contain exactly id and emotions")
        rid, labels = row["id"], row["emotions"]
        if not isinstance(rid, str) or rid not in expected_ids or rid in by_id:
            raise ValueError("Unknown or duplicate annotation id")
        if not isinstance(labels, list) or any(not isinstance(x, str) or x not in EMOTIONS for x in labels):
            raise ValueError("Unknown emotion label")
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate emotion label")
        by_id[rid] = [x for x in EMOTIONS if x in labels]
    if set(by_id) != set(expected_ids):
        raise ValueError("Missing annotation id")
    return [{"id": rid, "emotions": by_id[rid]} for rid in expected_ids]


def majority_vote(runs):
    """Independent per-emotion 2-of-3 vote; require exactly three valid runs."""
    if len(runs) != 3:
        raise ValueError("Exactly three valid runs are required")
    for labels in runs:
        if not isinstance(labels, (list, tuple, set)) or len(labels) != len(set(labels)) or any(e not in EMOTIONS for e in labels):
            raise ValueError("Invalid emotion set")
    counts = Counter(e for labels in runs for e in labels)
    return [e for e in EMOTIONS if counts[e] >= 2]


def run_consistency(runs):
    majority_vote(runs)  # Validate all three label sets.
    pairs = list(combinations([set(x) for x in runs], 2))
    return {"all_runs_identical": all(a == b for a, b in pairs),
            "pairwise_exact_agreement": sum(a == b for a, b in pairs) / 3,
            "pairwise_jaccard": sum(len(a & b) / len(a | b) if a | b else 1 for a, b in pairs) / 3}


class OllamaAdapter:
    def __init__(self, model="qwen3.5:9b", *, base_url=None, settings=None, transport=None, timeout=300):
        allowed_models = {"qwen3.5:9b", "qwen3.5:4b-q8_0"}
        if not isinstance(model, str) or model.lower() not in allowed_models:
            raise ValueError("Model must be an explicitly approved, already-installed local Qwen model")
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Invalid Ollama URL; embedded credentials are forbidden")
        self.settings = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 2048, **(settings or {})}
        if self.settings["temperature"] != 0:
            raise ValueError("Study inference requires temperature 0")
        if any(type(self.settings[k]) is not int or self.settings[k] <= 0 for k in ("num_ctx", "num_predict")):
            raise ValueError("num_ctx and num_predict must be positive integers")
        if self.settings["num_predict"] >= self.settings["num_ctx"]:
            raise ValueError("Output budget must be less than num_ctx")
        self.transport = transport or http_json
        self.timeout = timeout
        self._description = None

    def _call(self, method, path, payload=None):
        return self.transport(method, self.base_url + path, payload, {}, self.timeout)

    def describe(self):
        if self._description is None:
            version = self._call("GET", "/api/version")
            tags = self._call("GET", "/api/tags")
            matches = [x for x in tags.get("models", []) if (x.get("name", "").lower() == self.model.lower() or x.get("model", "").lower() == self.model.lower())]
            if len(matches) != 1 or not matches[0].get("digest"):
                raise InferenceError("Required model is not uniquely installed; no download attempted")
            show = self._call("POST", "/api/show", {"model": self.model})
            if show.get("remote_model") or show.get("remote_host"):
                raise InferenceError("A remote Ollama model does not satisfy the local-model condition")
            self._description = {"backend": "ollama", "base_url": self.base_url, "backend_version": version.get("version"),
                                 "model": self.model, "digest": matches[0]["digest"], "details": show.get("details", {}),
                                 "model_info": show.get("model_info", {}), "model_parameters": show.get("parameters"),
                                 "capabilities": show.get("capabilities", []), "settings": self.settings,
                                 "thinking": False, "truncation_policy": "reject_conservative_byte_bound"}
        return json.loads(canonical(self._description))

    def generate(self, system, user, schema):
        metadata = self.describe()
        # UTF-8 bytes conservatively upper-bound ordinary text tokens; reserve room
        # for chat template, schema and completion. Never silently trim the target.
        upper_bound = len((system + user + canonical(schema)).encode("utf-8")) + 1024
        if upper_bound + self.settings["num_predict"] > self.settings["num_ctx"]:
            raise InferenceError("Prompt exceeds conservative num_ctx budget; reduce batch/context size explicitly")
        raw = self._call("POST", "/api/chat", {"model": self.model, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema, "stream": False, "think": False, "options": self.settings})
        if raw.get("error") or raw.get("done") is not True or raw.get("done_reason") not in (None, "stop"):
            raise InferenceError("Ollama generation incomplete or failed", raw=raw)
        content = raw.get("message", {}).get("content")
        if not isinstance(content, str):
            raise InferenceError("Ollama response has no text", raw=raw)
        return {"text": content, "raw": raw, "metadata": metadata}


class GeminiAdapter:
    """Official REST adapter. Every network call requires explicit paid approval."""
    def __init__(self, model, *, allow_paid=False, api_key=None, settings=None, transport=None, timeout=120):
        if not isinstance(model, str) or not re.fullmatch(r"gemini-\d+\.\d+-flash(?:-\d{3}|-\d{2}-\d{2})?", model):
            raise ValueError("Provide an explicit stable Gemini Flash identifier; moving aliases/preview models are forbidden")
        self.model, self.allow_paid = model, allow_paid is True
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        thinking = {"thinkingLevel": "low"} if model.startswith("gemini-3.") else {"thinkingBudget": 0}
        self.settings = {"temperature": 0, "maxOutputTokens": 4096, "thinkingConfig": thinking, **(settings or {})}
        if self.settings["temperature"] != 0:
            raise ValueError("Study inference requires temperature 0")
        if self.settings["thinkingConfig"] != thinking:
            raise ValueError("Gemini thinking setting must match the frozen model-family policy")
        forbidden = {"responseMimeType", "responseSchema", "responseJsonSchema", "candidateCount"} & set(self.settings)
        if forbidden:
            raise ValueError("Response format and candidate count are fixed by the study protocol")
        self.transport, self.timeout = transport or http_json, timeout

    def describe(self):
        return {"backend": "gemini", "endpoint": "https://generativelanguage.googleapis.com/v1beta", "model": self.model,
                "settings": self.settings, "candidateCount": 1, "model_version_policy": "record_response_modelVersion",
                "quantization": "provider_managed_not_disclosed", "context_window": "provider_managed_not_verified"}

    def generate(self, system, user, schema):
        if not self.allow_paid:
            raise InferenceError("Gemini paid inference requires explicit approval; no request sent")
        if not self.api_key:
            raise InferenceError("GEMINI_API_KEY is unavailable; no request sent")
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": {**self.settings, "candidateCount": 1,
                       "responseMimeType": "application/json", "responseJsonSchema": schema}}
        raw = self.transport("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                             payload, {"x-goog-api-key": self.api_key}, self.timeout)
        candidates = raw.get("candidates", [])
        if raw.get("error") or len(candidates) != 1 or candidates[0].get("finishReason") != "STOP":
            raise InferenceError("Gemini response blocked, truncated or failed", raw=raw)
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought")]
        if not text_parts or not raw.get("modelVersion"):
            raise InferenceError("Gemini response lacks text or actual modelVersion", raw=raw)
        return {"text": "".join(text_parts), "raw": raw,
                "metadata": {**self.describe(), "actual_model_version": raw["modelVersion"], "usage": raw.get("usageMetadata", {})}}


def _openai_schema(schema):
    """Keep the common labels/shape while omitting unsupported array bounds."""
    if isinstance(schema, dict):
        return {key: _openai_schema(value) for key, value in schema.items() if key not in ("minItems", "maxItems")}
    if isinstance(schema, list):
        return [_openai_schema(value) for value in schema]
    return schema


class OpenAIAdapter:
    """Official Responses API adapter for the frozen GPT-6-Sol study arm."""
    def __init__(self, model="gpt-6-sol", *, allow_paid=False, api_key=None, transport=None, timeout=300):
        if model != "gpt-6-sol":
            raise ValueError("This study pins the OpenAI arm to gpt-6-sol")
        self.model = model
        self.allow_paid = allow_paid is True
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.transport = transport or http_json
        self.timeout = timeout
        self.settings = {"temperature": 0, "reasoning": {"effort": "none"}, "max_output_tokens": 2048,
                         "store": False, "stream": False}

    def describe(self):
        return {"backend": "openai", "endpoint": "https://api.openai.com/v1/responses", "model": self.model,
                "settings": self.settings, "model_version_policy": "record_response_model_and_system_fingerprint",
                "quantization": "provider_managed_not_disclosed", "context_window": 1050000}

    def generate(self, system, user, schema):
        if not self.allow_paid:
            raise InferenceError("OpenAI paid inference requires explicit approval; no request sent")
        if not self.api_key:
            raise InferenceError("OPENAI_API_KEY is unavailable; no request sent")
        payload = {"model": self.model, "input": [{"role": "system", "content": system},
                                                {"role": "user", "content": user}],
                   **self.settings, "text": {"format": {"type": "json_schema", "name": "emotion_annotations",
                                                         "strict": True, "schema": _openai_schema(schema)}}}
        raw = self.transport("POST", "https://api.openai.com/v1/responses", payload,
                             {"Authorization": "Bearer " + self.api_key}, self.timeout)
        if raw.get("error") or raw.get("status") != "completed" or raw.get("incomplete_details"):
            raise InferenceError("OpenAI response incomplete or failed", raw=raw)
        model = raw.get("model")
        if not isinstance(model, str) or (model != self.model and not model.startswith(self.model + "-")):
            raise InferenceError("OpenAI response lacks the requested model identity", raw=raw)
        output = raw.get("output")
        if not isinstance(output, list):
            raise InferenceError("OpenAI response lacks output items", raw=raw)
        messages = [item for item in output if isinstance(item, dict) and item.get("type") == "message"]
        if len(messages) != 1 or messages[0].get("status") != "completed":
            raise InferenceError("OpenAI response lacks one completed message", raw=raw)
        content = messages[0].get("content", [])
        if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict) or content[0].get("type") != "output_text" or not isinstance(content[0].get("text"), str):
            raise InferenceError("OpenAI response lacks one structured text output", raw=raw)
        fingerprint = raw.get("system_fingerprint")
        version = model + (":" + fingerprint if isinstance(fingerprint, str) and fingerprint else "")
        return {"text": content[0]["text"], "raw": raw,
                "metadata": {**self.describe(), "actual_model_version": version,
                             "system_fingerprint": fingerprint, "usage": raw.get("usage", {}),
                             "request_schema": payload["text"]["format"]["schema"]}}


class ClaudeAdapter:
    """Pinned Claude Haiku 5.5 Messages API adapter; live calls require approval."""
    MODEL = "claude-haiku-5-5"

    def __init__(self, model=MODEL, *, allow_paid=False, api_key=None, transport=None, timeout=120):
        if model != self.MODEL:
            raise ValueError("This study pins Claude to claude-haiku-5-5")
        self.model, self.allow_paid = model, allow_paid is True
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.transport, self.timeout = transport or http_json, timeout

    def describe(self):
        return {"backend": "claude", "endpoint": "https://api.anthropic.com/v1/messages",
                "model": self.model, "api_version": "2023-06-01", "max_tokens": 2048,
                "effort": "low", "thinking": "disabled", "temperature": "unsupported_omitted",
                "quantization": "provider_managed_not_disclosed",
                "context_window": "provider_managed_1m_advertised"}

    def generate(self, system, user, schema):
        if not self.allow_paid:
            raise InferenceError("Claude paid inference requires explicit approval; no request sent")
        if not self.api_key:
            raise InferenceError("ANTHROPIC_API_KEY is unavailable; no request sent")
        payload = {"model": self.model, "max_tokens": 2048, "system": system,
                   "messages": [{"role": "user", "content": user}],
                   "thinking": {"type": "disabled"},
                   "output_config": {"effort": "low", "format": {
                       "type": "json_schema", "schema": _openai_schema(schema)}}}
        raw = self.transport("POST", "https://api.anthropic.com/v1/messages", payload,
                             {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, self.timeout)
        if raw.get("model") != self.model or raw.get("stop_reason") != "end_turn":
            raise InferenceError("Claude response model or stop reason invalid", raw=raw)
        blocks = raw.get("content")
        if (not isinstance(blocks, list) or len(blocks) != 1 or
                not isinstance(blocks[0], dict) or blocks[0].get("type") != "text" or
                not isinstance(blocks[0].get("text"), str)):
            raise InferenceError("Claude response lacks one structured text block", raw=raw)
        return {"text": blocks[0]["text"], "raw": raw,
                "metadata": {**self.describe(), "actual_model_version": raw["model"],
                             "usage": raw.get("usage", {}),
                             "request_schema": payload["output_config"]["format"]["schema"]}}


def _context_payload(record, context):
    """Whitelist context fields: annotations/source_record can never enter prompts."""
    if not isinstance(context, dict) or not isinstance(context.get("target"), dict):
        raise ValueError("Context must identify its target")
    if context["target"].get("body") != record["raw_message"]:
        raise ValueError("Context target body differs from authoritative dataset text")
    if context.get("rules_version") != "preceding-only-v1":
        raise ValueError("Context must use the frozen preceding-only-v1 construction rules")
    target_time = context["target"].get("created_at")
    comments = context.get("preceding_comments", [])
    if not isinstance(comments, list):
        raise ValueError("preceding_comments must be a list")
    clean_comments = []
    previous_time = None
    for row in comments:
        if not isinstance(row, dict) or not isinstance(row.get("body"), str):
            raise ValueError("Context comment lacks body")
        if not target_time or not row.get("created_at"):
            raise ValueError("Preceding context requires known timestamps")
        parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        row_time, cutoff = parse(row["created_at"]), parse(target_time)
        if row_time.tzinfo is None or cutoff.tzinfo is None or row_time >= cutoff:
            raise ValueError("Context contains simultaneous or later comments")
        if previous_time is not None and row_time < previous_time:
            raise ValueError("Preceding comments are not ordered chronologically")
        previous_time = row_time
        if row.get("updated_at"):
            updated = parse(row["updated_at"])
            if updated.tzinfo is None or updated >= cutoff:
                raise ValueError("Preceding comment edit is not strictly before the target")
        clean_comments.append({k: row.get(k) for k in ("kind", "id", "created_at", "updated_at", "body", "author")})
    for name in ("pr_title", "pr_description"):
        if context.get(name) is not None and not isinstance(context[name], str):
            raise ValueError(f"{name} must be text or null")
    target_kind = context["target"].get("kind", record.get("target_kind"))
    if target_kind in ("pr_body", "pull_request") and context.get("pr_description"):
        raise ValueError("PR-body targets must not be duplicated as contextual descriptions")
    return {"pr_title": context.get("pr_title"), "pr_description": context.get("pr_description"),
            "preceding_comments": clean_comments, "rules_version": context["rules_version"],
            "target_created_at": target_time, "target_author": context["target"].get("author")}


def build_prompt(records, contexts_by_id, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    system = files("premo_study").joinpath("prompts/annotation.txt").read_text(encoding="utf-8")
    items = []
    for record in records:
        item = {"id": record["id"], "target": {"body": record["raw_message"]}}
        if condition == "context":
            item["context"] = _context_payload(record, contexts_by_id[record["id"]])
        items.append(item)
    return system, canonical({"condition": condition, "items": items})


def _immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing an immutable research artifact.
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(value) + "\n")
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Immutable artifact collision: {path.name}") from None


def _load_job(path, provenance, ids):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("provenance") != provenance or obj.get("response_sha256") != digest(obj.get("response")):
        raise ValueError("Cache provenance or response checksum mismatch")
    annotations = validate_response(obj["response"]["text"], ids)
    if annotations != obj.get("annotations"):
        raise ValueError("Cached labels differ from raw response")
    return obj


def inference_progress(output_dir):
    """Read a live run's immutable artifacts without contacting the model.

    Counts describe saved files, not final paired predictions. A file being
    written concurrently is reported as unreadable rather than guessed valid.
    """
    output = Path(output_dir)
    attempts = {"valid": 0, "failed": 0, "unreadable": 0}
    for path in (output / "attempts").glob("*.json"):
        try:
            status = json.loads(path.read_text(encoding="utf-8")).get("status")
        except (OSError, ValueError):
            status = None
        attempts[status if status in ("valid", "failed") else "unreadable"] += 1
    return {"output_dir": str(output),
            "active_lock": (output / ".inference.lock").exists(),
            "saved_requests": len(list((output / "requests").glob("*.json"))),
            "cached_jobs": len(list((output / "cache").glob("*.json"))),
            "attempts": attempts,
            "result_files": len(list((output / "results").glob("*.json")))}


def _run_job(output, provenance, ids, adapter, system, user, schema, max_attempts, retry_delay):
    key = digest(provenance)
    _immutable_json(output / "requests" / f"{key}.json", {
        "provenance": provenance, "system": system, "user": user, "schema": schema})
    cache = output / "cache" / f"{key}.json"
    if cache.exists():
        return _load_job(cache, provenance, ids)
    previous = sorted((output / "attempts").glob(f"{key}-*.json")) if (output / "attempts").exists() else []
    for path in previous:
        attempt = json.loads(path.read_text(encoding="utf-8"))
        if attempt.get("provenance") != provenance:
            raise ValueError("Attempt provenance mismatch")
        if attempt.get("status") == "valid":
            result = {k: attempt[k] for k in ("provenance", "response", "response_sha256", "annotations")}
            if result["response_sha256"] != digest(result["response"]) or validate_response(result["response"]["text"], ids) != result["annotations"]:
                raise ValueError("Attempt checksum or labels mismatch")
            _immutable_json(cache, result)
            return result
    for index in range(len(previous), max_attempts):
        attempt = {"provenance": provenance, "attempt": index + 1, "started_at": utc_now(),
                   "request": {"system": system, "user": user, "schema": schema}}
        started = time.monotonic()
        retryable = True
        try:
            response = adapter.generate(system, user, schema)
            attempt["response"] = response
            attempt["response_sha256"] = digest(response)
            attempt["annotations"] = validate_response(response["text"], ids)
            attempt["status"] = "valid"
        except (InferenceError, ValueError, KeyError, TypeError) as exc:
            attempt["status"] = "failed"
            attempt["error_type"] = type(exc).__name__
            attempt["error"] = str(exc)
            if isinstance(exc, InferenceError):
                attempt["raw_error"] = exc.raw
                retryable = exc.retryable
        attempt["finished_at"], attempt["elapsed_seconds"] = utc_now(), time.monotonic() - started
        _immutable_json(output / "attempts" / f"{key}-{uuid4().hex}.json", attempt)
        if attempt["status"] == "valid":
            result = {k: attempt[k] for k in ("provenance", "response", "response_sha256", "annotations")}
            _immutable_json(cache, result)
            return result
        if not retryable:
            break
        if index + 1 < max_attempts and retry_delay:
            time.sleep(min(retry_delay * 2 ** index, 30))
    return None


def _run_experiment(records, contexts_by_id, adapter, output_dir, *, dataset_sha256=None,
                   batch_size=10, repeats=3, max_attempts=3, retry_delay=1, context_max_chars=None):
    """Run three serial repetitions of both conditions on one eligible paired cohort.

    Returns predictions only for ids whose BOTH conditions have three valid runs.
    Existing malformed/tampered cache artifacts fail closed. Failed jobs are explicit
    exclusions, never empty label sets. No paid Gemini calls unless adapter opted in.
    """
    records = list(records)
    if repeats != 3 or type(batch_size) is not int or batch_size < 1 or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("Require three repeats and positive batch_size/max_attempts")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Record ids must be unique nonempty strings")
    if any(not isinstance(r.get("raw_message"), str) for r in records):
        raise ValueError("Every record requires authoritative raw_message text")
    exclusions, eligible = [], []
    for record in records:
        rid = record["id"]
        if rid not in contexts_by_id:
            exclusions.append({"id": rid, "reason": "context_unavailable"})
            continue
        try:
            payload = _context_payload(record, contexts_by_id[rid])
            if context_max_chars is not None and len(canonical(payload)) > context_max_chars:
                raise ValueError("Context exceeds explicit context_max_chars; no automatic truncation")
        except (ValueError, TypeError, KeyError) as exc:
            exclusions.append({"id": rid, "reason": "invalid_context", "detail": str(exc)})
            continue
        eligible.append(record)
    output = Path(output_dir)
    if eligible and isinstance(adapter, (GeminiAdapter, OpenAIAdapter, ClaudeAdapter)) and (not adapter.allow_paid or not adapter.api_key):
        raise InferenceError("Paid inference requires explicit approval and its API key; no request sent")
    description = adapter.describe() if eligible else {"backend": "not_contacted", "reason": "no_eligible_records"}
    source_hash = dataset_sha256 or digest([{k: r.get(k) for k in ("id", "project", "message_url", "raw_message")} for r in records])
    collected = {r["id"]: {c: [] for c in CONDITIONS} for r in eligible}
    failures = []
    actual_versions = set()
    for repeat in range(1, repeats + 1):
        for condition in CONDITIONS:
            for offset in range(0, len(eligible), batch_size):
                batch = eligible[offset:offset + batch_size]
                batch_ids = [r["id"] for r in batch]
                system, user = build_prompt(batch, contexts_by_id, condition)
                schema = response_schema(batch_ids)
                provenance = {"protocol_version": PROTOCOL_VERSION, "dataset_sha256": source_hash,
                              "model": description, "prompt_sha256": digest({"system": system, "user": user, "schema": schema}),
                              "condition": condition, "repeat": repeat, "ids": batch_ids,
                              "context_sha256": digest([_context_payload(r, contexts_by_id[r["id"]]) for r in batch]) if condition == "context" else None}
                result = _run_job(output, provenance, batch_ids, adapter, system, user, schema, max_attempts, retry_delay)
                if result is None:
                    failures.append({"job": digest(provenance), "ids": batch_ids, "condition": condition, "repeat": repeat})
                else:
                    actual_version = result["response"].get("metadata", {}).get("actual_model_version")
                    if isinstance(adapter, (GeminiAdapter, OpenAIAdapter, ClaudeAdapter)) and not actual_version:
                        raise InferenceError("Cached API response lacks actual model version")
                    if actual_version:
                        actual_versions.add(actual_version)
                        if len(actual_versions) > 1:
                            raise InferenceError("Mixed actual model versions across successful jobs; no aggregate produced")
                    for annotation in result["annotations"]:
                        collected[annotation["id"]][condition].append(annotation["emotions"])
    predictions, consistency = [], []
    for rid, conditions in collected.items():
        if any(len(conditions[c]) != repeats for c in CONDITIONS):
            exclusions.append({"id": rid, "reason": "incomplete_paired_inference"})
            continue
        for condition in CONDITIONS:
            predictions.append({"id": rid, "condition": condition, "emotions": majority_vote(conditions[condition])})
            consistency.append({"id": rid, "condition": condition, **run_consistency(conditions[condition])})
    result = {"protocol_version": PROTOCOL_VERSION, "dataset_sha256": source_hash, "adapter": description,
              "requested_ids": ids, "eligible_ids": [r["id"] for r in eligible],
              "complete_ids": [p["id"] for p in predictions if p["condition"] == "comment_only"],
              "batch_size": batch_size, "repeats": repeats, "max_attempts": max_attempts,
              "predictions": predictions, "consistency": consistency, "exclusions": exclusions, "failed_jobs": failures,
              "actual_model_version": next(iter(actual_versions)) if actual_versions else None,
              "prompt_version_sha256": digest(files("premo_study").joinpath("prompts/annotation.txt").read_text(encoding="utf-8"))}
    _immutable_json(output / "results" / f"{digest(result)}.json", result)
    return result


@contextmanager
def _output_lock(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".inference.lock"
    owner = {"pid": os.getpid(), "created_at": utc_now(), "nonce": uuid4().hex}
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(canonical(owner))
    except FileExistsError:
        raise InferenceError("Inference output directory is locked by another run; verify stale locks manually") from None
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) == owner:
            path.unlink()


def run_experiment(records, contexts_by_id, adapter, output_dir, *, dataset_sha256=None,
                   batch_size=10, repeats=3, max_attempts=3, retry_delay=1, context_max_chars=None):
    """Run the paired experiment under an exclusive output-directory lock.

    Do not run two local experiments against the same GPU under different output
    directories. The lock prevents duplicate concurrent jobs in this directory.
    """
    with _output_lock(output_dir):
        return _run_experiment(records, contexts_by_id, adapter, output_dir,
                               dataset_sha256=dataset_sha256, batch_size=batch_size, repeats=repeats,
                               max_attempts=max_attempts, retry_delay=retry_delay,
                               context_max_chars=context_max_chars)
