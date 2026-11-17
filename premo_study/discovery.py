"""Read-only discovery: never starts a service, pulls a model or runs inference."""
import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from .provenance import runtime_metadata


def _get(url):
    with urlopen(Request(url, headers={"User-Agent": "CSI5187-discovery/0.1"}), timeout=3) as response:
        return json.loads(response.read())


def discover_backends(get_json=None):
    get_json = get_json or _get
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    report = {**runtime_metadata(), "requested_model": "qwen3.5:9b", "ollama_base_url": base,
              "credentials_present": {k: bool(os.environ.get(k)) for k in ["GITHUB_TOKEN", "GH_TOKEN", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]},
              "probes": [], "model_verified": False,
              "policy": "Read-only HTTP discovery; no downloads, inference, service startup or changes to other projects."}
    for url in [base + "/api/version", base + "/api/tags",
                "http://localhost:1234/v1/models", "http://localhost:8000/v1/models",
                "http://localhost:8080/v1/models"]:
        try:
            data = get_json(url)
            report["probes"].append({"url": url, "status": "reachable", "response": data})
            if url == base + "/api/tags":
                report["model_verified"] = any(item.get("name", "").lower() == "qwen3.5:9b" for item in data.get("models", []))
        except (URLError, OSError, ValueError, TypeError) as error:
            report["probes"].append({"url": url, "status": "unavailable", "error_type": type(error).__name__})
    return report
