"""Read-only dependency manifests and runtime evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def file_manifest(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def verify_manifest(root, manifest):
    actual = file_manifest(root)
    return {"unchanged": actual == manifest,
            "changed": sorted(k for k in actual.keys() & manifest.keys() if actual[k] != manifest[k]),
            "added": sorted(actual.keys() - manifest.keys()), "removed": sorted(manifest.keys() - actual.keys())}


def git_revision(root):
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def runtime_metadata():
    return {"recorded_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform()}
