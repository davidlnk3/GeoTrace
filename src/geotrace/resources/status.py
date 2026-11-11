# src/geotrace/resources/status.py
import importlib.metadata as md
import json
import os
import platform
import re
import sys
from collections import defaultdict

from geotrace.core import mapillary, reverse_search
from geotrace.server import mcp

PACKAGE = "geotrace"


def _dist_name(requirement: str) -> str:
    return re.split(r"[\s\[<>=!~;@(]", requirement, maxsplit=1)[0]


def extras_status() -> dict:
    """Which optional extras are fully installed, read from GeoTrace's own metadata."""
    deps, includes = defaultdict(list), defaultdict(list)
    for req in md.requires(PACKAGE) or []:
        m = re.search(r'extra\s*==\s*["\']([^"\']+)["\']', req)
        if not m:
            continue
        extra, name = m.group(1), _dist_name(req)
        if name.lower() == PACKAGE:                       # p. ej. all = ["geotrace[ocr,sun,...]"]
            inner = re.search(r"\[([^\]]+)\]", req)
            includes[extra] += [e.strip() for e in inner.group(1).split(",")] if inner else []
        else:
            deps[extra].append(name)

    status = {}
    for extra, names in deps.items():
        missing = []
        for n in names:
            try:
                md.version(n)
            except md.PackageNotFoundError:
                missing.append(n)
        status[extra] = {"installed": not missing, "missing": missing}
    for extra, inner in includes.items():
        missing = [e for e in inner if not status.get(e, {}).get("installed")]
        status[extra] = {"installed": not missing, "missing_extras": missing}
    return dict(sorted(status.items()))


def credentials_status() -> dict:
    """Only whether each variable is set. Never return the values."""
    names = {
        "huggingface": "HF_TOKEN",
        "mapillary": mapillary.TOKEN_ENV,
        "serpapi": reverse_search.SERPAPI_ENV,
        "google_vision": reverse_search.VISION_ENV,
    }
    return {service: {"env_var": var, "configured": bool(os.environ.get(var))}
            for service, var in names.items()}


@mcp.resource(
    "geotrace://status",
    name="server_status",
    title="GeoTrace server status",
    description="Installed optional extras, configured API keys (yes/no only) and versions.",
    mime_type="application/json",
)
def server_status() -> str:
    try:
        version = md.version(PACKAGE)
    except md.PackageNotFoundError:
        version = "unknown"
    return json.dumps({
        "geotrace_version": version,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "extras": extras_status(),
        "credentials": credentials_status(),
    }, indent=2)