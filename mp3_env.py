"""Load .env and build OpenAI / Qdrant clients (OpenAI or OpenRouter)."""
from __future__ import annotations

import os
from pathlib import Path

_MP3_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _MP3_ROOT.parent.parent


def load_mp3_env() -> None:
    """Load local .env, then capstone fallbacks (no override of already-set vars)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_manual(_MP3_ROOT / ".env")
        return

    # Local .env wins (fixes stale $env:QDRANT_URL from an old PowerShell session).
    load_dotenv(_MP3_ROOT / ".env", override=True)
    candidates = [
        _MP3_ROOT.parent / "capstone" / ".env",
        _MP3_ROOT.parent.parent / "Week11" / "capstone" / ".env",
        _MP3_ROOT.parent.parent / "Week10" / "capstone" / ".env",
        _MP3_ROOT.parent.parent / "Week8" / "capstone" / ".env",
        _MP3_ROOT.parent.parent / "Week7" / "capstone" / ".env",
    ]
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)


def _load_env_manual(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key, "").strip():
            os.environ[key] = value


def resolve_corpus_dir() -> Path:
    """Sherlock corpus: local mp3/corpus, or Week 9 MP2 corpus if local is empty."""
    load_mp3_env()
    override = os.getenv("MP3_CORPUS_DIR", "").strip()
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = (_MP3_ROOT / path).resolve()
        return path

    local = _MP3_ROOT / "corpus"
    if any(local.glob("*.txt")):
        return local

    for rel in (
        "Week9_Capstone_mini/mp2/corpus",
        "Week9/mp2/corpus",
        (
            "Week9_Capstone_mini/Week 9_Graded Mini Project Documents/"
            "AI-RAG_MP2_MiniRAG/learner/corpus"
        ),
    ):
        cand = (_REPO_ROOT / rel).resolve()
        if any(cand.glob("*.txt")):
            return cand
    return local


def prefer_openrouter() -> bool:
    """MP3 defaults to OpenRouter when both keys exist (W7–W11 workflow)."""
    return os.getenv("MP3_USE_OPENROUTER", "true").lower() in {"1", "true", "yes"}


def active_llm_backend() -> str:
    load_mp3_env()
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    oa_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if or_key and (prefer_openrouter() or not oa_key):
        return "openrouter"
    if oa_key:
        return "openai"
    return ""


def uses_openrouter() -> bool:
    return active_llm_backend() == "openrouter"


def chat_model() -> str:
    raw = (
        os.environ.get("MP3_CHAT_MODEL")
        or os.environ.get("MP1_MODEL")
        or os.environ.get("CHAT_MODEL")
        or "gpt-4o-mini"
    ).strip()
    if uses_openrouter() and "/" not in raw:
        return f"openai/{raw}"
    return raw


def embedding_model() -> str:
    raw = (os.environ.get("EMBED_MODEL") or "text-embedding-3-small").strip()
    if uses_openrouter() and "/" not in raw:
        return f"openai/{raw}"
    return raw


def make_openai_client():
    """Same OpenAI SDK + base_url pattern as Week12 capstone ``make_sync_client``."""
    from openai import OpenAI

    backend = active_llm_backend()
    if backend == "openrouter":
        return OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"].strip(),
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ),
        )
    if backend == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
    return None


def capstone_settings():
    """Optional: reuse ``Settings.from_env`` after mp3 env load (for parity checks)."""
    capstone_root = _MP3_ROOT.parent / "capstone"
    if not (capstone_root / "src" / "settings.py").is_file():
        return None
    import sys

    root = str(capstone_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.settings import Settings  # noqa: PLC0415

    load_mp3_env()
    return Settings.from_env()


def qdrant_url_issues() -> list[str]:
    """Sanity-check QDRANT_URL before the client hits the network."""
    from urllib.parse import urlparse

    load_mp3_env()
    url = os.environ.get("QDRANT_URL", "").strip()
    issues: list[str] = []
    if not url:
        return ["QDRANT_URL is not set (check Week12/mp3/.env)"]
    upper = url.upper()
    if "YOUR-CLUSTER" in upper or "YOUR_CLUSTER" in upper:
        issues.append("QDRANT_URL still looks like the .env.example placeholder")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        issues.append(f"QDRANT_URL scheme must be http or https (got {parsed.scheme!r})")
    if not parsed.hostname:
        issues.append("QDRANT_URL has no hostname")
    elif " " in parsed.hostname:
        issues.append("QDRANT_URL hostname contains spaces (check quotes in .env)")
    return issues


def check_qdrant_dns() -> str | None:
    """Return an error message if the cluster host cannot be resolved."""
    import socket
    from urllib.parse import urlparse

    load_mp3_env()
    url = os.environ.get("QDRANT_URL", "").strip()
    if not url:
        return "QDRANT_URL is not set"
    host = urlparse(url).hostname
    if not host:
        return "QDRANT_URL has no hostname"
    try:
        socket.getaddrinfo(host, urlparse(url).port or 6333)
    except OSError as exc:
        return (
            f"Cannot resolve Qdrant host {host!r} ({exc}). "
            "Fix QDRANT_URL in Week12/mp3/.env, or clear a stale value with "
            'Remove-Item Env:QDRANT_URL -ErrorAction SilentlyContinue'
        )
    return None


def make_qdrant_client():
    from qdrant_client import QdrantClient

    issues = qdrant_url_issues()
    if issues:
        return None
    load_mp3_env()
    url = os.environ.get("QDRANT_URL", "").strip()
    prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "false").lower() in {"1", "true", "yes"}
    return QdrantClient(
        url=url,
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        timeout=int(os.environ.get("QDRANT_TIMEOUT", "60")),
        prefer_grpc=prefer_grpc,
        check_compatibility=False,
    )