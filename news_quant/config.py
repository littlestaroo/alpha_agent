from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(key: str, default: str) -> Path:
    value = os.getenv(key, default).strip() or default
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "0"))
MOCK_LLM = _env_bool("MOCK_LLM", default=not OPENAI_API_KEY)

DEFAULT_BODY_CHARS = int(os.getenv("BASELINE_MAX_BODY_CHARS", "1600"))
LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "Asia/Shanghai").strip() or "Asia/Shanghai"
OPENNEWSARCHIVE_RAW_DIR = _env_path(
    "OPENNEWSARCHIVE_RAW_DIR", "data/raw/opennewsarchive"
)
OPENNEWSARCHIVE_PREPARED_PATH = _env_path(
    "OPENNEWSARCHIVE_PREPARED_PATH", "data/prepared/opennewsarchive_news.jsonl"
)
OPENNEWSARCHIVE_EXPERIMENT_PATH = _env_path(
    "OPENNEWSARCHIVE_EXPERIMENT_PATH", "data/prepared/opennewsarchive_experiment_set.jsonl"
)
