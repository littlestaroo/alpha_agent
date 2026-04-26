from __future__ import annotations

import json
from hashlib import md5
from pathlib import Path
from typing import Any

import pandas as pd

from news_quant.config import LOCAL_TIMEZONE

SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}

FIELD_CANDIDATES: dict[str, list[str]] = {
    "id": [
        "id",
        "news_id",
        "article_id",
        "doc_id",
        "docid",
        "_id",
        "uuid",
    ],
    "title": [
        "title",
        "headline",
        "news_title",
        "article.title",
        "data.title",
        "meta.title",
    ],
    "body": [
        "body",
        "content",
        "text",
        "article",
        "news_content",
        "full_text",
        "description",
        "article.content",
        "article.text",
        "data.content",
        "data.text",
        "meta.description",
    ],
    "publish_time": [
        "publish_time",
        "published_at",
        "publish_date",
        "datetime",
        "date",
        "created_at",
        "time",
        "pub_time",
        "article.published_at",
        "data.published_at",
        "meta.published_at",
    ],
    "source": [
        "source",
        "media",
        "publisher",
        "site_name",
        "site",
        "provider",
        "article.source",
        "data.source",
    ],
    "url": [
        "url",
        "link",
        "article_url",
        "article.url",
        "data.url",
    ],
}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, list):
        return " ".join(_stringify(item) for item in value if _stringify(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            continue
        current_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_record(value, current_key))
        else:
            flat[current_key] = value
    return flat


def _pick_field(flat_record: dict[str, Any], field_name: str) -> str:
    lowered = {key.lower(): key for key in flat_record}
    for candidate in FIELD_CANDIDATES[field_name]:
        key = lowered.get(candidate.lower())
        if key is None:
            continue
        value = _stringify(flat_record[key])
        if value:
            return value
    return ""


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("data", "records", "items", "articles", "news"):
            value = raw.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [raw]
    return []


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def _read_records_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl_records(path)
    if suffix == ".json":
        return _read_json_records(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8").to_dict("records")
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"不支持的数据文件格式: {path}")


def _normalize_publish_times(raw_values: list[str]) -> tuple[list[str], list[str]]:
    series = pd.Series(raw_values, dtype="object")
    parsed = pd.to_datetime(series, errors="coerce")
    tz_info = getattr(parsed.dt, "tz", None)
    if tz_info is not None:
        parsed = parsed.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    publish_time = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    publish_time = publish_time.fillna(series.fillna("").astype(str))
    publish_date = parsed.dt.strftime("%Y-%m-%d").fillna("")
    return publish_time.tolist(), publish_date.tolist()


def normalize_news_records(
    records: list[dict[str, Any]],
    source_file: str = "",
    languages: set[str] | None = None,
) -> pd.DataFrame:
    normalized_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        flat_record = _flatten_record(record)
        raw_language = _stringify(
            flat_record.get("language", flat_record.get("article.language", ""))
        ).lower()
        if languages and raw_language and raw_language not in languages:
            continue
        title = _pick_field(flat_record, "title")
        body = _pick_field(flat_record, "body")
        if not title and not body:
            continue

        raw_id = _pick_field(flat_record, "id")
        publish_time = _pick_field(flat_record, "publish_time")
        source = _pick_field(flat_record, "source")
        url = _pick_field(flat_record, "url")
        if not raw_id:
            raw_id = md5(
                f"{source_file}|{idx}|{title}|{publish_time}".encode("utf-8")
            ).hexdigest()[:16]

        normalized_rows.append(
            {
                "id": raw_id,
                "title": title,
                "body": body,
                "publish_time": publish_time,
                "source": source,
                "url": url,
                "language": raw_language,
                "raw_source_file": source_file,
            }
        )

    if not normalized_rows:
        return pd.DataFrame(
            columns=[
                "id",
                "title",
                "body",
                "publish_time",
                "publish_date",
                "source",
                "url",
                "language",
                "raw_source_file",
            ]
        )

    df = pd.DataFrame(normalized_rows)
    normalized_time, normalized_date = _normalize_publish_times(df["publish_time"].tolist())
    df["publish_time"] = normalized_time
    df["publish_date"] = normalized_date
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["body"] = df["body"].fillna("").astype(str).str.strip()
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["id"] = df["id"].fillna("").astype(str).str.strip()

    missing_id_mask = df["id"] == ""
    if missing_id_mask.any():
        df.loc[missing_id_mask, "id"] = [
            f"n{i:08d}" for i in range(missing_id_mask.sum())
        ]

    df = df.drop_duplicates(subset=["id"], keep="first")
    df = df.drop_duplicates(
        subset=["title", "body", "publish_time"], keep="first"
    ).reset_index(drop=True)
    return df[
        [
            "id",
            "title",
            "body",
            "publish_time",
            "publish_date",
            "source",
            "url",
            "language",
            "raw_source_file",
        ]
    ]


def _collect_supported_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def prepare_opennewsarchive_dataset(
    raw_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    languages: set[str] | None = None,
) -> pd.DataFrame:
    source = Path(raw_path)
    if not source.exists():
        raise FileNotFoundError(f"原始新闻路径不存在: {source}")

    files = _collect_supported_files(source)
    if not files:
        raise ValueError(f"未找到支持的数据文件: {source}")

    frames: list[pd.DataFrame] = []
    remaining = limit
    for file_path in files:
        records = _read_records_from_file(file_path)
        if remaining is not None and remaining >= 0:
            records = records[:remaining]
        frame = normalize_news_records(
            records,
            source_file=str(
                file_path.relative_to(source if source.is_dir() else file_path.parent)
            ),
            languages=languages,
        )
        if not frame.empty:
            frames.append(frame)
            if remaining is not None:
                remaining -= len(frame)
                if remaining <= 0:
                    break

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["id", "title", "body", "publish_time"], keep="first"
        ).reset_index(drop=True)
    else:
        combined = normalize_news_records([], source_file="")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_json(output, orient="records", lines=True, force_ascii=False)
    return combined
