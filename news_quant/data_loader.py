from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from news_quant.config import LOCAL_TIMEZONE
from news_quant.dataset import normalize_news_records

NEWS_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "id": ["id", "news_id", "article_id", "doc_id", "docid"],
    "title": ["title", "headline", "news_title"],
    "body": [
        "body",
        "content",
        "text",
        "article",
        "news_content",
        "full_text",
        "description",
    ],
    "publish_time": [
        "publish_time",
        "published_at",
        "publish_date",
        "datetime",
        "date",
        "created_at",
        "time",
    ],
    "source": ["source", "media", "publisher", "site_name"],
    "url": ["url", "link", "article_url"],
}

STOCK_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "ts_code": ["ts_code", "stock_code", "code", "symbol"],
    "name": ["name", "stock_name", "company_name", "sec_name"],
    "industry": ["industry", "sector", "sw_industry"],
    "aliases": ["aliases", "alias", "stock_aliases"],
}

NEWS_SUFFIXES = {".csv", ".jsonl", ".json", ".parquet"}
ALIAS_SPLITTER = re.compile(r"[|,;；、/]+")


def _canonical_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        matched = lower_map.get(candidate.lower())
        if matched:
            return matched
    return None


def _read_single_news_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return normalize_news_records(
            pd.read_csv(path, encoding="utf-8").to_dict("records"),
            source_file=path.name,
        )
    if suffix == ".jsonl":
        records = pd.read_json(path, lines=True).to_dict("records")
        return normalize_news_records(records, source_file=path.name)
    if suffix == ".json":
        records = pd.read_json(path).to_dict("records")
        return normalize_news_records(records, source_file=path.name)
    if suffix == ".parquet":
        return normalize_news_records(
            pd.read_parquet(path).to_dict("records"),
            source_file=path.name,
        )
    raise ValueError(f"不支持的新闻文件格式: {path}")


def _normalize_publish_time(raw_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(raw_series, errors="coerce")
    tz_info = getattr(parsed.dt, "tz", None)
    if tz_info is not None:
        parsed = parsed.dt.tz_convert(LOCAL_TIMEZONE).dt.tz_localize(None)
    publish_time = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    publish_time = publish_time.fillna(raw_series.fillna("").astype(str))
    publish_date = parsed.dt.strftime("%Y-%m-%d").fillna("")
    return publish_time, publish_date


def load_news_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"新闻路径不存在: {target}")

    if target.is_dir():
        files = sorted(
            file_path
            for file_path in target.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in NEWS_SUFFIXES
        )
        if not files:
            raise ValueError(f"目录下未找到支持的新闻文件: {target}")
        frames = [_read_single_news_file(file_path) for file_path in files]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = _read_single_news_file(target)

    preferred_columns = [
        "id",
        "title",
        "body",
        "publish_time",
        "publish_date",
        "source",
        "url",
        "language",
    ]
    extra_columns = [column for column in df.columns if column not in preferred_columns]
    return df[preferred_columns + extra_columns].reset_index(drop=True)


def _split_aliases(raw_aliases: object, stock_name: str) -> list[str]:
    values: list[str] = []
    if raw_aliases is not None and not pd.isna(raw_aliases):
        values.extend(part.strip() for part in ALIAS_SPLITTER.split(str(raw_aliases)))
    values.append(stock_name.strip())

    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def load_stock_universe(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"股票池路径不存在: {target}")

    df = pd.read_csv(target, encoding="utf-8")
    rename_map: dict[str, str] = {}
    columns = list(df.columns)
    for target_column, candidates in STOCK_COLUMN_CANDIDATES.items():
        source_column = _canonical_column(columns, candidates)
        if source_column:
            rename_map[source_column] = target_column
    df = df.rename(columns=rename_map)

    missing = {"ts_code", "name"} - set(df.columns)
    if missing:
        raise ValueError(f"股票池缺少必要列: {sorted(missing)}")

    if "industry" not in df.columns:
        df["industry"] = ""
    if "aliases" not in df.columns:
        df["aliases"] = ""

    df["ts_code"] = df["ts_code"].fillna("").astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["industry"] = df["industry"].fillna("").astype(str).str.strip()
    df = df[(df["ts_code"] != "") & (df["name"] != "")].reset_index(drop=True)
    df["aliases"] = [
        _split_aliases(raw_aliases, stock_name)
        for raw_aliases, stock_name in zip(df["aliases"], df["name"], strict=False)
    ]
    return df[["ts_code", "name", "industry", "aliases"]]


def _parse_keyword_args(keywords: list[str] | None) -> list[str]:
    if not keywords:
        return []
    parsed: list[str] = []
    for raw in keywords:
        parts = re.split(r"[，,;；]", str(raw))
        parsed.extend(part.strip() for part in parts if part.strip())
    return parsed


def filter_news_rows(
    news_df: pd.DataFrame,
    keywords: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    df = news_df.copy()
    parsed_keywords = _parse_keyword_args(keywords)

    if parsed_keywords:
        combined_text = df["title"].fillna("") + "\n" + df["body"].fillna("")
        pattern = "|".join(re.escape(keyword) for keyword in parsed_keywords)
        df = df[combined_text.str.contains(pattern, na=False)]

    if date_from:
        df = df[df["publish_date"] >= date_from]
    if date_to:
        df = df[df["publish_date"] <= date_to]
    if limit is not None and limit > 0:
        df = df.head(limit)
    return df.reset_index(drop=True)
