from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from news_quant.data_loader import load_stock_universe
from news_quant.dataset import _flatten_record, _pick_field, _stringify

WHITESPACE_RE = re.compile(r"\s+")

THESIS_FINANCE_KEYWORDS = [
    "股票",
    "a股",
    "港股",
    "美股",
    "沪指",
    "深成指",
    "创业板",
    "公告",
    "财报",
    "业绩",
    "营收",
    "净利润",
    "归母净利润",
    "扣非",
    "市值",
    "股价",
    "估值",
    "分红",
    "回购",
    "增持",
    "减持",
    "融资",
    "收购",
    "并购",
    "订单",
    "销量",
    "产能",
    "产品",
    "渠道",
    "经销商",
    "合作",
]

THESIS_NEGATIVE_KEYWORDS = [
    "政府",
    "市委",
    "省政府",
    "人大",
    "政协",
    "招商",
    "签约",
    "大会",
    "论坛",
    "发布会",
    "开幕式",
    "文旅",
    "旅游",
    "景区",
    "世界杯",
    "比赛",
    "体育",
    "铁人三项",
    "乡村振兴",
    "考察",
    "观摩",
    "推介会",
]

STOCK_POSITIVE_KEYWORDS: dict[str, list[str]] = {
    "宁德时代": [
        "宁德时代",
        "宁王",
        "catl",
        "动力电池",
        "电池",
        "储能",
        "锂电",
        "碳酸锂",
        "电芯",
        "麒麟电池",
        "钠离子",
        "新能源车",
        "新能源汽车",
        "车企",
        "磷酸铁锂",
    ],
    "贵州茅台": [
        "贵州茅台",
        "茅台",
        "飞天茅台",
        "白酒",
        "酱香",
        "酒企",
        "高端白酒",
        "批价",
        "动销",
        "终端价",
        "经销商",
        "渠道",
        "酒业",
        "直营",
    ],
}

STOCK_NEGATIVE_KEYWORDS: dict[str, list[str]] = {
    "宁德时代": [
        "招商引资",
        "项目签约",
        "集中签约",
        "产业园",
        "开工仪式",
    ],
    "贵州茅台": [
        "敷在脸上的茅台",
        "xx界茅台",
        "赛道茅台",
        "旅游",
        "景区",
    ],
}


def _normalized_text(text: str) -> str:
    return WHITESPACE_RE.sub("", str(text)).lower()


def _stock_alias_rows(universe_path: str | Path) -> list[dict]:
    universe_df = load_stock_universe(universe_path)
    rows: list[dict] = []
    for _, row in universe_df.iterrows():
        aliases = row["aliases"] if isinstance(row["aliases"], list) else [row["name"]]
        normalized_aliases = []
        for alias in aliases:
            normalized = _normalized_text(alias)
            if len(normalized) >= 2:
                normalized_aliases.append((alias, normalized))
        rows.append(
            {
                "ts_code": row["ts_code"],
                "stock_name": row["name"],
                "industry": row["industry"],
                "aliases": normalized_aliases,
            }
        )
    return rows


def _match_record_to_stocks(record: dict, stock_rows: list[dict]) -> list[dict]:
    merged_text = _normalized_text(
        f"{record.get('title', '')}\n{record.get('body', '')}"
    )
    matches: list[dict] = []

    for stock in stock_rows:
        hit_aliases: list[str] = []
        best_alias_length = 0
        for alias, normalized_alias in stock["aliases"]:
            if normalized_alias in merged_text:
                hit_aliases.append(alias)
                best_alias_length = max(best_alias_length, len(normalized_alias))
        if hit_aliases:
            matches.append(
                {
                    "ts_code": stock["ts_code"],
                    "stock_name": stock["stock_name"],
                    "industry": stock["industry"],
                    "matched_aliases": hit_aliases,
                    "match_score": len(hit_aliases) * 10 + best_alias_length,
                }
            )

    matches.sort(key=lambda item: item["match_score"], reverse=True)
    return matches


def build_experiment_subset(
    input_path: str | Path,
    universe_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path | None = None,
    max_per_stock: int = 500,
    max_total: int = 2500,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, object]:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rejected_file = Path(rejected_path) if rejected_path else None
    if rejected_file is not None:
        rejected_file.parent.mkdir(parents=True, exist_ok=True)

    stock_rows = _stock_alias_rows(universe_path)
    counts_by_stock: dict[str, int] = defaultdict(int)

    total = 0
    kept = 0
    rejected = 0

    with source.open("r", encoding="utf-8") as reader, output.open(
        "w", encoding="utf-8"
    ) as writer:
        rejected_writer = (
            rejected_file.open("w", encoding="utf-8") if rejected_file else None
        )
        try:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                total += 1
                record = json.loads(line)

                publish_date = str(record.get("publish_date", "") or "")
                if date_from and publish_date and publish_date < date_from:
                    rejected += 1
                    continue
                if date_to and publish_date and publish_date > date_to:
                    rejected += 1
                    continue

                matches = _match_record_to_stocks(record, stock_rows)
                if not matches:
                    rejected += 1
                    if rejected_writer is not None:
                        rejected_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    continue

                kept_for_any_stock = False
                filtered_matches: list[dict] = []
                for match in matches:
                    if counts_by_stock[match["ts_code"]] >= max_per_stock:
                        continue
                    filtered_matches.append(match)

                if not filtered_matches:
                    rejected += 1
                    continue

                primary_match = filtered_matches[0]
                for match in filtered_matches:
                    counts_by_stock[match["ts_code"]] += 1
                kept_for_any_stock = True

                if kept_for_any_stock:
                    record["primary_ts_code"] = primary_match["ts_code"]
                    record["primary_stock_name"] = primary_match["stock_name"]
                    record["matched_ts_codes"] = "|".join(
                        match["ts_code"] for match in filtered_matches
                    )
                    record["matched_stock_names"] = "|".join(
                        match["stock_name"] for match in filtered_matches
                    )
                    record["matched_aliases"] = "|".join(
                        alias
                        for match in filtered_matches
                        for alias in match["matched_aliases"]
                    )
                    record["matched_stock_count"] = len(filtered_matches)
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1

                if max_total and kept >= max_total:
                    break
        finally:
            if rejected_writer is not None:
                rejected_writer.close()

    return {
        "total": total,
        "kept": kept,
        "rejected": rejected,
        "counts_by_stock": dict(sorted(counts_by_stock.items())),
    }


def _iter_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _normalize_raw_record(record: dict, source_file: str) -> dict:
    flat_record = _flatten_record(record)
    raw_language = _stringify(
        flat_record.get("language", flat_record.get("article.language", ""))
    ).lower()
    title = _pick_field(flat_record, "title")
    body = _pick_field(flat_record, "body")
    publish_time = _pick_field(flat_record, "publish_time")
    news_id = _pick_field(flat_record, "id")
    source = _pick_field(flat_record, "source")
    url = _pick_field(flat_record, "url")

    publish_date = ""
    if publish_time:
        publish_date = str(publish_time)[:10]

    return {
        "id": news_id,
        "title": title,
        "body": body,
        "publish_time": publish_time,
        "publish_date": publish_date,
        "source": source,
        "url": url,
        "language": raw_language,
        "raw_source_file": source_file,
    }


def build_experiment_subset_from_raw(
    raw_path: str | Path,
    universe_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path | None = None,
    max_per_stock: int = 0,
    max_total: int = 0,
    date_from: str | None = None,
    date_to: str | None = None,
    language: str = "zh",
) -> dict[str, object]:
    source = Path(raw_path)
    if not source.exists():
        raise FileNotFoundError(f"原始新闻路径不存在: {source}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rejected_file = Path(rejected_path) if rejected_path else None
    if rejected_file is not None:
        rejected_file.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        files = sorted(source.rglob("*.jsonl"))
    else:
        files = [source]

    stock_rows = _stock_alias_rows(universe_path)
    counts_by_stock: dict[str, int] = defaultdict(int)
    kept_ids: set[str] = set()

    total = 0
    kept = 0
    rejected = 0

    with output.open("w", encoding="utf-8") as writer:
        rejected_writer = (
            rejected_file.open("w", encoding="utf-8") if rejected_file else None
        )
        try:
            for file_path in files:
                rel = (
                    str(file_path.relative_to(source))
                    if source.is_dir()
                    else file_path.name
                )
                for raw_record in _iter_jsonl_records(file_path):
                    total += 1
                    record = _normalize_raw_record(raw_record, rel)

                    if language and record["language"] and record["language"] != language:
                        rejected += 1
                        continue
                    if not record["title"] and not record["body"]:
                        rejected += 1
                        continue

                    publish_date = str(record.get("publish_date", "") or "")
                    if date_from and publish_date and publish_date < date_from:
                        rejected += 1
                        continue
                    if date_to and publish_date and publish_date > date_to:
                        rejected += 1
                        continue

                    matches = _match_record_to_stocks(record, stock_rows)
                    if not matches:
                        rejected += 1
                        if rejected_writer is not None:
                            rejected_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                        continue

                    filtered_matches: list[dict] = []
                    for match in matches:
                        if max_per_stock and counts_by_stock[match["ts_code"]] >= max_per_stock:
                            continue
                        filtered_matches.append(match)

                    if not filtered_matches:
                        rejected += 1
                        continue

                    record_id = str(record.get("id", "") or "")
                    if record_id and record_id in kept_ids:
                        continue
                    if record_id:
                        kept_ids.add(record_id)

                    primary_match = filtered_matches[0]
                    for match in filtered_matches:
                        counts_by_stock[match["ts_code"]] += 1

                    record["primary_ts_code"] = primary_match["ts_code"]
                    record["primary_stock_name"] = primary_match["stock_name"]
                    record["matched_ts_codes"] = "|".join(
                        match["ts_code"] for match in filtered_matches
                    )
                    record["matched_stock_names"] = "|".join(
                        match["stock_name"] for match in filtered_matches
                    )
                    record["matched_aliases"] = "|".join(
                        alias
                        for match in filtered_matches
                        for alias in match["matched_aliases"]
                    )
                    record["matched_stock_count"] = len(filtered_matches)
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1

                    if max_total and kept >= max_total:
                        return {
                            "total": total,
                            "kept": kept,
                            "rejected": rejected,
                            "counts_by_stock": dict(sorted(counts_by_stock.items())),
                            "files_scanned": len(files),
                        }
        finally:
            if rejected_writer is not None:
                rejected_writer.close()

    return {
        "total": total,
        "kept": kept,
        "rejected": rejected,
        "counts_by_stock": dict(sorted(counts_by_stock.items())),
        "files_scanned": len(files),
    }


def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


def _alias_quality(record: dict) -> tuple[bool, bool]:
    stock_name = str(record.get("primary_stock_name", "") or "")
    aliases = [part.strip() for part in str(record.get("matched_aliases", "")).split("|") if part.strip()]
    has_full_name = stock_name in aliases or stock_name in f"{record.get('title', '')}\n{record.get('body', '')}"
    alias_only = bool(aliases) and not has_full_name
    return has_full_name, alias_only


def _thesis_relevance_decision(record: dict) -> dict[str, object]:
    stock_name = str(record.get("primary_stock_name", "") or "")
    title = str(record.get("title", "") or "")
    body = str(record.get("body", "") or "")
    title_text = title.lower()
    body_text = body.lower()
    merged_text = f"{title_text}\n{body_text}"

    finance_hits = _keyword_hits(merged_text, THESIS_FINANCE_KEYWORDS)
    negative_hits = _keyword_hits(merged_text, THESIS_NEGATIVE_KEYWORDS)
    stock_hits = _keyword_hits(merged_text, STOCK_POSITIVE_KEYWORDS.get(stock_name, []))
    stock_negative_hits = _keyword_hits(merged_text, STOCK_NEGATIVE_KEYWORDS.get(stock_name, []))
    has_full_name, alias_only = _alias_quality(record)

    score = 0.0
    if has_full_name:
        score += 4.0
    elif alias_only:
        score += 1.0

    for keyword in finance_hits:
        score += 1.8 if keyword.lower() in title_text else 0.7
    for keyword in stock_hits:
        score += 1.6 if keyword.lower() in title_text else 0.6
    for keyword in negative_hits:
        score -= 1.6 if keyword.lower() in title_text else 0.7
    for keyword in stock_negative_hits:
        score -= 2.0 if keyword.lower() in title_text else 1.0
    if not title.strip():
        score -= 0.5

    keep = False
    if has_full_name and score >= 4.5 and (finance_hits or len(stock_hits) >= 2):
        keep = True
    elif score >= 7.0 and (len(finance_hits) >= 2 or len(stock_hits) >= 3):
        keep = True

    if alias_only and score < 8.5:
        keep = False
    if negative_hits and not has_full_name and score < 9.5:
        keep = False

    return {
        "keep": keep,
        "score": round(score, 4),
        "finance_hits": finance_hits,
        "stock_hits": stock_hits,
        "negative_hits": negative_hits + stock_negative_hits,
        "has_full_name": has_full_name,
        "alias_only": alias_only,
    }


def refine_thesis_subset(
    input_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path | None = None,
) -> dict[str, object]:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rejected_file = Path(rejected_path) if rejected_path else None
    if rejected_file is not None:
        rejected_file.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    rejected = 0
    counts_by_stock: dict[str, int] = defaultdict(int)

    with source.open("r", encoding="utf-8") as reader, output.open(
        "w", encoding="utf-8"
    ) as writer:
        rejected_writer = (
            rejected_file.open("w", encoding="utf-8") if rejected_file else None
        )
        try:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                total += 1
                record = json.loads(line)
                decision = _thesis_relevance_decision(record)
                record["thesis_relevance_score"] = decision["score"]
                record["thesis_finance_hits"] = "|".join(decision["finance_hits"])
                record["thesis_stock_hits"] = "|".join(decision["stock_hits"])
                record["thesis_negative_hits"] = "|".join(decision["negative_hits"])
                record["thesis_has_full_name"] = int(bool(decision["has_full_name"]))
                record["thesis_alias_only"] = int(bool(decision["alias_only"]))

                if decision["keep"]:
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                    counts_by_stock[str(record.get("primary_ts_code", ""))] += 1
                else:
                    rejected += 1
                    if rejected_writer is not None:
                        rejected_writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            if rejected_writer is not None:
                rejected_writer.close()

    return {
        "total": total,
        "kept": kept,
        "rejected": rejected,
        "counts_by_stock": dict(sorted(counts_by_stock.items())),
    }


def select_thesis_llm_set(
    input_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path | None = None,
    min_score: float = 16.0,
    max_per_stock: int = 700,
) -> dict[str, object]:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rejected_file = Path(rejected_path) if rejected_path else None
    if rejected_file is not None:
        rejected_file.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    total = 0

    with source.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            score = float(record.get("thesis_relevance_score", 0.0))
            if score < min_score:
                continue
            grouped[str(record.get("primary_ts_code", ""))].append(record)

    kept_records: list[dict] = []
    rejected_records: list[dict] = []
    counts_by_stock: dict[str, int] = {}

    for ts_code, records in grouped.items():
        records.sort(
            key=lambda item: (
                -float(item.get("thesis_relevance_score", 0.0)),
                str(item.get("publish_date", "")),
                str(item.get("id", "")),
            )
        )
        if max_per_stock and len(records) > max_per_stock:
            selected = records[:max_per_stock]
            rejected_records.extend(records[max_per_stock:])
        else:
            selected = records
        kept_records.extend(selected)
        counts_by_stock[ts_code] = len(selected)

    kept_records.sort(
        key=lambda item: (
            str(item.get("publish_date", "")),
            str(item.get("primary_ts_code", "")),
            -float(item.get("thesis_relevance_score", 0.0)),
        )
    )

    with output.open("w", encoding="utf-8") as writer:
        for record in kept_records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    if rejected_file is not None:
        with rejected_file.open("w", encoding="utf-8") as writer:
            for record in rejected_records:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "total": total,
        "kept": len(kept_records),
        "rejected": total - len(kept_records),
        "counts_by_stock": dict(sorted(counts_by_stock.items())),
    }


def build_daily_topk_per_stock(
    input_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path | None = None,
    top_k: int = 3,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, object]:
    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rejected_file = Path(rejected_path) if rejected_path else None
    if rejected_file is not None:
        rejected_file.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    kept_ids: set[str] = set()
    total = 0

    with source.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            publish_date = str(record.get("publish_date", "") or "")
            ts_code = str(record.get("primary_ts_code", "") or "")
            record_id = str(record.get("id", "") or "")

            if not publish_date or not ts_code:
                continue
            if date_from and publish_date < date_from:
                continue
            if date_to and publish_date > date_to:
                continue
            if record_id and record_id in kept_ids:
                continue
            if record_id:
                kept_ids.add(record_id)

            total += 1
            grouped[(publish_date, ts_code)].append(record)

    kept_records: list[dict] = []
    rejected_records: list[dict] = []
    counts_by_stock: dict[str, int] = defaultdict(int)
    days_by_stock: dict[str, set[str]] = defaultdict(set)

    def _sort_key(item: dict) -> tuple[float, int, int, int, str]:
        return (
            -float(item.get("thesis_relevance_score", 0.0)),
            -int(item.get("thesis_has_full_name", 0) or 0),
            int(item.get("matched_stock_count", 0) or 0),
            -len(str(item.get("thesis_finance_hits", "") or "").split("|")),
            str(item.get("id", "") or ""),
        )

    for (publish_date, ts_code), records in sorted(grouped.items()):
        records.sort(key=_sort_key)
        selected = records[:top_k] if top_k > 0 else records
        dropped = records[top_k:] if top_k > 0 else []

        kept_records.extend(selected)
        rejected_records.extend(dropped)
        counts_by_stock[ts_code] += len(selected)
        if selected:
            days_by_stock[ts_code].add(publish_date)

    kept_records.sort(
        key=lambda item: (
            str(item.get("publish_date", "")),
            str(item.get("primary_ts_code", "")),
            -float(item.get("thesis_relevance_score", 0.0)),
            str(item.get("id", "")),
        )
    )

    with output.open("w", encoding="utf-8") as writer:
        for record in kept_records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    if rejected_file is not None:
        with rejected_file.open("w", encoding="utf-8") as writer:
            for record in rejected_records:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "total": total,
        "kept": len(kept_records),
        "rejected": len(rejected_records),
        "counts_by_stock": dict(sorted(counts_by_stock.items())),
        "days_by_stock": {
            ts_code: len(sorted(day_set)) for ts_code, day_set in sorted(days_by_stock.items())
        },
        "group_count": len(grouped),
    }
