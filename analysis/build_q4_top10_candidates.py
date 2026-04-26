from __future__ import annotations

import csv
import heapq
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/star/Desktop/agent")
RAW_DIR = ROOT / "data" / "raw" / "opennewsarchive"
UNIVERSE_PATH = ROOT / "data" / "stock_universe_thesis_20stocks.csv"
OUTPUT_PATH = ROOT / "data" / "prepared" / "opennewsarchive_thesis_20stocks_2023Q4_top10_perstock.jsonl"
SUMMARY_PATH = ROOT / "data" / "prepared" / "opennewsarchive_thesis_20stocks_2023Q4_top10_perstock_summary.json"

DATE_FROM = "2023-10-01"
DATE_TO = "2023-12-31"
LANGUAGE = "zh"
TOP_K = 10
BODY_EXCERPT_CHARS = 1600

FINANCE_KEYWORDS = [
    "股票",
    "a股",
    "港股",
    "美股",
    "公告",
    "财报",
    "业绩",
    "营收",
    "净利润",
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

NOISE_KEYWORDS = [
    "政府",
    "市委",
    "省政府",
    "人大",
    "政协",
    "招商",
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
    "乡村振兴",
    "考察",
    "观摩",
    "推介会",
]

AMBIGUOUS_SHORT_ALIASES = {
    "平安",
    "中信",
    "伊利",
    "美的",
    "三一",
    "海康",
    "神华",
    "中芯",
    "紫金",
    "恒瑞",
    "招行",
    "工行",
}

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub("", str(text or "")).lower()


def load_universe(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            name = (raw.get("name") or "").strip()
            aliases = [(raw.get("aliases") or "").strip()]
            alias_values: list[str] = []
            for alias_group in aliases:
                alias_values.extend(part.strip() for part in alias_group.split("|") if part.strip())
            alias_values.append(name)
            dedup: list[tuple[str, str]] = []
            seen: set[str] = set()
            for alias in alias_values:
                norm = normalize_text(alias)
                if not norm or len(norm) < 2 or norm in seen:
                    continue
                seen.add(norm)
                dedup.append((alias, norm))
            rows.append(
                {
                    "ts_code": (raw.get("ts_code") or "").strip(),
                    "name": name,
                    "industry": (raw.get("industry") or "").strip(),
                    "aliases": dedup,
                    "full_name_norm": normalize_text(name),
                }
            )
    return rows


def iter_jsonl_records(raw_dir: Path):
    for file_path in sorted(raw_dir.rglob("*.jsonl")):
        if "/en/" in str(file_path):
            continue
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record


def build_alias_index(universe: list[dict[str, object]]) -> tuple[dict[str, list[int]], re.Pattern[str]]:
    alias_to_stocks: dict[str, list[int]] = defaultdict(list)
    for idx, stock in enumerate(universe):
        for _, alias_norm in stock["aliases"]:  # type: ignore[index]
            alias_to_stocks[alias_norm].append(idx)
    pattern = re.compile("|".join(sorted((re.escape(alias) for alias in alias_to_stocks), key=len, reverse=True)))
    return alias_to_stocks, pattern


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


def score_stock_match(record: dict[str, str], stock: dict[str, object]) -> dict[str, object] | None:
    title = str(record.get("title", "") or "")
    body = str(record.get("body", "") or "")
    merged = normalize_text(f"{title}\n{body}")
    title_norm = normalize_text(title)
    body_norm = normalize_text(body)

    matched_aliases: list[str] = []
    title_hits = 0
    body_hits = 0
    max_alias_len = 0
    has_full_name = False

    for alias, alias_norm in stock["aliases"]:  # type: ignore[index]
        if alias_norm in title_norm:
            matched_aliases.append(alias)
            title_hits += 1
            max_alias_len = max(max_alias_len, len(alias_norm))
            if alias_norm == stock["full_name_norm"]:
                has_full_name = True
        elif alias_norm in body_norm:
            matched_aliases.append(alias)
            body_hits += 1
            max_alias_len = max(max_alias_len, len(alias_norm))
            if alias_norm == stock["full_name_norm"]:
                has_full_name = True

    if not matched_aliases:
        return None

    finance_hits = keyword_hits(f"{title}\n{body}", FINANCE_KEYWORDS)
    noise_hits = keyword_hits(f"{title}\n{body}", NOISE_KEYWORDS)

    score = 0.0
    if has_full_name:
        score += 8.0
    score += title_hits * 4.0 + body_hits * 1.5
    score += min(len(finance_hits), 6) * 0.8
    score -= min(len(noise_hits), 4) * 0.8
    if max_alias_len >= 4:
        score += 1.0
    elif max_alias_len == 3:
        score += 0.4
    else:
        score -= 0.2
    if not title.strip():
        score -= 0.5

    # Short and ambiguous aliases need extra support from finance context.
    if not has_full_name and any(alias in AMBIGUOUS_SHORT_ALIASES for alias in matched_aliases):
        score -= 1.0
        if len(finance_hits) < 2 and title_hits == 0:
            return None

    if not has_full_name and len(finance_hits) == 0 and title_hits == 0:
        return None
    if score < 2.5:
        return None

    return {
        "matched_aliases": matched_aliases,
        "finance_hits": finance_hits[:8],
        "noise_hits": noise_hits[:8],
        "has_full_name": int(has_full_name),
        "score": round(score, 4),
        "title_hits": title_hits,
        "body_hits": body_hits,
    }


def main() -> None:
    universe = load_universe(UNIVERSE_PATH)
    alias_to_stocks, alias_pattern = build_alias_index(universe)
    groups: dict[tuple[str, str], list[tuple[float, str, dict[str, object]]]] = defaultdict(list)
    seen_keys: set[tuple[str, str]] = set()
    total = 0
    kept_candidates = 0
    matched_articles = 0
    prefiltered_hits = 0

    for raw in iter_jsonl_records(RAW_DIR):
        total += 1
        language = str(raw.get("language", "") or "").lower()
        publish_date = str(raw.get("date", "") or "")
        title = str(raw.get("title", "") or "")
        body = str(raw.get("content", "") or "")
        news_id = str(raw.get("id", "") or "")

        if LANGUAGE and language != LANGUAGE:
            continue
        if not publish_date or publish_date < DATE_FROM or publish_date > DATE_TO:
            continue
        if not title and not body:
            continue

        excerpt = f"{title}\n{body[:BODY_EXCERPT_CHARS]}"
        excerpt_norm = normalize_text(excerpt)
        alias_hits = set(alias_pattern.findall(excerpt_norm))
        if not alias_hits:
            continue
        candidate_stock_indexes = sorted(
            {
                stock_idx
                for alias in alias_hits
                for stock_idx in alias_to_stocks.get(alias, [])
            }
        )
        if not candidate_stock_indexes:
            continue
        prefiltered_hits += 1

        matched_this_article = False
        for stock_idx in candidate_stock_indexes:
            stock = universe[stock_idx]
            decision = score_stock_match(
                {
                    "title": title,
                    "body": body[:BODY_EXCERPT_CHARS],
                },
                stock,
            )
            if decision is None:
                continue

            ts_code = str(stock["ts_code"])
            unique_key = (news_id, ts_code)
            if unique_key in seen_keys:
                continue
            seen_keys.add(unique_key)

            matched_this_article = True
            kept_candidates += 1
            group_key = (publish_date, ts_code)
            candidate = {
                "id": f"{news_id}__{ts_code}",
                "orig_news_id": news_id,
                "title": title,
                "body": body,
                "publish_date": publish_date,
                "publish_time": publish_date,
                "language": language,
                "source": "",
                "url": "",
                "raw_source_file": "",
                "primary_ts_code": ts_code,
                "primary_stock_name": str(stock["name"]),
                "industry": str(stock["industry"]),
                "matched_aliases": "|".join(decision["matched_aliases"]),  # type: ignore[index]
                "matched_stock_count": 1,
                "thesis_relevance_score": decision["score"],
                "thesis_has_full_name": decision["has_full_name"],
                "thesis_finance_hits": "|".join(decision["finance_hits"]),  # type: ignore[index]
                "thesis_negative_hits": "|".join(decision["noise_hits"]),  # type: ignore[index]
                "title_hit_count": decision["title_hits"],
                "body_hit_count": decision["body_hits"],
            }

            heap = groups[group_key]
            item = (float(decision["score"]), news_id, candidate)
            if len(heap) < TOP_K:
                heapq.heappush(heap, item)
            else:
                if item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        if matched_this_article:
            matched_articles += 1

        if total % 500000 == 0:
            print(
                json.dumps(
                    {
                        "raw_articles_scanned": total,
                        "prefiltered_hits": prefiltered_hits,
                        "matched_articles": matched_articles,
                        "candidate_matches_before_topk": kept_candidates,
                        "groups_so_far": len(groups),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts_by_stock: dict[str, int] = defaultdict(int)
    days_by_stock: dict[str, set[str]] = defaultdict(set)
    kept_records: list[dict[str, object]] = []

    for (publish_date, ts_code), heap in sorted(groups.items()):
        rows = [item[2] for item in sorted(heap, key=lambda x: (-x[0], x[1]))]
        for row in rows:
            kept_records.append(row)
            counts_by_stock[ts_code] += 1
            days_by_stock[ts_code].add(publish_date)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in kept_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "raw_articles_scanned": total,
        "prefiltered_hits": prefiltered_hits,
        "matched_articles": matched_articles,
        "candidate_matches_before_topk": kept_candidates,
        "topk_records_written": len(kept_records),
        "group_count": len(groups),
        "top_k": TOP_K,
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "counts_by_stock": dict(sorted(counts_by_stock.items())),
        "days_by_stock": {k: len(v) for k, v in sorted(days_by_stock.items())},
        "output_path": str(OUTPUT_PATH),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
