from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


STRONG_FINANCE_KEYWORDS = [
    "股票",
    "a股",
    "港股",
    "美股",
    "沪指",
    "深成指",
    "创业板",
    "科创板",
    "北交所",
    "证券",
    "券商",
    "上市公司",
    "财报",
    "业绩",
    "净利润",
    "营收",
    "扣非",
    "公告",
    "回购",
    "增持",
    "减持",
    "分红",
    "并购",
    "收购",
    "重组",
    "定增",
    "募资",
    "融资",
    "ipo",
    "市值",
    "估值",
    "股价",
    "涨停",
    "跌停",
    "龙虎榜",
    "停牌",
    "复牌",
]

WEAK_FINANCE_KEYWORDS = [
    "银行",
    "白酒",
    "新能源",
    "电池",
    "锂电",
    "光伏",
    "半导体",
    "芯片",
    "算力",
    "人工智能",
    "ai",
    "医药",
    "创新药",
    "军工",
    "地产",
    "汽车",
    "储能",
    "煤炭",
    "钢铁",
    "家电",
    "消费电子",
    "机器人",
    "稀土",
    "黄金",
    "原油",
    "期货",
    "基金",
    "信托",
    "保险",
]

EXCLUDE_KEYWORDS = [
    "nba",
    "cba",
    "世界杯",
    "欧冠",
    "英超",
    "足球",
    "篮球",
    "羽毛球",
    "网球",
    "比赛",
    "电影",
    "电视剧",
    "综艺",
    "明星",
    "演员",
    "歌手",
    "演唱会",
    "高考",
    "中考",
    "旅游",
    "景区",
    "航班",
    "酒店",
    "美食",
    "游戏",
    "动漫",
]

STRICT_EXCLUDE_KEYWORDS = [
    "税务局",
    "省委",
    "省政府",
    "政协",
    "人大",
    "乡村振兴",
    "现场观摩",
    "主题教育",
    "工作会议",
    "文旅",
    "景区",
]

COMPANY_OR_MARKET_HINTS = [
    "公司",
    "企业",
    "集团",
    "股份",
    "有限合伙",
    "上市",
    "板块",
    "龙头",
    "行情",
    "投资者",
    "机构",
    "资本市场",
    "市值",
    "证券时报",
    "财联社",
]


@dataclass
class FilterDecision:
    keep: bool
    score: float
    strong_hits: list[str]
    weak_hits: list[str]
    exclude_hits: list[str]


def _collect_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


def score_stock_news(title: str, body: str) -> FilterDecision:
    title_text = (title or "").lower()
    body_text = (body or "").lower()
    merged_text = f"{title_text}\n{body_text}"

    strong_hits = _collect_hits(merged_text, STRONG_FINANCE_KEYWORDS)
    weak_hits = _collect_hits(merged_text, WEAK_FINANCE_KEYWORDS)
    exclude_hits = _collect_hits(merged_text, EXCLUDE_KEYWORDS)

    score = 0.0
    for keyword in strong_hits:
        score += 3.0 if keyword.lower() in title_text else 2.0
    for keyword in weak_hits:
        score += 1.5 if keyword.lower() in title_text else 1.0
    for keyword in exclude_hits:
        score -= 3.0 if keyword.lower() in title_text else 2.0

    has_strong_signal = bool(strong_hits)
    has_weak_cluster = len(weak_hits) >= 2
    keep = (has_strong_signal or has_weak_cluster or score >= 3.0) and not (
        exclude_hits and not has_strong_signal and score < 4.0
    )
    return FilterDecision(
        keep=keep,
        score=round(score, 4),
        strong_hits=strong_hits,
        weak_hits=weak_hits,
        exclude_hits=exclude_hits,
    )


def score_stock_news_strict(title: str, body: str) -> FilterDecision:
    title_text = (title or "").lower()
    body_text = (body or "").lower()
    merged_text = f"{title_text}\n{body_text}"

    strong_hits = _collect_hits(merged_text, STRONG_FINANCE_KEYWORDS)
    weak_hits = _collect_hits(merged_text, WEAK_FINANCE_KEYWORDS)
    exclude_hits = _collect_hits(merged_text, EXCLUDE_KEYWORDS)
    strict_exclude_hits = _collect_hits(merged_text, STRICT_EXCLUDE_KEYWORDS)
    company_hits = _collect_hits(merged_text, COMPANY_OR_MARKET_HINTS)

    score = 0.0
    for keyword in strong_hits:
        score += 3.5 if keyword.lower() in title_text else 2.5
    for keyword in weak_hits:
        score += 1.2 if keyword.lower() in title_text else 0.8
    for keyword in company_hits:
        score += 0.8 if keyword.lower() in title_text else 0.4
    for keyword in exclude_hits:
        score -= 3.0 if keyword.lower() in title_text else 2.0
    for keyword in strict_exclude_hits:
        score -= 2.5 if keyword.lower() in title_text else 1.5

    has_strong_signal = bool(strong_hits)
    has_company_market_signal = bool(company_hits)
    keep = False
    if has_strong_signal and score >= 3.0:
        keep = True
    elif has_company_market_signal and len(weak_hits) >= 2 and score >= 3.5:
        keep = True
    elif score >= 5.0:
        keep = True

    if strict_exclude_hits and not has_strong_signal and score < 6.0:
        keep = False

    return FilterDecision(
        keep=keep,
        score=round(score, 4),
        strong_hits=strong_hits,
        weak_hits=weak_hits + company_hits,
        exclude_hits=exclude_hits + strict_exclude_hits,
    )


def filter_stock_news_file(
    input_path: str | Path,
    output_path: str | Path,
    rejected_path: str | Path | None = None,
    mode: str = "broad",
) -> dict[str, int]:
    source = Path(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    rejected_file = Path(rejected_path) if rejected_path else None
    if rejected_file is not None:
        rejected_file.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    rejected = 0

    with source.open("r", encoding="utf-8") as reader, target.open(
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
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    rejected += 1
                    continue

                title = str(record.get("title", "") or "")
                body = str(record.get("body", "") or "")
                decision = (
                    score_stock_news_strict(title, body)
                    if mode == "strict"
                    else score_stock_news(title, body)
                )
                record["filter_score"] = decision.score
                record["matched_strong_keywords"] = "|".join(decision.strong_hits)
                record["matched_weak_keywords"] = "|".join(decision.weak_hits)
                record["matched_exclude_keywords"] = "|".join(decision.exclude_hits)

                if decision.keep:
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    rejected += 1
                    if rejected_writer is not None:
                        rejected_writer.write(
                            json.dumps(record, ensure_ascii=False) + "\n"
                        )
        finally:
            if rejected_writer is not None:
                rejected_writer.close()

    return {"total": total, "kept": kept, "rejected": rejected}
