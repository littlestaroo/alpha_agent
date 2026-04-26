from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from news_quant.config import DEFAULT_BODY_CHARS, MOCK_LLM
from news_quant.data_loader import filter_news_rows, load_news_table, load_stock_universe
from news_quant.llm import call_llm_json
from news_quant.schemas import ArticleAnalysis, StockMention

BASELINE_SYSTEM_PROMPT = """你是金融文本舆情分析助手。目标是给“股票舆情因子挖掘 baseline”生成结构化结果。

请严格遵守：
1. 只能从候选股票列表中选择 ts_code 和 stock_name，不要新增股票。
2. 只有当新闻与候选股票真正相关时才输出到 mentions；如果都不相关，mentions 返回空列表。
3. 对每个相关股票输出：
   - 同一只股票在单条新闻中只能输出一条 mention；如果涉及多个事件，请合并为一个综合结果
   - event_type：只能是 业绩 / 经营 / 投融资 / 政策 / 风险 / 市场舆情 / 产品 / 其他
   - event_summary：一句话事件摘要
   - sentiment：[-1,1]，负面为负，正面为正
   - sentiment_strength：[0,1]
   - confidence：[0,1]
   - relevance：[0,1]
   - event_importance：[0,1]
   - risk_flag：0 或 1
   - trend_signal：上升 / 下降 / 震荡 / 未知
   - keywords：3 到 6 个关键词
4. article_summary 为全文摘要，overall_sentiment 为全文整体情绪。
5. 输出严格 JSON，不要 markdown，不要解释，不要额外字段。
"""

ARTICLE_COLUMNS = [
    "news_id",
    "publish_time",
    "publish_date",
    "source",
    "url",
    "title",
    "article_summary",
    "overall_sentiment",
    "ts_code",
    "stock_name",
    "industry",
    "matched_aliases",
    "coarse_score",
    "event_type",
    "event_summary",
    "merged_event_types",
    "merged_event_count",
    "sentiment",
    "sentiment_strength",
    "confidence",
    "relevance",
    "event_importance",
    "risk_flag",
    "trend_signal",
    "keywords",
]

PROFILE_COLUMNS = [
    "publish_date",
    "ts_code",
    "stock_name",
    "industry",
    "article_count",
    "mention_count",
    "net_sentiment_factor",
    "negative_shock_factor",
    "attention_factor",
    "sentiment_dispersion_factor",
    "event_density_factor",
    "risk_ratio",
    "positive_ratio",
    "neutral_ratio",
    "negative_ratio",
    "dominant_event_type",
    "top_keywords",
    "sentiment_delta_1d",
    "sentiment_trend_factor",
    "sentiment_momentum_factor",
    "ema3_sentiment_state",
    "ema5_sentiment_state",
    "negative_shock_carry",
    "event_novelty_factor",
    "earnings_event_factor",
    "operations_event_factor",
    "financing_event_factor",
    "market_buzz_event_factor",
    "risk_event_factor",
    "earnings_event_state",
    "operations_event_state",
    "financing_event_state",
    "market_buzz_event_state",
    "risk_event_state",
    "state_composite_factor",
    "composite_score",
    "profile_label",
]

SKIPPED_COLUMNS = [
    "news_id",
    "publish_time",
    "title",
    "reason",
    "candidate_count",
]

EVENT_TYPE_WEIGHTS = {
    "业绩": 1.15,
    "经营": 1.0,
    "投融资": 1.05,
    "政策": 1.05,
    "风险": 1.15,
    "市场舆情": 0.95,
    "产品": 1.0,
    "其他": 0.9,
}

EVENT_FACTOR_BASES = {
    "业绩": "earnings",
    "经营": "operations",
    "投融资": "financing",
    "市场舆情": "market_buzz",
    "风险": "risk",
}

EVENT_HINTS = {
    "业绩": ("业绩", "净利润", "营收", "预增", "亏损", "财报", "盈利"),
    "经营": ("经营", "订单", "产能", "扩产", "合作", "中标", "签约"),
    "投融资": ("融资", "投资", "增持", "减持", "收购", "并购", "回购"),
    "政策": ("政策", "央行", "降准", "逆回购", "监管", "指导意见"),
    "风险": ("风险", "处罚", "调查", "诉讼", "违约", "暴跌", "下滑", "承压"),
    "市场舆情": ("市场", "舆情", "热议", "关注", "波动", "情绪"),
    "产品": ("产品", "发布", "新品", "技术", "电池", "芯片"),
}

POSITIVE_HINTS = (
    "增长",
    "提升",
    "改善",
    "中标",
    "签约",
    "获批",
    "回暖",
    "突破",
    "定点",
    "创新高",
    "恢复",
)

NEGATIVE_HINTS = (
    "下滑",
    "亏损",
    "处罚",
    "调查",
    "风险",
    "承压",
    "违约",
    "诉讼",
    "减值",
    "下跌",
    "波动",
)

RISK_HINTS = ("风险", "处罚", "调查", "诉讼", "违约", "承压", "下滑", "亏损")
TREND_UP_HINTS = ("继续", "进一步", "加速", "改善", "恢复", "增长", "回暖")
TREND_DOWN_HINTS = ("承压", "下滑", "走弱", "恶化", "波动", "风险")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def _unique_non_empty(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _coarse_match_stocks(
    title: str,
    body: str,
    universe_df: pd.DataFrame,
    max_candidates: int = 8,
) -> list[dict]:
    blob = _normalized_text(f"{title}\n{body}")
    candidates: list[dict] = []

    for _, row in universe_df.iterrows():
        aliases = row["aliases"] if isinstance(row["aliases"], list) else [row["name"]]
        matched_aliases: list[str] = []
        longest_match = 0
        for alias in aliases:
            normalized_alias = _normalized_text(alias)
            if len(normalized_alias) < 2:
                continue
            if normalized_alias and normalized_alias in blob:
                matched_aliases.append(alias)
                longest_match = max(longest_match, len(normalized_alias))

        if matched_aliases:
            coarse_score = min(
                1.0, 0.68 + 0.03 * longest_match + 0.05 * (len(matched_aliases) - 1)
            )
            candidates.append(
                {
                    "ts_code": row["ts_code"],
                    "name": row["name"],
                    "industry": row["industry"],
                    "matched_aliases": _unique_non_empty(matched_aliases),
                    "coarse_score": round(coarse_score, 4),
                }
            )

    candidates.sort(
        key=lambda item: (
            item["coarse_score"],
            len(item["matched_aliases"]),
            max((len(alias) for alias in item["matched_aliases"]), default=0),
        ),
        reverse=True,
    )
    return candidates[:max_candidates]


def _detect_event_type(text: str) -> str:
    for event_type, keywords in EVENT_HINTS.items():
        if any(keyword in text for keyword in keywords):
            return event_type
    return "其他"


def _infer_trend_signal(text: str, sentiment: float) -> str:
    if any(keyword in text for keyword in TREND_UP_HINTS):
        return "上升" if sentiment >= -0.05 else "震荡"
    if any(keyword in text for keyword in TREND_DOWN_HINTS):
        return "下降" if sentiment <= 0.05 else "震荡"
    if sentiment > 0.15:
        return "上升"
    if sentiment < -0.15:
        return "下降"
    return "震荡"


def _mock_article_analysis(
    title: str,
    body: str,
    candidates: list[dict],
) -> ArticleAnalysis:
    text = f"{title}\n{body}"
    mentions: list[StockMention] = []

    for candidate in candidates:
        positive_hits = [keyword for keyword in POSITIVE_HINTS if keyword in text]
        negative_hits = [keyword for keyword in NEGATIVE_HINTS if keyword in text]
        risk_hits = [keyword for keyword in RISK_HINTS if keyword in text]
        event_type = _detect_event_type(text)

        sentiment = 0.18 * len(positive_hits) - 0.2 * len(negative_hits)
        if any(alias in title for alias in candidate["matched_aliases"]):
            sentiment += 0.08
        sentiment = round(_clamp(sentiment, -1.0, 1.0), 4)

        sentiment_strength = round(
            _clamp(abs(sentiment) + 0.08 * int(bool(positive_hits or negative_hits)), 0.05, 1.0),
            4,
        )
        confidence = round(
            _clamp(
                0.55
                + 0.10 * len(candidate["matched_aliases"])
                + 0.06 * int(bool(positive_hits or negative_hits or risk_hits)),
                0.0,
                0.95,
            ),
            4,
        )
        relevance = round(
            _clamp((candidate["coarse_score"] + 0.75) / 2, 0.0, 1.0),
            4,
        )
        event_importance = round(
            _clamp(EVENT_TYPE_WEIGHTS.get(event_type, 0.9) / 1.15, 0.2, 1.0),
            4,
        )
        risk_flag = 1 if risk_hits or event_type == "风险" else 0
        keywords = _unique_non_empty(
            candidate["matched_aliases"] + positive_hits + negative_hits + risk_hits
        )[:6]

        mentions.append(
            StockMention(
                ts_code=candidate["ts_code"],
                stock_name=candidate["name"],
                event_type=event_type,
                event_summary=f"{candidate['name']}相关{event_type}信息",
                sentiment=sentiment,
                sentiment_strength=sentiment_strength,
                confidence=confidence,
                relevance=relevance,
                event_importance=event_importance,
                risk_flag=risk_flag,
                trend_signal=_infer_trend_signal(text, sentiment),
                keywords=keywords,
            )
        )

    if mentions:
        weight_sum = sum(max(mention.relevance, 0.05) for mention in mentions)
        overall = sum(mention.sentiment * mention.relevance for mention in mentions) / weight_sum
    else:
        overall = 0.0

    return ArticleAnalysis(
        article_summary=title[:80] if title else "新闻摘要",
        overall_sentiment=round(_clamp(overall, -1.0, 1.0), 4),
        mentions=mentions,
    )


def _format_candidates_for_prompt(candidates: list[dict]) -> str:
    lines: list[str] = []
    for candidate in candidates:
        aliases = "、".join(candidate["matched_aliases"])
        lines.append(
            f"- {candidate['ts_code']} | {candidate['name']} | 行业:{candidate['industry']} | 命中别名:{aliases}"
        )
    return "\n".join(lines)


def _normalize_analysis_result(
    analysis: ArticleAnalysis,
    candidates: list[dict],
    fallback_title: str,
) -> ArticleAnalysis:
    candidate_map = {candidate["ts_code"]: candidate for candidate in candidates}
    normalized_mentions: list[StockMention] = []

    for mention in analysis.mentions:
        candidate = candidate_map.get(mention.ts_code)
        if not candidate:
            continue
        mention.stock_name = candidate["name"]
        mention.sentiment = round(_clamp(float(mention.sentiment), -1.0, 1.0), 4)
        mention.sentiment_strength = round(
            _clamp(float(mention.sentiment_strength or abs(mention.sentiment)), 0.0, 1.0),
            4,
        )
        mention.confidence = round(_clamp(float(mention.confidence), 0.0, 1.0), 4)
        mention.relevance = round(
            _clamp((float(mention.relevance) + candidate["coarse_score"]) / 2, 0.0, 1.0),
            4,
        )
        mention.event_importance = round(
            _clamp(float(mention.event_importance), 0.0, 1.0),
            4,
        )
        mention.risk_flag = 1 if int(mention.risk_flag) else 0
        mention.keywords = _unique_non_empty(mention.keywords + candidate["matched_aliases"])[:6]
        if not mention.event_summary.strip():
            mention.event_summary = f"{candidate['name']}相关{mention.event_type}舆情"
        normalized_mentions.append(mention)

    article_summary = analysis.article_summary.strip() or fallback_title[:80]
    overall_sentiment = round(_clamp(float(analysis.overall_sentiment), -1.0, 1.0), 4)

    return ArticleAnalysis(
        article_summary=article_summary,
        overall_sentiment=overall_sentiment,
        mentions=normalized_mentions,
    )


def _mention_merge_weight(mention: StockMention) -> float:
    return (
        max(float(mention.relevance), 0.05)
        * max(float(mention.confidence), 0.05)
        * max(float(mention.event_importance), 0.1)
        * (0.5 + 0.5 * max(float(mention.sentiment_strength), 0.0))
        * EVENT_TYPE_WEIGHTS.get(str(mention.event_type), 0.9)
    )


def _weighted_average(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 1e-6:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def _merge_mentions_by_stock(mentions: list[StockMention]) -> list[dict[str, object]]:
    grouped: dict[str, list[StockMention]] = {}
    for mention in mentions:
        grouped.setdefault(mention.ts_code, []).append(mention)

    merged_rows: list[dict[str, object]] = []
    for stock_mentions in grouped.values():
        weights = [_mention_merge_weight(mention) for mention in stock_mentions]
        ranked_mentions = [
            mention
            for _, mention in sorted(
                zip(weights, stock_mentions, strict=False),
                key=lambda item: item[0],
                reverse=True,
            )
        ]
        representative = ranked_mentions[0]

        event_type_scores: dict[str, float] = {}
        for mention, weight in zip(stock_mentions, weights):
            event_type_scores[mention.event_type] = (
                event_type_scores.get(mention.event_type, 0.0) + weight
            )
        merged_event_types = [
            event_type
            for event_type, _ in sorted(
                event_type_scores.items(),
                key=lambda item: (item[1], EVENT_TYPE_WEIGHTS.get(item[0], 0.9)),
                reverse=True,
            )
        ]
        dominant_event_type = merged_event_types[0]

        summary_candidates = _unique_non_empty(
            [mention.event_summary for mention in ranked_mentions]
        )
        if len(summary_candidates) <= 1:
            event_summary = summary_candidates[0] if summary_candidates else representative.event_summary
        elif len(summary_candidates) == 2:
            event_summary = "；".join(summary_candidates)
        else:
            event_summary = "；".join(summary_candidates[:2]) + "等"

        trend_signals = {
            mention.trend_signal
            for mention in stock_mentions
            if mention.trend_signal and mention.trend_signal != "未知"
        }
        if len(trend_signals) == 1:
            trend_signal = next(iter(trend_signals))
        elif len(trend_signals) > 1:
            trend_signal = "震荡"
        else:
            trend_signal = representative.trend_signal

        merged_keywords = _unique_non_empty(
            [keyword for mention in ranked_mentions for keyword in mention.keywords]
        )[:6]

        merged_mention = StockMention(
            ts_code=representative.ts_code,
            stock_name=representative.stock_name,
            event_type=dominant_event_type,
            event_summary=event_summary,
            sentiment=round(
                _clamp(
                    _weighted_average(
                        [float(mention.sentiment) for mention in stock_mentions], weights
                    ),
                    -1.0,
                    1.0,
                ),
                4,
            ),
            sentiment_strength=round(
                _clamp(
                    _weighted_average(
                        [float(mention.sentiment_strength) for mention in stock_mentions],
                        weights,
                    ),
                    0.0,
                    1.0,
                ),
                4,
            ),
            confidence=round(
                _clamp(
                    _weighted_average(
                        [float(mention.confidence) for mention in stock_mentions], weights
                    ),
                    0.0,
                    1.0,
                ),
                4,
            ),
            relevance=round(
                _clamp(
                    _weighted_average(
                        [float(mention.relevance) for mention in stock_mentions], weights
                    ),
                    0.0,
                    1.0,
                ),
                4,
            ),
            event_importance=round(
                _clamp(
                    max(float(mention.event_importance) for mention in stock_mentions),
                    0.0,
                    1.0,
                ),
                4,
            ),
            risk_flag=1 if any(int(mention.risk_flag) for mention in stock_mentions) else 0,
            trend_signal=trend_signal,
            keywords=merged_keywords,
        )
        merged_rows.append(
            {
                "mention": merged_mention,
                "merged_event_types": "|".join(merged_event_types),
                "merged_event_count": len(stock_mentions),
            }
        )

    return merged_rows


def analyze_article(
    title: str,
    body: str,
    candidates: list[dict],
    max_body_chars: int = DEFAULT_BODY_CHARS,
) -> ArticleAnalysis:
    trimmed_body = body[:max_body_chars]
    if MOCK_LLM:
        return _mock_article_analysis(title, trimmed_body, candidates)

    user_prompt = f"""候选股票列表：
{_format_candidates_for_prompt(candidates)}

请只在上述候选股票中做判断。

新闻标题：{title}
新闻正文：{trimmed_body}
"""
    raw_analysis = call_llm_json(BASELINE_SYSTEM_PROMPT, user_prompt, ArticleAnalysis)
    return _normalize_analysis_result(raw_analysis, candidates, title)


def _split_joined_keywords(raw_keywords: pd.Series) -> list[str]:
    values: list[str] = []
    for raw in raw_keywords.fillna("").astype(str):
        values.extend(part.strip() for part in raw.split("|") if part.strip())
    return values


def _mention_weight(row: pd.Series) -> float:
    event_weight = EVENT_TYPE_WEIGHTS.get(str(row["event_type"]), 0.9)
    return (
        max(float(row["relevance"]), 0.05)
        * max(float(row["confidence"]), 0.05)
        * max(float(row["event_importance"]), 0.1)
        * (0.5 + 0.5 * max(float(row["sentiment_strength"]), 0.0))
        * event_weight
    )


def _build_profile_label(row: pd.Series) -> str:
    attention_label = (
        "高关注"
        if row["attention_factor"] >= 0.60
        else "中关注"
        if row["attention_factor"] >= 0.35
        else "低关注"
    )
    tone_label = (
        "偏正面"
        if row["net_sentiment_factor"] >= 0.20
        else "偏负面"
        if row["net_sentiment_factor"] <= -0.20
        else "中性偏稳"
    )
    trend_label = (
        "情绪上行"
        if row["sentiment_momentum_factor"] >= 0.15
        else "情绪走弱"
        if row["sentiment_momentum_factor"] <= -0.15
        else "趋势平稳"
    )
    risk_label = (
        "风险升温"
        if row["negative_shock_factor"] >= 0.35 or row["risk_ratio"] >= 0.40
        else "风险可控"
    )
    return f"{attention_label}-{tone_label}-{trend_label}-{risk_label}"


def _build_decay_carry(series: pd.Series, decay: float = 0.65) -> pd.Series:
    carry = 0.0
    values: list[float] = []
    for item in series.fillna(0.0).astype(float):
        carry = decay * carry + item
        values.append(carry)
    return pd.Series(values, index=series.index)


def _safe_ewm(series: pd.Series, span: int) -> pd.Series:
    return series.fillna(0.0).astype(float).ewm(span=span, adjust=False).mean()


def build_daily_profiles(
    mentions_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    if mentions_df.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)

    df = mentions_df.copy()
    df = df[df["publish_date"].astype(str).str.len() > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=PROFILE_COLUMNS)

    df["weight"] = df.apply(_mention_weight, axis=1)
    industry_map = universe_df.set_index("ts_code")["industry"].to_dict()

    rows: list[dict] = []
    group_columns = ["publish_date", "ts_code", "stock_name"]
    for (publish_date, ts_code, stock_name), group in df.groupby(group_columns, sort=True):
        weights = group["weight"]
        weight_sum = max(float(weights.sum()), 1e-6)
        net_sentiment = float((group["sentiment"] * weights).sum() / weight_sum)
        negative_shock = float(((-group["sentiment"]).clip(lower=0) * weights).sum() / weight_sum)
        attention_raw = math.log1p(group["news_id"].nunique() + len(group) / 2) * float(
            group["relevance"].mean()
        )
        attention_factor = math.tanh(attention_raw / 2.0)
        dispersion = float(group["sentiment"].std(ddof=0) or 0.0)
        event_density = math.tanh(len(group) / 3.0)
        positive_ratio = float((group["sentiment"] > 0.15).mean())
        negative_ratio = float((group["sentiment"] < -0.15).mean())
        neutral_ratio = max(0.0, 1.0 - positive_ratio - negative_ratio)
        risk_ratio = float(group["risk_flag"].mean())
        dominant_event_type = Counter(group["event_type"].astype(str)).most_common(1)[0][0]
        top_keywords = "|".join(
            keyword for keyword, _ in Counter(_split_joined_keywords(group["keywords"])).most_common(5)
        )

        event_factor_values: dict[str, float] = {}
        for event_type, base_name in EVENT_FACTOR_BASES.items():
            event_mask = group["event_type"].astype(str) == event_type
            event_factor = float((group.loc[event_mask, "sentiment"] * weights.loc[event_mask]).sum() / weight_sum)
            event_factor_values[f"{base_name}_event_factor"] = round(event_factor, 4)

        rows.append(
            {
                "publish_date": publish_date,
                "ts_code": ts_code,
                "stock_name": stock_name,
                "industry": industry_map.get(ts_code, ""),
                "article_count": int(group["news_id"].nunique()),
                "mention_count": int(len(group)),
                "net_sentiment_factor": round(net_sentiment, 4),
                "negative_shock_factor": round(negative_shock, 4),
                "attention_factor": round(attention_factor, 4),
                "sentiment_dispersion_factor": round(dispersion, 4),
                "event_density_factor": round(event_density, 4),
                "risk_ratio": round(risk_ratio, 4),
                "positive_ratio": round(positive_ratio, 4),
                "neutral_ratio": round(neutral_ratio, 4),
                "negative_ratio": round(negative_ratio, 4),
                "dominant_event_type": dominant_event_type,
                "top_keywords": top_keywords,
                **event_factor_values,
            }
        )

    profiles_df = pd.DataFrame(rows)
    profiles_df["publish_date"] = pd.to_datetime(profiles_df["publish_date"])
    profiles_df = profiles_df.sort_values(["ts_code", "publish_date"]).reset_index(drop=True)

    grouped = profiles_df.groupby("ts_code", sort=False)
    profiles_df["sentiment_delta_1d"] = grouped["net_sentiment_factor"].diff().fillna(0.0)
    profiles_df["sentiment_trend_factor"] = grouped["net_sentiment_factor"].transform(
        lambda series: (series - series.shift(3)).fillna(0.0)
    )
    profiles_df["sentiment_momentum_factor"] = grouped["net_sentiment_factor"].transform(
        lambda series: (series - series.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    )
    profiles_df["ema3_sentiment_state"] = grouped["net_sentiment_factor"].transform(
        lambda series: _safe_ewm(series, span=3)
    )
    profiles_df["ema5_sentiment_state"] = grouped["net_sentiment_factor"].transform(
        lambda series: _safe_ewm(series, span=5)
    )
    profiles_df["negative_shock_carry"] = grouped["negative_shock_factor"].transform(
        lambda series: _build_decay_carry(series, decay=0.65)
    )
    profiles_df["event_novelty_factor"] = grouped["net_sentiment_factor"].transform(
        lambda series: (
            series.fillna(0.0).astype(float)
            - series.fillna(0.0).astype(float).shift(1).rolling(5, min_periods=1).mean()
        ).fillna(0.0)
    )

    for base_name in EVENT_FACTOR_BASES.values():
        factor_col = f"{base_name}_event_factor"
        state_col = f"{base_name}_event_state"
        profiles_df[state_col] = grouped[factor_col].transform(
            lambda series: _safe_ewm(series, span=5)
        )

    profiles_df["state_composite_factor"] = (
        0.35 * profiles_df["ema3_sentiment_state"]
        + 0.10 * profiles_df["ema5_sentiment_state"]
        - 0.20 * profiles_df["negative_shock_carry"]
        + 0.10 * profiles_df["attention_factor"]
        + 0.08 * profiles_df["earnings_event_state"]
        + 0.08 * profiles_df["operations_event_state"]
        + 0.07 * profiles_df["financing_event_state"]
        + 0.05 * profiles_df["market_buzz_event_state"]
        - 0.08 * profiles_df["risk_event_state"]
        - 0.05 * profiles_df["sentiment_dispersion_factor"]
    ).clip(-1.0, 1.0)
    profiles_df["composite_score"] = (
        0.45 * profiles_df["net_sentiment_factor"]
        - 0.25 * profiles_df["negative_shock_factor"]
        + 0.15 * profiles_df["attention_factor"]
        + 0.15 * profiles_df["sentiment_momentum_factor"].clip(-1.0, 1.0)
        - 0.10 * profiles_df["sentiment_dispersion_factor"]
    ).clip(-1.0, 1.0)
    profiles_df["profile_label"] = profiles_df.apply(_build_profile_label, axis=1)

    rounded_columns = [
        "sentiment_delta_1d",
        "sentiment_trend_factor",
        "sentiment_momentum_factor",
        "ema3_sentiment_state",
        "ema5_sentiment_state",
        "negative_shock_carry",
        "event_novelty_factor",
        "earnings_event_factor",
        "operations_event_factor",
        "financing_event_factor",
        "market_buzz_event_factor",
        "risk_event_factor",
        "earnings_event_state",
        "operations_event_state",
        "financing_event_state",
        "market_buzz_event_state",
        "risk_event_state",
        "state_composite_factor",
        "composite_score",
    ]
    for column in rounded_columns:
        profiles_df[column] = profiles_df[column].round(4)

    profiles_df["publish_date"] = profiles_df["publish_date"].dt.strftime("%Y-%m-%d")
    return profiles_df[PROFILE_COLUMNS]


def run_baseline(
    news_path: str | Path,
    universe_path: str | Path,
    output_path: str | Path | None = None,
    keywords: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int | None = None,
    max_candidates: int = 8,
    max_body_chars: int = DEFAULT_BODY_CHARS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    news_df = filter_news_rows(
        load_news_table(news_path),
        keywords=keywords,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )
    universe_df = load_stock_universe(universe_path)

    mention_rows: list[dict] = []
    skipped_rows: list[dict] = []

    total_news = len(news_df)

    for idx, (_, news_row) in enumerate(news_df.iterrows(), start=1):
        title = str(news_row["title"])
        body = str(news_row["body"])
        candidates = _coarse_match_stocks(
            title,
            body,
            universe_df,
            max_candidates=max_candidates,
        )

        if not candidates:
            skipped_rows.append(
                {
                    "news_id": news_row["id"],
                    "publish_time": news_row["publish_time"],
                    "title": title,
                    "reason": "no_alias_match",
                    "candidate_count": 0,
                }
            )
            continue

        try:
            analysis = analyze_article(
                title=title,
                body=body,
                candidates=candidates,
                max_body_chars=max_body_chars,
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "news_id": news_row["id"],
                    "publish_time": news_row["publish_time"],
                    "title": title,
                    "reason": f"llm_error:{type(exc).__name__}",
                    "candidate_count": len(candidates),
                }
            )
            continue
        merged_mentions = _merge_mentions_by_stock(analysis.mentions)
        if not merged_mentions:
            skipped_rows.append(
                {
                    "news_id": news_row["id"],
                    "publish_time": news_row["publish_time"],
                    "title": title,
                    "reason": "no_relevant_stock_after_llm",
                    "candidate_count": len(candidates),
                }
            )
            continue

        candidate_map = {candidate["ts_code"]: candidate for candidate in candidates}
        for merged in merged_mentions:
            mention = merged["mention"]
            candidate = candidate_map.get(mention.ts_code)
            if not candidate:
                continue
            mention_rows.append(
                {
                    "news_id": news_row["id"],
                    "publish_time": news_row["publish_time"],
                    "publish_date": news_row["publish_date"],
                    "source": news_row["source"],
                    "url": news_row["url"],
                    "title": title,
                    "article_summary": analysis.article_summary,
                    "overall_sentiment": analysis.overall_sentiment,
                    "ts_code": mention.ts_code,
                    "stock_name": mention.stock_name,
                    "industry": candidate["industry"],
                    "matched_aliases": "|".join(candidate["matched_aliases"]),
                    "coarse_score": candidate["coarse_score"],
                    "event_type": mention.event_type,
                    "event_summary": mention.event_summary,
                    "merged_event_types": merged["merged_event_types"],
                    "merged_event_count": merged["merged_event_count"],
                    "sentiment": mention.sentiment,
                    "sentiment_strength": mention.sentiment_strength,
                    "confidence": mention.confidence,
                    "relevance": mention.relevance,
                    "event_importance": mention.event_importance,
                    "risk_flag": mention.risk_flag,
                    "trend_signal": mention.trend_signal,
                    "keywords": "|".join(mention.keywords),
                }
            )

        if idx == 1 or idx % 10 == 0 or idx == total_news:
            print(
                f"[baseline] processed {idx}/{total_news} news, "
                f"mentions={len(mention_rows)}, skipped={len(skipped_rows)}",
                flush=True,
            )

    mentions_df = pd.DataFrame(mention_rows, columns=ARTICLE_COLUMNS)
    skipped_df = pd.DataFrame(skipped_rows, columns=SKIPPED_COLUMNS)
    profiles_df = build_daily_profiles(mentions_df, universe_df)

    if output_path:
        output_prefix = Path(output_path)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        mentions_df.to_csv(
            output_prefix.with_name(output_prefix.stem + "_article_mentions.csv"),
            index=False,
        )
        profiles_df.to_csv(
            output_prefix.with_name(output_prefix.stem + "_daily_profiles.csv"),
            index=False,
        )
        skipped_df.to_csv(
            output_prefix.with_name(output_prefix.stem + "_skipped_news.csv"),
            index=False,
        )

    return mentions_df, profiles_df, skipped_df
