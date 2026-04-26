from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    name: str
    label: str
    group: str
    weight: float
    direction: int = 1

    @classmethod
    def from_mapping(cls, item: dict[str, object]) -> "FactorSpec":
        return cls(
            name=str(item["name"]),
            label=str(item.get("label") or item["name"]),
            group=str(item.get("group") or "custom"),
            weight=float(item.get("weight", 1.0)),
            direction=1 if int(item.get("direction", 1)) >= 0 else -1,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_FACTOR_SPECS = [
    FactorSpec("net_sentiment_factor", "当日净情绪", "direct", 0.08, 1),
    FactorSpec("negative_shock_factor", "当日负面冲击", "direct", 0.04, -1),
    FactorSpec("attention_factor", "当日关注度", "direct", 0.08, 1),
    FactorSpec("composite_score", "基础综合分", "direct", 0.08, 1),
    FactorSpec("ema5_sentiment_state", "5日情绪状态", "state", 0.14, 1),
    FactorSpec("negative_shock_carry", "负面冲击记忆", "state", 0.09, -1),
    FactorSpec("state_composite_factor", "状态综合分", "state", 0.14, 1),
    FactorSpec("earnings_event_state", "业绩事件状态", "event", 0.05, 1),
    FactorSpec("operations_event_state", "经营事件状态", "event", 0.17, 1),
    FactorSpec("market_buzz_event_state", "市场舆情事件状态", "event", 0.05, 1),
    FactorSpec("risk_event_state", "风险事件状态", "event", 0.08, -1),
]

OPTIMIZED_FACTOR_SPECS = [
    FactorSpec("net_sentiment_factor", "当日净情绪", "direct", 0.06, 1),
    FactorSpec("negative_shock_factor", "当日负面冲击", "direct", 0.06, -1),
    FactorSpec("attention_factor", "当日关注度", "direct", 0.06, 1),
    FactorSpec("composite_score", "基础综合分", "direct", 0.06, 1),
    FactorSpec("event_novelty_factor", "事件新颖性", "direct", 0.05, 1),
    FactorSpec("ema5_sentiment_state", "5日情绪状态", "state", 0.10, 1),
    FactorSpec("negative_shock_carry", "负面冲击记忆", "state", 0.08, -1),
    FactorSpec("state_composite_factor", "状态综合分", "state", 0.10, 1),
    FactorSpec("sentiment_confirmation_factor", "情绪确认因子", "state", 0.12, 1),
    FactorSpec("coverage_reliability_factor", "覆盖可靠性因子", "reliability", 0.09, 1),
    FactorSpec("sentiment_dispersion_factor", "情绪分歧因子", "reliability", 0.08, -1),
    FactorSpec("earnings_event_state", "业绩事件状态", "event", 0.04, 1),
    FactorSpec("operations_event_state", "经营事件状态", "event", 0.12, 1),
    FactorSpec("market_buzz_event_state", "市场舆情事件状态", "event", 0.04, 1),
    FactorSpec("risk_event_state", "风险事件状态", "event", 0.10, -1),
    FactorSpec("event_density_factor", "事件密度因子", "event", 0.04, 1),
]

PRESET_FACTOR_NAMES = {
    "direct": {
        "net_sentiment_factor",
        "negative_shock_factor",
        "attention_factor",
        "composite_score",
    },
    "state": {
        "net_sentiment_factor",
        "negative_shock_factor",
        "attention_factor",
        "composite_score",
        "ema5_sentiment_state",
        "negative_shock_carry",
        "state_composite_factor",
    },
    "event": {spec.name for spec in DEFAULT_FACTOR_SPECS},
    "all": {spec.name for spec in DEFAULT_FACTOR_SPECS},
    "optimized": {spec.name for spec in OPTIMIZED_FACTOR_SPECS},
}


def load_factor_specs(path: Path | None = None, preset: str = "all") -> list[FactorSpec]:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload["factors"] if isinstance(payload, dict) and "factors" in payload else payload
        return [FactorSpec.from_mapping(item) for item in items]

    names = PRESET_FACTOR_NAMES.get(preset)
    if names is None:
        raise ValueError(f"未知排序模型 preset: {preset}")
    source_specs = OPTIMIZED_FACTOR_SPECS if preset == "optimized" else DEFAULT_FACTOR_SPECS
    return [spec for spec in source_specs if spec.name in names]


def cross_sectional_percentile(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(index=values.index, dtype=float)
    if len(valid) == 1 or valid.nunique(dropna=True) <= 1:
        filled = pd.Series(0.5, index=valid.index, dtype=float)
        return filled.reindex(values.index)
    ranked = valid.rank(method="average", pct=True)
    return ranked.reindex(values.index)


def _available_specs(df: pd.DataFrame, factor_specs: list[FactorSpec]) -> list[FactorSpec]:
    return [spec for spec in factor_specs if spec.name in df.columns]


def _clamp_series(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return values.clip(lower=lower, upper=upper)


def add_derived_ranking_features(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()

    if {"net_sentiment_factor", "ema5_sentiment_state"}.issubset(ranked.columns):
        net_sent = pd.to_numeric(ranked["net_sentiment_factor"], errors="coerce").fillna(0.0)
        state_sent = pd.to_numeric(ranked["ema5_sentiment_state"], errors="coerce").fillna(0.0)
        ranked["sentiment_confirmation_factor"] = (
            0.5 * (net_sent + state_sent) - 0.5 * (net_sent - state_sent).abs()
        )
        ranked["sentiment_confirmation_factor"] = _clamp_series(
            ranked["sentiment_confirmation_factor"], -1.0, 1.0
        )

    if {"article_count", "mention_count", "attention_factor", "sentiment_dispersion_factor"}.issubset(
        ranked.columns
    ):
        article_norm = _clamp_series(pd.to_numeric(ranked["article_count"], errors="coerce").fillna(0.0) / 3.0, 0.0, 1.0)
        mention_norm = _clamp_series(pd.to_numeric(ranked["mention_count"], errors="coerce").fillna(0.0) / 3.0, 0.0, 1.0)
        attention_norm = _clamp_series(
            pd.to_numeric(ranked["attention_factor"], errors="coerce").fillna(0.0) / 0.65,
            0.0,
            1.0,
        )
        consistency_norm = 1.0 - _clamp_series(
            pd.to_numeric(ranked["sentiment_dispersion_factor"], errors="coerce").fillna(0.0) / 0.8,
            0.0,
            1.0,
        )
        ranked["coverage_reliability_factor"] = (
            0.30 * article_norm
            + 0.20 * mention_norm
            + 0.20 * attention_norm
            + 0.30 * consistency_norm
        )
        ranked["coverage_reliability_factor"] = _clamp_series(
            ranked["coverage_reliability_factor"], 0.0, 1.0
        )

    return ranked


def build_reliability_multiplier(df: pd.DataFrame) -> pd.Series:
    if "coverage_reliability_factor" in df.columns:
        coverage = pd.to_numeric(df["coverage_reliability_factor"], errors="coerce").fillna(0.5)
    else:
        coverage = pd.Series(0.5, index=df.index, dtype=float)

    if "sentiment_dispersion_factor" in df.columns:
        consistency = 1.0 - _clamp_series(
            pd.to_numeric(df["sentiment_dispersion_factor"], errors="coerce").fillna(0.0) / 0.8,
            0.0,
            1.0,
        )
    else:
        consistency = pd.Series(0.5, index=df.index, dtype=float)

    if "event_novelty_factor" in df.columns:
        novelty = _clamp_series(
            pd.to_numeric(df["event_novelty_factor"], errors="coerce").fillna(0.0).abs() / 0.9,
            0.0,
            1.0,
        )
    else:
        novelty = pd.Series(0.5, index=df.index, dtype=float)

    multiplier = 0.70 + 0.20 * coverage + 0.10 * (0.6 * consistency + 0.4 * novelty)
    return _clamp_series(multiplier, 0.70, 1.00)


def build_factor_signals(df: pd.DataFrame, factor_specs: list[FactorSpec]) -> pd.DataFrame:
    ranked = add_derived_ranking_features(df)
    if "publish_date" not in ranked.columns:
        raise ValueError("输入因子表必须包含 publish_date 列")
    ranked["publish_date"] = pd.to_datetime(ranked["publish_date"])

    for spec in _available_specs(ranked, factor_specs):
        aligned_col = f"{spec.name}__aligned"
        signal_col = f"{spec.name}__signal"
        ranked[aligned_col] = pd.to_numeric(ranked[spec.name], errors="coerce") * spec.direction
        ranked[signal_col] = ranked.groupby("publish_date", group_keys=False)[aligned_col].apply(
            cross_sectional_percentile
        )
    return ranked


def weighted_group_score(df: pd.DataFrame, group_name: str, factor_specs: list[FactorSpec]) -> pd.Series:
    group_specs = [
        spec
        for spec in factor_specs
        if spec.group == group_name and f"{spec.name}__signal" in df.columns
    ]
    if not group_specs:
        return pd.Series(0.0, index=df.index, dtype=float)

    weight_sum = sum(abs(spec.weight) for spec in group_specs)
    if weight_sum == 0:
        return pd.Series(0.0, index=df.index, dtype=float)

    score = pd.Series(0.0, index=df.index, dtype=float)
    for spec in group_specs:
        score = score + spec.weight * df[f"{spec.name}__signal"].fillna(0.5)
    return score / weight_sum


def overall_ranking_score(df: pd.DataFrame, factor_specs: list[FactorSpec]) -> pd.Series:
    specs = [spec for spec in factor_specs if f"{spec.name}__signal" in df.columns]
    if not specs:
        raise ValueError("输入因子表没有任何可用于排序的因子列")

    weight_sum = sum(abs(spec.weight) for spec in specs)
    if weight_sum == 0:
        raise ValueError("排序因子权重之和不能为 0")

    score = pd.Series(0.0, index=df.index, dtype=float)
    for spec in specs:
        score = score + spec.weight * df[f"{spec.name}__signal"].fillna(0.5)
    return score / weight_sum


def build_rankings(
    df: pd.DataFrame,
    top_k: int,
    factor_specs: list[FactorSpec] | None = None,
) -> pd.DataFrame:
    specs = factor_specs or DEFAULT_FACTOR_SPECS
    ranked = build_factor_signals(df, specs)
    groups = list(dict.fromkeys(spec.group for spec in specs))
    for group_name in groups:
        ranked[f"{group_name}_group_score"] = weighted_group_score(ranked, group_name, specs)
    ranked["base_ranking_score"] = overall_ranking_score(ranked, specs)
    reliability_multiplier = build_reliability_multiplier(ranked)
    ranked["reliability_multiplier"] = reliability_multiplier
    ranked["ranking_score"] = 0.5 + (ranked["base_ranking_score"] - 0.5) * reliability_multiplier
    sort_cols = ["publish_date", "ranking_score"]
    ascending = [True, False]
    if "ts_code" in ranked.columns:
        sort_cols.append("ts_code")
        ascending.append(True)
    ranked = ranked.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    ranked["rank_in_date"] = ranked.groupby("publish_date")["ranking_score"].rank(
        method="first", ascending=False
    ).astype(int)
    ranked["selected_topk"] = (ranked["rank_in_date"] <= max(1, top_k)).astype(int)
    return ranked


def summarize_rankings(
    df: pd.DataFrame,
    top_k: int,
    factor_specs: list[FactorSpec] | None = None,
) -> dict[str, object]:
    specs = factor_specs or DEFAULT_FACTOR_SPECS
    counts = df.groupby("publish_date")["ts_code"].nunique() if "ts_code" in df.columns else pd.Series(dtype=int)
    topk = df[df["selected_topk"] == 1]
    group_names = list(dict.fromkeys(spec.group for spec in specs))
    topk_lists = []
    for publish_date, group in topk.groupby("publish_date", sort=True):
        stocks = []
        for _, row in group.sort_values("rank_in_date").iterrows():
            item = {
                "rank": int(row["rank_in_date"]),
                "ranking_score": round(float(row["ranking_score"]), 4),
            }
            for col in ("ts_code", "stock_name"):
                if col in row.index:
                    item[col] = row[col]
            for group_name in group_names:
                col = f"{group_name}_group_score"
                if col in row.index:
                    item[col] = round(float(row[col]), 4)
            stocks.append(item)

        topk_lists.append(
            {
                "publish_date": str(pd.to_datetime(publish_date).date()),
                "selected_count": int(len(group)),
                "selected_stocks": stocks,
            }
        )

    return {
        "top_k": top_k,
        "date_count": int(df["publish_date"].nunique()),
        "stock_count": int(df["ts_code"].nunique()) if "ts_code" in df.columns else 0,
        "avg_cross_section_size": round(float(counts.mean()), 4) if not counts.empty else 0.0,
        "factor_weights": [spec.to_dict() for spec in specs],
        "topk_by_date": topk_lists,
    }
