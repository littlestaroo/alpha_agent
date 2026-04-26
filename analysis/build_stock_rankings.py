#!/usr/bin/env python3
"""Build a cross-sectional stock ranking from daily factor profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/star/Desktop/agent")

DEFAULT_INPUT = (
    ROOT
    / "output"
    / "q4_20stocks_batches"
    / "chapter3_baseline_q4_20stocks_batch_000_smoke30_daily_profiles.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "output"
    / "q4_20stocks_batches"
    / "chapter3_baseline_q4_20stocks_batch_000_smoke30_ranked.csv"
)
DEFAULT_TOPK_OUTPUT = (
    ROOT
    / "output"
    / "q4_20stocks_batches"
    / "chapter3_baseline_q4_20stocks_batch_000_smoke30_topk.csv"
)
DEFAULT_SUMMARY = (
    ROOT
    / "output"
    / "q4_20stocks_batches"
    / "chapter3_baseline_q4_20stocks_batch_000_smoke30_rank_summary.json"
)


FACTOR_CONFIG = [
    {
        "name": "net_sentiment_factor",
        "label": "当日净情绪",
        "group": "direct",
        "weight": 0.08,
        "direction": 1,
    },
    {
        "name": "negative_shock_factor",
        "label": "当日负面冲击",
        "group": "direct",
        "weight": 0.04,
        "direction": -1,
    },
    {
        "name": "attention_factor",
        "label": "当日关注度",
        "group": "direct",
        "weight": 0.08,
        "direction": 1,
    },
    {
        "name": "composite_score",
        "label": "基础综合分",
        "group": "direct",
        "weight": 0.08,
        "direction": 1,
    },
    {
        "name": "ema5_sentiment_state",
        "label": "5日情绪状态",
        "group": "state",
        "weight": 0.14,
        "direction": 1,
    },
    {
        "name": "negative_shock_carry",
        "label": "负面冲击记忆",
        "group": "state",
        "weight": 0.09,
        "direction": -1,
    },
    {
        "name": "state_composite_factor",
        "label": "状态综合分",
        "group": "state",
        "weight": 0.14,
        "direction": 1,
    },
    {
        "name": "earnings_event_state",
        "label": "业绩事件状态",
        "group": "event",
        "weight": 0.05,
        "direction": 1,
    },
    {
        "name": "operations_event_state",
        "label": "经营事件状态",
        "group": "event",
        "weight": 0.17,
        "direction": 1,
    },
    {
        "name": "market_buzz_event_state",
        "label": "市场舆情事件状态",
        "group": "event",
        "weight": 0.05,
        "direction": 1,
    },
    {
        "name": "risk_event_state",
        "label": "风险事件状态",
        "group": "event",
        "weight": 0.08,
        "direction": -1,
    },
]

GROUP_ORDER = ["direct", "state", "event"]
GROUP_LABELS = {
    "direct": "direct_group_score",
    "state": "state_group_score",
    "event": "event_group_score",
}


def cross_sectional_percentile(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(index=values.index, dtype=float)
    if len(valid) == 1 or valid.nunique(dropna=True) <= 1:
        filled = pd.Series(0.5, index=valid.index, dtype=float)
        return filled.reindex(values.index)
    ranked = valid.rank(method="average", pct=True)
    return ranked.reindex(values.index)


def build_factor_signals(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["publish_date"] = pd.to_datetime(ranked["publish_date"])

    for spec in FACTOR_CONFIG:
        factor = spec["name"]
        if factor not in ranked.columns:
            continue
        aligned_col = f"{factor}__aligned"
        signal_col = f"{factor}__signal"
        ranked[aligned_col] = ranked[factor].astype(float) * float(spec["direction"])
        ranked[signal_col] = ranked.groupby("publish_date", group_keys=False)[aligned_col].apply(
            cross_sectional_percentile
        )

    return ranked


def weighted_group_score(df: pd.DataFrame, group_name: str) -> pd.Series:
    group_specs = [spec for spec in FACTOR_CONFIG if spec["group"] == group_name and f"{spec['name']}__signal" in df.columns]
    if not group_specs:
        return pd.Series(0.0, index=df.index)
    weight_sum = sum(float(spec["weight"]) for spec in group_specs)
    score = pd.Series(0.0, index=df.index, dtype=float)
    for spec in group_specs:
        score = score + float(spec["weight"]) * df[f"{spec['name']}__signal"].fillna(0.5)
    return score / weight_sum


def overall_ranking_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index, dtype=float)
    for spec in FACTOR_CONFIG:
        signal_col = f"{spec['name']}__signal"
        if signal_col not in df.columns:
            continue
        score = score + float(spec["weight"]) * df[signal_col].fillna(0.5)
    return score


def build_rankings(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    ranked = build_factor_signals(df)
    for group_name in GROUP_ORDER:
        ranked[GROUP_LABELS[group_name]] = weighted_group_score(ranked, group_name)
    ranked["ranking_score"] = overall_ranking_score(ranked)
    ranked = ranked.sort_values(["publish_date", "ranking_score", "ts_code"], ascending=[True, False, True]).reset_index(drop=True)
    ranked["rank_in_date"] = ranked.groupby("publish_date")["ranking_score"].rank(method="first", ascending=False).astype(int)
    ranked["selected_topk"] = (ranked["rank_in_date"] <= top_k).astype(int)
    return ranked


def summarize_rankings(df: pd.DataFrame, top_k: int) -> dict[str, object]:
    counts = df.groupby("publish_date")["ts_code"].nunique()
    topk = df[df["selected_topk"] == 1]
    topk_lists = []
    for publish_date, group in topk.groupby("publish_date", sort=True):
        topk_lists.append(
            {
                "publish_date": str(pd.to_datetime(publish_date).date()),
                "selected_count": int(len(group)),
                "selected_stocks": [
                    {
                        "rank": int(row["rank_in_date"]),
                        "ts_code": row["ts_code"],
                        "stock_name": row["stock_name"],
                        "ranking_score": round(float(row["ranking_score"]), 4),
                        "direct_group_score": round(float(row["direct_group_score"]), 4),
                        "state_group_score": round(float(row["state_group_score"]), 4),
                        "event_group_score": round(float(row["event_group_score"]), 4),
                    }
                    for _, row in group.sort_values("rank_in_date").iterrows()
                ],
            }
        )

    return {
        "top_k": top_k,
        "date_count": int(df["publish_date"].nunique()),
        "stock_count": int(df["ts_code"].nunique()),
        "avg_cross_section_size": round(float(counts.mean()), 4) if not counts.empty else 0.0,
        "factor_weights": FACTOR_CONFIG,
        "topk_by_date": topk_lists,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据日频因子做横截面股票排序，并选出每日 top k 股票",
    )
    parser.add_argument("--in", dest="input_path", type=Path, default=DEFAULT_INPUT, help="输入日频因子 CSV")
    parser.add_argument("--out", dest="output_path", type=Path, default=DEFAULT_OUTPUT, help="完整排序结果 CSV")
    parser.add_argument("--topk-out", dest="topk_output_path", type=Path, default=DEFAULT_TOPK_OUTPUT, help="仅保留 top k 的 CSV")
    parser.add_argument("--summary-out", dest="summary_path", type=Path, default=DEFAULT_SUMMARY, help="排序摘要 JSON")
    parser.add_argument("--top-k", type=int, default=5, help="每天选出前 k 只股票")
    args = parser.parse_args()

    input_path: Path = args.input_path
    output_path: Path = args.output_path
    topk_output_path: Path = args.topk_output_path
    summary_path: Path = args.summary_path
    top_k: int = max(1, args.top_k)

    if not input_path.exists():
        raise FileNotFoundError(f"输入因子文件不存在: {input_path}")

    df = pd.read_csv(input_path)
    ranked = build_rankings(df, top_k=top_k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    topk_output_path.parent.mkdir(parents=True, exist_ok=True)

    full_columns = [
        "publish_date",
        "ts_code",
        "stock_name",
        "industry",
        "article_count",
        "mention_count",
        "dominant_event_type",
        "direct_group_score",
        "state_group_score",
        "event_group_score",
        "ranking_score",
        "rank_in_date",
        "selected_topk",
        "profile_label",
    ]
    signal_columns = [f"{spec['name']}__signal" for spec in FACTOR_CONFIG if f"{spec['name']}__signal" in ranked.columns]
    available_columns = [col for col in full_columns if col in ranked.columns] + signal_columns

    ranked[available_columns].to_csv(output_path, index=False)
    ranked[ranked["selected_topk"] == 1][available_columns].to_csv(topk_output_path, index=False)

    summary = summarize_rankings(ranked, top_k=top_k)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
