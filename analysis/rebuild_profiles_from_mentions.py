#!/usr/bin/env python3
"""Rebuild global daily profiles from merged article mentions."""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/star/Desktop/agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_quant.baseline import ARTICLE_COLUMNS, build_daily_profiles
from news_quant.data_loader import load_stock_universe

NUMERIC_COLUMNS = [
    "overall_sentiment",
    "coarse_score",
    "merged_event_count",
    "sentiment",
    "sentiment_strength",
    "confidence",
    "relevance",
    "event_importance",
    "risk_flag",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从多批 article_mentions 合并结果重新构建全局 daily_profiles，避免批次级聚合重复",
    )
    parser.add_argument(
        "--mentions-glob",
        required=True,
        help="article_mentions 文件 glob，例如 /path/to/chapter3_baseline_q4_20stocks_batch_00[0-1]_article_mentions.csv",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        required=True,
        help="股票池 CSV",
    )
    parser.add_argument(
        "--merged-mentions-out",
        type=Path,
        required=True,
        help="合并后的文章级结果 CSV 输出路径",
    )
    parser.add_argument(
        "--profiles-out",
        type=Path,
        required=True,
        help="全局重建后的日频因子 CSV 输出路径",
    )
    args = parser.parse_args()

    mention_paths = [Path(path) for path in sorted(glob.glob(args.mentions_glob))]
    if not mention_paths:
        raise FileNotFoundError(f"未匹配到 article_mentions 文件: {args.mentions_glob}")

    rows: list[dict[str, object]] = []
    fieldnames: list[str] | None = None
    for path in mention_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or fieldnames
            rows.extend(reader)

    mentions_df = pd.DataFrame(rows)
    if fieldnames:
        for column in ARTICLE_COLUMNS:
            if column not in mentions_df.columns:
                mentions_df[column] = ""
        mentions_df = mentions_df[ARTICLE_COLUMNS]
    for column in NUMERIC_COLUMNS:
        if column in mentions_df.columns:
            mentions_df[column] = pd.to_numeric(mentions_df[column], errors="coerce").fillna(0.0)

    universe_df = load_stock_universe(args.universe)
    profiles_df = build_daily_profiles(mentions_df, universe_df)

    args.merged_mentions_out.parent.mkdir(parents=True, exist_ok=True)
    args.profiles_out.parent.mkdir(parents=True, exist_ok=True)
    mentions_df.to_csv(args.merged_mentions_out, index=False)
    profiles_df.to_csv(args.profiles_out, index=False)
    print(f"merged_mentions_rows={len(mentions_df)}")
    print(f"global_profiles_rows={len(profiles_df)}")


if __name__ == "__main__":
    main()
