#!/usr/bin/env python3
"""Build a cross-sectional stock ranking from daily factor profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path("/Users/star/Desktop/agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_quant.ranking import DEFAULT_FACTOR_SPECS, build_rankings, load_factor_specs, summarize_rankings

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


FACTOR_CONFIG = [spec.to_dict() for spec in DEFAULT_FACTOR_SPECS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据日频因子做横截面股票排序，并选出每日 top k 股票",
    )
    parser.add_argument("--in", dest="input_path", type=Path, default=DEFAULT_INPUT, help="输入日频因子 CSV")
    parser.add_argument("--out", dest="output_path", type=Path, default=DEFAULT_OUTPUT, help="完整排序结果 CSV")
    parser.add_argument("--topk-out", dest="topk_output_path", type=Path, default=DEFAULT_TOPK_OUTPUT, help="仅保留 top k 的 CSV")
    parser.add_argument("--summary-out", dest="summary_path", type=Path, default=DEFAULT_SUMMARY, help="排序摘要 JSON")
    parser.add_argument("--top-k", type=int, default=5, help="每天选出前 k 只股票")
    parser.add_argument(
        "--preset",
        choices=["direct", "state", "event", "all", "optimized"],
        default="all",
        help="排序模型预设：direct=直接因子，state=加入时序状态，event/all=加入事件状态，optimized=加入可靠性优化",
    )
    parser.add_argument(
        "--factor-config",
        type=Path,
        default=None,
        help="可选 JSON 因子配置，用于加入已有股票因子或调整权重",
    )
    args = parser.parse_args()

    input_path: Path = args.input_path
    output_path: Path = args.output_path
    topk_output_path: Path = args.topk_output_path
    summary_path: Path = args.summary_path
    top_k: int = max(1, args.top_k)
    factor_specs = load_factor_specs(args.factor_config, preset=args.preset)

    if not input_path.exists():
        raise FileNotFoundError(f"输入因子文件不存在: {input_path}")

    df = pd.read_csv(input_path)
    ranked = build_rankings(df, top_k=top_k, factor_specs=factor_specs)

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
    signal_columns = [f"{spec.name}__signal" for spec in factor_specs if f"{spec.name}__signal" in ranked.columns]
    group_columns = sorted(col for col in ranked.columns if col.endswith("_group_score"))
    available_columns = [col for col in full_columns if col in ranked.columns]
    available_columns = list(dict.fromkeys(available_columns + group_columns + signal_columns))

    ranked[available_columns].to_csv(output_path, index=False)
    ranked[ranked["selected_topk"] == 1][available_columns].to_csv(topk_output_path, index=False)

    summary = summarize_rankings(ranked, top_k=top_k, factor_specs=factor_specs)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
