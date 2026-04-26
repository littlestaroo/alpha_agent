#!/usr/bin/env python3
"""Run multiple ranking presets and compare top-k performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path("/Users/star/Desktop/agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate_stock_rankings import (  # noqa: E402
    attach_future_returns,
    build_price_cache,
    evaluate_topk,
    summarize_performance,
)
from news_quant.ranking import build_rankings, load_factor_specs  # noqa: E402


DEFAULT_INPUT = (
    ROOT
    / "output"
    / "q4_20stocks_batches"
    / "chapter3_baseline_q4_20stocks_batches_000_004_diverse_top3_daily_profiles_global.csv"
)
DEFAULT_OUT_DIR = ROOT / "output" / "chapter3_ranking_compare"
DEFAULT_PRICE_CACHE = ROOT / "data" / "market" / "q4_20stocks_prices_cache.csv"


def flatten_summary(model_name: str, top_k: int, summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    horizons = summary.get("horizons", {})
    for horizon, metrics in horizons.items():
        metric_map = metrics if isinstance(metrics, dict) else {}
        rows.append(
            {
                "model": model_name,
                "top_k": top_k,
                "horizon": int(horizon),
                "date_count": int(summary.get("date_count", 0)),
                "n_dates": metric_map.get("n_dates"),
                "avg_topk_return": metric_map.get("avg_topk_return"),
                "avg_universe_return": metric_map.get("avg_universe_return"),
                "avg_excess_return": metric_map.get("avg_excess_return"),
                "avg_top_bottom_spread": metric_map.get("avg_top_bottom_spread"),
                "hit_rate_vs_universe": metric_map.get("hit_rate_vs_universe"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="比较不同排序模型 preset 的 top-k 表现")
    parser.add_argument("--in", dest="input_path", type=Path, default=DEFAULT_INPUT, help="输入日频因子 CSV")
    parser.add_argument("--out-dir", dest="output_dir", type=Path, default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--price-cache", type=Path, default=DEFAULT_PRICE_CACHE, help="价格缓存 CSV")
    parser.add_argument("--top-k", type=int, default=5, help="每天选出前 k 只股票")
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["direct", "state", "event", "optimized"],
        help="要比较的模型 preset 列表",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"输入因子文件不存在: {args.input_path}")

    profiles_df = pd.read_csv(args.input_path)
    ts_codes = sorted(profiles_df["ts_code"].dropna().unique())
    prices_df = build_price_cache(ts_codes, args.price_cache, force_refresh=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    all_summaries: dict[str, object] = {}

    for preset in args.presets:
        factor_specs = load_factor_specs(None, preset=preset)
        ranked_df = build_rankings(profiles_df, top_k=args.top_k, factor_specs=factor_specs)
        panel_df = attach_future_returns(ranked_df, prices_df)
        daily_df = evaluate_topk(panel_df)
        summary = summarize_performance(daily_df)
        all_summaries[preset] = {
            "top_k": args.top_k,
            "factor_weights": [spec.to_dict() for spec in factor_specs],
            **summary,
        }
        summary_rows.extend(flatten_summary(preset, args.top_k, summary))

        ranked_df.to_csv(args.output_dir / f"{preset}_ranked.csv", index=False)
        panel_df.to_csv(args.output_dir / f"{preset}_rank_panel.csv", index=False)
        daily_df.to_csv(args.output_dir / f"{preset}_topk_performance.csv", index=False)
        (args.output_dir / f"{preset}_summary.json").write_text(
            json.dumps(all_summaries[preset], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    comparison_df = pd.DataFrame(summary_rows).sort_values(["horizon", "model"]).reset_index(drop=True)
    comparison_df.to_csv(args.output_dir / "preset_comparison.csv", index=False)
    (args.output_dir / "preset_comparison.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
