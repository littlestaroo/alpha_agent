#!/usr/bin/env python3
"""Run the chapter 4 multi-agent workflow and generate a report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path("/Users/star/Desktop/agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_quant.agent_pipeline import AgentPipelineConfig, run_agent_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="运行第四章多智能体工作流并生成报告")
    parser.add_argument("--news", type=Path, required=True, help="新闻文件或目录")
    parser.add_argument("--universe", type=Path, required=True, help="股票池 CSV")
    parser.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    parser.add_argument(
        "--ranking-preset",
        choices=["direct", "state", "event", "all", "optimized"],
        default="optimized",
        help="排序模型 preset",
    )
    parser.add_argument("--top-k", type=int, default=5, help="每天选出前 k 只股票")
    parser.add_argument("--date-from", type=str, default="", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--date-to", type=str, default="", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条新闻")
    parser.add_argument("--max-candidates", type=int, default=8, help="每条新闻最多候选股票数")
    parser.add_argument("--max-body-chars", type=int, default=1600, help="LLM 输入正文最大字符数")
    parser.add_argument("--price-cache", type=Path, default=None, help="价格缓存 CSV")
    args = parser.parse_args()

    config = AgentPipelineConfig(
        news_path=args.news,
        universe_path=args.universe,
        output_dir=args.out_dir,
        ranking_preset=args.ranking_preset,
        top_k=max(1, args.top_k),
        date_from=args.date_from or None,
        date_to=args.date_to or None,
        limit=args.limit or None,
        max_candidates=args.max_candidates,
        max_body_chars=args.max_body_chars,
        price_cache_path=args.price_cache,
    )
    summary = run_agent_pipeline(config)
    print(f"agent_workflow_summary={args.out_dir / 'agent_workflow_summary.json'}")
    print(f"agent_workflow_report={args.out_dir / 'agent_workflow_report.md'}")
    print(f"rank_date_count={summary['ranking_summary']['date_count']}")
    print(f"top_k={summary['top_k']}")


if __name__ == "__main__":
    main()
