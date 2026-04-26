from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from news_quant.baseline import build_daily_profiles, run_baseline
from news_quant.data_loader import load_stock_universe
from news_quant.ranking import build_rankings, load_factor_specs, summarize_rankings
from analysis.evaluate_stock_rankings import (
    attach_future_returns,
    build_price_cache,
    evaluate_topk,
    summarize_performance,
)


@dataclass
class AgentStepResult:
    agent_name: str
    status: str
    message: str
    output_path: str | None = None
    row_count: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AgentPipelineConfig:
    news_path: Path
    universe_path: Path
    output_dir: Path
    ranking_preset: str = "optimized"
    top_k: int = 5
    date_from: str | None = None
    date_to: str | None = None
    limit: int | None = None
    max_candidates: int = 8
    max_body_chars: int = 1600
    price_cache_path: Path | None = None


def _prefix(output_dir: Path, name: str) -> Path:
    return output_dir / name


def build_markdown_report(summary: dict[str, object]) -> str:
    ranking_summary = summary.get("ranking_summary", {})
    performance_summary = summary.get("performance_summary", {})
    steps = summary.get("steps", [])
    topk_by_date = ranking_summary.get("topk_by_date", [])
    horizons = performance_summary.get("horizons", {})

    lines = [
        "# 第四章 Agent 系统运行报告",
        "",
        "## 1. 系统概览",
        "",
        f"- 排序模型 preset：`{summary.get('ranking_preset', '')}`",
        f"- top k：`{summary.get('top_k', '')}`",
        f"- 输入新闻：`{summary.get('news_path', '')}`",
        f"- 股票池：`{summary.get('universe_path', '')}`",
        "",
        "## 2. Agent 执行结果",
        "",
    ]

    for step in steps:
        lines.append(
            f"- `{step['agent_name']}`：{step['status']}，{step['message']}"
        )

    lines.extend(
        [
            "",
            "## 3. 排序结果摘要",
            "",
            f"- 可排序交易日数：`{ranking_summary.get('date_count', 0)}`",
            f"- 股票池覆盖数：`{ranking_summary.get('stock_count', 0)}`",
            f"- 平均横截面股票数：`{ranking_summary.get('avg_cross_section_size', 0)}`",
            "",
        ]
    )

    preview = topk_by_date[:3]
    if preview:
        lines.append("前 3 个交易日 top k 股票如下：")
        lines.append("")
        for item in preview:
            selected = "、".join(stock.get("stock_name", stock.get("ts_code", "")) for stock in item.get("selected_stocks", []))
            lines.append(f"- `{item['publish_date']}`：{selected}")
        lines.append("")

    lines.extend(
        [
            "## 4. 排序表现摘要",
            "",
        ]
    )
    for horizon in ("1", "3", "5"):
        metrics = horizons.get(horizon, {})
        if not metrics:
            continue
        lines.append(
            f"- `future {horizon}D`：top{k_or_blank(summary.get('top_k'))}平均超额收益 `"
            f"{format_pct(metrics.get('avg_excess_return'))}`，命中率 `{format_pct(metrics.get('hit_rate_vs_universe'))}`"
        )

    lines.extend(
        [
            "",
            "## 5. 系统说明",
            "",
            "该工作流对应第四章中的多智能体处理链路：数据输入 agent -> 舆情分析 agent -> 因子构建 agent -> 排序选股 agent -> 评估报告 agent。",
        ]
    )
    return "\n".join(lines) + "\n"


def format_pct(value: object) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value) * 100:.4f}%"
    except (TypeError, ValueError):
        return "NA"


def k_or_blank(value: object) -> str:
    return str(value) if value not in (None, "") else ""


def run_agent_pipeline(config: AgentPipelineConfig) -> dict[str, object]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    steps: list[AgentStepResult] = []
    baseline_prefix = _prefix(output_dir, "agent_pipeline")
    mentions_path = Path(f"{baseline_prefix}_article_mentions.csv")
    profiles_path = Path(f"{baseline_prefix}_daily_profiles.csv")
    skipped_path = Path(f"{baseline_prefix}_skipped_news.csv")
    ranked_path = output_dir / "agent_ranked.csv"
    topk_path = output_dir / "agent_topk.csv"
    ranking_summary_path = output_dir / "agent_rank_summary.json"
    panel_path = output_dir / "agent_rank_panel.csv"
    performance_path = output_dir / "agent_topk_performance.csv"
    performance_summary_path = output_dir / "agent_topk_performance_summary.json"
    report_path = output_dir / "agent_workflow_report.md"
    pipeline_summary_path = output_dir / "agent_workflow_summary.json"

    mentions_df, profiles_df, skipped_df = run_baseline(
        news_path=config.news_path,
        universe_path=config.universe_path,
        output_path=baseline_prefix,
        date_from=config.date_from,
        date_to=config.date_to,
        limit=config.limit,
        max_candidates=config.max_candidates,
        max_body_chars=config.max_body_chars,
    )
    steps.append(
        AgentStepResult(
            agent_name="analysis_agent",
            status="ok",
            message=f"生成文章级结构化结果 {len(mentions_df)} 条，跳过新闻 {len(skipped_df)} 条",
            output_path=str(mentions_path),
            row_count=len(mentions_df),
        )
    )
    steps.append(
        AgentStepResult(
            agent_name="factor_agent",
            status="ok",
            message=f"生成股票日频因子 {len(profiles_df)} 条",
            output_path=str(profiles_path),
            row_count=len(profiles_df),
        )
    )

    factor_specs = load_factor_specs(None, preset=config.ranking_preset)
    ranked_df = build_rankings(profiles_df, top_k=config.top_k, factor_specs=factor_specs)
    ranked_df.to_csv(ranked_path, index=False)
    ranked_df[ranked_df["selected_topk"] == 1].to_csv(topk_path, index=False)
    ranking_summary = summarize_rankings(ranked_df, top_k=config.top_k, factor_specs=factor_specs)
    ranking_summary_path.write_text(
        json.dumps(ranking_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    steps.append(
        AgentStepResult(
            agent_name="ranking_agent",
            status="ok",
            message=f"完成横截面排序并选出每日 top {config.top_k} 股票",
            output_path=str(ranked_path),
            row_count=len(ranked_df),
        )
    )

    price_cache = config.price_cache_path or (config.output_dir / "agent_price_cache.csv")
    ts_codes = sorted(ranked_df["ts_code"].dropna().unique())
    prices_df = build_price_cache(ts_codes, price_cache, force_refresh=False)
    panel_df = attach_future_returns(ranked_df, prices_df)
    daily_df = evaluate_topk(panel_df)
    performance_summary = summarize_performance(daily_df)
    panel_df.to_csv(panel_path, index=False)
    daily_df.to_csv(performance_path, index=False)
    performance_summary_path.write_text(
        json.dumps(performance_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    steps.append(
        AgentStepResult(
            agent_name="evaluation_agent",
            status="ok",
            message=f"完成 top {config.top_k} 未来收益评估",
            output_path=str(performance_path),
            row_count=len(daily_df),
        )
    )

    summary = {
        "news_path": str(config.news_path),
        "universe_path": str(config.universe_path),
        "output_dir": str(output_dir),
        "ranking_preset": config.ranking_preset,
        "top_k": config.top_k,
        "steps": [step.to_dict() for step in steps],
        "ranking_summary": ranking_summary,
        "performance_summary": performance_summary,
        "artifacts": {
            "mentions_path": str(mentions_path),
            "profiles_path": str(profiles_path),
            "skipped_path": str(skipped_path),
            "ranked_path": str(ranked_path),
            "topk_path": str(topk_path),
            "panel_path": str(panel_path),
            "performance_path": str(performance_path),
            "report_path": str(report_path),
        },
    }
    report_path.write_text(build_markdown_report(summary), encoding="utf-8")
    pipeline_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

