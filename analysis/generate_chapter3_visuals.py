from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/star/Desktop/agent")
ARTICLE_PATH = (
    ROOT
    / "output"
    / "chapter3_baseline_thesis_2stocks_2023-11_top3_perstock_final_article_mentions.csv"
)
PROFILE_PATH = (
    ROOT
    / "output"
    / "chapter3_baseline_thesis_2stocks_2023-11_top3_perstock_final_daily_profiles.csv"
)
OUTPUT_DIR = ROOT / "output" / "chapter3_visuals"

COLORS = {
    "daily": "#2f6fed",
    "ema3": "#ef6c00",
    "state": "#0f9d58",
    "neg": "#c62828",
    "宁德时代": "#1259c3",
    "贵州茅台": "#8b1e3f",
    "业绩": "#2563eb",
    "经营": "#059669",
    "投融资": "#d97706",
    "市场舆情": "#7c3aed",
    "产品": "#db2777",
    "其他": "#6b7280",
}

EVENT_ORDER = ["经营", "市场舆情", "投融资", "业绩", "产品", "其他"]


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2937; }",
        ".title { font-size: 24px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #4b5563; }",
        ".axis { stroke: #cbd5e1; stroke-width: 1; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; stroke-dasharray: 4 4; }",
        ".small { font-size: 12px; fill: #475569; }",
        ".legend { font-size: 13px; }",
        ".panel-title { font-size: 18px; font-weight: 600; }",
        "</style>",
    ]


def _svg_footer(lines: list[str]) -> str:
    return "\n".join(lines + ["</svg>"])


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if math.isclose(hi, lo):
        return (out_lo + out_hi) / 2
    ratio = (value - lo) / (hi - lo)
    return out_lo + ratio * (out_hi - out_lo)


def _line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    first_x, first_y = points[0]
    commands = [f"M {first_x:.2f} {first_y:.2f}"]
    for x, y in points[1:]:
        commands.append(f"L {x:.2f} {y:.2f}")
    return " ".join(commands)


def build_enhanced_factor_demo(profiles_df: pd.DataFrame, article_df: pd.DataFrame) -> pd.DataFrame:
    df = profiles_df.copy()
    df["publish_date"] = pd.to_datetime(df["publish_date"])
    df = df.sort_values(["ts_code", "publish_date"]).reset_index(drop=True)

    df["ema3_sentiment_state"] = (
        df.groupby("ts_code")["net_sentiment_factor"]
        .transform(lambda s: s.ewm(span=3, adjust=False).mean())
        .round(4)
    )
    df["ema5_sentiment_state"] = (
        df.groupby("ts_code")["net_sentiment_factor"]
        .transform(lambda s: s.ewm(span=5, adjust=False).mean())
        .round(4)
    )

    carry_values: list[float] = []
    for _, group in df.groupby("ts_code", sort=False):
        carry = 0.0
        for shock in group["negative_shock_factor"].tolist():
            carry = 0.65 * carry + float(shock)
            carry_values.append(round(carry, 4))
    df["negative_shock_carry"] = carry_values
    df["state_composite_demo"] = (
        0.50 * df["ema3_sentiment_state"]
        + 0.20 * df["ema5_sentiment_state"]
        - 0.25 * df["negative_shock_carry"]
        + 0.10 * df["attention_factor"]
    ).round(4)

    article = article_df.copy()
    article["publish_date"] = pd.to_datetime(article["publish_date"])
    article["weighted_sentiment"] = (
        article["sentiment"]
        * article["relevance"]
        * article["confidence"]
        * article["event_importance"]
    )

    pivot = (
        article.pivot_table(
            index=["publish_date", "ts_code", "stock_name"],
            columns="event_type",
            values="weighted_sentiment",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    pivot.columns.name = None
    for event_type in EVENT_ORDER:
        if event_type not in pivot.columns:
            pivot[event_type] = 0.0

    merged = df.merge(
        pivot,
        on=["publish_date", "ts_code", "stock_name"],
        how="left",
    )
    for event_type in EVENT_ORDER:
        merged[event_type] = merged[event_type].fillna(0.0).round(4)

    merged["publish_date"] = merged["publish_date"].dt.strftime("%Y-%m-%d")
    return merged


def render_state_chart(enhanced_df: pd.DataFrame, output_path: Path) -> None:
    width, height = 1400, 720
    lines = _svg_header(width, height)
    lines.append('<text x="50" y="46" class="title">日度直用值 vs 时序状态值</text>')
    lines.append(
        '<text x="50" y="72" class="subtitle">蓝线：当日净情绪因子；橙线：3日EMA；绿线：带负面冲击记忆的状态值。这个图用来说明只看当天值会更噪声，加入时间累计后会更稳定。</text>'
    )

    chart_x = 70
    chart_w = 1260
    panel_h = 240
    panel_gap = 100
    chart_top = 130

    for idx, stock_name in enumerate(["宁德时代", "贵州茅台"]):
        panel_y = chart_top + idx * (panel_h + panel_gap)
        panel = enhanced_df[enhanced_df["stock_name"] == stock_name].copy()
        panel = panel.sort_values("publish_date")

        y_values = (
            panel["net_sentiment_factor"].tolist()
            + panel["ema3_sentiment_state"].tolist()
            + panel["state_composite_demo"].tolist()
        )
        y_min = min(y_values) - 0.12
        y_max = max(y_values) + 0.12

        lines.append(f'<text x="{chart_x}" y="{panel_y - 18}" class="panel-title">{stock_name}</text>')
        lines.append(
            f'<rect x="{chart_x}" y="{panel_y}" width="{chart_w}" height="{panel_h}" fill="#ffffff" stroke="#e5e7eb"/>'
        )

        for tick in range(5):
            value = y_min + (y_max - y_min) * tick / 4
            y = _scale(value, y_min, y_max, panel_y + panel_h - 24, panel_y + 24)
            lines.append(f'<line x1="{chart_x}" x2="{chart_x + chart_w}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
            lines.append(f'<text x="{chart_x - 10}" y="{y + 4:.2f}" text-anchor="end" class="small">{value:.2f}</text>')

        points_daily: list[tuple[float, float]] = []
        points_ema3: list[tuple[float, float]] = []
        points_state: list[tuple[float, float]] = []

        for pos, (_, row) in enumerate(panel.iterrows()):
            x = _scale(pos, 0, max(len(panel) - 1, 1), chart_x + 18, chart_x + chart_w - 18)
            y_daily = _scale(float(row["net_sentiment_factor"]), y_min, y_max, panel_y + panel_h - 24, panel_y + 24)
            y_ema3 = _scale(float(row["ema3_sentiment_state"]), y_min, y_max, panel_y + panel_h - 24, panel_y + 24)
            y_state = _scale(float(row["state_composite_demo"]), y_min, y_max, panel_y + panel_h - 24, panel_y + 24)
            points_daily.append((x, y_daily))
            points_ema3.append((x, y_ema3))
            points_state.append((x, y_state))

            if pos < len(panel) - 1:
                lines.append(f'<line x1="{x:.2f}" y1="{panel_y + panel_h - 24}" x2="{x:.2f}" y2="{panel_y + panel_h - 20}" class="axis"/>')
            if pos % 4 == 0 or pos == len(panel) - 1:
                lines.append(
                    f'<text x="{x:.2f}" y="{panel_y + panel_h + 14}" text-anchor="middle" class="small">{str(row["publish_date"])[5:]}</text>'
                )

        lines.append(f'<path d="{_line_path(points_daily)}" fill="none" stroke="{COLORS["daily"]}" stroke-width="2.5"/>')
        lines.append(f'<path d="{_line_path(points_ema3)}" fill="none" stroke="{COLORS["ema3"]}" stroke-width="2.5"/>')
        lines.append(f'<path d="{_line_path(points_state)}" fill="none" stroke="{COLORS["state"]}" stroke-width="3"/>')

        legend_x = chart_x + chart_w - 330
        legend_y = panel_y + 18
        legend_items = [
            ("当日净情绪", COLORS["daily"]),
            ("3日EMA状态", COLORS["ema3"]),
            ("状态值(含负面冲击记忆)", COLORS["state"]),
        ]
        for i, (label, color) in enumerate(legend_items):
            yy = legend_y + i * 24
            lines.append(f'<line x1="{legend_x}" x2="{legend_x + 26}" y1="{yy}" y2="{yy}" stroke="{color}" stroke-width="3"/>')
            lines.append(f'<text x="{legend_x + 34}" y="{yy + 4}" class="legend">{label}</text>')

    output_path.write_text(_svg_footer(lines), encoding="utf-8")


def render_event_chart(article_df: pd.DataFrame, output_path: Path) -> None:
    width, height = 1400, 760
    lines = _svg_header(width, height)
    lines.append('<text x="50" y="46" class="title">事件结构与情绪差异</text>')
    lines.append(
        '<text x="50" y="72" class="subtitle">左侧看事件数量结构，右侧看不同事件类型的平均情绪。这个图用来说明仅用总舆情值会把“业绩利好”和“市场补跌”混在一起。</text>'
    )

    counts = pd.crosstab(article_df["stock_name"], article_df["event_type"])
    avg_sent = (
        article_df.groupby(["stock_name", "event_type"])["sentiment"]
        .mean()
        .reset_index()
    )

    left_x, left_y, left_w, left_h = 70, 130, 560, 250
    right_x, right_y, right_w, right_h = 720, 130, 590, 420

    lines.append(f'<rect x="{left_x}" y="{left_y}" width="{left_w}" height="{left_h}" fill="#ffffff" stroke="#e5e7eb"/>')
    lines.append(f'<text x="{left_x}" y="{left_y - 16}" class="panel-title">事件数量结构</text>')

    stock_names = ["宁德时代", "贵州茅台"]
    max_total = int(counts.reindex(stock_names).fillna(0).sum(axis=1).max())
    bar_height = 48
    bar_gap = 60

    for idx, stock_name in enumerate(stock_names):
        y = left_y + 50 + idx * (bar_height + bar_gap)
        lines.append(f'<text x="{left_x}" y="{y - 10}" class="small">{stock_name}</text>')
        current_x = left_x + 90
        total = int(counts.loc[stock_name].sum()) if stock_name in counts.index else 0
        for event_type in EVENT_ORDER:
            value = int(counts.loc[stock_name, event_type]) if event_type in counts.columns and stock_name in counts.index else 0
            if value <= 0:
                continue
            width_part = (left_w - 130) * value / max_total
            lines.append(
                f'<rect x="{current_x:.2f}" y="{y}" width="{width_part:.2f}" height="{bar_height}" fill="{COLORS[event_type]}"/>'
            )
            if width_part > 26:
                lines.append(
                    f'<text x="{current_x + width_part / 2:.2f}" y="{y + 29}" text-anchor="middle" fill="#ffffff" style="font-size:12px;font-weight:600;">{value}</text>'
                )
            current_x += width_part
        lines.append(f'<text x="{left_x + left_w - 24}" y="{y + 29}" text-anchor="end" class="small">{total} 条</text>')

    legend_x = left_x + 24
    legend_y = left_y + left_h + 28
    for i, event_type in enumerate(EVENT_ORDER):
        x = legend_x + (i % 3) * 170
        y = legend_y + (i // 3) * 24
        lines.append(f'<rect x="{x}" y="{y - 10}" width="14" height="14" fill="{COLORS[event_type]}"/>')
        lines.append(f'<text x="{x + 22}" y="{y + 2}" class="legend">{event_type}</text>')

    lines.append(f'<rect x="{right_x}" y="{right_y}" width="{right_w}" height="{right_h}" fill="#ffffff" stroke="#e5e7eb"/>')
    lines.append(f'<text x="{right_x}" y="{right_y - 16}" class="panel-title">分事件平均情绪</text>')

    x0 = right_x + 70
    y0 = right_y + right_h - 40
    usable_w = right_w - 120
    usable_h = right_h - 90
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0 + usable_w}" y2="{y0}" class="axis"/>')
    lines.append(f'<line x1="{x0}" y1="{right_y + 30}" x2="{x0}" y2="{y0}" class="axis"/>')

    for tick_val in [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]:
        y = _scale(tick_val, -0.4, 0.6, y0, right_y + 30)
        lines.append(f'<line x1="{x0}" x2="{x0 + usable_w}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        lines.append(f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" class="small">{tick_val:.1f}</text>')

    category_gap = usable_w / max(len(EVENT_ORDER), 1)
    bar_group_w = category_gap * 0.6
    single_bar_w = bar_group_w / 2.5

    for idx, event_type in enumerate(EVENT_ORDER):
        center_x = x0 + category_gap * idx + category_gap / 2
        lines.append(f'<text x="{center_x:.2f}" y="{y0 + 20}" text-anchor="middle" class="small">{event_type}</text>')
        for stock_pos, stock_name in enumerate(stock_names):
            row = avg_sent[(avg_sent["stock_name"] == stock_name) & (avg_sent["event_type"] == event_type)]
            value = float(row["sentiment"].iloc[0]) if not row.empty else 0.0
            x = center_x - bar_group_w / 2 + stock_pos * (single_bar_w + 14)
            y = _scale(max(value, 0.0), -0.4, 0.6, y0, right_y + 30)
            zero_y = _scale(0.0, -0.4, 0.6, y0, right_y + 30)
            if value >= 0:
                rect_y = y
                rect_h = max(zero_y - y, 1.5)
            else:
                rect_y = zero_y
                rect_h = max(y - zero_y, 1.5)
            lines.append(
                f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{single_bar_w:.2f}" height="{rect_h:.2f}" fill="{COLORS[stock_name]}" opacity="0.88"/>'
            )

    legend_y2 = right_y + right_h + 28
    for i, stock_name in enumerate(stock_names):
        x = right_x + 20 + i * 180
        lines.append(f'<rect x="{x}" y="{legend_y2 - 10}" width="14" height="14" fill="{COLORS[stock_name]}"/>')
        lines.append(f'<text x="{x + 22}" y="{legend_y2 + 2}" class="legend">{stock_name}</text>')

    output_path.write_text(_svg_footer(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    article_df = pd.read_csv(ARTICLE_PATH)
    profiles_df = pd.read_csv(PROFILE_PATH)
    enhanced_df = build_enhanced_factor_demo(profiles_df, article_df)
    enhanced_df.to_csv(OUTPUT_DIR / "chapter3_enhanced_factor_demo.csv", index=False)
    render_state_chart(enhanced_df, OUTPUT_DIR / "chapter3_state_comparison.svg")
    render_event_chart(article_df, OUTPUT_DIR / "chapter3_event_decomposition.svg")
    print(f"wrote: {OUTPUT_DIR / 'chapter3_enhanced_factor_demo.csv'}")
    print(f"wrote: {OUTPUT_DIR / 'chapter3_state_comparison.svg'}")
    print(f"wrote: {OUTPUT_DIR / 'chapter3_event_decomposition.svg'}")


if __name__ == "__main__":
    main()
