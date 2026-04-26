from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path("/Users/star/Desktop/agent")
ARTICLE_PATH = ROOT / "output" / "chapter3_baseline_thesis_2stocks_2023-11_top3_perstock_final_article_mentions.csv"
OUTPUT_DIR = ROOT / "output" / "chapter3_teacher"

COLORS = {
    "bg": "#ffffff",
    "text": "#0f172a",
    "muted": "#475569",
    "line": "#dbe4ee",
    "blue": "#2563eb",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
    "panel": "#f8fafc",
}


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{COLORS["bg"]}"/>',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        ".title { font-size: 26px; font-weight: 700; fill: #0f172a; }",
        ".subtitle { font-size: 14px; fill: #475569; }",
        ".label { font-size: 14px; fill: #334155; }",
        ".small { font-size: 12px; fill: #475569; }",
        ".card-title { font-size: 16px; font-weight: 700; fill: #0f172a; }",
        "</style>",
    ]


def _svg_footer(lines: list[str]) -> str:
    return "\n".join(lines + ["</svg>"])


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if math.isclose(lo, hi):
        return (out_lo + out_hi) / 2
    ratio = (value - lo) / (hi - lo)
    return out_lo + ratio * (out_hi - out_lo)


def _mean_from_rows(rows: list[dict[str, str]], key: str) -> float:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw in ("", None):
            continue
        values.append(float(raw))
    return sum(values) / len(values) if values else 0.0


def render_parameter_meaning(article_rows: list[dict[str, str]], output_path: Path) -> None:
    stats = {
        "sentiment": {
            "title": "sentiment",
            "subtitle": "情绪方向值",
            "range": (-1.0, 1.0),
            "mean": _mean_from_rows(article_rows, "sentiment"),
            "desc1": "越大越正面，越小越负面",
            "desc2": "-1 极负面，0 中性，+1 极正面",
            "color": COLORS["blue"],
        },
        "sentiment_strength": {
            "title": "sentiment_strength",
            "subtitle": "情绪强度",
            "range": (0.0, 1.0),
            "mean": _mean_from_rows(article_rows, "sentiment_strength"),
            "desc1": "越大代表措辞越强烈",
            "desc2": "0 接近平淡，1 接近强烈表态",
            "color": COLORS["amber"],
        },
        "confidence": {
            "title": "confidence",
            "subtitle": "模型把握度",
            "range": (0.0, 1.0),
            "mean": _mean_from_rows(article_rows, "confidence"),
            "desc1": "越大表示模型越有把握",
            "desc2": "低值说明这条判断更不确定",
            "color": COLORS["green"],
        },
        "relevance": {
            "title": "relevance",
            "subtitle": "与股票的相关度",
            "range": (0.0, 1.0),
            "mean": _mean_from_rows(article_rows, "relevance"),
            "desc1": "越大越像“明确在讲这只股票”",
            "desc2": "低值更像行业/板块顺带提及",
            "color": COLORS["blue"],
        },
        "event_importance": {
            "title": "event_importance",
            "subtitle": "事件重要度",
            "range": (0.0, 1.0),
            "mean": _mean_from_rows(article_rows, "event_importance"),
            "desc1": "越大越可能影响股价或预期",
            "desc2": "业绩、回购、提价通常更高",
            "color": COLORS["green"],
        },
        "risk_flag": {
            "title": "risk_flag",
            "subtitle": "风险标记",
            "range": (0.0, 1.0),
            "mean": _mean_from_rows(article_rows, "risk_flag"),
            "desc1": "1 表示新闻中含明显风险提示",
            "desc2": "0 表示没有明确风险警示",
            "color": COLORS["red"],
        },
    }

    width, height = 1420, 920
    lines = _svg_header(width, height)
    lines.append('<text x="50" y="48" class="title">单条新闻数值化参数都是什么意思</text>')
    lines.append('<text x="50" y="74" class="subtitle">这张图回答的是：一条新闻经过 LLM 之后，为什么会变成一组数值，以及每个数值分别在表达什么。</text>')

    card_w, card_h = 420, 220
    start_x, start_y = 50, 120
    gap_x, gap_y = 30, 32
    keys = list(stats.keys())

    for idx, key in enumerate(keys):
        row = idx // 3
        col = idx % 3
        x = start_x + col * (card_w + gap_x)
        y = start_y + row * (card_h + gap_y)
        item = stats[key]
        lo, hi = item["range"]
        mean = item["mean"]
        bar_x = x + 26
        bar_y = y + 118
        bar_w = card_w - 52
        marker_x = _scale(mean, lo, hi, bar_x, bar_x + bar_w)

        lines.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="16" fill="{COLORS["panel"]}" stroke="{COLORS["line"]}"/>')
        lines.append(f'<text x="{x + 24}" y="{y + 34}" class="card-title">{item["title"]}</text>')
        lines.append(f'<text x="{x + 24}" y="{y + 58}" class="label">{item["subtitle"]}</text>')
        lines.append(f'<text x="{x + 24}" y="{y + 84}" class="small">{item["desc1"]}</text>')
        lines.append(f'<text x="{x + 24}" y="{y + 102}" class="small">{item["desc2"]}</text>')

        lines.append(f'<line x1="{bar_x}" x2="{bar_x + bar_w}" y1="{bar_y}" y2="{bar_y}" stroke="{COLORS["line"]}" stroke-width="8" stroke-linecap="round"/>')
        if key == "sentiment":
            zero_x = _scale(0.0, lo, hi, bar_x, bar_x + bar_w)
            lines.append(f'<line x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="{bar_y - 12}" y2="{bar_y + 12}" stroke="{COLORS["muted"]}" stroke-width="2"/>')
            lines.append(f'<text x="{zero_x:.1f}" y="{bar_y + 28}" text-anchor="middle" class="small">0</text>')
        lines.append(f'<circle cx="{marker_x:.1f}" cy="{bar_y}" r="9" fill="{item["color"]}"/>')
        lines.append(f'<text x="{bar_x}" y="{bar_y + 28}" text-anchor="start" class="small">{lo:.1f}</text>')
        lines.append(f'<text x="{bar_x + bar_w}" y="{bar_y + 28}" text-anchor="end" class="small">{hi:.1f}</text>')
        lines.append(f'<text x="{x + 24}" y="{y + 166}" class="small">当前样本平均值：{mean:.3f}</text>')

        if key == "risk_flag":
            lines.append(f'<text x="{x + 24}" y="{y + 188}" class="small">当前样本中约 {mean * 100:.1f}% 的新闻被标记为风险新闻。</text>')
        else:
            lines.append(f'<text x="{x + 24}" y="{y + 188}" class="small">这个参数会参与后面的文章权重与日频因子聚合。</text>')

    output_path.write_text(_svg_footer(lines), encoding="utf-8")


def render_numeric_to_factor_flow(output_path: Path) -> None:
    width, height = 1500, 860
    lines = _svg_header(width, height)
    lines.append('<text x="50" y="48" class="title">从数值化到因子，是怎么一步步做出来的</text>')
    lines.append('<text x="50" y="74" class="subtitle">这张图回答的是：单条新闻先变成哪些数值，再怎样聚合成股票级因子，最后怎么拿去检验。</text>')

    boxes = [
        (70, 120, 280, 110, "步骤1：新闻文本", ["标题、正文、日期", "例：提价、回购、业绩、风险"]),
        (420, 120, 320, 180, "步骤2：LLM 数值化", ["event_type", "sentiment / sentiment_strength", "confidence / relevance", "event_importance / risk_flag"]),
        (820, 120, 320, 180, "步骤3：单条新闻权重", ["weight = relevance × confidence", "× event_importance × (0.5+0.5×strength)", "× event_type_weight"]),
        (1180, 120, 250, 180, "步骤4：同日聚合", ["按 股票-日期 聚合", "得到当天的文章数、mention 数", "并计算加权平均情绪"]),
        (140, 400, 300, 220, "步骤5：基础日频因子", ["net_sentiment_factor", "negative_shock_factor", "attention_factor", "sentiment_momentum_factor", "composite_score"]),
        (560, 400, 360, 220, "步骤6：增强因子", ["ema3 / ema5 sentiment state", "negative_shock_carry", "event_novelty_factor", "state_composite_factor"]),
        (1020, 400, 360, 220, "步骤7：事件拆分因子", ["earnings_event_state", "operations_event_state", "financing_event_state", "market_buzz_event_state"]),
        (420, 690, 620, 110, "步骤8：因子检验", ["未来 1 / 3 / 5 个交易日收益", "看相关系数、高低分组收益差、方向命中率", "回答：哪个因子更有解释力"]),
    ]

    for x, y, w, h, title, bullets in boxes:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{COLORS["panel"]}" stroke="{COLORS["line"]}"/>')
        lines.append(f'<text x="{x + 22}" y="{y + 34}" class="card-title">{title}</text>')
        for idx, bullet in enumerate(bullets):
            lines.append(f'<text x="{x + 22}" y="{y + 64 + idx * 22}" class="label">{bullet}</text>')

    arrows = [
        ((350, 175), (420, 175)),
        ((740, 210), (820, 210)),
        ((1140, 210), (1180, 210)),
        ((1305, 300), (1305, 400)),
        ((1060, 210), (900, 400)),
        ((980, 300), (720, 400)),
        ((780, 300), (300, 400)),
        ((290, 620), (500, 745)),
        ((740, 620), (730, 690)),
        ((1200, 620), (960, 745)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS["blue"]}" stroke-width="3"/>')
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 10
        arrow_a = angle - math.pi / 6
        arrow_b = angle + math.pi / 6
        xa = x2 - arrow_len * math.cos(arrow_a)
        ya = y2 - arrow_len * math.sin(arrow_a)
        xb = x2 - arrow_len * math.cos(arrow_b)
        yb = y2 - arrow_len * math.sin(arrow_b)
        lines.append(f'<polygon points="{x2},{y2} {xa:.1f},{ya:.1f} {xb:.1f},{yb:.1f}" fill="{COLORS["blue"]}"/>')

    lines.append('<text x="1080" y="760" class="small">要点：单条新闻阶段主要解决“看懂文本”；股票级聚合阶段才真正进入“因子”这一步。</text>')
    output_path.write_text(_svg_footer(lines), encoding="utf-8")


def render_factor_system_map(output_path: Path) -> None:
    width, height = 1560, 980
    lines = _svg_header(width, height)
    lines.append('<text x="50" y="48" class="title">第三章因子体系总览图</text>')
    lines.append('<text x="50" y="74" class="subtitle">这张图回答的是：从单条新闻数值化开始，最后到底形成了哪些因子，以及哪些因子带有时间累计影响。</text>')

    def draw_box(x: int, y: int, w: int, h: int, title: str, bullets: list[str], fill: str = COLORS["panel"]) -> None:
        lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="{COLORS["line"]}"/>')
        lines.append(f'<text x="{x + 22}" y="{y + 34}" class="card-title">{title}</text>')
        for idx, bullet in enumerate(bullets):
            lines.append(f'<text x="{x + 22}" y="{y + 64 + idx * 22}" class="label">{bullet}</text>')

    def draw_arrow(x1: int, y1: int, x2: int, y2: int, color: str = COLORS["blue"]) -> None:
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3"/>')
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 10
        arrow_a = angle - math.pi / 6
        arrow_b = angle + math.pi / 6
        xa = x2 - arrow_len * math.cos(arrow_a)
        ya = y2 - arrow_len * math.sin(arrow_a)
        xb = x2 - arrow_len * math.cos(arrow_b)
        yb = y2 - arrow_len * math.sin(arrow_b)
        lines.append(f'<polygon points="{x2},{y2} {xa:.1f},{ya:.1f} {xb:.1f},{yb:.1f}" fill="{color}"/>')

    draw_box(
        60,
        120,
        330,
        220,
        "第1层：单条新闻数值化",
        [
            "sentiment：情绪方向",
            "sentiment_strength：情绪强度",
            "confidence：模型把握度",
            "relevance：与股票相关度",
            "event_importance：事件重要度",
            "risk_flag：风险标记",
            "event_type：业绩/经营/风险等类别",
        ],
        fill="#eef6ff",
    )

    draw_box(
        460,
        120,
        300,
        180,
        "第2层：新闻权重",
        [
            "weight = relevance × confidence",
            "× event_importance",
            "× (0.5 + 0.5 × strength)",
            "× event_type_weight",
        ],
        fill="#f8fafc",
    )

    draw_box(
        820,
        120,
        320,
        210,
        "第3层：基础日频因子",
        [
            "net_sentiment_factor",
            "negative_shock_factor",
            "attention_factor",
            "sentiment_dispersion_factor",
            "event_density_factor",
            "risk_ratio / positive_ratio / negative_ratio",
        ],
        fill="#effcf5",
    )

    draw_box(
        1200,
        120,
        300,
        190,
        "第4层：时间变化因子",
        [
            "sentiment_delta_1d",
            "sentiment_trend_factor",
            "sentiment_momentum_factor",
            "回答：情绪是在改善还是走弱？",
        ],
        fill="#fff7ed",
    )

    draw_box(
        110,
        430,
        370,
        240,
        "第5层：时序累计因子",
        [
            "ema3_sentiment_state",
            "ema5_sentiment_state",
            "negative_shock_carry",
            "event_novelty_factor",
            "特点：不是只看今天，而是把前几天的",
            "情绪或利空压力累计到今天。",
        ],
        fill="#fff1f2",
    )

    draw_box(
        560,
        430,
        390,
        240,
        "第6层：事件拆分因子",
        [
            "earnings_event_factor：业绩事件",
            "operations_event_factor：经营事件",
            "financing_event_factor：投融资事件",
            "market_buzz_event_factor：市场舆情事件",
            "risk_event_factor：风险事件",
        ],
        fill="#f5f3ff",
    )

    draw_box(
        1030,
        430,
        420,
        240,
        "第7层：事件状态因子",
        [
            "earnings_event_state",
            "operations_event_state",
            "financing_event_state",
            "market_buzz_event_state",
            "risk_event_state",
            "特点：某一类事件是否在最近几天持续发酵。",
        ],
        fill="#eefcf3",
    )

    draw_box(
        340,
        760,
        360,
        150,
        "第8层：综合因子",
        [
            "composite_score：更偏当天综合值",
            "state_composite_factor：更偏累计状态值",
            "用于排序、比较和最终画像展示",
        ],
        fill="#eff6ff",
    )

    draw_box(
        820,
        760,
        430,
        150,
        "第9层：第三章重点展示建议",
        [
            "基础因子：净情绪、负面冲击、关注度",
            "累计因子：ema3/ema5、负面冲击记忆",
            "事件状态因子：经营/业绩/风险事件状态",
        ],
        fill="#f8fafc",
    )

    draw_arrow(390, 220, 460, 220)
    draw_arrow(760, 220, 820, 220)
    draw_arrow(1140, 220, 1200, 220)
    draw_arrow(970, 330, 870, 430)
    draw_arrow(980, 330, 1250, 430)
    draw_arrow(980, 330, 300, 430)
    draw_arrow(480, 550, 520, 550)
    draw_arrow(950, 550, 1030, 550)
    draw_arrow(1230, 670, 1040, 760)
    draw_arrow(300, 670, 520, 760)

    lines.append('<text x="80" y="710" class="small">关键理解 1：只有聚合到“股票-日期”之后，才真正进入因子这一步。</text>')
    lines.append('<text x="80" y="736" class="small">关键理解 2：本文已经不是只用当天舆情值，而是有“时序累计因子”和“事件状态因子”。</text>')
    lines.append('<text x="80" y="940" class="small">汇报时一句话总结：当前体系已经形成“单条新闻数值化 -> 基础因子 -> 时序累计因子 -> 事件状态因子 -> 综合因子”的完整链路。</text>')

    output_path.write_text(_svg_footer(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with ARTICLE_PATH.open("r", encoding="utf-8") as f:
        article_rows = list(csv.DictReader(f))
    render_parameter_meaning(article_rows, OUTPUT_DIR / "chapter3_numeric_parameters.svg")
    render_numeric_to_factor_flow(OUTPUT_DIR / "chapter3_numeric_to_factor_flow.svg")
    render_factor_system_map(OUTPUT_DIR / "chapter3_factor_system_map.svg")
    print(f"wrote: {OUTPUT_DIR / 'chapter3_numeric_parameters.svg'}")
    print(f"wrote: {OUTPUT_DIR / 'chapter3_numeric_to_factor_flow.svg'}")
    print(f"wrote: {OUTPUT_DIR / 'chapter3_factor_system_map.svg'}")


if __name__ == "__main__":
    main()
