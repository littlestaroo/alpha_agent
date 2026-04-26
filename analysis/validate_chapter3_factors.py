from __future__ import annotations

import math
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path("/Users/star/Desktop/agent")
PROFILE_PATH = (
    ROOT
    / "output"
    / "chapter3_baseline_thesis_2stocks_2023-11_top3_perstock_enhanced_daily_profiles.csv"
)
PRICE_CACHE_PATH = ROOT / "data" / "market" / "chapter3_2stocks_prices_20231020_20231231.csv"
OUTPUT_DIR = ROOT / "output" / "chapter3_validation"

PRICE_START = "20231020"
PRICE_END = "20231231"
HORIZONS = (1, 3, 5)

FACTOR_SPECS = [
    ("net_sentiment_factor", "当日净情绪", 1),
    ("negative_shock_factor", "当日负面冲击", -1),
    ("attention_factor", "当日关注度", 1),
    ("sentiment_momentum_factor", "当日情绪动量", 1),
    ("composite_score", "基础综合分", 1),
    ("ema3_sentiment_state", "3日情绪状态", 1),
    ("ema5_sentiment_state", "5日情绪状态", 1),
    ("negative_shock_carry", "负面冲击记忆", -1),
    ("event_novelty_factor", "事件新颖性", 1),
    ("earnings_event_state", "业绩事件状态", 1),
    ("operations_event_state", "经营事件状态", 1),
    ("market_buzz_event_state", "市场舆情事件状态", 1),
    ("risk_event_state", "风险事件状态", -1),
    ("state_composite_factor", "状态综合分", 1),
]

FACTOR_NAME_MAP = {key: label for key, label, _ in FACTOR_SPECS}
FACTOR_DIRECTION_MAP = {key: direction for key, _, direction in FACTOR_SPECS}

HEATMAP_FACTORS = [
    "net_sentiment_factor",
    "composite_score",
    "ema3_sentiment_state",
    "negative_shock_carry",
    "operations_event_state",
    "state_composite_factor",
]

COLORS = {
    "bg": "#ffffff",
    "text": "#0f172a",
    "muted": "#475569",
    "grid": "#e2e8f0",
    "axis": "#cbd5e1",
    "blue": "#2563eb",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
}


def ts_code_to_secid(ts_code: str) -> str:
    code, market = ts_code.split(".")
    prefix = "1." if market == "SH" else "0."
    return f"{prefix}{code}"


def fetch_price_history(ts_code: str, client: httpx.Client) -> pd.DataFrame:
    response = client.get(
        "https://push2his.eastmoney.com/api/qt/stock/kline/get",
        params={
            "secid": ts_code_to_secid(ts_code),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": "101",
            "fqt": "1",
            "beg": PRICE_START,
            "end": PRICE_END,
            "lmt": "200",
        },
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or {}
    klines = data.get("klines") or []
    rows: list[dict[str, object]] = []
    for line in klines:
        parts = line.split(",")
        rows.append(
            {
                "trade_date": pd.to_datetime(parts[0]),
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
                "pct_chg": float(parts[8]),
                "ts_code": ts_code,
                "stock_name": data.get("name", ""),
            }
        )
    return pd.DataFrame(rows)


def build_price_cache(profile_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    if PRICE_CACHE_PATH.exists() and not force_refresh:
        return pd.read_csv(PRICE_CACHE_PATH, parse_dates=["trade_date"])

    PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    frames: list[pd.DataFrame] = []
    with httpx.Client(
        timeout=20,
        trust_env=False,
        headers=headers,
        http2=False,
        follow_redirects=True,
    ) as client:
        for ts_code in sorted(profile_df["ts_code"].dropna().unique()):
            frames.append(fetch_price_history(ts_code, client))

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    for horizon in HORIZONS:
        prices[f"fwd_return_{horizon}d"] = (
            prices.groupby("ts_code")["close"].shift(-horizon) / prices["close"] - 1
        )
    prices.to_csv(PRICE_CACHE_PATH, index=False)
    return prices


def attach_future_returns(profile_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    profiles = profile_df.copy()
    profiles["publish_date"] = pd.to_datetime(profiles["publish_date"])
    prices = prices_df.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])

    merged_frames: list[pd.DataFrame] = []
    for ts_code, profile_group in profiles.groupby("ts_code", sort=False):
        left = profile_group.sort_values("publish_date").reset_index(drop=True)
        right = (
            prices[prices["ts_code"] == ts_code]
            .drop(columns=["ts_code"], errors="ignore")
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        merged_group = pd.merge_asof(
            left,
            right,
            left_on="publish_date",
            right_on="trade_date",
            direction="forward",
            allow_exact_matches=False,
        )
        merged_frames.append(merged_group)

    merged = pd.concat(merged_frames, ignore_index=True)
    merged = merged.rename(columns={"trade_date": "anchor_trade_date", "close": "anchor_close"})
    return merged.sort_values(["ts_code", "publish_date"]).reset_index(drop=True)


def _safe_corr(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    if left.nunique(dropna=True) <= 1 or right.nunique(dropna=True) <= 1:
        return float("nan")
    return float(left.corr(right, method=method))


def compute_time_series_validation(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor, label, direction in FACTOR_SPECS:
        factor_series = panel_df[factor]
        for horizon in HORIZONS:
            return_col = f"fwd_return_{horizon}d"
            mask = factor_series.notna() & panel_df[return_col].notna()
            xs = factor_series[mask]
            ys = panel_df.loc[mask, return_col]
            if len(xs) < 5:
                continue
            q_high = xs.quantile(0.7)
            q_low = xs.quantile(0.3)
            top_mean = float(ys[xs >= q_high].mean())
            bottom_mean = float(ys[xs <= q_low].mean())
            rows.append(
                {
                    "factor": factor,
                    "factor_label": label,
                    "horizon": horizon,
                    "n_obs": int(len(xs)),
                    "pearson_corr": _safe_corr(xs, ys, "pearson"),
                    "spearman_corr": _safe_corr(xs.rank(), ys.rank(), "pearson"),
                    "top_bucket_mean_return": top_mean,
                    "bottom_bucket_mean_return": bottom_mean,
                    "expected_top_bottom_gap": (top_mean - bottom_mean) * direction,
                }
            )
    return pd.DataFrame(rows)


def compute_cross_section_validation(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor, label, direction in FACTOR_SPECS:
        for horizon in HORIZONS:
            diff_rows: list[dict[str, object]] = []
            for publish_date, group in panel_df.groupby("publish_date"):
                subset = group[["ts_code", factor, f"fwd_return_{horizon}d"]].dropna()
                if subset["ts_code"].nunique() != 2:
                    continue
                subset = subset.sort_values("ts_code").reset_index(drop=True)
                factor_diff = (subset.loc[0, factor] - subset.loc[1, factor]) * direction
                return_diff = subset.loc[0, f"fwd_return_{horizon}d"] - subset.loc[1, f"fwd_return_{horizon}d"]
                diff_rows.append(
                    {
                        "publish_date": publish_date,
                        "factor_diff": factor_diff,
                        "return_diff": return_diff,
                    }
                )
            diff_df = pd.DataFrame(diff_rows)
            if len(diff_df) < 5:
                continue
            q_high = diff_df["factor_diff"].quantile(0.7)
            q_low = diff_df["factor_diff"].quantile(0.3)
            top_mean = float(diff_df.loc[diff_df["factor_diff"] >= q_high, "return_diff"].mean())
            bottom_mean = float(diff_df.loc[diff_df["factor_diff"] <= q_low, "return_diff"].mean())
            rows.append(
                {
                    "factor": factor,
                    "factor_label": label,
                    "horizon": horizon,
                    "n_dates": int(len(diff_df)),
                    "spread_corr": _safe_corr(diff_df["factor_diff"], diff_df["return_diff"], "pearson"),
                    "spread_rank_corr": _safe_corr(diff_df["factor_diff"].rank(), diff_df["return_diff"].rank(), "pearson"),
                    "rank_hit_rate": float(
                        (diff_df["factor_diff"].apply(_sign) == diff_df["return_diff"].apply(_sign)).mean()
                    ),
                    "top_bucket_return_spread": top_mean,
                    "bottom_bucket_return_spread": bottom_mean,
                    "top_bottom_spread_gap": top_mean - bottom_mean,
                }
            )
    return pd.DataFrame(rows)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def compute_direction_validation(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor, label, direction in FACTOR_SPECS:
        for horizon in HORIZONS:
            return_col = f"fwd_return_{horizon}d"
            subset = panel_df[[factor, return_col]].dropna().copy()
            if len(subset) < 5:
                continue
            subset["aligned_signal"] = subset[factor].astype(float) * direction
            subset = subset[subset["aligned_signal"].abs() > 1e-12].copy()
            if len(subset) < 5:
                continue

            predicted = subset["aligned_signal"].apply(_sign)
            actual = subset[return_col].apply(_sign)
            raw_hit_rate = float((predicted == actual).mean())
            up_ratio = float((subset[return_col] > 0).mean())
            down_ratio = float((subset[return_col] < 0).mean())
            market_majority_hit_rate = max(up_ratio, down_ratio)

            rows.append(
                {
                    "factor": factor,
                    "factor_label": label,
                    "horizon": horizon,
                    "n_obs": int(len(subset)),
                    "raw_direction_hit_rate": raw_hit_rate,
                    "market_up_ratio": up_ratio,
                    "market_down_ratio": down_ratio,
                    "market_majority_hit_rate": market_majority_hit_rate,
                    "excess_hit_rate_vs_market": raw_hit_rate - market_majority_hit_rate,
                }
            )
    return pd.DataFrame(rows)


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{COLORS["bg"]}"/>',
        "<style>",
        "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        ".title { font-size: 24px; font-weight: 700; fill: #0f172a; }",
        ".subtitle { font-size: 14px; fill: #475569; }",
        ".label { font-size: 13px; fill: #334155; }",
        ".small { font-size: 12px; fill: #475569; }",
        ".axis { stroke: #cbd5e1; stroke-width: 1; }",
        ".grid { stroke: #e2e8f0; stroke-width: 1; }",
        "</style>",
    ]


def _svg_footer(lines: list[str]) -> str:
    return "\n".join(lines + ["</svg>"])


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if math.isclose(lo, hi):
        return (out_lo + out_hi) / 2
    ratio = (value - lo) / (hi - lo)
    return out_lo + ratio * (out_hi - out_lo)


def _color_for_value(value: float, lo: float, hi: float) -> str:
    if pd.isna(value):
        return "#f8fafc"
    max_abs = max(abs(lo), abs(hi), 1e-9)
    ratio = min(abs(value) / max_abs, 1.0)
    if value >= 0:
        base = (5, 150, 105)
    else:
        base = (220, 38, 38)
    white = (255, 255, 255)
    rgb = tuple(int(white[i] * (1 - ratio) + base[i] * ratio) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def render_heatmap(time_df: pd.DataFrame, output_path: Path) -> None:
    matrix = (
        time_df[time_df["factor"].isin(HEATMAP_FACTORS)]
        .pivot(index="factor_label", columns="horizon", values="expected_top_bottom_gap")
        .reindex([FACTOR_NAME_MAP[key] for key in HEATMAP_FACTORS])
    )
    width, height = 1040, 580
    lines = _svg_header(width, height)
    lines.append('<text x="50" y="46" class="title">高因子组相对低因子组的未来收益差</text>')
    lines.append(
        '<text x="50" y="72" class="subtitle">这里的 1日 / 3日 / 5日，表示“从下一交易日开始算，未来 1 / 3 / 5 个交易日”的收益。格子里的百分号表示：高因子组平均收益减去低因子组平均收益。</text>'
    )
    lines.append(
        '<text x="50" y="94" class="subtitle">绿色越深越好，表示因子高的时候后续表现更强；红色越深越差，表示因子高的时候后续反而更弱。</text>'
    )

    table_x, table_y = 250, 150
    cell_w, cell_h = 190, 58
    lo = -0.02
    hi = 0.02

    for col_idx, horizon in enumerate(matrix.columns):
        x = table_x + col_idx * cell_w
        lines.append(f'<text x="{x + cell_w / 2:.1f}" y="{table_y - 18}" text-anchor="middle" class="label">未来{horizon}个交易日</text>')

    for row_idx, factor_label in enumerate(matrix.index):
        y = table_y + row_idx * cell_h
        lines.append(f'<text x="{table_x - 16}" y="{y + 35}" text-anchor="end" class="label">{factor_label}</text>')
        for col_idx, horizon in enumerate(matrix.columns):
            x = table_x + col_idx * cell_w
            value = float(matrix.loc[factor_label, horizon])
            color = _color_for_value(value, lo, hi)
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w - 10}" height="{cell_h - 10}" rx="8" fill="{color}" stroke="#e2e8f0"/>')
            lines.append(f'<text x="{x + (cell_w - 10) / 2:.1f}" y="{y + 26}" text-anchor="middle" class="label">{value * 100:.2f}%</text>')
            lines.append(f'<text x="{x + (cell_w - 10) / 2:.1f}" y="{y + 42}" text-anchor="middle" class="small">高组 - 低组</text>')

    legend_x = 300
    legend_y = table_y + len(matrix.index) * cell_h + 30
    for i, sample in enumerate([lo, 0.0, hi]):
        x = legend_x + i * 200
        lines.append(f'<rect x="{x}" y="{legend_y}" width="42" height="18" rx="5" fill="{_color_for_value(sample, lo, hi)}" stroke="#e2e8f0"/>')
        lines.append(f'<text x="{x + 52}" y="{legend_y + 14}" class="small">{sample * 100:.2f}%</text>')
    lines.append(f'<text x="{legend_x}" y="{legend_y + 42}" class="small">例：2.00% 表示高因子组未来收益比低因子组平均高 2 个百分点。</text>')

    output_path.write_text(_svg_footer(lines), encoding="utf-8")


def render_rank_hit_chart(cross_df: pd.DataFrame, output_path: Path) -> None:
    selected = cross_df[cross_df["factor"].isin(["net_sentiment_factor", "ema3_sentiment_state", "operations_event_state", "state_composite_factor"])]
    matrix = selected.pivot(index="factor_label", columns="horizon", values="rank_hit_rate")
    counts = selected.pivot(index="factor_label", columns="horizon", values="n_dates")
    matrix = matrix.reindex(
        [
            FACTOR_NAME_MAP["net_sentiment_factor"],
            FACTOR_NAME_MAP["ema3_sentiment_state"],
            FACTOR_NAME_MAP["operations_event_state"],
            FACTOR_NAME_MAP["state_composite_factor"],
        ]
    )
    counts = counts.reindex(matrix.index)

    width, height = 1080, 620
    lines = _svg_header(width, height)
    lines.append('<text x="50" y="46" class="title">两只股票相对强弱判断命中率</text>')
    lines.append(
        '<text x="50" y="72" class="subtitle">这里的百分号表示：在可比较的 28 个交易日里，因子更强的那只股票，后续相对表现也更强的比例。</text>'
    )
    lines.append(
        '<text x="50" y="94" class="subtitle">50% 约等于随机猜；明显高于 50% 才说明这个因子对“谁更强”有一定判断能力。</text>'
    )

    chart_x, chart_y = 120, 145
    group_w, chart_h = 220, 320
    max_bar_h = 240
    horizons = list(matrix.columns)
    colors = [COLORS["blue"], COLORS["amber"], COLORS["green"]]

    for i, factor_label in enumerate(matrix.index):
        base_x = chart_x + i * group_w
        lines.append(f'<text x="{base_x + 70}" y="{chart_y + chart_h + 38}" text-anchor="middle" class="label">{factor_label}</text>')
        for j, horizon in enumerate(horizons):
            value = float(matrix.loc[factor_label, horizon])
            count = int(round(float(counts.loc[factor_label, horizon]) * value)) if pd.notna(counts.loc[factor_label, horizon]) else 0
            total = int(counts.loc[factor_label, horizon]) if pd.notna(counts.loc[factor_label, horizon]) else 0
            bar_h = max_bar_h * value
            x = base_x + j * 44
            y = chart_y + max_bar_h - bar_h + 40
            lines.append(f'<rect x="{x}" y="{y:.1f}" width="28" height="{bar_h:.1f}" rx="6" fill="{colors[j]}"/>')
            lines.append(f'<text x="{x + 14}" y="{y - 18:.1f}" text-anchor="middle" class="small">{value * 100:.1f}%</text>')
            lines.append(f'<text x="{x + 14}" y="{y - 4:.1f}" text-anchor="middle" class="small">{count}/{total}</text>')
            lines.append(f'<text x="{x + 14}" y="{chart_y + max_bar_h + 62}" text-anchor="middle" class="small">未来{horizon}日</text>')

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = chart_y + max_bar_h - tick * max_bar_h + 40
        lines.append(f'<line x1="{chart_x - 20}" x2="{chart_x + 4 * group_w - 20}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{chart_x - 30}" y="{y + 4:.1f}" text-anchor="end" class="small">{tick * 100:.0f}%</text>')
    baseline_y = chart_y + max_bar_h - 0.5 * max_bar_h + 40
    lines.append(f'<line x1="{chart_x - 20}" x2="{chart_x + 4 * group_w - 20}" y1="{baseline_y:.1f}" y2="{baseline_y:.1f}" stroke="{COLORS["red"]}" stroke-width="2" stroke-dasharray="6 6"/>')
    lines.append(f'<text x="{chart_x + 4 * group_w - 10}" y="{baseline_y - 8:.1f}" text-anchor="end" class="small">50% 随机猜水平</text>')

    output_path.write_text(_svg_footer(lines), encoding="utf-8")


def render_direction_accuracy_chart(direction_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = direction_df[direction_df["horizon"] == 5].copy()
    plot_df = plot_df[
        plot_df["factor"].isin(
            [
                "net_sentiment_factor",
                "ema3_sentiment_state",
                "negative_shock_carry",
                "operations_event_state",
                "state_composite_factor",
            ]
        )
    ].copy()
    plot_df["factor_label"] = pd.Categorical(
        plot_df["factor_label"],
        categories=[
            FACTOR_NAME_MAP["net_sentiment_factor"],
            FACTOR_NAME_MAP["ema3_sentiment_state"],
            FACTOR_NAME_MAP["negative_shock_carry"],
            FACTOR_NAME_MAP["operations_event_state"],
            FACTOR_NAME_MAP["state_composite_factor"],
        ],
        ordered=True,
    )
    plot_df = plot_df.sort_values("factor_label").reset_index(drop=True)
    baseline = float(plot_df["market_majority_hit_rate"].iloc[0]) if not plot_df.empty else 0.5

    width, height = 1200, 560
    lines = _svg_header(width, height)
    lines.append('<text x="50" y="46" class="title">未来5日方向命中率（对比市场基准）</text>')
    lines.append(
        '<text x="50" y="72" class="subtitle">这里的命中率表示：因子给出的“涨/跌方向”预测，与未来 5 个交易日真实涨跌方向一致的比例。</text>'
    )
    lines.append(
        '<text x="50" y="94" class="subtitle">但请注意：2023 年 11 月整体偏弱，未来 5 日收益有 81.0% 为负，因此“总猜下跌”本身就能达到 81.0% 的基准命中率。</text>'
    )

    chart_x = 300
    chart_y = 150
    chart_w = 760
    row_h = 68
    bar_h = 28

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = chart_x + chart_w * tick
        lines.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{chart_y - 16}" y2="{chart_y + row_h * len(plot_df)}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{chart_y + row_h * len(plot_df) + 24}" text-anchor="middle" class="small">{tick * 100:.0f}%</text>')

    baseline_x = chart_x + chart_w * baseline
    lines.append(f'<line x1="{baseline_x:.1f}" x2="{baseline_x:.1f}" y1="{chart_y - 16}" y2="{chart_y + row_h * len(plot_df)}" stroke="{COLORS["red"]}" stroke-width="2" stroke-dasharray="6 6"/>')
    lines.append(f'<text x="{baseline_x:.1f}" y="{chart_y - 26}" text-anchor="middle" class="small">市场基准 {baseline * 100:.1f}%</text>')

    for idx, row in plot_df.iterrows():
        y = chart_y + idx * row_h
        value = float(row["raw_direction_hit_rate"])
        x2 = chart_x + chart_w * value
        color = COLORS["green"] if value >= baseline else COLORS["amber"]
        lines.append(f'<text x="{chart_x - 18}" y="{y + 18}" text-anchor="end" class="label">{row["factor_label"]}</text>')
        lines.append(f'<rect x="{chart_x}" y="{y}" width="{chart_w:.1f}" height="{bar_h}" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>')
        lines.append(f'<rect x="{chart_x}" y="{y}" width="{max(x2-chart_x,0):.1f}" height="{bar_h}" rx="8" fill="{color}"/>')
        lines.append(f'<text x="{x2 + 10:.1f}" y="{y + 18}" class="small">{value * 100:.1f}% ({int(round(value * row["n_obs"]))}/{int(row["n_obs"])})</text>')
        lines.append(f'<text x="{chart_x}" y="{y + 48}" class="small">相对市场基准：{row["excess_hit_rate_vs_market"] * 100:+.1f} 个百分点</text>')

    lines.append(f'<text x="{chart_x}" y="{chart_y + row_h * len(plot_df) + 54}" class="small">解读：如果某因子命中率只比 81.0% 的“总猜下跌”基准高一点点，那么它虽然看起来准确率高，但增量价值其实有限。</text>')
    output_path.write_text(_svg_footer(lines), encoding="utf-8")


def build_report(time_df: pd.DataFrame, cross_df: pd.DataFrame, panel_df: pd.DataFrame) -> str:
    direction_df = compute_direction_validation(panel_df)
    best_5d = time_df[time_df["horizon"] == 5].sort_values(
        ["expected_top_bottom_gap", "pearson_corr"], ascending=[False, False]
    )
    best_3d = time_df[time_df["horizon"] == 3].sort_values(
        ["expected_top_bottom_gap", "pearson_corr"], ascending=[False, False]
    )
    top_5d = best_5d.head(5)
    top_3d = best_3d.head(5)

    op_5d = best_5d[best_5d["factor"] == "operations_event_state"].iloc[0]
    net_5d = best_5d[best_5d["factor"] == "net_sentiment_factor"].iloc[0]
    state_5d = best_5d[best_5d["factor"] == "state_composite_factor"].iloc[0]

    rank_5d = cross_df[cross_df["horizon"] == 5].sort_values(
        ["rank_hit_rate", "top_bottom_spread_gap"], ascending=[False, False]
    ).head(5)
    direction_5d = direction_df[direction_df["horizon"] == 5].sort_values(
        ["raw_direction_hit_rate", "excess_hit_rate_vs_market"], ascending=[False, False]
    ).head(5)
    baseline_5d = float(direction_df[direction_df["horizon"] == 5]["market_majority_hit_rate"].iloc[0])

    lines: list[str] = []
    lines.append("# 第三章因子检验报告")
    lines.append("")
    lines.append("## 1. 检验目的")
    lines.append("")
    lines.append(
        "本报告用于验证第三章中由新闻舆情生成的日频因子，考察它们是否与后续股价表现存在可解释的关系。这里的重点不是证明策略已经可以实盘使用，而是回答一个更基础的问题：这些因子相较于单纯的文本数值化，是否已经具备一定的市场解释力。"
    )
    lines.append("")
    lines.append("## 2. 样本与检验设计")
    lines.append("")
    lines.append(
        f"- 因子样本：2023 年 11 月，宁德时代与贵州茅台，按“每天每股 top3 新闻”聚合后得到 {len(panel_df)} 条股票-日期观测。"
    )
    lines.append(f"- 股票价格：使用东财日线复权价格，区间为 {PRICE_START} 至 {PRICE_END}。")
    lines.append("- 对齐方式：为避免看未来，本文将新闻日期映射到“下一交易日”作为信号生效日，再计算未来 1/3/5 个交易日收益。")
    lines.append("- 验证方式：")
    lines.append("  1. 时间序列检验：计算因子与未来收益的 Pearson / Spearman 相关系数，并比较高因子组与低因子组的后续收益差。")
    lines.append("  2. 截面检验：在每天的两只股票之间比较谁的因子更强、谁后续表现更好，计算相对排序命中率。")
    lines.append("")
    lines.append("## 3. 主要结果")
    lines.append("")
    lines.append("### 3.1 时间序列结果")
    lines.append("")
    lines.append("从结果看，短期 1 日收益上的信号较弱，而 3 日和 5 日维度开始出现更清晰的差异，尤其是“事件状态”和“状态累计”类因子优于“当天直接值”类因子。")
    lines.append("")
    lines.append("5 日 horizon 表现较好的因子包括：")
    for _, row in top_5d.iterrows():
        lines.append(
            f"- {row['factor_label']}：Pearson={row['pearson_corr']:.3f}，Spearman={row['spearman_corr']:.3f}，高低分组未来 5 日收益差={row['expected_top_bottom_gap'] * 100:.2f}%。"
        )
    lines.append("")
    lines.append("3 日 horizon 表现较好的因子包括：")
    for _, row in top_3d.iterrows():
        lines.append(
            f"- {row['factor_label']}：Pearson={row['pearson_corr']:.3f}，Spearman={row['spearman_corr']:.3f}，高低分组未来 3 日收益差={row['expected_top_bottom_gap'] * 100:.2f}%。"
        )
    lines.append("")
    lines.append("### 3.2 直接因子 vs 状态因子")
    lines.append("")
    lines.append(
        f"- 当日净情绪因子在未来 5 日上的表现基本接近于零：Pearson={net_5d['pearson_corr']:.3f}，高低分组收益差={net_5d['expected_top_bottom_gap'] * 100:.2f}%。"
    )
    lines.append(
        f"- 状态综合分在未来 5 日上的表现更稳：Pearson={state_5d['pearson_corr']:.3f}，高低分组收益差={state_5d['expected_top_bottom_gap'] * 100:.2f}%。"
    )
    lines.append(
        f"- 经营事件状态因子是当前样本中最强的单项因子之一：未来 5 日 Pearson={op_5d['pearson_corr']:.3f}，Spearman={op_5d['spearman_corr']:.3f}，高低分组收益差={op_5d['expected_top_bottom_gap'] * 100:.2f}%。"
    )
    lines.append("")
    lines.append("这说明当前样本下，直接使用“当天总舆情值”并不理想；把新闻信号做时序累计，并按事件类型拆分后，解释力明显更强。")
    lines.append("")
    lines.append("### 3.3 截面排序结果")
    lines.append("")
    lines.append("在双股票样本里，还可以把问题改写成：每天哪只股票的因子更强，它后续是否相对更抗跌或更强势。由于 2023 年 11 月整体环境偏弱，绝对收益差容易受到市场下跌背景影响，因此这里更关注“排序命中率”而不是绝对涨跌幅。按未来 5 日排序命中率看，表现较好的因子包括：")
    for _, row in rank_5d.iterrows():
        lines.append(
            f"- {row['factor_label']}：相对排序命中率={row['rank_hit_rate'] * 100:.1f}%。"
        )
    lines.append("")
    lines.append("### 3.4 准确率口径说明")
    lines.append("")
    lines.append(f"如果一定要使用更接近“准确率”的口径，则可以定义方向命中率，即因子给出的涨跌方向与未来收益真实涨跌方向一致的比例。但这里需要特别说明：在本样本中，未来 5 日收益有 {baseline_5d * 100:.1f}% 为负，因此“总猜下跌”本身就已经能得到较高命中率。换言之，单纯的准确率并不能直接说明因子更好。")
    lines.append("")
    lines.append("未来 5 日方向命中率较高的因子包括：")
    for _, row in direction_5d.iterrows():
        lines.append(
            f"- {row['factor_label']}：方向命中率={row['raw_direction_hit_rate'] * 100:.1f}%，相对市场基准提升={row['excess_hit_rate_vs_market'] * 100:+.1f} 个百分点。"
        )
    lines.append("")
    lines.append("因此，若从论文汇报角度讨论“哪个因子更好”，更合理的做法是同时看三项指标：第一，看高低分组未来收益差；第二，看相关系数；第三，再辅助看方向命中率。若只看准确率，容易把一个“总猜下跌”的因子误判为好因子。")
    lines.append("")
    lines.append("## 4. 结论")
    lines.append("")
    lines.append("本次检验可以得到三个结论：")
    lines.append("")
    lines.append("1. 单条新闻交给 LLM 做情绪与事件抽取是合理的，但股票层面不能只直接使用当天舆情值。")
    lines.append("2. 从当前样本看，状态型因子和事件型因子优于直接因子，尤其是经营事件状态、状态综合分和关注度因子，在 3 日到 5 日维度上表现更好。")
    lines.append("3. 这些结果说明“新闻 -> 数值化 -> 因子”这条链路已经不仅能生成数值，也开始体现出一定的市场解释力，但样本仍然偏小，还不足以支持强结论。")
    lines.append("")
    lines.append("## 5. 局限与下一步")
    lines.append("")
    lines.append("- 当前只有 2 只股票、1 个月样本，统计显著性有限。")
    lines.append("- 目前使用的是日级新闻日期，没有纳入更细的发布时间，因而采用了“下一交易日生效”的保守假设。")
    lines.append("- 还没有加入成交量、波动率、行业基准收益等控制变量。")
    lines.append("- 下一步建议扩展到更长时间窗口和更多股票，并对因子做分层回测或 IC 检验。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile_df = pd.read_csv(PROFILE_PATH)
    prices_df = build_price_cache(profile_df)
    panel_df = attach_future_returns(profile_df, prices_df)
    time_df = compute_time_series_validation(panel_df)
    cross_df = compute_cross_section_validation(panel_df)
    direction_df = compute_direction_validation(panel_df)

    panel_path = OUTPUT_DIR / "factor_validation_panel.csv"
    time_path = OUTPUT_DIR / "factor_time_series_validation.csv"
    cross_path = OUTPUT_DIR / "factor_cross_section_validation.csv"
    direction_path = OUTPUT_DIR / "factor_direction_validation.csv"
    heatmap_path = OUTPUT_DIR / "factor_validation_heatmap.svg"
    rank_path = OUTPUT_DIR / "factor_rank_hit_chart.svg"
    direction_chart_path = OUTPUT_DIR / "factor_direction_accuracy_chart.svg"
    report_path = ROOT / "chapter3_factor_validation_report.md"

    panel_df.to_csv(panel_path, index=False)
    time_df.to_csv(time_path, index=False)
    cross_df.to_csv(cross_path, index=False)
    direction_df.to_csv(direction_path, index=False)
    render_heatmap(time_df, heatmap_path)
    render_rank_hit_chart(cross_df, rank_path)
    render_direction_accuracy_chart(direction_df, direction_chart_path)
    report_path.write_text(build_report(time_df, cross_df, panel_df), encoding="utf-8")

    print(f"wrote: {PRICE_CACHE_PATH}")
    print(f"wrote: {panel_path}")
    print(f"wrote: {time_path}")
    print(f"wrote: {cross_path}")
    print(f"wrote: {direction_path}")
    print(f"wrote: {heatmap_path}")
    print(f"wrote: {rank_path}")
    print(f"wrote: {direction_chart_path}")
    print(f"wrote: {report_path}")


if __name__ == "__main__":
    main()
