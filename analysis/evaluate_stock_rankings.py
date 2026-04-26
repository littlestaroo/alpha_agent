#!/usr/bin/env python3
"""Evaluate top-k stock rankings with future return metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path("/Users/star/Desktop/agent")

DEFAULT_RANKED = (
    ROOT
    / "output"
    / "q4_20stocks_batches"
    / "chapter3_baseline_q4_20stocks_batch_000_ranked.csv"
)
DEFAULT_PANEL = (
    ROOT
    / "output"
    / "q4_20stocks_batches"
    / "chapter3_baseline_q4_20stocks_batch_000_rank_panel.csv"
)
DEFAULT_DAILY = (
    ROOT
    / "output"
    / "q4_20stocks_batches"
    / "chapter3_baseline_q4_20stocks_batch_000_topk_performance.csv"
)
DEFAULT_SUMMARY = (
    ROOT
    / "output"
    / "q4_20stocks_batches"
    / "chapter3_baseline_q4_20stocks_batch_000_topk_performance_summary.json"
)
DEFAULT_PRICE_CACHE = ROOT / "data" / "market" / "q4_20stocks_prices_cache.csv"

PRICE_START = "20230920"
PRICE_END = "20240131"
HORIZONS = (1, 3, 5)


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
            "lmt": "400",
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
                "stock_name_px": data.get("name", ""),
            }
        )
    return pd.DataFrame(rows)


def build_price_cache(ts_codes: list[str], cache_path: Path, force_refresh: bool = False) -> pd.DataFrame:
    if cache_path.exists() and not force_refresh:
        prices = pd.read_csv(cache_path, parse_dates=["trade_date"])
        cached_codes = set(prices["ts_code"].dropna().unique())
        if set(ts_codes).issubset(cached_codes):
            return prices

    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    frames: list[pd.DataFrame] = []
    with httpx.Client(
        timeout=20,
        trust_env=False,
        headers=headers,
        http2=False,
        follow_redirects=True,
    ) as client:
        for ts_code in sorted(ts_codes):
            frames.append(fetch_price_history(ts_code, client))

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    for horizon in HORIZONS:
        prices[f"fwd_return_{horizon}d"] = (
            prices.groupby("ts_code")["close"].shift(-horizon) / prices["close"] - 1
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path, index=False)
    return prices


def attach_future_returns(ranked_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    ranked = ranked_df.copy()
    ranked["publish_date"] = pd.to_datetime(ranked["publish_date"])
    prices = prices_df.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])

    merged_frames: list[pd.DataFrame] = []
    for ts_code, rank_group in ranked.groupby("ts_code", sort=False):
        left = rank_group.sort_values("publish_date").reset_index(drop=True)
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
    return merged.sort_values(["publish_date", "rank_in_date", "ts_code"]).reset_index(drop=True)


def evaluate_topk(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for publish_date, group in panel_df.groupby("publish_date", sort=True):
        selected = group[group["selected_topk"] == 1].copy()
        if selected.empty:
            continue
        row: dict[str, object] = {
            "publish_date": pd.to_datetime(publish_date).date().isoformat(),
            "cross_section_size": int(group["ts_code"].nunique()),
            "topk_size": int(len(selected)),
        }
        for horizon in HORIZONS:
            return_col = f"fwd_return_{horizon}d"
            valid_group = group[group[return_col].notna()].copy()
            valid_selected = selected[selected[return_col].notna()].copy()
            if valid_group.empty or valid_selected.empty:
                row[f"topk_mean_return_{horizon}d"] = None
                row[f"universe_mean_return_{horizon}d"] = None
                row[f"topk_excess_return_{horizon}d"] = None
                row[f"bottomk_mean_return_{horizon}d"] = None
                row[f"top_bottom_spread_{horizon}d"] = None
                continue
            topk_mean = float(valid_selected[return_col].mean())
            universe_mean = float(valid_group[return_col].mean())
            bottomk_count = min(len(valid_selected), len(valid_group))
            bottomk = valid_group.sort_values("ranking_score", ascending=True).head(bottomk_count)
            bottomk_mean = float(bottomk[return_col].mean())
            row[f"topk_mean_return_{horizon}d"] = topk_mean
            row[f"universe_mean_return_{horizon}d"] = universe_mean
            row[f"topk_excess_return_{horizon}d"] = topk_mean - universe_mean
            row[f"bottomk_mean_return_{horizon}d"] = bottomk_mean
            row[f"top_bottom_spread_{horizon}d"] = topk_mean - bottomk_mean
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_performance(daily_df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "date_count": int(len(daily_df)),
        "horizons": {},
    }
    for horizon in HORIZONS:
        excess_col = f"topk_excess_return_{horizon}d"
        spread_col = f"top_bottom_spread_{horizon}d"
        topk_col = f"topk_mean_return_{horizon}d"
        universe_col = f"universe_mean_return_{horizon}d"
        subset = daily_df[[topk_col, universe_col, excess_col, spread_col]].dropna()
        if subset.empty:
            summary["horizons"][str(horizon)] = {
                "n_dates": 0,
                "avg_topk_return": None,
                "avg_universe_return": None,
                "avg_excess_return": None,
                "avg_top_bottom_spread": None,
                "hit_rate_vs_universe": None,
            }
            continue
        summary["horizons"][str(horizon)] = {
            "n_dates": int(len(subset)),
            "avg_topk_return": float(subset[topk_col].mean()),
            "avg_universe_return": float(subset[universe_col].mean()),
            "avg_excess_return": float(subset[excess_col].mean()),
            "avg_top_bottom_spread": float(subset[spread_col].mean()),
            "hit_rate_vs_universe": float((subset[excess_col] > 0).mean()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 top-k 股票排序结果的未来收益表现")
    parser.add_argument("--ranked", type=Path, default=DEFAULT_RANKED, help="排序结果 CSV")
    parser.add_argument("--panel-out", type=Path, default=DEFAULT_PANEL, help="价格对齐后的面板 CSV")
    parser.add_argument("--daily-out", type=Path, default=DEFAULT_DAILY, help="按日期汇总的 top-k 表现 CSV")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY, help="表现摘要 JSON")
    parser.add_argument("--price-cache", type=Path, default=DEFAULT_PRICE_CACHE, help="价格缓存 CSV")
    parser.add_argument("--force-refresh", action="store_true", help="强制重新抓取价格数据")
    args = parser.parse_args()

    ranked_path = args.ranked
    if not ranked_path.exists():
        raise FileNotFoundError(f"排序结果不存在: {ranked_path}")

    ranked_df = pd.read_csv(ranked_path)
    ts_codes = sorted(ranked_df["ts_code"].dropna().unique())
    prices_df = build_price_cache(ts_codes, args.price_cache, force_refresh=args.force_refresh)
    panel_df = attach_future_returns(ranked_df, prices_df)
    daily_df = evaluate_topk(panel_df)
    summary = summarize_performance(daily_df)

    args.panel_out.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(args.panel_out, index=False)
    daily_df.to_csv(args.daily_out, index=False)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
