from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/star/Desktop/agent")
DEFAULT_MENTIONS = ROOT / "output" / "chapter3_baseline_thesis_2stocks_2023-11_top3_perstock_final_article_mentions.csv"
DEFAULT_SELECTED = ROOT / "output" / "chapter3_validation" / "chapter3_2stocks_diverse_mentions.csv"
DEFAULT_SUMMARY = ROOT / "output" / "chapter3_validation" / "chapter3_2stocks_diverse_mentions_summary.json"

MAX_PER_GROUP = 3


def to_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def normalize_event_types(row: dict[str, str]) -> list[str]:
    merged = (row.get("merged_event_types") or "").strip()
    if merged:
        return [part.strip() for part in merged.split("|") if part.strip()]
    event_type = (row.get("event_type") or "").strip()
    return [event_type] if event_type else ["其他"]


def mention_score(row: dict[str, str]) -> float:
    return (
        0.35 * to_float(row.get("relevance"))
        + 0.20 * to_float(row.get("confidence"))
        + 0.20 * to_float(row.get("event_importance"))
        + 0.15 * to_float(row.get("sentiment_strength"))
        + 0.05 * to_float(row.get("coarse_score"))
        + 0.05 * to_float(row.get("merged_event_count"), 1.0)
    )


def select_diverse_rows(rows: list[dict[str, str]], max_per_group: int = 3) -> list[dict[str, str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -mention_score(row),
            -to_float(row.get("relevance")),
            -to_float(row.get("confidence")),
            row.get("news_id", ""),
        ),
    )
    selected: list[dict[str, str]] = []
    covered_events: set[str] = set()

    # First pass: prefer new event types.
    for row in ordered:
        row_events = normalize_event_types(row)
        if any(event not in covered_events for event in row_events):
            selected.append(row)
            covered_events.update(row_events)
        if len(selected) >= max_per_group:
            return selected

    # Second pass: fill remaining slots by score.
    selected_ids = {row.get("news_id", "") for row in selected}
    for row in ordered:
        if row.get("news_id", "") in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row.get("news_id", ""))
        if len(selected) >= max_per_group:
            break

    return selected


def main(
    mentions_path: Path = DEFAULT_MENTIONS,
    selected_path: Path = DEFAULT_SELECTED,
    summary_path: Path = DEFAULT_SUMMARY,
    max_per_group: int = MAX_PER_GROUP,
) -> None:
    if not mentions_path.exists():
        raise FileNotFoundError(f"文章级结果不存在: {mentions_path}")

    with mentions_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        publish_date = (row.get("publish_date") or "").strip()
        ts_code = (row.get("ts_code") or "").strip()
        if not publish_date or not ts_code:
            continue
        row["selection_score"] = f"{mention_score(row):.4f}"
        row["diverse_event_types"] = "|".join(normalize_event_types(row))
        grouped[(publish_date, ts_code)].append(row)

    selected_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, object]] = []
    counts_by_stock: dict[str, int] = defaultdict(int)

    for (publish_date, ts_code), group_rows in sorted(grouped.items()):
        picked = select_diverse_rows(group_rows, max_per_group=max_per_group)
        selected_rows.extend(picked)
        counts_by_stock[ts_code] += len(picked)
        summary_rows.append(
            {
                "publish_date": publish_date,
                "ts_code": ts_code,
                "stock_name": group_rows[0].get("stock_name", ""),
                "input_count": len(group_rows),
                "selected_count": len(picked),
                "selected_event_types": sorted(
                    {event for row in picked for event in normalize_event_types(row)}
                ),
            }
        )

    selected_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selected_rows[0].keys()) if selected_rows else []
    with selected_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    summary = {
        "input_rows": len(rows),
        "selected_rows": len(selected_rows),
        "group_count": len(grouped),
        "max_per_group": max_per_group,
        "counts_by_stock": dict(sorted(counts_by_stock.items())),
        "groups": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="按日期-股票对文章级结果做事件多样性筛选，优先保留不同 event_type 的新闻",
    )
    parser.add_argument(
        "--mentions",
        type=Path,
        default=DEFAULT_MENTIONS,
        help="文章级结果 CSV 路径",
    )
    parser.add_argument(
        "--selected-out",
        type=Path,
        default=DEFAULT_SELECTED,
        help="筛选后的文章级结果 CSV 输出路径",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="筛选汇总 JSON 输出路径",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=MAX_PER_GROUP,
        help="每个 日期-股票 分组最多保留多少条",
    )
    args = parser.parse_args()
    main(
        mentions_path=args.mentions,
        selected_path=args.selected_out,
        summary_path=args.summary_out,
        max_per_group=args.max_per_group,
    )
