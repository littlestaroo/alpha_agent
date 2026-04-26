#!/usr/bin/env python3
"""Run news_quant on selected Q4 candidate batches with resume support."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


DEFAULT_MANIFEST = Path(
    "/Users/star/Desktop/agent/data/prepared/q4_20stocks_top10_batches/manifest.json"
)
DEFAULT_UNIVERSE = Path(
    "/Users/star/Desktop/agent/data/stock_universe_thesis_20stocks.csv"
)
DEFAULT_OUT_DIR = Path("/Users/star/Desktop/agent/output/q4_20stocks_batches")
DEFAULT_PYTHON = Path("/Users/star/Desktop/agent/.venv_news/bin/python")


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_indices(
    batch_count: int,
    batch_index: int | None,
    start: int | None,
    end: int | None,
) -> list[int]:
    if batch_index is not None:
        if not 0 <= batch_index < batch_count:
            raise ValueError(f"batch_index {batch_index} 超出范围 0..{batch_count - 1}")
        return [batch_index]

    resolved_start = 0 if start is None else start
    resolved_end = batch_count - 1 if end is None else end
    if resolved_start < 0 or resolved_end >= batch_count or resolved_start > resolved_end:
        raise ValueError(
            f"批次区间非法: start={resolved_start}, end={resolved_end}, batch_count={batch_count}"
        )
    return list(range(resolved_start, resolved_end + 1))


def iter_selected_batches(manifest: dict, indices: Iterable[int]) -> list[dict]:
    selected = []
    manifest_batches = manifest["batches"]
    for idx in indices:
        selected.append(manifest_batches[idx])
    return selected


def output_prefix(out_dir: Path, batch_index: int) -> Path:
    return out_dir / f"chapter3_baseline_q4_20stocks_batch_{batch_index:03d}"


def expected_outputs(prefix: Path) -> dict[str, Path]:
    return {
        "article_mentions": Path(f"{prefix}_article_mentions.csv"),
        "daily_profiles": Path(f"{prefix}_daily_profiles.csv"),
        "skipped_news": Path(f"{prefix}_skipped_news.csv"),
    }


def build_command(
    python_bin: Path,
    batch_path: Path,
    universe_path: Path,
    out_prefix: Path,
    limit_per_batch: int,
) -> list[str]:
    command = [
        str(python_bin),
        "-m",
        "news_quant",
        "--news",
        str(batch_path),
        "--universe",
        str(universe_path),
        "--out",
        str(out_prefix),
    ]
    if limit_per_batch > 0:
        command.extend(["--limit", str(limit_per_batch)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按批次运行 20 只股票 2023Q4 候选新闻的 baseline"
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="批次 manifest JSON")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE), help="股票池 CSV")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON), help="虚拟环境 python")
    parser.add_argument("--batch-index", type=int, help="只跑单个批次")
    parser.add_argument("--start", type=int, help="起始批次编号")
    parser.add_argument("--end", type=int, help="结束批次编号（含）")
    parser.add_argument(
        "--limit-per-batch",
        type=int,
        default=0,
        help="每个批次最多处理多少条新闻；0 表示不限制",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="每批之间暂停秒数，用于控制 API 压力",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标 article_mentions 已存在，则跳过该批次",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的命令，不实际运行",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    universe_path = Path(args.universe)
    out_dir = Path(args.out_dir)
    python_bin = Path(args.python_bin)

    manifest = load_manifest(manifest_path)
    indices = build_indices(
        batch_count=manifest["batch_count"],
        batch_index=args.batch_index,
        start=args.start,
        end=args.end,
    )
    batches = iter_selected_batches(manifest, indices)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary: list[dict] = []
    for batch in batches:
        batch_index = batch["batch_index"]
        batch_path = Path(batch["path"])
        prefix = output_prefix(out_dir, batch_index)
        outputs = expected_outputs(prefix)
        command = build_command(
            python_bin=python_bin,
            batch_path=batch_path,
            universe_path=universe_path,
            out_prefix=prefix,
            limit_per_batch=args.limit_per_batch,
        )

        if args.skip_existing and outputs["article_mentions"].exists():
            print(f"[skip] batch_{batch_index:03d} -> 已存在 {outputs['article_mentions']}")
            run_summary.append(
                {
                    "batch_index": batch_index,
                    "status": "skipped_existing",
                    "article_mentions": str(outputs["article_mentions"]),
                }
            )
            continue

        print(f"[run] batch_{batch_index:03d}")
        print(" ".join(command))
        if args.dry_run:
            run_summary.append({"batch_index": batch_index, "status": "dry_run"})
            continue

        completed = subprocess.run(command, check=False)
        status = "ok" if completed.returncode == 0 else f"failed:{completed.returncode}"
        run_summary.append(
            {
                "batch_index": batch_index,
                "status": status,
                "return_code": completed.returncode,
                "article_mentions_exists": outputs["article_mentions"].exists(),
                "daily_profiles_exists": outputs["daily_profiles"].exists(),
                "skipped_news_exists": outputs["skipped_news"].exists(),
            }
        )

        if completed.returncode != 0:
            print(f"[error] batch_{batch_index:03d} 失败，停止后续批次", file=sys.stderr)
            break

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary_path = out_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "manifest": str(manifest_path),
                "universe": str(universe_path),
                "selected_batches": indices,
                "limit_per_batch": args.limit_per_batch,
                "dry_run": args.dry_run,
                "results": run_summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
