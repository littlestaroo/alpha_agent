from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def merge_csv_group(paths: list[Path], output_path: Path) -> int:
    rows_written = 0
    fieldnames: list[str] | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = None
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                if reader.fieldnames is None:
                    continue
                if writer is None:
                    fieldnames = list(reader.fieldnames)
                    writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1

    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将多批 baseline 输出的 article/skipped CSV 合并成总表",
    )
    parser.add_argument(
        "--batch-output-dir",
        type=Path,
        required=True,
        help="存放批次输出前缀结果的目录",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="chapter3_baseline_q4_20stocks",
        help="批次输出文件的公共前缀",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="合并后的输出目录",
    )
    args = parser.parse_args()

    batch_output_dir: Path = args.batch_output_dir
    out_dir: Path = args.out_dir
    prefix: str = args.prefix

    article_paths = sorted(batch_output_dir.glob(f"{prefix}_batch_*_article_mentions.csv"))
    skipped_paths = sorted(batch_output_dir.glob(f"{prefix}_batch_*_skipped_news.csv"))

    merged_articles = out_dir / f"{prefix}_merged_article_mentions.csv"
    merged_skipped = out_dir / f"{prefix}_merged_skipped_news.csv"
    summary_path = out_dir / f"{prefix}_merge_summary.json"

    article_rows = merge_csv_group(article_paths, merged_articles)
    skipped_rows = merge_csv_group(skipped_paths, merged_skipped)

    summary = {
        "batch_output_dir": str(batch_output_dir),
        "article_file_count": len(article_paths),
        "skipped_file_count": len(skipped_paths),
        "merged_article_rows": article_rows,
        "merged_skipped_rows": skipped_rows,
        "merged_articles_path": str(merged_articles),
        "merged_skipped_path": str(merged_skipped),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
