from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将候选新闻 jsonl 按固定条数切分为可分批跑 baseline 的批次文件",
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        required=True,
        help="输入候选新闻 jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="批次输出目录",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="每个批次保留多少条新闻",
    )
    args = parser.parse_args()

    input_path: Path = args.input_path
    out_dir: Path = args.out_dir
    batch_size: int = max(1, args.batch_size)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    total = 0
    batch_index = 0
    current_count = 0
    current_handle = None
    current_path: Path | None = None
    manifest: list[dict[str, object]] = []

    try:
        with input_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                if not line.strip():
                    continue
                if current_handle is None or current_count >= batch_size:
                    if current_handle is not None and current_path is not None:
                        current_handle.close()
                        manifest.append(
                            {
                                "batch_index": batch_index - 1,
                                "path": str(current_path),
                                "record_count": current_count,
                            }
                        )
                    current_count = 0
                    current_path = out_dir / f"batch_{batch_index:03d}.jsonl"
                    current_handle = current_path.open("w", encoding="utf-8")
                    batch_index += 1

                current_handle.write(line)
                current_count += 1
                total += 1

        if current_handle is not None and current_path is not None:
            current_handle.close()
            manifest.append(
                {
                    "batch_index": batch_index - 1,
                    "path": str(current_path),
                    "record_count": current_count,
                }
            )
    finally:
        if current_handle is not None and not current_handle.closed:
            current_handle.close()

    summary = {
        "input_path": str(input_path),
        "batch_size": batch_size,
        "total_records": total,
        "batch_count": len(manifest),
        "batches": manifest,
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
